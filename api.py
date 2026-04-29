"""
Blue Jays API — a lean FastAPI service exposing Toronto Blue Jays / MLB stats
for chat-tool consumption (Heechee iOS app via Anthropic tool calls).

Mirrors the deployment shape of grimsby-dashboard/api.py.

Data sources:
  - MLB Stats API (statsapi)        → standings, schedule, results, live, roster, basic stats
  - Statcast (pybaseball)           → xwOBA, exit velocity, launch angle, barrel%, hard-hit%, whiff%, spin
  - Computed-from-Statcast          → AVG, OBP, SLG, OPS, wOBA, ISO, BABIP, K%, BB%, FIP, K/9, BB/9 etc.

NOTE: FanGraphs scraping (wRC+, WAR, OPS+, ERA+, xFIP, SIERA) is currently blocked
by Cloudflare protection. We compute everything we can from Statcast events
(richer than basic MLB Stats API output) and document what's missing in the README.

All response shapes:  {"status": "ok", "data": ...}  or  {"status": "error", "data": {"detail": "..."}}.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import statsapi
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# pybaseball is heavy & noisy — silence its logger and disable its disk cache
# (Railway disks aren't persistent; we have our own in-process cache).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("blue-jays-api")

import pybaseball  # noqa: E402  (after logging config)

try:
    pybaseball.cache.disable()
except Exception:
    pass

from cache import cached  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────────

JAYS_TEAM_ID = 141            # MLB Stats API team id for Toronto Blue Jays
AL_EAST_DIV_ID = 201          # Division id (AL East)
ALL_LEAGUES = "103,104"       # AL=103, NL=104

# Statcast-derived thresholds
HARD_HIT_MPH = 95.0
SWEET_SPOT_MIN, SWEET_SPOT_MAX = 8.0, 32.0

# 2024 FanGraphs wOBA linear weights (close enough for current-season chat answers)
WOBA_W = {
    "walk": 0.690,
    "hit_by_pitch": 0.722,
    "single": 0.881,
    "double": 1.244,
    "triple": 1.567,
    "home_run": 2.011,
}
# FIP constant for ~2024 (varies year-to-year by 0.1 or so; close enough)
FIP_CONSTANT = 3.10


def current_season() -> int:
    """Season year. After Nov 15 we roll to the next year (offseason chat shouldn't break)."""
    today = datetime.now()
    if today.month >= 11 and today.day >= 15:
        return today.year + 1
    return today.year


def season_start(year: int | None = None) -> str:
    y = year or current_season()
    return f"{y}-03-15"  # spring training cutoff; covers any pre-Opening-Day pitches


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Blue Jays API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def ok(data: Any) -> dict:
    return {"status": "ok", "data": data}


def err(detail: str) -> dict:
    return {"status": "error", "data": {"detail": detail}}


def safe(value: Any) -> Any:
    """Convert NaN / numpy types to JSON-safe primitives."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item"):  # numpy scalar
        try:
            v = value.item()
            return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        except Exception:
            return value
    return value


def round_floats(d: dict, decimals: int = 3) -> dict:
    """Round float values for cleaner JSON. Preserves None."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if math.isnan(v) or math.isinf(v) else round(v, decimals)
        else:
            out[k] = safe(v)
    return out


# ── MLB Stats API helpers ───────────────────────────────────────────────────

@cached(ttl_seconds=600)
def _standings_data() -> dict:
    return statsapi.standings_data(leagueId=ALL_LEAGUES, season=current_season())


@cached(ttl_seconds=600)
def _team_record(team_id: int = JAYS_TEAM_ID) -> dict:
    """Find the Jays' row inside the standings_data tree."""
    data = _standings_data()
    for div_id, div in data.items():
        for row in div.get("teams", []):
            if row.get("team_id") == team_id:
                return {
                    "team": row.get("name"),
                    "wins": row.get("w"),
                    "losses": row.get("l"),
                    "win_pct": row.get("pct"),
                    "games_back": row.get("gb"),
                    "wildcard_games_back": row.get("wc_gb"),
                    "elimination_number": row.get("elim_num"),
                    "division_id": div_id,
                    "division_rank": row.get("div_rank"),
                    "league_rank": row.get("league_rank"),
                    "streak": row.get("streak"),
                    "last_ten": row.get("last_ten"),
                }
    return {}


@cached(ttl_seconds=600)
def _schedule(start_date: str, end_date: str) -> list[dict]:
    return statsapi.schedule(team=JAYS_TEAM_ID, start_date=start_date, end_date=end_date)


@cached(ttl_seconds=3600)
def _roster() -> list[dict]:
    raw = statsapi.roster(JAYS_TEAM_ID, rosterType="active")
    rows = []
    for line in raw.strip().splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            jersey, position, name = parts
            rows.append({
                "jersey": jersey.lstrip("#"),
                "position": position,
                "name": name.strip(),
            })
    return rows


def _format_game(g: dict) -> dict:
    is_jays_home = g.get("home_id") == JAYS_TEAM_ID
    opponent = g.get("away_name") if is_jays_home else g.get("home_name")
    jays_score = g.get("home_score") if is_jays_home else g.get("away_score")
    opp_score = g.get("away_score") if is_jays_home else g.get("home_score")
    return {
        "game_id": g.get("game_id"),
        "date": g.get("game_date"),
        "datetime": g.get("game_datetime"),
        "status": g.get("status"),
        "opponent": opponent,
        "venue": g.get("venue_name"),
        "is_home": is_jays_home,
        "jays_score": jays_score,
        "opponent_score": opp_score,
        "winning_team": g.get("winning_team"),
        "losing_team": g.get("losing_team"),
        "winning_pitcher": g.get("winning_pitcher"),
        "losing_pitcher": g.get("losing_pitcher"),
        "save_pitcher": g.get("save_pitcher"),
        "home_probable_pitcher": g.get("home_probable_pitcher"),
        "away_probable_pitcher": g.get("away_probable_pitcher"),
        "summary": g.get("summary"),
    }


# ── Player resolution ───────────────────────────────────────────────────────

def _name_match(roster_name: str, candidate: str) -> bool:
    """Loose match: handle Jr./Sr./III, accents, missing middle names."""
    rn = roster_name.lower().replace(".", "").replace(",", "")
    cn = candidate.lower().replace(".", "").replace(",", "")
    rt = set(rn.split())
    ct = set(cn.split())
    # Drop suffix tokens for comparison
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    rt -= suffixes
    ct -= suffixes
    return bool(rt & ct) and (rt.issubset(ct) or ct.issubset(rt))


@cached(ttl_seconds=24 * 3600)
def _resolve_player(name: str) -> dict | None:
    """Resolve name → {mlbam_id, full_name, on_jays}. Prefers active Jays."""
    name = name.strip()
    if " " not in name:
        last, first = name, ""
    else:
        parts = name.split()
        first = parts[0]
        last = " ".join(parts[1:])

    try:
        df = pybaseball.playerid_lookup(last, first, fuzzy=True)
    except Exception as e:
        log.warning("playerid_lookup failed for %r: %s", name, e)
        return None

    if df is None or df.empty:
        return None

    season = current_season()
    df = df.copy()
    df["mlb_played_last"] = pd.to_numeric(df["mlb_played_last"], errors="coerce")
    df = df[df["mlb_played_last"].notna() & (df["mlb_played_last"] >= season - 1)]
    if df.empty:
        return None

    roster_names = [r["name"] for r in _roster()]
    df["full_name"] = (
        df["name_first"].fillna("").str.title() + " " + df["name_last"].fillna("").str.title()
    ).str.strip()
    df["is_jay"] = df["full_name"].apply(
        lambda fn: any(_name_match(rn, fn) for rn in roster_names)
    )
    df = df.sort_values(by=["is_jay", "mlb_played_last"], ascending=[False, False])

    row = df.iloc[0]
    mlbam = int(row["key_mlbam"]) if pd.notna(row["key_mlbam"]) else None
    return {
        "mlbam_id": mlbam,
        "full_name": str(row["full_name"]),
        "on_jays": bool(row["is_jay"]),
    }


# ── Statcast: per-player season fetches ─────────────────────────────────────

@cached(ttl_seconds=12 * 3600)
def _statcast_batter_season(mlbam_id: int) -> pd.DataFrame:
    end = datetime.now().date()
    start_str = season_start()
    log.info("fetching Statcast batter %s [%s..%s]", mlbam_id, start_str, end)
    df = pybaseball.statcast_batter(start_str, end.isoformat(), mlbam_id)
    return df if df is not None else pd.DataFrame()


@cached(ttl_seconds=12 * 3600)
def _statcast_pitcher_season(mlbam_id: int) -> pd.DataFrame:
    end = datetime.now().date()
    start_str = season_start()
    log.info("fetching Statcast pitcher %s [%s..%s]", mlbam_id, start_str, end)
    df = pybaseball.statcast_pitcher(start_str, end.isoformat(), mlbam_id)
    return df if df is not None else pd.DataFrame()


@cached(ttl_seconds=3600)
def _statcast_batter_window(mlbam_id: int, days: int) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=days)
    log.info("fetching Statcast batter %s [%s..%s]", mlbam_id, start, end)
    df = pybaseball.statcast_batter(start.isoformat(), end.isoformat(), mlbam_id)
    return df if df is not None else pd.DataFrame()


@cached(ttl_seconds=3600)
def _statcast_pitcher_window(mlbam_id: int, days: int) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=days)
    log.info("fetching Statcast pitcher %s [%s..%s]", mlbam_id, start, end)
    df = pybaseball.statcast_pitcher(start.isoformat(), end.isoformat(), mlbam_id)
    return df if df is not None else pd.DataFrame()


# ── Stat aggregations ───────────────────────────────────────────────────────

# Statcast event values that count as Plate Appearances
PA_EVENTS = {
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play", "fielders_choice",
    "fielders_choice_out", "double_play", "triple_play",
    "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play",
    "field_error", "catcher_interf",
}
AB_EVENTS = PA_EVENTS - {"walk", "hit_by_pitch", "sac_fly", "sac_bunt",
                          "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}


def _aggregate_batter_basic(df: pd.DataFrame) -> dict:
    """Compute season AVG/OBP/SLG/OPS/wOBA/ISO/BABIP/K%/BB% from Statcast events."""
    if df is None or df.empty:
        return {}

    events = df["events"].dropna() if "events" in df.columns else pd.Series([], dtype=object)
    if events.empty:
        return {}

    counts = events.value_counts().to_dict()
    h = sum(counts.get(e, 0) for e in HIT_EVENTS)
    hr = counts.get("home_run", 0)
    bb = counts.get("walk", 0)
    hbp = counts.get("hit_by_pitch", 0)
    k = counts.get("strikeout", 0) + counts.get("strikeout_double_play", 0)
    sf = counts.get("sac_fly", 0) + counts.get("sac_fly_double_play", 0)

    pa = sum(counts.get(e, 0) for e in PA_EVENTS)
    ab = sum(counts.get(e, 0) for e in AB_EVENTS)

    singles = counts.get("single", 0)
    doubles = counts.get("double", 0)
    triples = counts.get("triple", 0)
    total_bases = singles + 2 * doubles + 3 * triples + 4 * hr

    avg = h / ab if ab else None
    obp = (h + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) else None
    slg = total_bases / ab if ab else None
    ops = (obp + slg) if (obp is not None and slg is not None) else None
    iso = (slg - avg) if (avg is not None and slg is not None) else None
    babip_denom = ab - k - hr + sf
    babip = ((h - hr) / babip_denom) if babip_denom > 0 else None

    # wOBA numerator/denominator
    woba_num = (
        WOBA_W["walk"] * (bb - counts.get("intent_walk", 0))  # intentional walks excluded
        + WOBA_W["hit_by_pitch"] * hbp
        + WOBA_W["single"] * singles
        + WOBA_W["double"] * doubles
        + WOBA_W["triple"] * triples
        + WOBA_W["home_run"] * hr
    )
    woba_denom = ab + bb - counts.get("intent_walk", 0) + sf + hbp
    woba = woba_num / woba_denom if woba_denom else None

    return round_floats({
        "pa": int(pa),
        "ab": int(ab),
        "h": int(h),
        "doubles": int(doubles),
        "triples": int(triples),
        "hr": int(hr),
        "bb": int(bb),
        "hbp": int(hbp),
        "k": int(k),
        "sf": int(sf),
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": ops,
        "iso": iso,
        "babip": babip,
        "woba": woba,
        "k_pct": (k / pa) if pa else None,
        "bb_pct": (bb / pa) if pa else None,
    }, decimals=3)


def _aggregate_batter_quality(df: pd.DataFrame) -> dict:
    """Statcast quality-of-contact aggregation (xwOBA, exit velo, barrel%, etc.)."""
    if df is None or df.empty:
        return {}
    bbe = df.dropna(subset=["launch_speed", "launch_angle"]) if "launch_speed" in df.columns else pd.DataFrame()
    bip = len(bbe)
    if bip == 0:
        return {"batted_balls": 0}

    barrels = (bbe["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bbe.columns else 0
    hard_hit = (bbe["launch_speed"] >= HARD_HIT_MPH).sum()
    sweet = bbe["launch_angle"].between(SWEET_SPOT_MIN, SWEET_SPOT_MAX).sum()

    xwoba_col = "estimated_woba_using_speedangle"
    xba_col = "estimated_ba_using_speedangle"
    xslg_col = "estimated_slg_using_speedangle"
    sprint = df["sprint_speed"].mean() if "sprint_speed" in df.columns else None

    return round_floats({
        "batted_balls": int(bip),
        "avg_exit_velocity": float(bbe["launch_speed"].mean()),
        "max_exit_velocity": float(bbe["launch_speed"].max()),
        "avg_launch_angle": float(bbe["launch_angle"].mean()),
        "barrel_pct": float(barrels) / bip if bip else None,
        "hard_hit_pct": float(hard_hit) / bip if bip else None,
        "sweet_spot_pct": float(sweet) / bip if bip else None,
        "xwoba": float(bbe[xwoba_col].mean()) if xwoba_col in bbe.columns else None,
        "xba": float(bbe[xba_col].mean()) if xba_col in bbe.columns else None,
        "xslg": float(bbe[xslg_col].mean()) if xslg_col in bbe.columns else None,
        "sprint_speed": float(sprint) if sprint is not None and not pd.isna(sprint) else None,
    }, decimals=3)


def _aggregate_pitcher_basic(df: pd.DataFrame) -> dict:
    """Pitcher season aggregation: K/9, BB/9, FIP, K-BB%, ERA-style from Statcast events."""
    if df is None or df.empty:
        return {}
    events = df["events"].dropna() if "events" in df.columns else pd.Series([], dtype=object)
    if events.empty:
        return {}

    counts = events.value_counts().to_dict()
    h = sum(counts.get(e, 0) for e in HIT_EVENTS)
    hr = counts.get("home_run", 0)
    bb = counts.get("walk", 0)
    hbp = counts.get("hit_by_pitch", 0)
    k = counts.get("strikeout", 0) + counts.get("strikeout_double_play", 0)

    pa = sum(counts.get(e, 0) for e in PA_EVENTS)
    ab = sum(counts.get(e, 0) for e in AB_EVENTS)

    # Rough IP estimate: outs / 3
    out_events = pa - h - bb - hbp - counts.get("catcher_interf", 0) - counts.get("field_error", 0)
    # Add double-play extra outs
    out_events += counts.get("grounded_into_double_play", 0)
    out_events += counts.get("strikeout_double_play", 0)
    out_events += counts.get("double_play", 0)
    out_events += 2 * counts.get("triple_play", 0)
    ip = out_events / 3.0 if out_events else 0.0

    fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT) if ip > 0 else None
    return round_floats({
        "ip": ip,
        "h": int(h),
        "hr": int(hr),
        "bb": int(bb),
        "hbp": int(hbp),
        "k": int(k),
        "pa_against": int(pa),
        "ab_against": int(ab),
        "k_per_9": (9 * k / ip) if ip > 0 else None,
        "bb_per_9": (9 * bb / ip) if ip > 0 else None,
        "k_pct": (k / pa) if pa else None,
        "bb_pct": (bb / pa) if pa else None,
        "k_minus_bb_pct": ((k - bb) / pa) if pa else None,
        "avg_against": (h / ab) if ab else None,
        "whip": ((h + bb) / ip) if ip > 0 else None,
        "fip": fip,
    }, decimals=3)


def _aggregate_pitcher_quality(df: pd.DataFrame) -> dict:
    """Pitcher quality from pitch-level data: velocity, spin, whiff%, swstr%."""
    if df is None or df.empty:
        return {}
    pitches = len(df)
    swing_descs = {"swinging_strike", "swinging_strike_blocked", "foul",
                   "hit_into_play", "foul_tip", "missed_bunt"}
    whiff_descs = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
    desc = df.get("description", pd.Series([], dtype=object))
    swings = int(desc.isin(swing_descs).sum())
    whiffs = int(desc.isin(whiff_descs).sum())
    return round_floats({
        "pitches": int(pitches),
        "avg_velocity_mph": float(df["release_speed"].mean()) if "release_speed" in df.columns else None,
        "max_velocity_mph": float(df["release_speed"].max()) if "release_speed" in df.columns else None,
        "avg_spin_rate_rpm": float(df["release_spin_rate"].mean()) if "release_spin_rate" in df.columns else None,
        "swings": swings,
        "whiffs": whiffs,
        "whiff_pct": (whiffs / swings) if swings else None,
        "swstr_pct": (whiffs / pitches) if pitches else None,
    }, decimals=3)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/jays/season")
def jays_season():
    try:
        rec = _team_record()
        if not rec:
            return JSONResponse(err("Jays not found in standings"), status_code=200)
        return ok(rec)
    except Exception as e:
        log.exception("season")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/schedule")
def jays_schedule(n: int = Query(10, ge=1, le=50)):
    try:
        today = datetime.now().date()
        end = today + timedelta(days=60)
        games = _schedule(today.isoformat(), end.isoformat())
        upcoming = [
            _format_game(g) for g in games
            if g.get("status") in ("Preview", "Scheduled", "Pre-Game", "Warmup", "Delayed Start")
        ][:n]
        return ok(upcoming)
    except Exception as e:
        log.exception("schedule")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/results")
def jays_results(n: int = Query(10, ge=1, le=50)):
    try:
        today = datetime.now().date()
        start = today - timedelta(days=45)
        games = _schedule(start.isoformat(), today.isoformat())
        finals = [
            _format_game(g) for g in games
            if g.get("status") in ("Final", "Game Over", "Completed Early")
        ]
        finals.reverse()  # newest first
        return ok(finals[:n])
    except Exception as e:
        log.exception("results")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/live")
def jays_live():
    try:
        today = datetime.now().date()
        games = _schedule(today.isoformat(), today.isoformat())
        live = next((g for g in games if g.get("status") in (
            "In Progress", "Manager challenge", "Umpire Review", "Delayed", "Suspended",
        )), None)
        if not live:
            return ok(None)

        gid = live.get("game_id")
        try:
            line = statsapi.linescore(gid)
        except Exception:
            line = None

        return ok({
            **_format_game(live),
            "current_inning": live.get("current_inning"),
            "inning_state": live.get("inning_state"),
            "linescore_text": line,
        })
    except Exception as e:
        log.exception("live")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/roster")
def jays_roster():
    try:
        return ok(_roster())
    except Exception as e:
        log.exception("roster")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/player")
def jays_player(name: str = Query(..., min_length=2)):
    """Basic season line via MLB Stats API (AVG/HR/RBI for hitters; W-L/ERA/WHIP for pitchers)."""
    try:
        resolved = _resolve_player(name)
        if not resolved or not resolved["mlbam_id"]:
            return JSONResponse(err(f"Player not found: {name}"), status_code=200)

        mlbam = resolved["mlbam_id"]
        hitting = statsapi.player_stat_data(mlbam, group="hitting", type="season")
        pitching = statsapi.player_stat_data(mlbam, group="pitching", type="season")

        out = {
            "name": resolved["full_name"],
            "mlbam_id": mlbam,
            "on_jays": resolved["on_jays"],
            "hitting": None,
            "pitching": None,
        }
        h_stats = (hitting.get("stats") or [None])[0] if hitting else None
        if h_stats and h_stats.get("stats"):
            out["hitting"] = h_stats["stats"]
        p_stats = (pitching.get("stats") or [None])[0] if pitching else None
        if p_stats and p_stats.get("stats"):
            out["pitching"] = p_stats["stats"]

        return ok(out)
    except Exception as e:
        log.exception("player")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/player/advanced")
def jays_player_advanced(name: str = Query(..., min_length=2)):
    """Season sabermetrics computed from Statcast events.

    Hitters get:  AVG/OBP/SLG/OPS/wOBA/ISO/BABIP/K%/BB%/xwOBA/xBA/xSLG/exit-velo/barrel%/hard-hit%
    Pitchers get: IP/K/BB/H/HR/K-9/BB-9/K%/BB%/K-BB%/AVG-against/WHIP/FIP/velocity/spin/whiff%
    """
    try:
        resolved = _resolve_player(name)
        if not resolved or not resolved["mlbam_id"]:
            return JSONResponse(err(f"Player not found: {name}"), status_code=200)

        mlbam = resolved["mlbam_id"]
        out: dict = {
            "name": resolved["full_name"],
            "mlbam_id": mlbam,
            "season": current_season(),
            "on_jays": resolved["on_jays"],
            "batting": None,
            "pitching": None,
        }

        try:
            bdf = _statcast_batter_season(mlbam)
            if not bdf.empty:
                merged = {**_aggregate_batter_basic(bdf), **_aggregate_batter_quality(bdf)}
                if merged:
                    out["batting"] = merged
        except Exception as e:
            log.info("batter season for %s: %s", mlbam, e)

        try:
            pdf = _statcast_pitcher_season(mlbam)
            if not pdf.empty:
                merged = {**_aggregate_pitcher_basic(pdf), **_aggregate_pitcher_quality(pdf)}
                if merged:
                    out["pitching"] = merged
        except Exception as e:
            log.info("pitcher season for %s: %s", mlbam, e)

        if out["batting"] is None and out["pitching"] is None:
            return JSONResponse(err(f"No Statcast season data for {resolved['full_name']}"), status_code=200)

        return ok(out)
    except Exception as e:
        log.exception("player/advanced")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/player/statcast")
def jays_player_statcast(
    name: str = Query(..., min_length=2),
    days: int = Query(60, ge=7, le=365),
):
    """Statcast quality-of-contact + velocity/spin metrics over a recent window."""
    try:
        resolved = _resolve_player(name)
        if not resolved or not resolved["mlbam_id"]:
            return JSONResponse(err(f"Player not found: {name}"), status_code=200)

        mlbam = resolved["mlbam_id"]
        out: dict = {
            "name": resolved["full_name"],
            "mlbam_id": mlbam,
            "window_days": days,
            "on_jays": resolved["on_jays"],
            "batting": None,
            "pitching": None,
        }

        try:
            bat = _statcast_batter_window(mlbam, days)
            agg = _aggregate_batter_quality(bat)
            if agg:
                out["batting"] = agg
        except Exception as e:
            log.info("statcast batter window %s: %s", mlbam, e)

        try:
            pit = _statcast_pitcher_window(mlbam, days)
            agg = _aggregate_pitcher_quality(pit)
            if agg:
                out["pitching"] = agg
        except Exception as e:
            log.info("statcast pitcher window %s: %s", mlbam, e)

        if out["batting"] is None and out["pitching"] is None:
            return JSONResponse(err(f"No Statcast events for {resolved['full_name']} in last {days} days"), status_code=200)

        return ok(out)
    except Exception as e:
        log.exception("player/statcast")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/jays/recent-form")
def jays_recent_form(
    name: str = Query(..., min_length=2),
    days: int = Query(14, ge=3, le=60),
):
    """Hot-streak / cold-streak: last N days BA/OBP/SLG/wOBA/exit-velo for hitters; rate stats for pitchers."""
    try:
        resolved = _resolve_player(name)
        if not resolved or not resolved["mlbam_id"]:
            return JSONResponse(err(f"Player not found: {name}"), status_code=200)

        mlbam = resolved["mlbam_id"]
        out: dict = {
            "name": resolved["full_name"],
            "mlbam_id": mlbam,
            "window_days": days,
            "on_jays": resolved["on_jays"],
            "batting": None,
            "pitching": None,
        }

        try:
            bat = _statcast_batter_window(mlbam, days)
            if bat is not None and not bat.empty:
                merged = {**_aggregate_batter_basic(bat), **_aggregate_batter_quality(bat)}
                if merged:
                    out["batting"] = merged
        except Exception as e:
            log.info("recent-form batter %s: %s", mlbam, e)

        try:
            pit = _statcast_pitcher_window(mlbam, days)
            if pit is not None and not pit.empty:
                merged = {**_aggregate_pitcher_basic(pit), **_aggregate_pitcher_quality(pit)}
                if merged:
                    out["pitching"] = merged
        except Exception as e:
            log.info("recent-form pitcher %s: %s", mlbam, e)

        if out["batting"] is None and out["pitching"] is None:
            return JSONResponse(err(f"No recent Statcast events for {resolved['full_name']}"), status_code=200)

        return ok(out)
    except Exception as e:
        log.exception("recent-form")
        return JSONResponse(err(str(e)), status_code=200)


# ── Standings ───────────────────────────────────────────────────────────────

def _div_table(div_id: int) -> list[dict]:
    data = _standings_data()
    rows = data.get(div_id, {}).get("teams", [])
    return [{
        "team": r.get("name"),
        "wins": r.get("w"),
        "losses": r.get("l"),
        "win_pct": r.get("pct"),
        "games_back": r.get("gb"),
        "wildcard_games_back": r.get("wc_gb"),
        "div_rank": r.get("div_rank"),
        "league_rank": r.get("league_rank"),
        "streak": r.get("streak"),
        "last_ten": r.get("last_ten"),
    } for r in rows]


@app.get("/api/al-east-standings")
def al_east_standings():
    try:
        return ok(_div_table(AL_EAST_DIV_ID))
    except Exception as e:
        log.exception("al-east-standings")
        return JSONResponse(err(str(e)), status_code=200)


@app.get("/api/mlb-standings")
def mlb_standings():
    try:
        data = _standings_data()
        out = {}
        for div_id, div in data.items():
            out[div.get("div_name", str(div_id))] = _div_table(div_id)
        return ok(out)
    except Exception as e:
        log.exception("mlb-standings")
        return JSONResponse(err(str(e)), status_code=200)
