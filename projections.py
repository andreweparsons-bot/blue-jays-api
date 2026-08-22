"""
Heechee projections — Marcel-style systems computed from data we
already retrieve (MLB StatsAPI + Baseball Savant via pybaseball).

We do NOT reproduce Steamer or OOPSY (proprietary weights, park
factors, aging curves). We implement the public skeleton they share,
Tom Tango's Marcel, and label the output as ours:

  MARCEL   — box-score results. Seasons weighted 5/4/3 (current
             partial season, last year, two years ago); each rate
             regressed toward league average by adding REGRESS_PA
             plate appearances of league-average play; a Marcel age
             factor; rates scaled to SCALE_PA (600) so bench bats and
             regulars compare.
  MARCEL-X — the OOPSY idea on the same skeleton: AVG / SLG / wOBA
             driven by Statcast EXPECTED rates (xBA, xSLG, xwOBA from
             the Savant leaderboard for each season) instead of
             results; HR, SB, K%, BB% still from MARCEL (Savant's
             expected stats don't split those).

Pitchers: MARCEL on K%, BB%, HR% per batter faced (weights 3/2/1,
same regression), reported per 600 BF with FIP (league-calibrated
constant) — ERA is presented as FIP-equivalent, not a separate model.

wRC+ here is PARK-NEUTRAL: ((wOBA − lgwOBA) / wOBA_SCALE + lgR/PA)
/ lgR/PA × 100, with wOBA_SCALE an approximate constant. Every input
number is retrieved; the method is the only editorial content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

WEIGHTS_HIT = (5.0, 4.0, 3.0)       # current season, last, two ago
WEIGHTS_PIT = (3.0, 2.0, 1.0)
REGRESS_PA = 1200.0
REGRESS_BF = 1200.0
SCALE_PA = 600.0
SCALE_BF = 600.0
WOBA_SCALE = 1.25                   # approximate; disclosed in method
WOBA_W = {"ubb": 0.69, "hbp": 0.72, "single": 0.88,
          "double": 1.24, "triple": 1.57, "hr": 2.01}

HIT_COMPONENTS = ("single", "double", "triple", "hr", "ubb", "hbp",
                  "so", "sf", "sb", "cs")
PIT_COMPONENTS = ("so", "bb", "hbp", "hr", "h")


# ── Pure math ───────────────────────────────────────────────────────────────

def age_factor(age: Optional[float]) -> float:
    """Marcel's aging rule: peak at 29; +0.6%/yr younger, −0.3%/yr older."""
    if age is None:
        return 1.0
    if age < 29:
        return 1.0 + 0.006 * (29 - age)
    return 1.0 - 0.003 * (age - 29)


def regressed_rate(counts: list[float], exposures: list[float],
                   weights: tuple, league_rate: float, regress_n: float) -> float:
    """Weighted player rate pulled toward league_rate by regress_n
    exposures of league-average play. counts[i]/exposures[i] are the
    i-th season (most recent first); missing seasons are simply absent."""
    num = sum(w * c for w, c in zip(weights, counts)) + regress_n * league_rate
    den = sum(w * e for w, e in zip(weights, exposures)) + regress_n
    return num / den if den > 0 else league_rate


def weighted_league_rate(league_rates: list[float], exposures: list[float],
                         weights: tuple) -> float:
    """League rate blended the same way the player's seasons are, so a
    player whose 3 seasons span different run environments regresses
    toward the matching mix. Falls back to the most recent season."""
    num = sum(w * e * r for w, e, r in zip(weights, exposures, league_rates))
    den = sum(w * e for w, e in zip(weights, exposures))
    if den > 0:
        return num / den
    return league_rates[0] if league_rates else 0.0


def project_hitter(seasons: list[dict], league: list[dict], age: Optional[float],
                   weights=WEIGHTS_HIT, regress=REGRESS_PA, scale=SCALE_PA) -> dict:
    """seasons: [{pa, single, double, triple, hr, ubb, hbp, so, sf, sb, cs}],
    most recent first (current partial season first). league: same keys
    per matching season (rates derived from league totals)."""
    seasons = [s for s in seasons if s.get("pa", 0) > 0][:len(weights)]
    if not seasons:
        return {}
    exposures = [float(s["pa"]) for s in seasons]
    lg = league[:len(seasons)]
    af = age_factor(age)
    rates: dict[str, float] = {}
    for c in HIT_COMPONENTS:
        counts = [float(s.get(c, 0)) for s in seasons]
        lg_rates = [l.get(c, 0) / l["pa"] if l.get("pa") else 0.0 for l in lg]
        lgr = weighted_league_rate(lg_rates, exposures, weights)
        r = regressed_rate(counts, exposures, weights, lgr, regress)
        # Age: positive outcomes up for the young, strikeouts the
        # other way; neutral components untouched.
        if c in ("single", "double", "triple", "hr", "ubb", "hbp", "sb"):
            r *= af
        elif c == "so":
            r /= af
        rates[c] = r
    return _hitter_line(rates, scale, lg[0] if lg else {})


def _hitter_line(rates: dict, scale: float, league_now: dict) -> dict:
    n = {c: rates[c] * scale for c in HIT_COMPONENTS}
    h = n["single"] + n["double"] + n["triple"] + n["hr"]
    ab = scale - n["ubb"] - n["hbp"] - n["sf"]
    tb = n["single"] + 2 * n["double"] + 3 * n["triple"] + 4 * n["hr"]
    avg = h / ab if ab else 0.0
    obp = (h + n["ubb"] + n["hbp"]) / (ab + n["ubb"] + n["hbp"] + n["sf"])
    slg = tb / ab if ab else 0.0
    woba = (sum(WOBA_W[k] * n[k] for k in WOBA_W)
            / (ab + n["ubb"] + n["sf"] + n["hbp"]))
    out = {
        "pa": round(scale), "hr": round(n["hr"], 1), "sb": round(n["sb"], 1),
        "bb": round(n["ubb"], 1), "so": round(n["so"], 1),
        "avg": round(avg, 3), "obp": round(obp, 3), "slg": round(slg, 3),
        "ops": round(obp + slg, 3), "iso": round(slg - avg, 3),
        "woba": round(woba, 3),
        "k_pct": round(n["so"] / scale, 3), "bb_pct": round(n["ubb"] / scale, 3),
    }
    wrc = wrc_plus(woba, league_now)
    if wrc is not None:
        out["wrc_plus"] = wrc
    return out


def league_woba(league: dict) -> Optional[float]:
    pa = league.get("pa")
    if not pa:
        return None
    ab = pa - league.get("ubb", 0) - league.get("hbp", 0) - league.get("sf", 0)
    denom = ab + league.get("ubb", 0) + league.get("sf", 0) + league.get("hbp", 0)
    if denom <= 0:
        return None
    return sum(WOBA_W[k] * league.get(k, 0) for k in WOBA_W) / denom


def wrc_plus(woba: float, league: dict) -> Optional[int]:
    """Park-neutral wRC+ (100 = league average)."""
    lg_woba = league_woba(league)
    pa, runs = league.get("pa"), league.get("runs")
    if lg_woba is None or not pa or runs is None:
        return None
    r_pa = runs / pa
    if r_pa <= 0:
        return None
    return round(((woba - lg_woba) / WOBA_SCALE + r_pa) / r_pa * 100)


def project_hitter_x(seasons: list[dict], league: list[dict], age: Optional[float],
                     base: dict, weights=WEIGHTS_HIT, regress=REGRESS_PA) -> dict:
    """MARCEL-X: regress the player's Statcast EXPECTED rates (xba,
    xslg, xwoba per season, with that season's PA as exposure) toward
    the league's expected averages, age-adjust, and overwrite the
    contact-quality lines of the MARCEL projection `base`. seasons:
    [{pa, xba, xslg, xwoba}] most recent first; league: [{xba, xslg,
    xwoba}] per season (PA-weighted leaderboard means)."""
    usable = [(s, l) for s, l in zip(seasons, league)
              if s.get("pa", 0) > 0 and all(s.get(k) is not None for k in ("xba", "xslg", "xwoba"))]
    if not usable or not base:
        return {}
    usable = usable[:len(weights)]
    exposures = [float(s["pa"]) for s, _ in usable]
    af = age_factor(age)
    out = dict(base)
    for key in ("xba", "xslg", "xwoba"):
        # expected rate × PA = "expected count"; regress like a count
        counts = [s[key] * s["pa"] for s, _ in usable]
        lg_rates = [l.get(key, 0.0) for _, l in usable]
        lgr = weighted_league_rate(lg_rates, exposures, weights)
        r = regressed_rate(counts, exposures, weights, lgr, regress) * af
        out[{"xba": "avg", "xslg": "slg", "xwoba": "woba"}[key]] = round(r, 3)
    out["iso"] = round(out["slg"] - out["avg"], 3)
    # OBP: keep MARCEL walk/HBP rates, swap in the expected hit rate.
    obp_delta = out["avg"] - base["avg"]
    out["obp"] = round(base["obp"] + obp_delta * 0.9, 3)   # hits' share of OBP denom ≈ AB/PA
    out["ops"] = round(out["obp"] + out["slg"], 3)
    lg_now = league[0] if league else {}
    if "lg_totals" in lg_now:
        w = wrc_plus(out["woba"], lg_now["lg_totals"])
        if w is not None:
            out["wrc_plus"] = w
    out["source"] = "xBA/xSLG/xwOBA regressed; HR, SB, K%, BB% from MARCEL"
    return out


def project_pitcher(seasons: list[dict], league: list[dict], age: Optional[float],
                    weights=WEIGHTS_PIT, regress=REGRESS_BF, scale=SCALE_BF) -> dict:
    """seasons: [{bf, so, bb, hbp, hr, h, ip}] most recent first; league:
    [{bf, so, bb, hbp, hr, h, ip, er}] per season."""
    seasons = [s for s in seasons if s.get("bf", 0) > 0][:len(weights)]
    if not seasons:
        return {}
    exposures = [float(s["bf"]) for s in seasons]
    lg = league[:len(seasons)]
    af = age_factor(age)
    rates: dict[str, float] = {}
    for c in PIT_COMPONENTS:
        counts = [float(s.get(c, 0)) for s in seasons]
        lg_rates = [l.get(c, 0) / l["bf"] if l.get("bf") else 0.0 for l in lg]
        lgr = weighted_league_rate(lg_rates, exposures, weights)
        r = regressed_rate(counts, exposures, weights, lgr, regress)
        # Younger arms: more K, fewer BB/HR; older the reverse.
        if c == "so":
            r *= af
        elif c in ("bb", "hr", "hbp", "h"):
            r /= af
        rates[c] = r
    n = {c: rates[c] * scale for c in PIT_COMPONENTS}
    lg_now = lg[0] if lg else {}
    ip_per_bf = (lg_now.get("ip", 0) / lg_now["bf"]) if lg_now.get("bf") else 0.26
    ip = scale * ip_per_bf
    cfip = fip_constant(lg_now)
    fip = ((13 * n["hr"] + 3 * (n["bb"] + n["hbp"]) - 2 * n["so"]) / ip + cfip) if ip else None
    return {
        "bf": round(scale), "ip": round(ip, 1),
        "so": round(n["so"], 1), "bb": round(n["bb"], 1), "hr": round(n["hr"], 1),
        "k_pct": round(n["so"] / scale, 3), "bb_pct": round(n["bb"] / scale, 3),
        "k_per_9": round(n["so"] / ip * 9, 2) if ip else None,
        "bb_per_9": round(n["bb"] / ip * 9, 2) if ip else None,
        "hr_per_9": round(n["hr"] / ip * 9, 2) if ip else None,
        "fip": round(fip, 2) if fip is not None else None,
        "whip": round((n["h"] + n["bb"]) / ip, 2) if ip else None,
    }


def fip_constant(league: dict) -> float:
    """cFIP = lgERA − (13·HR + 3·(BB+HBP) − 2·K) / IP, from league totals."""
    ip = league.get("ip")
    if not ip:
        return 3.10
    era = (league.get("er", 0) * 9) / ip
    return era - (13 * league.get("hr", 0) + 3 * (league.get("bb", 0) + league.get("hbp", 0))
                  - 2 * league.get("so", 0)) / ip


# ── Shape adapters (StatsAPI rows → component dicts) ────────────────────────

def hitter_components(stat: dict) -> dict:
    """StatsAPI hitting stat dict → component counts."""
    h = int(stat.get("hits", 0) or 0)
    d = int(stat.get("doubles", 0) or 0)
    t = int(stat.get("triples", 0) or 0)
    hr = int(stat.get("homeRuns", 0) or 0)
    bb = int(stat.get("baseOnBalls", 0) or 0)
    ibb = int(stat.get("intentionalWalks", 0) or 0)
    return {
        "pa": int(stat.get("plateAppearances", 0) or 0),
        "single": h - d - t - hr, "double": d, "triple": t, "hr": hr,
        "ubb": bb - ibb, "hbp": int(stat.get("hitByPitch", 0) or 0),
        "so": int(stat.get("strikeOuts", 0) or 0),
        "sf": int(stat.get("sacFlies", 0) or 0),
        "sb": int(stat.get("stolenBases", 0) or 0),
        "cs": int(stat.get("caughtStealing", 0) or 0),
        "runs": int(stat.get("runs", 0) or 0),
    }


def pitcher_components(stat: dict) -> dict:
    ip_raw = str(stat.get("inningsPitched", "0") or "0")
    whole, _, frac = ip_raw.partition(".")
    ip = int(whole or 0) + {"1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)
    return {
        "bf": int(stat.get("battersFaced", 0) or 0),
        "so": int(stat.get("strikeOuts", 0) or 0),
        "bb": int(stat.get("baseOnBalls", 0) or 0),
        "hbp": int(stat.get("hitByPitch", 0) or 0),
        "hr": int(stat.get("homeRuns", 0) or 0),
        "h": int(stat.get("hits", 0) or 0),
        "ip": ip, "er": int(stat.get("earnedRuns", 0) or 0),
    }


def merge_seasons(splits: list[dict], adapter) -> dict[int, dict]:
    """yearByYear splits (one per team-stint) → {season: summed components}."""
    out: dict[int, dict] = {}
    for sp in splits:
        try:
            season = int(sp.get("season"))
        except (TypeError, ValueError):
            continue
        comp = adapter(sp.get("stat") or {})
        acc = out.setdefault(season, {})
        for k, v in comp.items():
            acc[k] = acc.get(k, 0) + v
    return out


METHOD = {
    "marcel": {
        "label": "Marcel",
        "summary": "Box-score results: seasons weighted 5/4/3, regressed toward "
                   "league average with 1200 PA of average play, Marcel aging "
                   "(peak 29), scaled to 600 PA. Park-neutral.",
    },
    "marcel_x": {
        "label": "Marcel-X",
        "summary": "Same skeleton on Statcast EXPECTED rates: xBA, xSLG and "
                   "xwOBA (Savant, per season) regressed and aged drive AVG/SLG/"
                   "wOBA/wRC+; HR, SB, K% and BB% come from Marcel.",
    },
    "pitching": {
        "label": "Marcel (pitching)",
        "summary": "K, BB, HR and hits per batter faced, seasons weighted 3/2/1, "
                   "regressed with 1200 BF, aged; FIP with a league-calibrated "
                   "constant. ERA shown as FIP-equivalent.",
    },
    "wrc_plus": "Park-neutral: ((wOBA − lg wOBA) / 1.25 + lg R/PA) / lg R/PA × 100.",
}
