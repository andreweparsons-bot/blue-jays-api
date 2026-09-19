"""
Pitch-by-pitch, SQL-queryable.

Andy, 2026-09-16: "I want the full pitch by pitch database to be SQL
queryable by the Johnny Heechee chat, so if I want to know how Ernie
Clement has done in 1-2 counts across the last month he should be able
to write a SQL query to that effect."

Shape of it
-----------
Every pitch thrown in a Blue Jays game is available from Statcast,
which is where this service's arsenal and locations endpoints already
get their data. It takes TWO pulls, not one:

  - `statcast(start, end, team='TOR')` gives Toronto PITCHING. Despite
    reading like a filter on "games involving Toronto", it returns only
    the half-innings where the other side bats. Measured on the live
    table: twenty-three thousand pitches, every one with the opposing
    team at the plate, and not a single Blue Jays hitter in it.
  - `statcast_batter(start, end, id)` per Jays hitter gives Toronto
    BATTING. There is no team-level equivalent, so it is one request
    per name on the active roster, pitchers excluded.

The two are concatenated and de-duplicated on the pitch itself
(game_pk, at_bat_number, pitch_number), because a Jay batting in a Jays
game appears in both pulls.

That frame is held in memory and queried with DuckDB, which reads a
pandas DataFrame directly with no copy. So there is no database to
provision, no volume to attach and nothing to migrate. The cost is that
a cold container re-pulls from Statcast, which takes a minute or two
for a season, so the frame is cached and refreshed on a TTL.

Two things had to be added on top of raw Statcast to make it usable
from a chat:

1. NAMES. Statcast identifies batter and pitcher by MLBAM id, and its
   own `player_name` column is the PITCHER. A question about Ernie
   Clement is unanswerable without a name column, so ids are resolved
   once per refresh and `batter_name` / `pitcher_name` are joined on.
2. A TRIMMED COLUMN SET. Statcast ships about ninety columns, most of
   them raw tracking values nobody will ask about. Keeping roughly
   thirty holds the memory down on a small Railway container, which is
   metered by the megabyte.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timedelta

import pandas as pd
import pybaseball
import statsapi

log = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:      # pragma: no cover
    duckdb = None

# The columns worth keeping, in the order a human would read them.
# Anything not listed here is dropped on load to keep the frame small.
COLUMNS = [
    # when and where
    "game_date", "game_pk", "game_year", "home_team", "away_team",
    "inning", "inning_topbot",
    # who
    "batter_name", "pitcher_name", "batter", "pitcher", "stand", "p_throws",
    # the situation, which is the point of the whole exercise
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "at_bat_number", "pitch_number",
    # the pitch
    "pitch_type", "pitch_name", "release_speed", "release_spin_rate",
    "plate_x", "plate_z", "zone",
    # what happened
    "description", "events", "bb_type",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "delta_run_exp",
]

#: A human-readable schema, handed to the model so it can write valid
#: SQL without a round trip to ask what the columns are.
SCHEMA_NOTE = """Table: pitches (one row per pitch in a Blue Jays game, both teams).
Columns:
  game_date DATE, game_pk INT, game_year INT, home_team TEXT, away_team TEXT,
  inning INT, inning_topbot TEXT ('Top'/'Bot'),
  batter_name TEXT, pitcher_name TEXT ('First Last'), batter INT, pitcher INT,
  stand TEXT ('L'/'R', the batter's side), p_throws TEXT ('L'/'R'),
  balls INT, strikes INT (the count BEFORE the pitch), outs_when_up INT,
  on_1b/on_2b/on_3b INT (runner's player id, NULL when the base is empty),
  at_bat_number INT, pitch_number INT (pitch within the at-bat),
  pitch_type TEXT ('FF','SL','CH'...), pitch_name TEXT ('4-Seam Fastball'...),
  release_speed DOUBLE (mph), release_spin_rate DOUBLE (rpm),
  plate_x DOUBLE, plate_z DOUBLE (feet, catcher's view), zone INT,
  description TEXT (per-pitch outcome: 'called_strike','swinging_strike','foul','hit_into_play','ball'...),
  events TEXT (at-bat result on the final pitch only: 'single','home_run','strikeout','walk','field_out'...; NULL on other pitches),
  bb_type TEXT ('ground_ball','line_drive','fly_ball','popup'),
  launch_speed DOUBLE (mph), launch_angle DOUBLE (degrees), hit_distance_sc DOUBLE (feet),
  estimated_woba_using_speedangle DOUBLE (xwOBA on contact), woba_value DOUBLE, woba_denom DOUBLE,
  delta_run_exp DOUBLE (run expectancy change on the pitch)
Notes:
  A 1-2 count is `balls = 1 AND strikes = 2`.
  Plate appearance outcomes live in `events` and are NULL except on the last pitch of the at-bat.
  For a batter's line use batter_name; for a pitcher's use pitcher_name.
  Toronto is 'TOR' in home_team/away_team."""

_lock = threading.Lock()
# One cached table per window (years_back), each with its own clock.
_frames: dict[int, pd.DataFrame] = {}
_loaded_at: dict[int, datetime] = {}
_loaded_range: dict[int, tuple[str, str]] = {}
_refreshing: set[int] = set()
_ready: dict[int, threading.Event] = {}
_TTL = timedelta(hours=6)
# How long a question will wait for a table that does not exist yet
# before being told the data is still loading. Kept well under the
# app's tool timeout so Johnny gets a clear answer, not a dead call.
_COLD_WAIT_SECONDS = 8


def _season_bounds(years_back: int = 0) -> tuple[str, str]:
    """Regular season runs late March to early November. Pull a whole
    calendar span rather than guessing exact dates."""
    today = datetime.now().date()
    start_year = today.year - years_back
    return f"{start_year}-03-01", today.isoformat()


JAYS_TEAM_ID = 141


def _jays_batter_ids() -> list[int]:
    """MLBAM ids for everyone on the active roster who bats.

    Needed because of a Statcast quirk that cost an afternoon: asking
    for `statcast(team='TOR')` returns ONLY the half-innings where the
    OTHER side is batting, i.e. Toronto pitching. Checked against the
    live table, every row came back with the opposing team at the plate
    and there was not one Blue Jays hitter in twenty-three thousand
    pitches. So the batting side has to be fetched per player."""
    try:
        raw = statsapi.get("team_roster", {"teamId": JAYS_TEAM_ID, "rosterType": "active"})
    except Exception as exc:                # pragma: no cover
        log.warning("roster lookup failed, batting side will be missing: %s", exc)
        return []
    ids = []
    for row in raw.get("roster", []):
        person = row.get("person") or {}
        pid = person.get("id")
        # Pitchers bat so rarely in the DH era that their plate
        # appearances are not worth a request each.
        code = ((row.get("position") or {}).get("abbreviation") or "").upper()
        if pid and code != "P":
            ids.append(int(pid))
    return ids


def _batting_side(start: str, end: str) -> pd.DataFrame:
    """Every pitch seen by a Blue Jays hitter, one request per hitter."""
    frames = []
    for pid in _jays_batter_ids():
        try:
            df = pybaseball.statcast_batter(start, end, pid)
        except Exception as exc:            # pragma: no cover
            log.warning("statcast_batter %s failed: %s", pid, exc)
            continue
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _attach_names(df: pd.DataFrame) -> pd.DataFrame:
    """Statcast gives ids; a chat needs names.

    One reverse lookup per refresh covers every batter and pitcher that
    appeared, which is a few hundred people. If the lookup fails the
    frame is still returned, with empty name columns, because a pitch
    table without names is worth more than no table."""
    ids = pd.unique(pd.concat([
        df["batter"].dropna(), df["pitcher"].dropna()
    ]).astype(int)) if {"batter", "pitcher"} <= set(df.columns) else []
    names: dict[int, str] = {}
    if len(ids):
        try:
            look = pybaseball.playerid_reverse_lookup(list(map(int, ids)), key_type="mlbam")
            for _, r in look.iterrows():
                first = str(r.get("name_first") or "").strip().title()
                last = str(r.get("name_last") or "").strip().title()
                full = f"{first} {last}".strip()
                if full:
                    names[int(r["key_mlbam"])] = full
        except Exception as exc:      # pragma: no cover
            log.warning("player name lookup failed, names will be blank: %s", exc)
    df["batter_name"] = df["batter"].map(names) if "batter" in df.columns else None
    df["pitcher_name"] = df["pitcher"].map(names) if "pitcher" in df.columns else None
    return df


def _fetch(years_back: int) -> tuple[pd.DataFrame, tuple[str, str]]:
    """The slow part: every Jays pitch in the window, both sides, named.
    Ten seconds warm and a good deal more cold, which is why nothing a
    user is waiting on may call this directly."""
    start, end = _season_bounds(years_back)
    log.info("fetching every Jays pitch [%s..%s]", start, end)
    # Two halves of the same season. The team pull gives Toronto
    # PITCHING; the per-hitter pull gives Toronto BATTING. Neither
    # on its own answers both kinds of question.
    pitching = pybaseball.statcast(start_dt=start, end_dt=end, team="TOR")
    batting = _batting_side(start, end)
    parts = [f for f in (pitching, batting) if f is not None and not f.empty]
    raw = pd.concat(parts, ignore_index=True) if parts else None
    if raw is None or raw.empty:
        log.warning("Statcast returned nothing for [%s..%s]", start, end)
        return pd.DataFrame(columns=COLUMNS), (start, end)
    # A hitter's pull and the team pull can both contain the same pitch
    # when a Jay bats in a Jays game, so key on the pitch itself.
    keys = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in raw.columns]
    if keys:
        raw = raw.drop_duplicates(subset=keys)
    raw = _attach_names(raw)
    keep = [c for c in COLUMNS if c in raw.columns]
    df = raw[keep].copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    log.info("pitch table ready: %s rows, %s columns", len(df), len(df.columns))
    return df, (start, end)


def _refresh(years_back: int) -> None:
    """Rebuild one window's table off the request path and swap it in."""
    try:
        df, rng = _fetch(years_back)
        with _lock:
            _frames[years_back] = df
            _loaded_at[years_back] = datetime.now()
            _loaded_range[years_back] = rng
    except Exception as exc:                  # pragma: no cover
        log.warning("pitch refresh failed, keeping the old table: %s", exc)
    finally:
        with _lock:
            _refreshing.discard(years_back)
            _ready.setdefault(years_back, threading.Event()).set()


def _kick(years_back: int) -> None:
    """Start a background refresh for a window unless one is running."""
    with _lock:
        if years_back in _refreshing:
            return
        _refreshing.add(years_back)
        _ready.setdefault(years_back, threading.Event()).clear()
    threading.Thread(target=_refresh, args=(years_back,), daemon=True,
                     name=f"pitches-refresh-{years_back}").start()


def warm() -> None:
    """Load this season's table at boot, so the first question of the
    day does not pay for it (Andy, 2026-09-19: Johnny could not run the
    analysis because the load happened inside his tool call)."""
    _kick(0)


def load(years_back: int = 0, force: bool = False) -> pd.DataFrame:
    """The pitch table for a window, WITHOUT ever blocking on a rebuild
    when a table already exists.

    This used to rebuild inline once the six hour cache lapsed, so the
    first question after a lapse waited on a full Statcast re-pull, ten
    seconds warm and more cold. The app gives a tool twelve seconds, so
    Johnny's call died and he told Andy he could not do it. Now a stale
    table is served at once and refreshed behind it; only a window that
    has never loaded waits, briefly, and then says so plainly."""
    with _lock:
        df = _frames.get(years_back)
        at = _loaded_at.get(years_back)
    stale = df is None or at is None or datetime.now() - at >= _TTL
    if stale or force:
        _kick(years_back)
    if df is not None:
        return df
    # Never loaded. Give it a moment, then report honestly.
    ready = _ready.setdefault(years_back, threading.Event())
    ready.wait(timeout=_COLD_WAIT_SECONDS)
    with _lock:
        df = _frames.get(years_back)
    if df is None:
        raise QueryError("warming: the pitch database is loading after a restart and "
                         "will be ready in about a minute. Tell the user that plainly "
                         "and offer to run the query again shortly.")
    return df


def status() -> dict:
    df = _frames.get(0)
    at = _loaded_at.get(0)
    rng = _loaded_range.get(0)
    return {
        "rows": 0 if df is None else int(len(df)),
        "columns": [] if df is None else list(df.columns),
        "loaded_at": at.isoformat() if at else None,
        "range": list(rng) if rng else None,
        "refreshing": 0 in _refreshing,
    }


# ── The query guard ──────────────────────────────────────────────────
#
# The SQL is written by a language model, so it is treated as untrusted
# input even though it cannot reach a real database: DuckDB here is
# pointed at an in-memory DataFrame, so there is nothing to drop and
# nothing to leak. The guard exists to keep a bad query from wedging the
# container or returning a reply too big to put in a chat.

_FORBIDDEN = re.compile(
    r"\b(attach|copy|install|load|export|import|create|insert|update|delete|drop|"
    r"alter|pragma|call|set|read_csv|read_parquet|read_json|glob)\b",
    re.IGNORECASE,
)
MAX_ROWS = 200


class QueryError(Exception):
    pass


def _guard(sql: str) -> str:
    q = sql.strip().rstrip(";").strip()
    if not q:
        raise QueryError("empty query")
    if ";" in q:
        raise QueryError("one statement at a time, no semicolons")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise QueryError("only SELECT (or WITH ... SELECT) is allowed")
    if _FORBIDDEN.search(q):
        raise QueryError("that query uses a keyword this endpoint does not allow")
    # A model that forgets LIMIT should not return forty thousand rows
    # into a chat message.
    if not re.search(r"\blimit\s+\d+", q, re.IGNORECASE):
        q = f"{q}\nLIMIT {MAX_ROWS}"
    return q


def run_sql(sql: str, years_back: int = 0) -> dict:
    """Run one read-only SELECT against the pitch table."""
    if duckdb is None:
        raise QueryError("duckdb is not installed on this service")
    guarded = _guard(sql)
    pitches = load(years_back=years_back)      # noqa: F841 — DuckDB reads it by name
    if pitches.empty:
        raise QueryError("the pitch table is empty, Statcast returned nothing")

    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"SET statement_timeout = '{20}s'")
    except Exception:
        pass                                    # older DuckDB, the row cap still applies
    try:
        con.register("pitches", pitches)
        out = con.execute(guarded).fetchdf()
    except Exception as exc:
        raise QueryError(str(exc).split("\n")[0]) from exc
    finally:
        con.close()

    if len(out) > MAX_ROWS:
        out = out.head(MAX_ROWS)
    # NaN is not JSON, and a chat reads "null" better than a crash.
    out = out.where(pd.notna(out), None)
    return {
        "sql": guarded,
        "row_count": int(len(out)),
        "rows": out.to_dict(orient="records"),
        "truncated": bool(len(out) >= MAX_ROWS),
    }
