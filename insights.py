"""Layer-0 insight functions — deterministic answers computed from
retrieved data. The LLM never computes these; it narrates them.

Pure logic lives here (unit-testable, data injected); api.py provides
the fetchers and routes.
"""

from __future__ import annotations

from typing import Any, Optional

# ── Bullpen freshness (mirror of the app's card, server-side so the
#    chat pack and gameday DATA can carry it) ───────────────────────

def pen_rows(relievers: list[dict], day_keys: list[str]) -> list[dict]:
    """relievers: [{name, hand, pitches_by_date: {iso: n}}].
    day_keys: 3 ISO dates, today first. Returns tagged rows."""
    out = []
    for r in relievers:
        by = r.get("pitches_by_date") or {}
        days = [by.get(k) for k in day_keys]
        total = sum(d for d in days if d)
        b2b = (days[0] is not None and days[1] is not None) or \
              (days[1] is not None and days[2] is not None)
        if b2b or total >= 30:
            tag = "HEAVY"
        elif days[0] is None and days[1] is None and total <= 12:
            tag = "FRESH"
        else:
            tag = "WORKED"
        out.append({"name": r["name"], "hand": r.get("hand"),
                    "days": days, "total": total, "tag": tag})
    rank = {"FRESH": 0, "WORKED": 1, "HEAVY": 2}
    out.sort(key=lambda x: (rank[x["tag"]], x["total"], x["name"]))
    return out


# ── Milestone watch ─────────────────────────────────────────────────

# stat key -> (round step, "within" threshold)
_MILESTONE_STEPS_HIT = {"homeRuns": (10, 3), "hits": (50, 5),
                        "rbi": (25, 3), "stolenBases": (10, 2),
                        "runs": (25, 3)}
_MILESTONE_STEPS_PIT = {"strikeOuts": (50, 5), "wins": (5, 1),
                        "saves": (10, 2)}


def milestones(players: list[dict]) -> list[dict]:
    """players: [{name, group: 'hitting'|'pitching', season: {...},
    career_best: {...}}]. Returns labeled facts: approaching round
    numbers and career-high watches."""
    facts = []
    for p in players:
        steps = _MILESTONE_STEPS_HIT if p["group"] == "hitting" else _MILESTONE_STEPS_PIT
        season = p.get("season") or {}
        best = p.get("career_best") or {}
        for key, (step, within) in steps.items():
            v = season.get(key)
            if not isinstance(v, int) or v <= 0:
                continue
            nxt = ((v // step) + 1) * step
            gap = nxt - v
            if 0 < gap <= within:
                facts.append({"player": p["name"], "stat": key,
                              "current": v, "target": nxt,
                              "label": f"{p['name']} is {gap} {key} from {nxt}"})
            prior = best.get(key)
            if isinstance(prior, int) and prior > 0:
                if v > prior:
                    facts.append({"player": p["name"], "stat": key,
                                  "current": v, "target": prior,
                                  "label": f"{p['name']} has a career-high {v} {key} (prior best {prior})"})
                elif 0 < prior - v <= within:
                    facts.append({"player": p["name"], "stat": key,
                                  "current": v, "target": prior,
                                  "label": f"{p['name']} is {prior - v} {key} from his career high ({prior})"})
    return facts


# ── Generic roster comparison (the Clement pattern) ─────────────────

_COMPARE_WHITELIST = {
    "hitting": {"hits", "homeRuns", "rbi", "stolenBases", "baseOnBalls",
                "strikeOuts", "avg", "obp", "slg", "ops",
                "plateAppearances", "runs", "doubles"},
    "pitching": {"strikeOuts", "baseOnBalls", "era", "whip", "wins",
                 "saves", "inningsPitched", "homeRuns"},
}


def compare(roster: list[dict], group: str, stat: str, op: str,
            value: float) -> Optional[dict]:
    """roster: [{name, stats: {...}}] (season stats for `group`).
    Returns every player whose `stat` satisfies `op value`, with the
    values — an exhaustive, code-computed list."""
    if stat not in _COMPARE_WHITELIST.get(group, set()):
        return None
    ops = {"lt": lambda a: a < value, "gt": lambda a: a > value,
           "lte": lambda a: a <= value, "gte": lambda a: a >= value}
    fn = ops.get(op)
    if fn is None:
        return None
    matches, skipped = [], []
    for r in roster:
        raw = (r.get("stats") or {}).get(stat)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            skipped.append(r["name"])
            continue
        if fn(v):
            matches.append({"name": r["name"], "value": raw})
    matches.sort(key=lambda m: float(m["value"]),
                 reverse=op in ("gt", "gte"))
    return {"stat": stat, "op": op, "value": value,
            "matches": matches, "no_data": skipped}
