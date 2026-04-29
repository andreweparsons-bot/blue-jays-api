# Blue Jays API

Lean FastAPI service that exposes Toronto Blue Jays / MLB stats to the Heechee iOS app via Anthropic tool calls. Sister project to `grimsby-dashboard`.

## Data sources

- **MLB Stats API** (`MLB-StatsAPI`) — standings, schedule, results, live game state, roster, basic player stats
- **Statcast** (`pybaseball.statcast_batter` / `statcast_pitcher`) — pitch-level event data, used for both quality-of-contact metrics (xwOBA, exit velocity, barrel%, hard-hit%, sprint speed, whiff%) and computed season aggregates (AVG/OBP/SLG/OPS/wOBA/ISO/BABIP/K%/BB%/FIP/K-9/BB-9 etc.)

> **Note**: Direct FanGraphs scraping (for wRC+, OPS+, ERA+, WAR, xFIP, SIERA) is currently blocked by Cloudflare protection — even with browser-equivalent headers, requests return 403 from a "Just a moment…" challenge page. Everything else is computed from Statcast events using the standard linear-weights formulas (FanGraphs 2024 wOBA constants) and FIP constant 3.10. wRC+ and WAR are not available because they need league-wide context (park factors, replacement-level baselines) we can't fetch.

## Endpoints

| Path | Source | TTL |
|---|---|---|
| `GET /health` | static | — |
| `GET /api/jays/season` | MLB Stats API | 10min |
| `GET /api/jays/schedule?n=10` | MLB Stats API | 10min |
| `GET /api/jays/results?n=10` | MLB Stats API | 10min |
| `GET /api/jays/live` | MLB Stats API | 30s |
| `GET /api/jays/roster` | MLB Stats API | 1hr |
| `GET /api/jays/player?name=...` | MLB Stats API | 1hr |
| `GET /api/jays/player/advanced?name=...` | Statcast (full season) | 12hr |
| `GET /api/jays/player/statcast?name=...&days=60` | Statcast (window) | 1hr |
| `GET /api/jays/recent-form?name=...&days=14` | Statcast (window) | 1hr |
| `GET /api/al-east-standings` | MLB Stats API | 10min |
| `GET /api/mlb-standings` | MLB Stats API | 10min |

All responses: `{"status": "ok", "data": ...}` or `{"status": "error", "data": {"detail": "..."}}` (status code stays 200 in both cases — the iOS client checks `status` not HTTP code).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
python start.py
# uvicorn on http://localhost:8000  (or PORT env var)
```

First call to a Statcast endpoint takes ~10s (cold cache); subsequent calls are instant from in-memory cache.

## Deploy
Railway, dockerfile builder. **Set the public networking port to `8080`** (matches the `PORT` env var Railway injects). The `Dockerfile`, `start.py`, and `railway.toml` are copied verbatim from the working `grimsby-dashboard` setup.
