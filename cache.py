"""
Tiny in-process TTL cache.

Used to avoid hammering MLB Stats API / FanGraphs / Statcast on every request.
Single-replica Railway service, so a process-local dict is enough — no Redis.

Usage:
    from cache import cached

    @cached(ttl_seconds=600)
    def expensive_lookup(team_id: int):
        return slow_upstream_call(team_id)

The decorator keys on (function name, args, sorted kwargs). On a hit within TTL,
returns the cached value; otherwise calls the function and stores the result.

If the upstream call raises, we serve stale cache (if any) rather than propagating
the error — preferring "slightly stale data" over "no data". Set strict=True to
disable that fallback.
"""

import time
import functools
import logging
import sys
from typing import Any, Callable

log = logging.getLogger(__name__)

# key -> (stored_at, value). Access times tracked separately for LRU.
_store: dict[str, tuple[float, Any]] = {}
_touched: dict[str, float] = {}

# -- Eviction policy --------------------------------------------------
# Solo-user service on a small Railway container: the cache must be a
# convenience, never a leak. pybaseball DataFrames are 10-50MB each and
# keys accumulate per player/season, so we bound BOTH entry count and
# approximate total bytes, evicting least-recently-used first.
MAX_ENTRIES = 200
MAX_BYTES = 500 * 1024 * 1024   # ~500MB of cached values


def _approx_bytes(value: Any) -> int:
    try:
        mem = getattr(value, "memory_usage", None)   # pandas DataFrame
        if callable(mem):
            return int(value.memory_usage(deep=True).sum())
    except Exception:
        pass
    try:
        return sys.getsizeof(value)
    except Exception:
        return 1024


def _evict_if_needed() -> None:
    total = sum(_approx_bytes(v) for _, v in _store.values())
    while _store and (len(_store) > MAX_ENTRIES or total > MAX_BYTES):
        oldest = min(_store, key=lambda k: _touched.get(k, 0.0))
        total -= _approx_bytes(_store[oldest][1])
        _store.pop(oldest, None)
        _touched.pop(oldest, None)
        log.info("cache evicted %s (entries=%d)", oldest.split(":")[0], len(_store))


def clear_heavy() -> int:
    # Emergency valve for the memory watchdog: drop every cached value
    # over ~1MB (the DataFrames), keep the cheap lookups.
    heavy = [k for k, (_, v) in _store.items() if _approx_bytes(v) > 1024 * 1024]
    for k in heavy:
        _store.pop(k, None)
        _touched.pop(k, None)
    return len(heavy)


def cached(ttl_seconds: int, strict: bool = False) -> Callable:
    """Cache the decorated function's return value for `ttl_seconds`.

    On upstream failure, falls back to the most recent cached value if any (unless strict).
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__name__}:{args}:{sorted(kwargs.items())}"
            now = time.time()
            entry = _store.get(key)

            if entry is not None:
                ts, value = entry
                if now - ts < ttl_seconds:
                    _touched[key] = now
                    return value

            try:
                value = fn(*args, **kwargs)
                _store[key] = (now, value)
                _touched[key] = now
                _evict_if_needed()
                return value
            except Exception as e:
                if not strict and entry is not None:
                    log.warning("upstream failed for %s, serving stale cache: %s", key, e)
                    return entry[1]
                raise

        return wrapper
    return decorator


def clear() -> None:
    """For tests / debugging."""
    _store.clear()
