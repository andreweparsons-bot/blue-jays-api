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
from typing import Any, Callable

log = logging.getLogger(__name__)

_store: dict[str, tuple[float, Any]] = {}


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
                    return value

            try:
                value = fn(*args, **kwargs)
                _store[key] = (now, value)
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
