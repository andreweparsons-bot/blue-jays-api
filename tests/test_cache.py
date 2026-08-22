"""TTL cache: hit inside TTL, recompute after expiry, stale fallback on error."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import cache as C


def test_cache_hits_then_expires(monkeypatch):
    C._store.clear()
    calls = {"n": 0}
    now = {"t": 1000.0}
    monkeypatch.setattr(C.time, "time", lambda: now["t"])

    @C.cached(ttl_seconds=60)
    def f(x):
        calls["n"] += 1
        return x * 2

    assert f(2) == 4 and f(2) == 4
    assert calls["n"] == 1                 # second call served from cache
    now["t"] += 59
    f(2); assert calls["n"] == 1           # still inside TTL
    now["t"] += 2
    f(2); assert calls["n"] == 2           # expired → recomputed
    f(3); assert calls["n"] == 3           # different args → different key


def test_cache_serves_stale_on_upstream_error(monkeypatch):
    C._store.clear()
    now = {"t": 1000.0}
    monkeypatch.setattr(C.time, "time", lambda: now["t"])
    state = {"fail": False}

    @C.cached(ttl_seconds=10)
    def g():
        if state["fail"]:
            raise RuntimeError("upstream down")
        return "fresh"

    assert g() == "fresh"
    now["t"] += 100
    state["fail"] = True
    assert g() == "fresh"                  # stale value rather than an error
