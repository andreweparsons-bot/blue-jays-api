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


def test_eviction_bounds_entries(monkeypatch):
    C._store.clear(); C._touched.clear()
    monkeypatch.setattr(C, "MAX_ENTRIES", 5)
    now = {"t": 1000.0}
    monkeypatch.setattr(C.time, "time", lambda: now["t"])

    @C.cached(ttl_seconds=9999)
    def f(x):
        return x

    for i in range(10):
        now["t"] += 1
        f(i)
    assert len(C._store) <= 5
    # newest entries survive, oldest evicted
    assert any(":(9,)" in k for k in C._store)
    assert not any(":(0,)" in k for k in C._store)


def test_eviction_bounds_bytes(monkeypatch):
    C._store.clear(); C._touched.clear()
    monkeypatch.setattr(C, "MAX_BYTES", 10_000)
    now = {"t": 1000.0}
    monkeypatch.setattr(C.time, "time", lambda: now["t"])

    @C.cached(ttl_seconds=9999)
    def blob(i):
        return "x" * 5_000

    for i in range(6):
        now["t"] += 1
        blob(i)
    total = sum(C._approx_bytes(v) for _, v in C._store.values())
    assert total <= 10_000 + 6_000     # at most one entry over before eviction

def test_clear_heavy_drops_only_big_values():
    C._store.clear(); C._touched.clear()
    C._store["big:1"] = (0.0, "x" * (2 * 1024 * 1024))
    C._store["small:1"] = (0.0, "tiny")
    C._touched.update({"big:1": 1.0, "small:1": 1.0})
    dropped = C.clear_heavy()
    assert dropped == 1
    assert "small:1" in C._store and "big:1" not in C._store
