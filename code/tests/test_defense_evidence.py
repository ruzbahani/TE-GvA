import math

import numpy as np
import pytest

from cpsdefense.core.types import EvidenceItem
from cpsdefense.defense.evidence import EvidenceLog
from cpsdefense.model.evidence import derived_validity, validity


def item(eid, arrival=10.0, valid_until=100.0, subject="sensor:reactor", supersedes=(), parents=None, **payload):
    if parents is not None:
        payload["derived_from"] = list(parents)
    return EvidenceItem(evidence_id=eid, source="network", kind="link_health", subject_id=subject,
                        event_time=arrival - 1, arrival_time=arrival, valid_until=valid_until, payload=payload,
                        provenance="test", supersedes=tuple(supersedes))


def test_validity_predicate_boundaries():
    log = EvidenceLog([item("a", arrival=10.0, valid_until=20.0)])
    assert not log.valid("a", np.nextafter(10.0, -math.inf))
    assert log.valid("a", 10.0)
    assert log.valid("a", np.nextafter(20.0, -math.inf))
    assert not log.valid("a", 20.0)
    log.retract("a", 15.0)
    assert log.valid("a", np.nextafter(15.0, -math.inf))
    assert not log.valid("a", 15.0)
    assert log.retracted_at("a") == 15.0
    assert log.known("a", 15.0) and not log.known("a", 9.0)


def test_unknown_ids_and_duplicates():
    log = EvidenceLog([item("a")])
    for bad in ("b", "", None, 3, ("a",)):
        assert not log.valid(bad, 50.0)
        assert log.get(bad) is None
        assert bad not in log
    with pytest.raises(ValueError):
        log.add(item("a", arrival=12.0))
    with pytest.raises(ValueError):
        log.add(item(""))
    assert len(log) == 1 and list(log) == ["a"] and log.get("a").arrival_time == 10.0


def test_nan_metadata_fails_closed():
    log = EvidenceLog([item("a", arrival=math.nan), item("b", valid_until=math.nan), item("c")])
    for t in (0.0, 10.0, 50.0, 1e9):
        assert not log.valid("a", t) and not log.valid("b", t)
    assert log.invalid_reason(["a"], 50.0) == "unknown_evidence"
    assert log.invalid_reason(["b"], 50.0) == "stale_evidence"
    with pytest.raises(ValueError):
        log.retract("c", math.nan)
    log.add(item("d", arrival=math.nan, supersedes=["c"]))
    assert not log.valid("c", 50.0)


def test_retraction_is_permanent_and_keeps_the_earliest_time():
    log = EvidenceLog([item("a", arrival=10.0, valid_until=math.inf)])
    log.retract("a", 30.0)
    log.retract("a", 50.0)
    assert log.retracted_at("a") == 30.0
    log.retract("a", 20.0)
    assert log.retracted_at("a") == 20.0
    assert all(not log.valid("a", t) for t in np.linspace(20.0, 1e6, 500))
    assert log.valid("a", 19.0)


def test_retraction_recorded_before_arrival_applies():
    log = EvidenceLog()
    log.retract("late", 12.0)
    log.add(item("late", arrival=10.0))
    assert log.valid("late", 11.0) and not log.valid("late", 12.0)


def test_supersession_from_the_superseding_arrival():
    log = EvidenceLog([item("old", arrival=10.0, valid_until=500.0)])
    log.add(item("new", arrival=40.0, supersedes=["old"]))
    assert log.valid("old", 39.9) and not log.valid("old", 40.0)
    assert log.retracted_at("old") == 40.0
    log.add(item("newer", arrival=30.0, supersedes=["old"]))
    assert not log.valid("old", 30.0)
    log.retract("new", 45.0)
    assert not log.valid("old", 100.0)
    log.add(item("fix", arrival=60.0, supersedes=["pending"]))
    log.add(item("pending", arrival=50.0))
    assert log.valid("pending", 55.0) and not log.valid("pending", 60.0)
    with pytest.raises(ValueError):
        log.add(item("self", supersedes=["self"]))


def test_derived_items_need_valid_parents():
    log = EvidenceLog([item("p1", arrival=10.0, valid_until=100.0), item("p2", arrival=30.0, valid_until=80.0)])
    log.add(item("d", arrival=20.0, valid_until=200.0, parents=["p1", "p2"]))
    log.add(item("g", arrival=35.0, valid_until=300.0, parents=["d"]))
    assert not log.valid("d", 25.0)
    assert log.valid("d", 30.0) and log.valid("g", 35.0)
    assert not log.valid("d", 80.0) and not log.valid("g", 80.0)
    assert log.retracted_at("d") == math.inf
    log.retract("p1", 50.0)
    assert log.retracted_at("d") == 50.0 and log.retracted_at("g") == 50.0
    assert log.valid("g", 49.0) and not log.valid("g", 50.0) and not log.valid("d", 50.0)
    log.add(item("p3", arrival=10.0))
    log.add(item("d3", arrival=10.0, parents=["p3"]))
    log.add(item("s", arrival=40.0, supersedes=["p3"]))
    assert log.valid("d3", 39.0) and not log.valid("d3", 40.0)
    assert log.invalid_reason(["d3"], 45.0) == "stale_evidence"


def test_derived_from_must_name_known_items():
    log = EvidenceLog([item("p")])
    with pytest.raises(ValueError):
        log.add(item("d", parents=["missing"]))
    with pytest.raises(ValueError):
        log.add(item("d", parents=["d"]))
    with pytest.raises(ValueError):
        log.add(item("d", derived_from="p"))
    assert "d" not in log


def test_invalid_reason_precedence():
    log = EvidenceLog([item("a", arrival=10.0, valid_until=20.0), item("b", arrival=10.0, valid_until=90.0),
                       item("future", arrival=70.0)])
    assert log.invalid_reason([], 50.0) is None
    assert log.invalid_reason(["b"], 50.0) is None
    assert log.invalid_reason(["a", "b"], 50.0) == "stale_evidence"
    assert log.invalid_reason(["a", "missing"], 50.0) == "unknown_evidence"
    assert log.invalid_reason(["future"], 50.0) == "unknown_evidence"
    assert log.invalid_reason(["future"], 70.0) is None


@pytest.mark.parametrize("seed", range(10))
def test_log_matches_the_tensor_contract(seed):
    rng = np.random.default_rng(seed)
    log = EvidenceLog()
    arrival, until, retracted, parents = {}, {}, {}, {}
    for i in range(60):
        eid = f"e{i}"
        a = float(rng.integers(0, 100))
        u = a + float(rng.integers(0, 60)) if rng.random() < 0.9 else math.inf
        ps = list(rng.choice(i, size=min(i, int(rng.integers(0, 3))), replace=False)) if i and rng.random() < 0.3 else []
        ps = [f"e{p}" for p in ps]
        sup = [f"e{int(rng.integers(0, i))}"] if i and rng.random() < 0.15 else []
        log.add(item(eid, arrival=a, valid_until=u, supersedes=sup, parents=ps or None))
        arrival[eid], until[eid], parents[eid] = a, u, ps
        retracted.setdefault(eid, math.inf)
        for old in sup:
            retracted[old] = min(retracted[old], a)
        if rng.random() < 0.2:
            victim = f"e{int(rng.integers(0, i + 1))}"
            t = float(rng.integers(0, 150))
            log.retract(victim, t)
            retracted[victim] = min(retracted[victim], t)

    def meta(eid):
        rows = [meta(p) for p in parents[eid]] + [(arrival[eid], until[eid], retracted[eid])]
        a, u, r = derived_validity(*(np.array(col) for col in zip(*rows)), axis=0)
        return float(a), float(u), float(r)

    ids = list(arrival)
    for t in np.arange(-1.0, 170.0, 0.5):
        a, u, r = (np.array(col) for col in zip(*(meta(e) for e in ids)))
        want = validity(np.ones(len(ids), bool), a, u, r, t)
        got = np.array([log.valid(e, t) for e in ids])
        np.testing.assert_array_equal(got, want, err_msg=f"t={t}")
        assert log.valid_ids(t) == tuple(e for e, ok in zip(ids, want) if ok)
    for e in ids:
        assert log.retracted_at(e) == meta(e)[2]
