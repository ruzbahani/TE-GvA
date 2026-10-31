import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cpsdefense.experience.dataset import belief_labels, candidate_labels, voi_of
from cpsdefense.model.direct import pad_candidates
from trip_penalty import (
    LUMP, decision_table, dev_split, horizon_stats, label_stats, lump_view, new_counts, replicate_checks, rounded,
    shifted_decision)

COSTS = {"physical": {"shutdown": 100.0, "near_limit_per_h": 5.0},
         "service": {"production_shortfall_per_h": 10.0, "degraded_per_h": 0.5}}


def toy_split():
    decision = np.repeat(np.arange(4), 2)[:-1]
    observe = np.array([True, False, True, False, True, False, True])
    J = np.array([5.0, 4.0, 2.0, 2.0, 1.0, 3.0, 1.0])
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    Jb = np.array([5.0, 6.0, 2.0, 2.0 + 1e-12, 1.0, 3.0, 1.0])
    pb = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.5, 0.0])
    flat = {"J": J, "p_trip": p, "J_belief": Jb, "p_trip_belief": pb, "observe": observe,
            "voi": np.concatenate([voi_of(J[decision == i], observe[decision == i]) for i in range(4)]),
            "voi_belief": np.concatenate([voi_of(Jb[decision == i], observe[decision == i]) for i in range(4)])}
    mask, pad = pad_candidates(decision, 4, flat)
    dec = {"family": np.array([True, True, False, False]), "backup_available": np.array([True, False, True, True]),
           "episode_seed": np.array([1, 2, 3, 3])}
    return {"name": "toy", "n": 4, "mask": mask, "pad": pad, "dec": dec,
            "cand": {"decision": decision, "J": J, "p_trip": p, "J_belief": Jb, "p_trip_belief": pb}}


def test_lump_view_shifts_both_label_sets_and_recomputes_voi():
    split = toy_split()
    same = lump_view(split, LUMP)
    for k in ("J", "J_belief", "voi", "voi_belief"):
        np.testing.assert_array_equal(same["pad"][k], split["pad"][k])
    view = lump_view(split, 25)
    m = split["mask"]
    np.testing.assert_allclose(view["pad"]["J"][m], split["pad"]["J"][m] - 75 * split["pad"]["p_trip"][m])
    np.testing.assert_allclose(view["cand"]["J_belief"], split["cand"]["J_belief"] - 75 * split["cand"]["p_trip_belief"])
    assert view["pad"]["voi_belief"][0, 0] == pytest.approx(-1.25)
    assert np.isnan(view["pad"]["voi"][3, 0])
    np.testing.assert_array_equal(view["pad"]["p_trip"], split["pad"]["p_trip"])


def test_decision_table_follows_the_dataset_stats_definitions():
    t = decision_table(lump_view(toy_split(), LUMP), LUMP)
    assert t["both"].tolist() == [True, True, True, False]
    assert t["gather_belief"].tolist() == [True, False, True, False]
    assert t["gather_hindsight"].tolist() == [False, False, True, False]
    assert t["best_hindsight"][1].tolist() == [True, True]
    np.testing.assert_allclose(t["regret"], [1.0, 0.0, 0.0, 0.0], atol=1e-11)
    np.testing.assert_allclose(t["regret_trip"], [3.0, 0.0, 0.0, 0.0])


def test_label_stats_pool_splits_and_compare_with_L100():
    split = toy_split()
    at100, at25 = label_stats([split, split], LUMP), label_stats([split], 25)
    assert at100["gather_belief_ambiguous"] == {"k": 2, "n": 4, "share": 0.5}
    assert at100["gather_belief_ambiguous_by_backup"] == {"available": {"n": 2, "share": 1.0},
                                                          "unavailable": {"n": 2, "share": 0.0}}
    assert at100["gather_hindsight_all"] == {"k": 2, "n": 6, "share": pytest.approx(1 / 3)}
    reg = at100["hindsight_best_belief_regret_ambiguous"]
    assert reg["mean"] == pytest.approx(0.5) and reg["trip_part_mean"] == pytest.approx(1.5)
    assert reg["positive"]["k"] == 2 and reg["by_backup"]["available"]["mean"] == pytest.approx(1.0)
    assert at100["gather_flipped_ambiguous"]["k"] == 0 and at100["belief_best_changed_ambiguous"]["k"] == 0
    assert at25["gather_belief_ambiguous"]["k"] == 0
    assert at25["gather_flipped_ambiguous"] == {"k": 1, "n": 2, "share": 0.5}
    assert at25["belief_best_changed_ambiguous"] == {"k": 1, "n": 2, "share": 0.5}
    assert at25["hindsight_best_changed_all"]["k"] == 1
    assert at25["mean_J_trip_part"] == pytest.approx(25 * 0.5 / 7)
    assert at25["mean_J"] == pytest.approx((18.0 - 75 * 0.5) / 7)


def test_horizon_stats_count_early_trips_and_short_horizon_stakes():
    decision = np.array([0, 0, 1, 1])
    nan = np.nan
    event = np.array([[1.0, 0.0, nan], [0.0, 0.0, nan], [1.0, 1.0, nan], [1.0, 0.0, nan]])
    time_ = np.array([[5000.0, 10800.0, nan], [10800.0, 10800.0, nan], [600.0, 4000.0, nan], [3600.0, 10800.0, nan]])
    flat = {"trip_event": event, "trip_time": time_, "p_trip": np.array([0.5, 0.0, 1.0, 0.5])}
    mask, pad = pad_candidates(decision, 2, flat)
    h = horizon_stats([{"name": "toy", "n": 2, "mask": mask, "pad": pad}])
    assert (h["replicates"], h["trips"], h["trips_within_1h"]) == (8, 4, 2)
    assert h["trip_time_min"]["median"] == pytest.approx(3800.0 / 60)
    assert h["high_stakes"]["k"] == 2 and h["high_stakes_1h"]["k"] == 0 and h["high_stakes_still_at_1h"]["k"] == 0
    flat["p_trip"] = np.array([0.5, 0.0, 0.5, 0.5])
    mask, pad = pad_candidates(decision, 2, flat)
    with pytest.raises(ValueError):
        horizon_stats([{"name": "toy", "n": 2, "mask": mask, "pad": pad}])


def _result(candidate, replicate, other, trip=False, tts=None, physical=0.0, service=0.0):
    total = other + physical + service
    return {"candidate": candidate, "replicate": replicate, "total": total, "shutdown": trip,
            "time_to_shutdown": tts, "horizon_s": 10800.0, "resolved": True, "mitigated": True,
            "first_status": "succeeded", "first_duration": 60.0, "first_started": 0.0,
            "cost": {"attack": 0.0, "service": service, "physical": physical, "compute": other}}


def toy_decision():
    cands = [{"family": "observe"}, {"family": "respond"}]
    results = [_result(0, 0, 0.2), _result(0, 1, 0.2, physical=1.0),
               _result(1, 0, 0.4, trip=True, tts=1800.0, physical=101.0), _result(1, 1, 0.4)]
    twin = [_result(0, 0, 0.2, trip=True, tts=3000.0, physical=100.0), _result(0, 1, 0.2),
            _result(1, 0, 0.4, service=2.0), _result(1, 1, 0.4)]
    evidence = {"prior": [0.7, 0.3], "params": {"trace_accuracy": 0.9, "operator_accuracy": 0.9, "causes": [0, 1]},
                "deterministic_identical": True, "windows": [], "traces": [], "operator": []}
    return {"candidates": cands, "results": results, "twin": {"belief": evidence, "results": twin}}


def test_shifted_totals_give_the_label_identity():
    d = toy_decision()
    lab, bl = candidate_labels(d), belief_labels(d, True)
    assert bl["posterior"] == pytest.approx(0.7) and not bl["belief_fallback"]
    for L in (25, 50, 100, 200, 400):
        dL = shifted_decision(d, L)
        np.testing.assert_allclose(candidate_labels(dL)["J"], lab["J"] + (L - LUMP) * lab["p_trip"], atol=1e-12)
        np.testing.assert_allclose(belief_labels(dL, True)["J_belief"],
                                   bl["J_belief"] + (L - LUMP) * bl["p_trip_belief"], atol=1e-12)
    assert d["results"][2]["total"] == pytest.approx(101.4)


def test_replicate_checks_flag_costs_after_a_trip():
    acc = new_counts()
    replicate_checks(toy_decision()["results"], COSTS, acc)
    assert (acc["replicates"], acc["trips"], acc["trips_within_1h"]) == (4, 1, 1)
    assert acc["physical_tripped"] == [101.0, 101.0] and acc["physical_untripped_max"] == 1.0
    assert sum(v for k, v in acc.items() if isinstance(v, int) and k not in ("replicates", "trips",
                                                                             "trips_within_1h")) == 0
    bad = [_result(0, 0, 0.0, trip=True, tts=600.0, physical=110.0),
           _result(0, 1, 0.0, trip=True, tts=600.0, physical=100.0, service=5.0),
           _result(1, 0, 0.0, physical=100.0),
           _result(1, 1, 0.0, trip=True, tts=None, physical=100.0)]
    acc = new_counts()
    replicate_checks(bad, COSTS, acc)
    assert acc["near_limit_cost_exceeds_time"] == 2 and acc["service_cost_exceeds_time"] == 1
    assert acc["lump_without_trip_or_trip_without_lump"] == 1 and acc["trip_time_inconsistent"] == 1


def test_only_development_splits_are_read(tmp_path):
    with pytest.raises(AssertionError):
        dev_split(tmp_path, "test")


def test_rounded_keeps_small_errors_visible():
    assert rounded({"a": [1.23456789, 1.1e-13, 0.0, 3]}) == {"a": [1.2346, 1.1e-13, 0.0, 3]}
