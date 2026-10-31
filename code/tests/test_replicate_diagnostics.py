import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replicate_diagnostics import (
    argmin_mask, dev_splits, gather_optimal, identical_rows, remembered_trace, replicate_labels, resample,
    resample_mean, split_halves, subset_masks, ties, twin_memory)


def test_only_development_splits_are_accepted():
    assert dev_splits(["train", "val"]) == ("train", "val")
    assert dev_splits(["val"]) == ("val",)
    for bad in (["test"], ["train", "test"], []):
        with pytest.raises(AssertionError):
            dev_splits(bad)


def test_argmin_set_and_gather_answer_follow_tol():
    J = np.array([[1.0, 1.0 + 1e-12, 2.0], [3.0, 0.5, 0.5]])
    assert argmin_mask(J).tolist() == [[True, True, False], [False, True, True]]
    observe = np.array([True, False, False])
    assert gather_optimal(np.array([0.5, 1.0, 2.0]), observe)
    assert not gather_optimal(np.array([1.0, 1.0 + 1e-12, 2.0]), observe)
    assert gather_optimal(J, observe).tolist() == [False, False]


def test_replicate_labels_take_the_same_columns_in_both_worlds():
    P = np.array([[0.0, 10.0, 0.0, 10.0], [4.0, 4.0, 4.0, 4.0]])
    T = np.array([[10.0, 0.0, 10.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    w = (0.6, 0.4)
    Jh, Jb = replicate_labels(P, T, w, np.arange(4))
    np.testing.assert_array_equal(Jh, P.mean(1))
    np.testing.assert_array_equal(Jb, 0.6 * P.mean(1) + 0.4 * T.mean(1))
    idx = np.array([[0, 0, 1, 1], [3, 3, 3, 3]])
    Jh, Jb = replicate_labels(P, T, w, idx)
    assert Jh.shape == Jb.shape == (2, 2)
    for b in range(2):
        np.testing.assert_allclose(Jb[b], 0.6 * P[:, idx[b]].mean(1) + 0.4 * T[:, idx[b]].mean(1))
    unmixed = replicate_labels(P, None, (1.0, 0.0), idx)
    np.testing.assert_array_equal(unmixed[0], unmixed[1])


def test_resampling_is_joint_across_worlds():
    P = np.array([[0.0, 10.0, 0.0, 10.0], [4.0, 4.0, 4.0, 4.0]])
    T = np.array([[10.0, 0.0, 10.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    observe = np.array([True, False])
    idx = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 2, 3], [2, 2, 2, 2]])
    out = resample(P, T, (0.5, 0.5), observe, True, idx)
    assert out["hindsight_disjoint"] == out["hindsight_differs"] == 0.5
    assert out["hindsight_flip"] == 0.5
    assert out["belief_disjoint"] == out["belief_differs"] == out["belief_flip"] == 0.0
    assert np.isnan(resample(P, T, (0.5, 0.5), observe, False, idx)["belief_flip"])


def test_disjoint_and_differing_argmin_sets_are_told_apart():
    P = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 2.0, 2.0]])
    out = resample(P, None, (1.0, 0.0), np.array([False, False, True]), True, np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
    assert out["hindsight_disjoint"] == 0.0 and out["hindsight_differs"] == 0.0
    out = resample(P, None, (1.0, 0.0), np.array([False, False, True]), True, np.array([[0, 0, 0, 0], [0, 1, 1, 1]]))
    assert out["hindsight_disjoint"] == 0.5 and out["hindsight_differs"] == 0.5 and out["hindsight_flip"] == 0.5
    P2 = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 0.0], [3.0, 3.0, 3.0, 3.0]])
    out = resample(P2, None, (1.0, 0.0), np.array([False, False, True]), True, np.array([[1, 1, 1, 1], [3, 3, 3, 3]]))
    assert out["hindsight_disjoint"] == 0.0 and out["hindsight_differs"] == 1.0


def test_split_halves_compare_the_first_and_second_replicate_pairs():
    P = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0]])
    observe = np.array([True, False])
    out = split_halves(P, None, (1.0, 0.0), observe, True)
    assert out == {"hindsight_overlap": False, "hindsight_gather_agree": False,
                   "belief_overlap": False, "belief_gather_agree": False}
    T = np.array([[10.0, 10.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    out = split_halves(P, T, (0.5, 0.5), observe, False)
    assert out["belief_overlap"] and out["belief_gather_agree"] is None and not out["hindsight_overlap"]


def test_resample_mean_and_its_monte_carlo_se():
    p, se = resample_mean([0.5, 0.0, 1.0], 401)
    assert p == pytest.approx(0.5)
    assert se == pytest.approx(np.sqrt(0.25 / 400) / 3)
    assert resample_mean([], 400) == (None, None)


def test_identical_rows_within_tol():
    M = np.array([[1.0, 2.0], [1.0 + 1e-12, 2.0], [2.0, 1.0]])
    assert identical_rows(M, np.array([0, 1]))
    assert not identical_rows(M, np.array([0, 2]))


def trace_item(eid, link, event, arrival):
    return {"evidence_id": eid, "kind": "trace_link", "subject_id": link, "event_time": event, "arrival_time": arrival}


def test_remembered_trace_is_the_latest_valid_trace_of_the_group_link():
    items = {i["evidence_id"]: i for i in (
        trace_item("tool:trace_link:1", "sensor:reactor", 100.0, 220.0),
        trace_item("tool:trace_link:2", "sensor:reactor", 400.0, 530.0),
        trace_item("tool:trace_link:3", "sensor:stripper", 600.0, 700.0),
        trace_item("tool:trace_link:4", "sensor:reactor", 650.0, 760.0))}
    traces = [{"time": 100.0, "reported_healthy": True, "healthy": [True, True]},
              {"time": 400.0, "reported_healthy": True, "healthy": [False, True]},
              {"time": 650.0, "reported_healthy": False, "healthy": [False, True]}]
    d = {"valid_evidence": ["tool:trace_link:1", "tool:trace_link:2", "tool:trace_link:3", "win:reactor:9"],
         "twin": {"belief": {"traces": traces}}}
    assert remembered_trace(items, d, "reactor") == (True, True)
    d1 = {**d, "valid_evidence": ["tool:trace_link:1"]}
    assert remembered_trace(items, d1, "reactor") == (False, False)
    assert remembered_trace(items, {**d, "valid_evidence": ["tool:trace_link:3"]}, "reactor") is None
    d2 = {**d1, "twin": {"belief": {"traces": traces + [dict(traces[0])]}}}
    with pytest.raises(ValueError):
        remembered_trace(items, d2, "reactor")


def toy_record(family=True, twin_weight=0.3, flag=False, trace=None, P=None, T=None, tools=None, p=(0.0, 0.0)):
    P = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 5.0]]) if P is None else P
    w = (1.0 - twin_weight, twin_weight) if T is not None and twin_weight > 0 else (1.0, 0.0)
    J = P.mean(1)
    Jb = J if w[1] == 0 else w[0] * J + w[1] * T.mean(1)
    return {"row": {"family": family, "twin_record": family, "p_min": p[0], "p_max": p[1],
                    "cand": {"tool": tools or ["monitor", "monitor", "escalate"]}},
            "P": P, "T": T, "w": w, "observe": np.array([False, False, True]), "has_both": True, "J": J,
            "J_belief": Jb, "twin_weight": twin_weight if family else 0.0, "twin_labelled": T is not None,
            "flag": flag if family else None, "trace": trace, "citation_changed": flag, "citation_replaced": False}


def test_twin_memory_flag_rule_and_groups():
    T = np.array([[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]])
    records = [toy_record(flag=True, trace=(True, True), T=T, twin_weight=0.02),
               toy_record(flag=False, trace=(False, False), T=T, twin_weight=0.5),
               toy_record(flag=False, trace=None, T=T, twin_weight=0.0),
               toy_record(family=False)]
    masks = subset_masks(records)
    assert masks["twin_weight_pos"].tolist() == [True, True, False, False]
    out = twin_memory(records, masks)
    assert out["flag_share_family"] == {"k": 1, "n": 3, "share": 1 / 3}
    assert out["flag_equals_trace_rule"]["k"] == 3
    assert out["flag_equals_trace_rule"]["crosstab"]["flag=True,trace_rule=True"] == 1
    tp = out["twin_weight_pos"]
    assert tp["flagged_share"]["k"] == 1 and tp["flagged_share"]["n"] == 2
    assert tp["by_flag"]["flagged"]["twin_weight"]["median"] == pytest.approx(0.02)
    assert tp["by_flag"]["flagged"]["remembered_trace_wrong"]["k"] == 1
    assert tp["by_flag"]["unflagged"]["remembered_trace_wrong"] == {"k": 0, "n": 1, "share": 0.0}
    np.testing.assert_allclose(tp["by_flag"]["flagged"]["abs_J_belief_minus_J_per_candidate"]["mean"],
                               0.02 * np.mean([1.0, 1.0, 4.0]))
    assert tp["citation_changed"]["k"] == 1


def test_ties_identical_trajectories_and_monitor_only():
    noisy = np.array([[0.0, 2.0], [2.0, 0.0], [3.0, 5.0]])
    T = np.array([[2.0, 2.0], [2.0, 3.0], [9.0, 9.0]])
    records = [toy_record(),
               toy_record(P=noisy, tools=["monitor", "repair_link", "escalate"], p=(0.0, 0.5)),
               toy_record(T=T, twin_weight=0.0001)]
    out = ties(records, subset_masks(records))
    h, b = out["hindsight"], out["belief"]
    assert h["all"]["k"] == 3 and h["high_stakes"] == {"k": 1, "n": 1, "share": 1.0}
    assert h["tied_identical_every_replicate"]["k"] == 2
    assert h["tied_monitor_only"]["k"] == 2
    assert b["all"]["k"] == 2 and b["tied_identical_every_replicate"]["k"] == 1
