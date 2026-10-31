import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import label_diagnostics as LD
from cpsdefense.experience import belief
from cpsdefense.experience.dataset import to_item


def _cand(tool, target, family="respond"):
    return {"tool_id": tool, "target_id": target, "family": family}


def _row(classes, J, Jb, seed=1):
    action = np.array(classes)
    return {"action": action, "info": np.array([LD.INFO_CLASS[c] for c in action]), "J": np.array(J, float),
            "Jb": np.array(Jb, float), "seed": seed}


def test_action_classes_are_relative_to_the_group():
    g = "reactor"
    cases = [(_cand("trace_link", "sensor:reactor", "observe"), "trace_link_pair"),
             (_cand("trace_link", "actuator:reactor", "observe"), "other_observe"),
             (_cand("trace_link", "sensor:feeds", "observe"), "other_observe"),
             (_cand("verify_sensor", "source:reactor", "observe"), "verify_sensor_pair"),
             (_cand("query_historian", "source:reactor", "observe"), "other_observe"),
             (_cand("escalate", "incident:1", "escalate"), "escalate"),
             (_cand("switch_to_backup", "sensor:reactor"), "switch_to_backup_pair"),
             (_cand("switch_to_backup", "sensor:feeds"), "other"),
             (_cand("restore_primary", "sensor:reactor"), "restore_primary"),
             (_cand("repair_link", "sensor:reactor"), "repair_pair"),
             (_cand("repair_link", "actuator:reactor"), "other"),
             (_cand("replace_io_module", "source:reactor"), "repair_pair"),
             (_cand("replace_io_module", "source:feeds"), "other"),
             (_cand("recalibrate_sensor", "reactor_level"), "recalibrate_sensor"),
             (_cand("set_production", "plant"), "set_production"),
             (_cand("monitor", "plant", "monitor"), "monitor")]
    for c, expected in cases:
        assert LD.action_class(c, g) == expected, c
    assert set(LD.INFO_CLASS) == set(LD.ACTION_CLASSES) and set(LD.INFO_CLASS.values()) == set(LD.INFO_CLASSES)
    observe = {LD.action_class(c, g) for c, _ in cases if c["family"] == "observe"}
    assert observe == set(LD.DEFINITIONS["D0_observe_family"])


def test_strictly_best_needs_both_sides_and_a_margin():
    J = np.array([1.0, 2.0, 1.0 + 1e-12])
    assert LD.strictly_best(J, np.array([True, False, False])) is False
    assert LD.strictly_best(J, np.array([True, False, True])) is True
    assert LD.strictly_best(J, np.array([False, False, False])) is None
    assert LD.strictly_best(J, np.array([True, True, True])) is None
    assert LD.argmin_set(J) == frozenset({0, 2})


def test_class_table_counts_ties_and_unique_argmins():
    rows = [_row(["trace_link_pair", "monitor", "monitor"], J=[5, 1, 1], Jb=[0, 1, 1]),
            _row(["trace_link_pair", "monitor", "repair_pair"], J=[2, 2, 1], Jb=[1, 1, 3])]
    b = LD.class_table(rows, "action", "belief", LD.ACTION_CLASSES)
    assert b["n"] == 2 and b["tie_share"] == 0.5 and b["cross_class_tie_share"] == 0.5
    assert b["classes"]["trace_link_pair"] == {"co_optimal": 1.0, "unique_argmin": 0.5, "candidate": 1.0}
    assert b["classes"]["monitor"]["co_optimal"] == 0.5 and b["classes"]["monitor"]["unique_argmin"] == 0.0
    h = LD.class_table(rows, "action", "hindsight", LD.ACTION_CLASSES)
    assert h["tie_share"] == 0.5 and h["cross_class_tie_share"] == 0.0
    assert h["classes"]["repair_pair"]["unique_argmin"] == 0.5
    info = LD.class_table(rows, "info", "belief", LD.INFO_CLASSES)
    assert info["classes"]["diagnostic"]["co_optimal"] == 1.0 and info["classes"]["wait"]["co_optimal"] == 0.5


@pytest.mark.filterwarnings("ignore:only . clusters")
def test_definition_table_denominators_and_paired_difference():
    rows = [_row(["trace_link_pair", "monitor"], J=[2, 1], Jb=[0, 1], seed=1),
            _row(["switch_to_backup_pair", "monitor"], J=[0, 1], Jb=[0, 1], seed=2),
            _row(["trace_link_pair", "switch_to_backup_pair"], J=[1, 1], Jb=[1, 2], seed=3)]
    t = LD.definition_table(rows, n_boot=50, seed=0)
    d0 = t["D0_observe_family"]
    assert d0["with_both"] == 2 and d0["decisions"] == 3
    assert d0["belief"]["strictly_best"] == {"count": 2, "share": 1.0}
    assert d0["hindsight"]["strictly_best"] == {"count": 0, "share": 0.0}
    assert d0["hindsight"]["co_optimal"] == {"count": 1, "share": 1 / 3}
    assert d0["belief_minus_hindsight"]["point"] == pytest.approx(1.0)
    d2 = t["D2_diagnostic_or_backup_probe"]
    assert d2["with_both"] == 2
    assert d2["belief"]["co_optimal"]["count"] == 3 and d2["hindsight"]["strictly_best"]["count"] == 1


def test_groupwide_staleness_follows_the_playbook_rule():
    def alert(eid, group, measurement, category="stale"):
        return eid, {"evidence_id": eid, "kind": "alert", "subject_id": f"source:{group}",
                     "payload": {"category": category, "measurement": measurement}}
    items = dict([alert("a1", "a", "m1"), alert("a2", "a", "m2"), alert("a2b", "a", "m2"), alert("a3", "a", "m3"),
                  alert("b1", "b", "x1"), alert("b2", "b", "x2"), alert("c1", "c", "y1", "deviation"),
                  alert("c2", "c", "y2", "deviation")])
    sizes = {"a": 5, "b": 3, "c": 2}
    assert LD.groupwide_stale(items, ["a1", "a2", "a2b"], sizes, 0.5) == set()
    assert LD.groupwide_stale(items, ["a1", "a2", "a3", "b1", "b2"], sizes, 0.5) == {"a", "b"}
    assert LD.groupwide_stale(items, ["b1", "c1", "c2", "missing"], sizes, 0.5) == set()


def test_shifted_world_moves_the_link_fault_only():
    states = [(100.0, "up", False), (200.0, "down", True), (300.0, "backup", True)]
    w0 = LD.shifted_world(belief.LINK, 150.0, 1.8, states, 0.0)
    assert w0["states"] == states and w0["onset_time"] == 150.0
    later = LD.shifted_world(belief.LINK, 150.0, 1.8, states, 100.0)
    assert later["onset_time"] == 250.0
    assert later["states"] == [(100.0, "up", False), (200.0, "up", False), (300.0, "backup", True)]
    earlier = LD.shifted_world(belief.LINK, 150.0, 1.8, states, -100.0)
    assert earlier["states"] == [(100.0, "down", True), (200.0, "down", True), (300.0, "backup", True)]
    module = LD.shifted_world(belief.MODULE, 148.2, 1.8, [(100.0, "up", False)], -100.0)
    assert module["states"] == [(100.0, "up", False)] and module["onset_time"] == pytest.approx(48.2)


def _toy_episode():
    def item(eid, kind, subject, t, values=None):
        return {"evidence_id": eid, "source": "net", "kind": kind, "subject_id": subject, "event_time": t,
                "arrival_time": t + 2.0, "valid_until": t + 900.0, "payload": {"values": values or [0.0, 0.0]},
                "provenance": "toy"}
    items = [item("win:g:10", "window_summary", "source:g", 360.0), item("link:sensor:g:6", "link_health",
                                                                            "sensor:g", 216.0, [1.0, 0.0])]
    scenario = {"warmup_s": 0.0, "incident": {"kind": belief.LINK, "target": "sensor:g", "onset_s": 150.0}}
    twin = {"scenario": {"warmup_s": 0.0, "incident": {"kind": belief.MODULE, "target": "g", "onset_s": 148.2}},
            "primary_states": [(0.0, "up", False)], "states": [(0.0, "up", False)]}
    params = {"p_detect_down": 0.05, "p_false": 0.0003, "report_backup": True, "delay_s": 30.0, "delay_sigma": 0.5,
              "evidence_scale": 1.0, "window_s": 36.0, "trace_accuracy": 0.97, "operator_accuracy": 0.85,
              "causes": ["none", belief.LINK, belief.MODULE]}
    worlds = [belief.world(belief.LINK, 150.0, 1.8, twin["primary_states"]),
              belief.world(belief.MODULE, 148.2, 1.8, twin["states"])]
    stored = belief.belief_evidence([to_item(i) for i in items], 400.0, "g", worlds, params, [0.5, 0.5])
    return {"items": items, "scenario": scenario, "twin": twin}, stored


def test_onset_evidence_reproduces_and_shifts_the_stored_evidence():
    episode, stored = _toy_episode()
    items = LD.onset_items(episode, stored["tau"], "g")
    assert [i.evidence_id for i in items] == ["win:g:10", "link:sensor:g:6"]
    ev0 = LD.onset_evidence(episode, stored, items, 1.8)
    assert ev0["windows"] == stored["windows"] and [w["w"] for w in ev0["windows"]] == list(range(5, 12))
    later = LD.onset_evidence(episode, stored, items, 1.8, 36.0)
    assert [w["w"] for w in later["windows"]] == list(range(6, 12))
    earlier = LD.onset_evidence(episode, stored, items, 1.8, -36.0)
    assert [w["w"] for w in earlier["windows"]] == list(range(4, 12))
    p0 = belief.posterior(stored)
    np.testing.assert_allclose(LD.posterior_from_loglik(stored["prior"], belief.log_likelihoods(ev0)), p0)
    assert p0[0] > 0.5


def test_mean_loglik_averages_likelihoods_per_cause():
    lls = [[np.log(0.5), np.log(0.1)], [np.log(0.1), -np.inf]]
    np.testing.assert_allclose(np.exp(LD.mean_loglik(lls)), [0.3, 0.05])
    assert LD.mean_loglik([[0.0, -np.inf]])[1] == -np.inf
    np.testing.assert_allclose(LD.posterior_from_loglik([0.5, 0.5], LD.mean_loglik(lls)), [0.3 / 0.35, 0.05 / 0.35])


def test_null_ece_is_zero_for_certain_posteriors_and_reproducible():
    assert LD.null_ece(np.array([0.0, 1.0, 1.0, 0.0]), 20, seed=0) == (0.0, 0.0)
    p = np.random.default_rng(3).random(400)
    a, b = LD.null_ece(p, 50, seed=7), LD.null_ece(p, 50, seed=7)
    assert a == b and 0 < a[0] <= a[1] < 0.2


def test_prior_components_match_pair_prior():
    incidents = {"types": {"link_fault": 0.35, "io_module_fault": 0.2, "sensor_fault": 0.45},
                 "link_fault": {"links": {"sensor:a": 2, "sensor:b": 1, "actuator:a": 1}},
                 "io_module_fault": {"groups": {"a": 2, "b": 1}}}
    out = LD.prior_components(incidents, ["a", "b"])
    assert (out["W_L"], out["W_L_sensor"], out["W_L_actuator"], out["W_M"]) == (4.0, 3.0, 1.0, 3.0)
    assert out["pi_L"] == pytest.approx(0.35) and out["pi_M"] == pytest.approx(0.2)
    for g in ("a", "b"):
        assert out["groups"][g]["prior_link"] == pytest.approx(belief.pair_prior(incidents, g)[belief.LINK])
    assert out["groups"]["a"]["prior_link"] == pytest.approx(out["groups"]["b"]["prior_link"])
    assert out["groups"]["a"]["prior_link"] == pytest.approx(0.35 * 2 / 4 / (0.35 * 2 / 4 + 0.2 * 2 / 3))


def test_calibration_subgroups_and_range():
    rows = [{"since_onset_s": s, "onset_happened": s >= 0, "settled": i % 3 == 0, "backup": "available",
             "index": i % 2} for i, s in enumerate([-10.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])]
    groups, edges = LD.calibration_subgroups(rows)
    names = dict(groups)
    assert "backup_available" not in names
    assert names["pre_onset"].tolist() == [True] + [False] * 6
    assert sum(names[f"since_onset_t{k}"].sum() for k in (1, 2, 3)) == 6
    assert edges[0] < edges[1]
    sub = {"all": {"ece": 0.5, "below_null_q95": False}, "index_0": {"ece": 0.02, "below_null_q95": True},
           "x:since_onset_t2": {"ece": 0.01, "below_null_q95": True}}
    rng = LD.ece_range(sub)
    assert (rng["min"], rng["min_subgroup"], rng["max"], rng["max_subgroup"]) == (0.01, "x:since_onset_t2", 0.02,
                                                                                 "index_0")
    assert rng["all_below_null_q95"]


def test_only_development_splits_are_read(tmp_path):
    for splits in (("train", "test"), ("test",), ()):
        with pytest.raises(ValueError):
            LD.load_dev(tmp_path, splits)
    assert LD.load_dev(tmp_path, ("val",)) == {"val": []}


def test_rounded_keeps_types():
    out = LD.rounded({"a": np.float64(0.123456), "b": [1, np.int64(2), 0.98766], "c": np.nan, "d": True})
    assert out == {"a": 0.1235, "b": [1, 2, 0.9877], "c": None, "d": True}
