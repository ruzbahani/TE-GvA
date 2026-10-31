import copy
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

from cpsdefense.experience import belief
from cpsdefense.experience.dataset import (BELIEF_DECISION_KEYS, BELIEF_KEYS, SPLITS, baseline_index, belief_labels,
                                           candidate_labels, in_family, split_of)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dataset_stats.py"
SEEDS = {s: [k for k in range(2000) if split_of(k) == s][:2] for s in SPLITS}


@pytest.fixture(scope="module")
def ds():
    spec = importlib.util.spec_from_file_location("dataset_stats_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cand(tool, family, target="sensor:reactor", args=None):
    args = args or {}
    text = ",".join(f"{k}={v}" for k, v in args.items())
    return {"action_id": f"{tool}({target};{text})", "tool_id": tool, "family": family, "target_id": target,
            "args": args, "expected_duration": 60.0, "duration_q90": 100.0}


def _decision(tau, onset, cands, totals, trips=None, statuses=None, kind="link_fault",
              cause="link_fault:sensor:reactor", onset_happened=True, baselines=None, behaviour=None, oracle=None,
              compute=None, valid=("e1", "e2"), submitted=None):
    results = []
    for i, row in enumerate(totals):
        for k, total in enumerate(row):
            trip = bool(trips[i][k]) if trips else False
            tool = compute[i] if compute else 0.0
            results.append({
                "candidate": i, "replicate": k, "tool_id": cands[i]["tool_id"], "target_id": cands[i]["target_id"],
                "args": cands[i]["args"], "family": cands[i]["family"], "total": float(total),
                "cost": {"attack": 0.0, "service": total - 100.0 * trip - tool, "physical": 100.0 * trip,
                         "compute": tool},
                "shutdown": trip, "time_to_shutdown": 1800.0 if trip else None,
                "resolved": None if kind == "disturbance" else True, "mitigated": None if kind == "disturbance" else True,
                "first_status": statuses[i][k] if statuses else "succeeded", "first_reason": None,
                "first_duration": None if statuses and statuses[i][k] in ("running", "accepted") else 60.0,
                "horizon_s": 10800.0})
    out = {"tau": tau, "obs_version": 1, "valid_evidence": list(valid), "cited_evidence": ["e1"],
           "candidates": cands, "results": results, "baselines": baselines, "behaviour": behaviour,
           "hidden": {"cause": cause, "kind": kind, "onset_time": onset, "onset_happened": onset_happened,
                      "oracle_time_to_shutdown": oracle}}
    if submitted is not None:
        out["submitted_action_ids"] = list(submitted)
    return out


def _receipt(n, t, status, tool="trace_link", target="sensor:reactor", action="", retracted=np.inf, supersedes=()):
    payload = {"tool_id": tool, "status": status}
    if action is not None:
        payload.update(action_id=action or f"{tool}({target})@1000.000", idempotency_key=f"x:{action}")
    return {"evidence_id": f"receipt:{n}", "kind": "receipt", "subject_id": target, "event_time": t,
            "arrival_time": t, "valid_until": np.inf if status == "running" else t + 3600.0,
            "retracted_at": retracted, "supersedes": list(supersedes), "payload": payload}


def _tool_result(n, t, kind="trace_link", target="sensor:reactor", action=""):
    payload = {"values": [1.0, 0.0]}
    if action is not None:
        payload.update(action_id=action or f"trace_link({target})@1000.000", idempotency_key=f"x:{action}")
    return {"evidence_id": f"tool:{kind}:{n}", "kind": kind, "subject_id": target, "event_time": t,
            "arrival_time": t, "valid_until": t + 900.0, "retracted_at": np.inf, "supersedes": [], "payload": payload}


def _episode(seed, kind, target, decisions, mode="act_first", background=(), fault_kind=None, items=None,
             backup=True):
    cause = f"{kind}:{target}" + (f":{fault_kind}" if fault_kind else "")
    if items is None:
        items = [{"evidence_id": "e1", "kind": "alert"}, {"evidence_id": "e2", "kind": "window_summary"}] \
            if decisions else []
    return {"seed": seed, "decisions": decisions, "items": items, "seconds": 1.0,
            "scenario": {"seed": seed, "incident": {"kind": kind, "target": target, "onset_s": 600.0,
                                                    "fault_kind": fault_kind, "magnitude": 0.0},
                         "background": background, "warmup_s": 3600.0, "backup_available": backup},
            "outcome": {"total": 10.0, "shutdown": None, "hidden": {"cause": cause}, "behaviour_mode": mode}}


def _behaviour(mode, explored, actions, probabilities):
    return {"mode": mode, "epsilon": 0.3, "explored": explored, "actions": actions, "probabilities": probabilities,
            "empty_plan_probability": 0.0 if actions and not explored else 0.7}


def _episodes():
    c1 = [_cand("trace_link", "observe"), _cand("switch_to_backup", "respond"),
          _cand("monitor", "monitor", "plant", {"review_after_s": 300}),
          _cand("monitor", "monitor", "plant", {"review_after_s": 900})]
    d1 = _decision(4500.0, 4200.0, c1, [[10, 12], [110, 16], [11, 13], [11, 13]],
                   trips=[[0, 0], [1, 0], [0, 0], [0, 0]], oracle=3600.0,
                   statuses=[["succeeded", "running"]] + [["succeeded"] * 2] * 3,
                   baselines={"act_first": [c1[1]["action_id"]], "investigate_first": [c1[0]["action_id"]]},
                   behaviour=_behaviour("act_first", False, [c1[1]["action_id"]], [0.775]),
                   valid=("e1", "e2", "receipt:1", "receipt:2", "receipt:3", "receipt:7"),
                   submitted=[c1[1]["action_id"]])
    c2 = [_cand("repair_link", "respond"), _cand("monitor", "monitor", "plant", {"review_after_s": 300})]
    d2 = _decision(9000.0, 4200.0, c2, [[110, 110], [105, 105]], trips=[[1, 1], [1, 1]],
                   statuses=[["rejected"] * 2, ["rejected"] * 2],
                   baselines={"act_first": [], "investigate_first": [c2[0]["action_id"]]},
                   behaviour=_behaviour("act_first", True, [c2[0]["action_id"]], [0.15]))
    c3 = [_cand("monitor", "monitor", "plant", {"review_after_s": 300}), _cand("escalate", "escalate", "incident:1"),
          _cand("query_historian", "observe", "source:reactor")]
    d3 = _decision(3900.0, 4400.0, c3, [[1, 1], [5, 5], [1.04, 1.04]], kind="disturbance", cause="disturbance:IDV4",
                   onset_happened=False, baselines={"act_first": [], "investigate_first": [c3[2]["action_id"]]},
                   behaviour=_behaviour("investigate_first", False, [], []), compute=[0.0, 4.0, 0.04])
    own = c1[1]["action_id"]
    receipts = [_receipt(1, 4400.0, "accepted"), _receipt(2, 4400.0, "rejected"),
                _receipt(3, 4410.0, "accepted", "repair_link"),
                _receipt(4, 4415.4, "running", "repair_link", retracted=4495.0),
                _receipt(5, 4500.0, "accepted", "switch_to_backup", action=own),
                _receipt(6, 4505.4, "running", "switch_to_backup", action=own, retracted=4540.0),
                _receipt(7, 4495.0, "succeeded", "repair_link", supersedes=["receipt:4"]),
                _receipt(8, 4540.0, "succeeded", "switch_to_backup", action=own, supersedes=["receipt:6"])]
    link_items = [{"evidence_id": "e1", "kind": "alert"}, {"evidence_id": "e2", "kind": "window_summary"}, *receipts,
                  _tool_result(1, 4530.0)]
    return {"train": [_episode(SEEDS["train"][0], "link_fault", "sensor:reactor", [d1, d2], items=link_items),
                      _episode(SEEDS["train"][1], "sensor_fault", "feed_A_flow", [], fault_kind="drift")],
            "val": [_episode(SEEDS["val"][0], "disturbance", "IDV4", [d3], mode="investigate_first", background=(8,),
                             items=[{"evidence_id": "e1", "kind": "alert"}], backup=False)]}


def test_pairwise_noise_compares_paired_differences_with_independent_replicates(ds):
    total = np.array([[10.0, 12.0], [11.0, 13.0], [20.0, 16.0], [11.0, 13.0]])
    p = ds.pairwise_noise(total)
    np.testing.assert_allclose(p["sd_diff"], [0.0, np.sqrt(18.0), 0.0, np.sqrt(18.0), 0.0, np.sqrt(18.0)])
    np.testing.assert_allclose(p["sd_indep"], [2.0, np.sqrt(10.0), 2.0, np.sqrt(10.0), 2.0, np.sqrt(10.0)])
    assert p["identical"].tolist() == [False, False, False, False, True, False]


def test_loader_selects_splits_by_seed_and_never_opens_test_files(ds, tmp_path):
    eps = _episodes()
    for split in ("train", "val"):
        for e in eps[split]:
            (tmp_path / f"ep_{e['seed']:06d}.pkl").write_bytes(pickle.dumps(e))
    (tmp_path / f"ep_{SEEDS['test'][0]:06d}.pkl").write_bytes(b"not a pickle: must never be opened")
    loaded = ds.load_episodes(tmp_path)
    assert sorted(e["seed"] for e in loaded["train"]) == sorted(SEEDS["train"])
    assert [e["seed"] for e in loaded["val"]] == [SEEDS["val"][0]]
    with pytest.raises(ValueError):
        ds.load_episodes(tmp_path, ("train", "test"))
    with pytest.raises(ValueError):
        ds.load_arrays(tmp_path, ("test",))
    meta = {"fractions": [0.6, 0.2, 0.2], "splits": {
        "train": {"episodes": 1, "decisions": 2, "candidates": 6, "seeds": [SEEDS["train"][0]]},
        "val": {"episodes": 1, "decisions": 1, "candidates": 3, "seeds": [SEEDS["val"][0]]},
        "test": {"episodes": 1, "decisions": 4, "candidates": 40, "seeds": [SEEDS["test"][0]]}}}
    table = ds.split_table(meta, tmp_path)
    assert table["train"]["episodes_without_decisions"] == [SEEDS["train"][1]]
    assert table["test"]["episode_files"] == 1 and table["test"]["decisions"] == 4
    assert table["total"] == {"episode_files": 4, "episodes_with_decisions": 3, "decisions": 7, "candidates": 49}


def test_a_split_without_any_decision_still_has_a_row_and_its_files_count(ds, tmp_path):
    for s in ("train", "val", "test"):
        (tmp_path / f"ep_{SEEDS[s][0]:06d}.pkl").write_bytes(b"never opened")
    meta = {"fractions": [0.6, 0.2, 0.2], "splits": {
        "train": {"episodes": 1, "decisions": 2, "candidates": 6, "seeds": [SEEDS["train"][0]]},
        "test": {"episodes": 1, "decisions": 4, "candidates": 40, "seeds": [SEEDS["test"][0]]}}}
    table = ds.split_table(meta, tmp_path)
    assert table["val"] == {"episode_files": 1, "episodes_with_decisions": 0, "decisions": 0, "candidates": 0,
                            "episodes_without_decisions": [SEEDS["val"][0]], "seeds": []}
    assert table["total"] == {"episode_files": 3, "episodes_with_decisions": 2, "decisions": 6, "candidates": 46}
    stats = ds.compute(_episodes())
    stats["splits"], stats["provenance"] = table, {"kernel": {"match": True}}
    head = ds.headline(stats)
    assert head["decisions_by_split"] == {"train": 2, "val": 0, "test": 4} and head["episodes"] == 3


def test_block_statistics_have_the_known_answers(ds):
    dev = ds.jsonable(ds.compute(_episodes()))["development"]
    ep, dec = dev["episodes"], dev["decisions"]
    assert ep["n"] == 3 and ep["with_decisions"] == 2
    assert [e["seed"] for e in ep["without_decisions"]] == [SEEDS["train"][1]]
    assert ep["incident_kind"]["counts"] == {"link_fault": 1, "sensor_fault": 1, "disturbance": 1}
    assert ep["background"]["counts"] == {"none": 2, "IDV8": 1}
    assert ep["backup"]["counts"] == {"available": 2, "unavailable": 1}
    assert dec["backup"]["counts"] == {"available": 2, "unavailable": 1}
    assert dec["n"] == 3 and dec["pre_onset_share"] == pytest.approx(1 / 3)
    assert dec["since_onset_min"]["median"] == pytest.approx((5.0 + 80.0) / 2)
    assert dec["before_onset_min"]["max"] == pytest.approx(500.0 / 60)
    assert dec["oracle_deadline_share"] == pytest.approx(0.5)

    st = dev["stakes"]
    assert st["high_stakes_share"] == pytest.approx(1 / 3) and st["unavoidable_trip_share"] == pytest.approx(1 / 3)
    assert st["loose_definition_share"] == pytest.approx(1 / 3) and st["loose_definition_without_trip_difference"] == 0
    assert "train_direct_definition_share" not in st
    assert st["trip_free_share"] == pytest.approx(1 / 3) and st["decisive_share"] == 0.0
    assert st["trip_difference_ge_half_share"] == pytest.approx(1 / 3)
    assert st["tool_cost_only_share"] == pytest.approx(1 / 3)
    assert st["outcome_range_per_decision"]["max"] == pytest.approx(52.0)
    assert st["high_stakes_by_backup"] == {"available": {"n": 2, "share": 0.5}, "unavailable": {"n": 1, "share": 0.0}}

    best = dev["best"]
    assert best["tool_share"] == pytest.approx({"monitor": 2 / 3, "trace_link": 1 / 3})
    assert best["tie_share"] == 0.0 and best["wait_only_argmin_share"] == pytest.approx(2 / 3)
    assert best["gap_to_runner_up"]["min"] == pytest.approx(0.04) and best["gap_to_runner_up"]["max"] == 5.0

    voi = dev["voi"]
    assert voi["decisions_with_both"] == 2 and voi["gather_beats_act_share"] == pytest.approx(0.5)
    assert voi["observe_candidates"]["max"] == pytest.approx(1.0)
    assert voi["observe_candidates"]["min"] == pytest.approx(-0.04)
    assert voi["positive_share"] == pytest.approx(0.5)
    assert voi["gather_beats_act_tool_cost_only_share"] == 0.0
    assert voi["gather_margin_by_component"]["service"]["mean"] == pytest.approx(1.0)
    assert voi["gather_beats_act_by_backup"] == {"available": {"n": 1, "share": 1.0},
                                                 "unavailable": {"n": 1, "share": 0.0}}
    assert voi["gather_beats_act_high_stakes_by_backup"] == {"available": {"n": 1, "share": 1.0}}

    pb = dev["playbooks"]
    assert pb["act_first"]["regret"]["mean"] == pytest.approx(52.0 / 3)
    assert pb["act_first"]["zero_regret_share"] == pytest.approx(2 / 3)
    assert pb["investigate_first"]["regret"]["mean"] == pytest.approx((0.0 + 5.0 + 0.04) / 3)
    assert pb["act_first"]["wait_share"] == pytest.approx(2 / 3) and pb["modes_agree_share"] == 0.0
    assert pb["reference"]["random_candidate_regret"]["mean"] == pytest.approx((13.5 + 2.5 + 7.04 / 3 - 1.0) / 3)

    beh = dev["behaviour"]
    assert beh["explored_share"] == pytest.approx(1 / 3) and beh["empty_plan_share"] == pytest.approx(1 / 3)
    assert beh["empty_plan_implied_probability"]["counts"] == {"0.7": 1}
    assert beh["empty_plan_recorded_probability"]["counts"] == {"0.7": 1}
    assert beh["regret"]["mean"] == pytest.approx((52.0 + 5.0 + 0.0) / 3)
    assert beh["first_action_probability"]["n"] == 2 and beh["taken_probability"]["n"] == 3
    assert beh["taken_probability"]["max"] == pytest.approx(0.775) and beh["taken_probability"]["median"] == 0.7

    ch = dev["checks"]
    assert ch["all_candidates_rejected_share"] == pytest.approx(1 / 3)
    assert ch["monitor_pair_identical_share"] == 1.0 and ch["monitor_pair_identical_labels_share"] == 1.0
    assert ch["gather_beats_act_tool_cost_only_share"] == 0.0
    assert ch["first_action_censored_share"] == pytest.approx(1 / 18)
    assert ch["first_action_censored_by_tool"] == pytest.approx(
        {"trace_link": 0.5, "switch_to_backup": 0.0, "monitor": 0.0, "repair_link": 0.0, "escalate": 0.0,
         "query_historian": 0.0})
    assert ch["receipt_items"] == 8 and ch["episodes_with_receipt_items_share"] == pytest.approx(0.5)
    assert ch["decisions_with_valid_receipt_share"] == pytest.approx(1 / 3)
    assert ch["receipts_valid_by_time_not_known"] == 1
    assert ch["other_items_valid_by_time_not_known"] == 0
    assert ch["refusals_at_accept_time"] == 1
    assert ch["decisions_with_valid_evidence_missing_share"] == pytest.approx(1 / 3)
    assert ch["own_action_receipts_stored"] == 3 and ch["receipt_items_without_action_id"] == 0
    assert ch["own_action_tool_results_stored"] == 0 and ch["tool_results_without_action_id"] == 0
    assert dev["labels"]["first_duration_min_by_tool"]["trace_link"]["censored"] == 1
    assert dev["labels"]["p_success_by_tool"]["trace_link"] == 1.0

    q = dev["quality"]
    assert q["label_identical_candidates"] == 1 and q["monitor_pair_identical_share"] == 1.0
    assert q["all_candidates_rejected_decisions"] == 1 and q["candidates_rejected_in_every_replicate"] == 2
    assert q["duplicate_decisions_same_episode_time"] == 0 and q["all_J_equal_decisions"] == 0

    noise = dev["noise"]
    assert noise["pooled_variance_ratio"] == pytest.approx(13824.0 / 13268.0)
    assert noise["pairs"] == 6 + 1 + 3 and noise["pairs_identical_share"] == pytest.approx(0.1)
    assert noise["trip_disagreement_share"] == pytest.approx(1 / 9)
    assert noise["variance_share_by_component"]["physical"] == pytest.approx(5000 / 5024)

    assert dev["candidates"]["n"] == 9 and dev["candidates"]["distinct_outcomes_per_decision"]["mean"] == \
        pytest.approx((3 + 2 + 3) / 3)
    assert dev["labels"]["p_trip"]["one_share"] == pytest.approx(2 / 9)
    val = ds.jsonable(ds.compute(_episodes()))["val"]
    assert val["decisions"]["n"] == 1 and val["stakes"]["trip_free_share"] == 1.0


def test_consistency_detects_arrays_that_do_not_reproduce_from_raw(ds):
    eps = _episodes()["train"]
    decs = [(e, d) for e in eps for d in e["decisions"]]
    labs = [candidate_labels(d) for _, d in decs]
    arrays = {f"y_{k}": np.concatenate([lab[k] for lab in labs]) for k in ("J", "J_std", "p_trip", "voi", "p_resolved",
                                                                            "p_success", "duration")}
    arrays.update({f"y_{k}": np.concatenate([lab[k] for lab in labs]).astype(np.float64)
                   for k in ("first_censored", "duration_time", "duration_censored")})
    arrays.update({"dec_tau": np.array([d["tau"] for _, d in decs]),
                   "dec_episode_seed": np.array([e["seed"] for e, _ in decs]), "dec_kind": np.array([1, 1]),
                   "dec_explored": np.array([False, True]), "dec_n_candidates": np.array([4, 2]),
                   "dec_baseline_act_first": np.array([baseline_index(d, "act_first") for _, d in decs]),
                   "dec_baseline_investigate_first": np.array([0, 0]), "dec_backup_available": np.array([True, True])})
    beliefs = [belief_labels(d, in_family(e)) for e, d in decs]
    arrays.update({f"y_{k}": np.concatenate([b[k] for b in beliefs]) for k in BELIEF_KEYS})
    arrays.update({f"dec_{k}": np.array([b[k] for b in beliefs]) for k in BELIEF_DECISION_KEYS})
    assert all(v["match"] for v in ds.consistency(eps, arrays).values())
    assert arrays["y_first_censored"][0].tolist() == [0.0, 1.0]
    arrays["y_J"] = arrays["y_J"] + np.r_[0.0, 0.5, np.zeros(arrays["y_J"].size - 2)]
    arrays["y_first_censored"] = np.zeros_like(arrays["y_first_censored"])
    arrays["dec_backup_available"] = np.array([True, False])
    arrays["y_J_belief"] = arrays["y_J_belief"] + 0.25
    arrays["dec_belief_fallback"] = ~arrays["dec_belief_fallback"]
    out = ds.consistency(eps, arrays)
    assert not out["y_J"]["match"] and out["y_J"]["max_abs_diff"] == pytest.approx(0.5)
    assert out["y_p_trip"]["match"] and not out["y_first_censored"]["match"]
    assert not out["dec_backup_available"]["match"]
    assert not out["y_J_belief"]["match"] and not out["dec_belief_fallback"]["match"]
    assert out["dec_posterior"]["match"]
    del arrays["y_duration_time"]
    assert ds.consistency(eps, arrays)["y_duration_time"] == {"match": None, "missing_in_arrays": True}


def _rows(decision_times, slots):
    m = max(len(s) for s in slots)
    item_id = np.array([list(s) + [-1] * (m - len(s)) for s in slots], np.int64)
    return {"decision_time": np.asarray(decision_times, np.float64), "item_id": item_id, "present": item_id >= 0}


def test_array_check_finds_receipts_of_a_rows_own_actions_valid_or_not(ds):
    eps = _episodes()["train"]
    keys = [[0, "e1"], [0, "receipt:6"], [0, "receipt:1"], [0, "receipt:5"], [0, "receipt:3"]]
    clean = _rows([4500.0, 9000.0], [[0, 2, 4], [0, 3]])
    zero = {"rows": 2, "own_action_receipt_slots": 0, "rows_with_own_action_receipts": 0,
            "later_action_receipt_slots": 0, "receipt_slots_without_action_id": 0, "own_action_tool_result_slots": 0,
            "rows_with_own_action_tool_results": 0, "later_action_tool_result_slots": 0,
            "tool_result_slots_without_action_id": 0}
    assert ds.own_action_items_in_rows(eps, clean, keys) == zero
    leaky = _rows([4500.0, 9000.0], [[0, 1, 2], [0]])
    assert ds.own_action_items_in_rows(eps, leaky, keys)["own_action_receipt_slots"] == 1
    assert ds.own_action_items_in_rows(eps, _rows([4500.0, 9000.0], [[3], [0]]),
                                       keys)["rows_with_own_action_receipts"] == 1
    later = copy.deepcopy(eps)
    later[0]["items"].append(_receipt(9, 4503.6, "accepted", "escalate", "incident:1", action="escalate@4503.6"))
    later[0]["submissions"] = [{"time": 4500.0, "action_id": later[0]["decisions"][0]["submitted_action_ids"][0]},
                               {"time": 4503.6, "action_id": "escalate@4503.6"}]
    rows = _rows([4500.0, 9000.0], [[0, 5], [0, 5]])
    counted = ds.own_action_items_in_rows(later, rows, keys + [[0, "receipt:9"]])
    assert counted["later_action_receipt_slots"] == 1 and counted["own_action_receipt_slots"] == 0
    unnamed = _episodes()["train"]
    del unnamed[0]["items"][2]["payload"]["action_id"]
    assert ds.own_action_items_in_rows(unnamed, clean, keys)["receipt_slots_without_action_id"] == 1
    with pytest.raises(ValueError):
        ds.own_action_items_in_rows(eps, _rows([4500.0, 9001.0], [[0], [0]]), keys)
    with pytest.raises(ValueError):
        ds.own_action_items_in_rows(eps, clean, [[1, "e1"]] + keys[1:])
    stats = ds.compute(_episodes(), {"train": leaky}, None, {"train": keys})
    assert stats["array_checks"]["train"]["own_action_receipt_slots"] == 1
    stats["splits"] = {s: {"decisions": 1, "episode_files": 1} for s in SPLITS}
    stats["splits"]["total"] = {"decisions": 3, "episode_files": 3}
    stats["provenance"] = {"kernel": {"match": True}}
    head = ds.jsonable(ds.headline(stats))
    assert head["own_action_receipts_in_rows"] == 1 and head["receipt_slots_without_action_id"] == 0
    assert head["later_action_receipts_in_rows"] == 0
    assert head["checks"]["own_action_receipts_stored"] == 3
    assert head["gather_beats_act_by_backup"]["unavailable"] == {"n": 1, "share": 0.0}
    assert ds.headline({**stats, "array_checks": None})["own_action_receipts_in_rows"] is None


def _with_own_trace(eps):
    eps = copy.deepcopy(eps)
    link = eps["train"][0]
    d1 = link["decisions"][0]
    trace = d1["candidates"][0]["action_id"]
    d1["behaviour"]["actions"].append(trace)
    d1["behaviour"]["probabilities"].append(0.075)
    d1["submitted_action_ids"].append(trace)
    link["submissions"] = [{"time": 4500.0, "action_id": a} for a in d1["submitted_action_ids"]] + [
        {"time": 4503.6, "action_id": "escalate@4503.6"}]
    link["items"] += [_tool_result(2, 4580.0, action=trace),
                      _tool_result(3, 4563.6, "operator_report", "incident:1", action="escalate@4503.6"),
                      _tool_result(4, 4600.0, action=None)]
    return eps


def test_array_check_finds_tool_results_of_a_rows_own_and_later_actions(ds):
    eps = _with_own_trace(_episodes())
    keys = [[0, "e1"], [0, "tool:trace_link:1"], [0, "tool:trace_link:2"], [0, "tool:operator_report:3"],
            [0, "tool:trace_link:4"]]
    rows = _rows([4500.0, 9000.0], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    assert ds.own_action_items_in_rows(eps["train"], rows, keys) == {
        "rows": 2, "own_action_receipt_slots": 0, "rows_with_own_action_receipts": 0, "later_action_receipt_slots": 0,
        "receipt_slots_without_action_id": 0, "own_action_tool_result_slots": 1, "rows_with_own_action_tool_results": 1,
        "later_action_tool_result_slots": 1, "tool_result_slots_without_action_id": 2}
    clean = _rows([4500.0, 9000.0], [[0, 1], [0, 1, 2, 3]])
    assert not any(ds.own_action_items_in_rows(eps["train"], clean, keys)[k] for k in (
        "own_action_tool_result_slots", "later_action_tool_result_slots", "tool_result_slots_without_action_id"))
    stats = ds.compute(eps, {"train": rows}, None, {"train": keys})
    stats["splits"] = {s: {"decisions": 1, "episode_files": 1} for s in SPLITS}
    stats["splits"]["total"] = {"decisions": 3, "episode_files": 3}
    stats["provenance"] = {"kernel": {"match": True}}
    head = ds.jsonable(ds.headline(stats))
    assert (head["own_action_tool_results_in_rows"], head["later_action_tool_results_in_rows"],
            head["tool_result_slots_without_action_id"]) == (1, 1, 2)
    assert head["own_action_receipts_in_rows"] == 0
    assert ds.headline({**stats, "array_checks": None})["own_action_tool_results_in_rows"] is None


def test_tool_results_of_a_decisions_own_actions_stored_with_the_episode_are_counted(ds):
    ch = ds.jsonable(ds.compute(_with_own_trace(_episodes())))["development"]["checks"]
    assert ch["own_action_tool_results_stored"] == 1
    assert ch["tool_results_without_action_id"] == 1
    assert ch["own_action_receipts_stored"] == 3 and ch["receipt_items_without_action_id"] == 0


def test_receipts_that_name_no_action_are_counted(ds):
    eps = _episodes()
    del eps["train"][0]["items"][3]["payload"]["action_id"]
    del eps["train"][0]["items"][4]["payload"]["action_id"]
    assert ds.jsonable(ds.compute(eps))["development"]["checks"]["receipt_items_without_action_id"] == 2


def test_monitor_pair_compares_every_outcome_label_not_only_J(ds):
    cands = [_cand("monitor", "monitor", "plant", {"review_after_s": 300}),
             _cand("monitor", "monitor", "plant", {"review_after_s": 900}), _cand("trace_link", "observe")]
    same_total = _decision(4500.0, 4200.0, cands, [[3, 3], [3, 3], [4, 4]], compute=[0.0, 1.0, 0.2])
    J, labels = ds.monitor_pair(same_total, ds.outcome_labels(same_total), ds.result_matrix(same_total, "total"))
    assert J and not labels
    identical = _decision(4500.0, 4200.0, cands, [[3, 3], [3, 3], [4, 4]])
    assert ds.monitor_pair(identical, ds.outcome_labels(identical), ds.result_matrix(identical, "total")) == \
        (True, True)
    one = _decision(4500.0, 4200.0, cands[::2], [[3, 3], [4, 4]])
    assert ds.monitor_pair(one, ds.outcome_labels(one), ds.result_matrix(one, "total")) == (None, None)


def test_headline_uses_the_renamed_high_stakes_key_and_carries_the_checks(ds):
    stats = ds.compute(_episodes())
    stats["splits"] = {s: {"decisions": 1, "episode_files": 1} for s in SPLITS}
    stats["splits"]["total"] = {"decisions": 3, "episode_files": 3}
    stats["provenance"] = {"kernel": {"match": True}}
    head = ds.jsonable(ds.headline(stats))
    assert head["loose_high_stakes_without_trip_difference"] == 0
    assert "train_direct_high_stakes_without_trip_difference" not in head
    assert head["checks"]["refusals_at_accept_time"] == 1 and "first_action_censored_by_tool" not in head["checks"]
    assert head["checks"]["first_action_censored_share"] == pytest.approx(1 / 18)


def _twin_evidence(prior=(0.5, 0.5), identical=True, windows=()):
    return {"group": "reactor", "causes": ["link_fault", "io_module_fault"], "prior": list(prior), "tau": 5000.0,
            "params": belief.load_params(), "deterministic_identical": identical,
            "first_difference": None if identical else [4900.0, "win:reactor:1"], "windows": list(windows),
            "traces": [], "operator": []}


def _twinned(ep, twin_totals, evidence, receipt_differs=False, first=None):
    d = ep["decisions"][0]
    d["twin"] = {"cause": "twin", "labelled": True, "belief": evidence, "cited_evidence": [],
                 "results": _decision(d["tau"], 4200.0, d["candidates"], twin_totals)["results"],
                 "candidates_unmatched": [], "hidden": None, "pi_after_memory_differs": False}
    ep["twin"] = {"divergence": {
        "receipt_status_differs": receipt_differs,
        "receipt_differences": [{"key": "k", "primary": [["succeeded", None]], "twin": [["rejected", "stale_state"]]}]
        if receipt_differs else [],
        "first_deterministic_difference": first}, "replayed": [{"matched": True, "citation_changed": False}],
        "not_replayed": [], "outcome": {"shutdown": None}}
    return ep


def _belief_episodes():
    trace, switch, repair = _cand("trace_link", "observe"), _cand("switch_to_backup", "respond"), \
        _cand("repair_link", "respond")
    wait = _cand("monitor", "monitor", "plant", {"review_after_s": 300})
    beh = _behaviour("act_first", False, [], [])
    base = {"act_first": [], "investigate_first": []}
    a = _twinned(_episode(SEEDS["train"][0], "link_fault", "sensor:reactor",
                          [_decision(4500.0, 4200.0, [trace, switch, wait], [[12, 12], [10, 10], [11, 11]],
                                     behaviour=beh, baselines=base)]),
                 [[12, 12], [30, 30], [25, 25]], _twin_evidence(),
                 first={"time": 4800.0, "kind": "window_summary", "since_onset_s": 600.0})
    b = _twinned(_episode(SEEDS["train"][1], "io_module_fault", "reactor",
                          [_decision(4500.0, 4200.0, [trace, repair], [[12, 12], [10, 10]], kind="io_module_fault",
                                     cause="io_module_fault:reactor", behaviour=beh, baselines=base)], backup=False),
                 [[12, 12], [40, 40]], _twin_evidence(identical=False), receipt_differs=True)
    report = {"w": 1, "end": 4000.0, "modes": ["down", "up"], "report": [1.0, 0.0], "age": 1000.0}
    c = _twinned(_episode(SEEDS["val"][0], "link_fault", "sensor:stripper",
                          [_decision(4500.0, 4200.0, [trace, repair], [[5, 5], [8, 8]],
                                     cause="link_fault:sensor:stripper", behaviour=beh, baselines=base)], backup=False),
                 [[5, 5], [8, 8]], _twin_evidence(windows=[report]))
    d = _episode(SEEDS["val"][1], "disturbance", "IDV6",
                 [_decision(4500.0, 4200.0, [_cand("query_historian", "observe", "source:reactor"), wait],
                            [[1.04, 1.04], [1, 1]], kind="disturbance", cause="disturbance:IDV6", behaviour=beh,
                            baselines=base)])
    return {"train": [a, b], "val": [c, d]}


def test_belief_statistics_have_the_known_answers(ds):
    dev = ds.jsonable(ds.compute(_belief_episodes()))["development"]
    bs = dev["belief"]
    assert (bs["family_decisions"], bs["belief_decisions"], bs["decisions_with_both"]) == (3, 3, 4)
    assert bs["twin_labelled_share_in_family"] == 1.0 and bs["fallback_share_in_family"] == 0.0
    assert bs["gather_beats_act_share"] == pytest.approx(0.5) and bs["gather_beats_act_family_share"] == \
        pytest.approx(2 / 3)
    assert bs["gather_beats_act_by_backup"] == {"available": {"n": 2, "share": 0.5},
                                                "unavailable": {"n": 2, "share": 0.5}}
    assert bs["gather_beats_act_family_by_backup"] == {"available": {"n": 1, "share": 1.0},
                                                       "unavailable": {"n": 2, "share": 0.5}}
    assert bs["gather_beats_act_by_group"] == {"outside_family": {"n": 1, "share": 0.0},
                                               "reactor": {"n": 2, "share": 0.5}, "stripper": {"n": 1, "share": 1.0}}
    assert dev["voi"]["gather_beats_act_share"] == pytest.approx(0.25)
    hb = bs["hindsight_vs_belief"]
    assert hb["gather_beats_act_hindsight_share"] == pytest.approx(1 / 3)
    assert hb["gather_beats_act_belief_share"] == pytest.approx(2 / 3)
    assert hb["gather_only_under_belief_share"] == pytest.approx(1 / 3) and hb["gather_only_under_hindsight_share"] == 0
    assert hb["argmin_agree_share"] == pytest.approx(2 / 3)
    assert hb["hindsight_choice_belief_regret"]["max"] == pytest.approx(8.0)
    assert hb["hindsight_choice_belief_regret"]["mean"] == pytest.approx(8.0 / 3)
    assert hb["belief_choice_hindsight_regret"]["max"] == pytest.approx(2.0)
    assert hb["twin_excluded_share"] == pytest.approx(1 / 3)
    p_c = 0.5 * 0.05 / (0.5 * 0.05 + 0.5 * 0.0003)
    assert hb["posterior_true_cause"]["max"] == pytest.approx(1.0) and hb["posterior_true_cause"]["min"] == 0.5
    assert bs["voi"]["observe_candidates"]["max"] == pytest.approx(6.0)
    assert bs["voi"]["positive_share"] == pytest.approx(0.5) and bs["voi"]["positive_share_family"] == \
        pytest.approx(2 / 3)
    assert bs["voi"]["by_group"]["reactor"]["n"] == 2 and bs["voi"]["by_backup"]["unavailable"]["max"] == pytest.approx(3.0)
    cal = bs["calibration"]
    assert cal["n"] == 3 and [b["n"] for b in cal["bins"]] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert cal["bins"][5] == {"bin": [0.5, pytest.approx(0.6)], "n": 1, "mean_p_link": 0.5, "link_frequency": 1.0}
    assert cal["bins"][9]["mean_p_link"] == pytest.approx(p_c) and cal["bins"][0]["link_frequency"] == 0.0
    assert cal["brier"] == pytest.approx((0.25 + 0.0 + (1 - p_c) ** 2) / 3)
    assert cal["ece"] == pytest.approx((0.5 + 0.0 + (1 - p_c)) / 3)
    assert bs["calibration_post_onset"]["n"] == 3
    tw = dev["twins"]
    assert (tw["family_episodes"], tw["twinned_episodes"]) == (3, 3)
    assert tw["receipt_status_differs_share"] == pytest.approx(1 / 3) and tw["receipt_differences"] == 1
    assert tw["diverged_share"] == pytest.approx(1 / 3) and tw["first_difference_since_onset_min"]["median"] == 10.0
    assert tw["first_difference_kind"]["counts"] == {"window_summary": 1}
    assert tw["decisions"] == {"n": 3, "labelled_share": 1.0, "deterministic_identical_share": pytest.approx(2 / 3),
                               "candidates_unmatched": 0, "pi_after_memory_differs_share": 0.0}
    assert tw["replayed_submissions"] == 3 and tw["replays_citation_changed"] == 0
    stats = ds.compute(_belief_episodes())
    stats["splits"] = {s: {"decisions": 1, "episode_files": 1} for s in SPLITS}
    stats["splits"]["total"] = {"decisions": 3, "episode_files": 3}
    stats["provenance"] = {"kernel": {"match": True}}
    head = ds.jsonable(ds.headline(stats))["belief"]
    assert head["gather_beats_act_family_share"] == pytest.approx(2 / 3)
    assert head["hindsight_gather_beats_act_family_share"] == pytest.approx(1 / 3)
    assert head["calibration_ece"] == pytest.approx(cal["ece"]) and head["twin_receipt_status_differs_share"] == \
        pytest.approx(1 / 3)


def test_family_decisions_without_twins_fall_back_and_are_counted(ds):
    bs = ds.jsonable(ds.compute(_episodes()))["development"]["belief"]
    assert bs["family_decisions"] == 2 and bs["belief_decisions"] == 0
    assert bs["family_without_twin_record"] == 2 and bs["posterior_undefined_in_family"] == 2
    assert bs["fallback_share_in_family"] == 1.0 and bs["calibration"]["n"] == 0
    assert bs["gather_beats_act_share"] == pytest.approx(0.5)
    d = _episodes()["train"][0]["decisions"][0]
    row = ds.belief_fields({"seed": 1, "scenario": {}}, d, candidate_labels(d))
    assert (row["family"], row["family_group"], row["belief_fallback"], row["posterior"]) == (
        False, "outside_family", True, 1.0)
    np.testing.assert_array_equal(row["J_belief"], candidate_labels(d)["J"])


def test_summary_and_json_conversion_handle_missing_values(ds):
    s = ds.summary([1.0, None, np.nan, 3.0], scale=2.0)
    assert s["n"] == 2 and s["mean"] == 4.0 and s["max"] == 6.0
    assert ds.summary([None, np.inf]) == {"n": 0}
    assert ds.jsonable({"a": np.float64(np.nan), "b": np.int64(3), "c": (np.bool_(True),)}) == \
        {"a": None, "b": 3, "c": [True]}
