import copy
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from cpsdefense.agents.playbook import Playbook, citable
from cpsdefense.core.types import EvidenceItem
from cpsdefense.defense.evidence import EvidenceLog
from cpsdefense.defense.registry import load_registry
from cpsdefense.experience import belief
from cpsdefense.experience.dataset import (ACTION_ITEM_KINDS, BELIEF_DECISION_KEYS, BELIEF_KEYS, LABEL_FORMAT,
                                           LABEL_KEYS, SPLITS, TOOL_RESULT_KINDS, belief_labels, build,
                                           candidate_labels, in_family, split_of)
from cpsdefense.utils.config import load_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SEED = {s: next(k for k in range(1000) if split_of(k) == s) for s in SPLITS}


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _script("generate_experience")


@pytest.fixture(scope="module")
def bd():
    return _script("build_dataset")


def _result(candidate, replicate, total, shutdown=False, tts=None, resolved=True, status="succeeded", dur=60.0,
            **extra):
    return {"candidate": candidate, "replicate": replicate, "total": total, "shutdown": shutdown,
            "time_to_shutdown": tts, "horizon_s": 3600.0, "resolved": resolved, "mitigated": resolved,
            "first_status": status, "first_duration": dur, **extra}


def test_split_is_deterministic_by_scenario_and_roughly_proportional():
    seeds = range(5000)
    splits = [split_of(s) for s in seeds]
    assert splits == [split_of(s) for s in seeds]
    share = {s: splits.count(s) / len(splits) for s in SPLITS}
    assert share["train"] == pytest.approx(0.6, abs=0.03) and share["val"] == pytest.approx(0.2, abs=0.03)


def test_candidate_labels_aggregate_replicates_and_define_voi_against_the_best_action():
    decision = {
        "candidates": [{"family": "observe"}, {"family": "respond"}, {"family": "monitor"}],
        "results": [_result(0, 0, 10.0), _result(0, 1, 14.0),
                    _result(1, 0, 20.0, shutdown=True, tts=1800.0, resolved=False, status="failed", dur=None),
                    _result(1, 1, 16.0),
                    _result(2, 0, 11.0, resolved=None), _result(2, 1, 13.0, resolved=None)],
    }
    lab = candidate_labels(decision)
    np.testing.assert_allclose(lab["J"], [12.0, 18.0, 12.0])
    np.testing.assert_allclose(lab["regret"], [0.0, 6.0, 0.0])
    np.testing.assert_allclose(lab["p_trip"], [0.0, 0.5, 0.0])
    np.testing.assert_allclose(lab["trip_time"], [[3600, 3600], [1800, 3600], [3600, 3600]])
    assert lab["trip_event"].tolist() == [[False, False], [True, False], [False, False]]
    assert lab["voi"][0] == pytest.approx(12.0 - 12.0) and np.isnan(lab["voi"][1:]).all()
    assert lab["p_success"].tolist() == [1.0, 0.5, 1.0]
    assert lab["duration"][1] == pytest.approx(60.0)
    assert lab["p_resolved"][1] == pytest.approx(0.5) and np.isnan(lab["p_resolved"][2])
    assert not lab["first_censored"].any() and not lab["duration_censored"].any()


def test_first_actions_still_open_when_the_branch_ends_are_censored_not_failed():
    decision = {
        "candidates": [{"family": "respond"}, {"family": "respond"}, {"family": "observe"}],
        "results": [_result(0, 0, 5.0, dur=3000.0), _result(0, 1, 5.0, status="failed", dur=2900.0),
                    _result(0, 2, 3.0, status="running", dur=None, first_started=5.4),
                    _result(0, 3, 105.0, shutdown=True, tts=1800.0, status="running", dur=None),
                    _result(1, 0, 4.0, status="running", dur=None, first_started=5.4),
                    _result(1, 1, 4.0, status="accepted", dur=None),
                    _result(1, 2, 4.0, status="running", dur=None, first_started=5.4),
                    _result(1, 3, 104.0, shutdown=True, tts=3.6, status="accepted", dur=None),
                    _result(2, 0, 4.0, status="rejected", dur=None), _result(2, 1, 4.0, dur=100.0),
                    _result(2, 2, 4.0, dur=140.0), _result(2, 3, 4.0, dur=120.0)],
    }
    lab = candidate_labels(decision)
    assert lab["first_censored"].tolist() == [[False, False, True, True], [True] * 4, [False] * 4]
    assert lab["duration_censored"].tolist() == lab["first_censored"].tolist()
    assert lab["p_success"][0] == pytest.approx(0.5)
    assert np.isnan(lab["p_success"][1])
    assert lab["p_success"][2] == pytest.approx(0.75)
    np.testing.assert_allclose(lab["duration_time"], [[3000.0, 2900.0, 3600.0 - 5.4, 1800.0],
                                                      [3600.0 - 5.4, 0.0, 3600.0 - 5.4, 0.0],
                                                      [np.nan, 100.0, 140.0, 120.0]])
    assert lab["duration"][0] == pytest.approx(2950.0) and np.isnan(lab["duration"][1])
    assert lab["duration"][2] == pytest.approx(120.0)


class _Rng:
    def __init__(self, u, index=0):
        self.u, self.index = u, index

    def random(self):
        return self.u

    def integers(self, n):
        return self.index


@pytest.fixture(scope="module")
def playbook_states():
    pytest.importorskip("tep_studio")
    from cpsdefense.experience.rollout import relevant_candidates, relevant_evidence
    from cpsdefense.sim.incidents import Incident, Scenario
    from cpsdefense.sim.tep_env import TEPEnv

    env = TEPEnv(Scenario(seed=3, incident=Incident("link_fault", "sensor:stripper", 600.0), background=(),
                          horizon_s=3 * 3600))
    env.reset()
    pb = Playbook("act_first")
    states = {}
    while not env.done and "planning" not in states:
        pb.update(env.advance_to_decision(), env)
        key = "planning" if copy.deepcopy(pb).plan(env) else "quiet"
        if key not in states:
            states[key] = (env.snapshot(), copy.deepcopy(pb), relevant_candidates(env), relevant_evidence(env))
    assert set(states) == {"quiet", "planning"}
    return env, states


def _state(playbook_states, key):
    env, states = playbook_states
    snap, pb, cands, cited = states[key]
    env.restore(snap)
    return env, copy.deepcopy(pb), cands, cited


def test_behaviour_probabilities_keep_the_playbook_share_of_an_explored_pick(gen, playbook_states):
    env, pb, cands, cited = _state(playbook_states, "planning")
    plan = copy.deepcopy(pb).plan(env)
    ids = [a.action_id for a in cands]
    first, n = plan[0][0], len(cands)
    other = next(i for i, a in enumerate(cands) if a.action_id not in {p.action_id for p, _ in plan})
    _, rec = gen.behaviour_step(pb, env, cands, cited, 0.3, _Rng(0.1, index=ids.index(first.action_id)))
    assert rec["explored"] and rec["actions"] == [first.action_id]
    assert rec["probabilities"] == [pytest.approx(0.3 / n + 0.7)] and rec["empty_plan_probability"] == 0.0
    assert (first.tool_id, first.target_id) in pb.done
    assert all((a.tool_id, a.target_id) not in pb.done for a, _ in plan[1:])
    env, pb, cands, cited = _state(playbook_states, "planning")
    _, rec = gen.behaviour_step(pb, env, cands, cited, 0.3, _Rng(0.1, index=other))
    assert rec["probabilities"] == [pytest.approx(0.3 / n)] and rec["empty_plan_probability"] == 0.0
    env, pb, cands, cited = _state(playbook_states, "planning")
    _, rec = gen.behaviour_step(pb, env, cands, cited, 0.3, _Rng(0.9))
    assert not rec["explored"] and rec["actions"] == [a.action_id for a, _ in plan]
    assert rec["probabilities"] == [pytest.approx((0.3 / n if a.action_id in ids else 0.0) + 0.7) for a, _ in plan]


def test_behaviour_records_the_probability_of_an_empty_plan(gen, playbook_states):
    env, pb, cands, cited = _state(playbook_states, "quiet")
    plan, rec = gen.behaviour_step(pb, env, cands, cited, 0.3, _Rng(0.9))
    assert plan == [] and rec["actions"] == [] and rec["probabilities"] == []
    assert rec["empty_plan_probability"] == pytest.approx(0.7)
    env, pb, cands, cited = _state(playbook_states, "quiet")
    _, rec = gen.behaviour_step(pb, env, cands, cited, 0.3, _Rng(0.1, index=0))
    assert rec["actions"] == [cands[0].action_id] and rec["probabilities"] == [pytest.approx(0.3 / len(cands))]
    assert rec["empty_plan_probability"] == pytest.approx(0.7)


def test_replayed_submissions_cite_what_the_twin_can_and_mirror_the_behaviours_bookkeeping(gen, playbook_states):
    env, pb, _, _ = _state(playbook_states, "planning")
    plan = copy.deepcopy(pb).plan(env)
    switch, alerts = next((a, ev) for a, ev in plan if a.tool_id == "switch_to_backup")
    cands = env.candidates()
    repair = next(a for a in cands if a.tool_id == "repair_link" and a.target_id == switch.target_id)
    monitor = next(a for a in cands if a.tool_id == "monitor")
    escalate = next(a for a in cands if a.tool_id == "escalate")

    def sub(action, evidence, escalated=False):
        return {"time": env.now, "decision_id": "ep3d9", "action_id": action.action_id, "tool_id": action.tool_id,
                "target_id": action.target_id, "args": dict(action.bounded_arguments), "evidence_ids": list(evidence),
                "escalated_after": escalated}

    unknown = "link:sensor:stripper:999999"
    twin_pb = Playbook("act_first")
    expected_fallback = gen.shared_evidence(env, [repair.target_id])
    (fallback,) = expected_fallback
    assert env.log.get(fallback).kind in gen.SHARED_KINDS and env.log.get(fallback).subject_id == "source:stripper"
    assert fallback in citable(env, env.valid_evidence()) and fallback == max(
        (i for i in (env.log.get(e) for e in citable(env, env.valid_evidence()))
         if i.kind in gen.SHARED_KINDS and i.subject_id == "source:stripper"), key=lambda i: i.arrival_time).evidence_id
    assert [gen.target_group(env, t) for t in ("sensor:stripper", "source:stripper", "stripper_level", "plant")] == \
        ["stripper", "stripper", "stripper", None]
    shared = [i for i in (env.log.get(e) for e in citable(env, env.valid_evidence())) if i.kind in gen.SHARED_KINDS]
    assert gen.shared_evidence(env, ["plant"]) == [max(shared, key=lambda i: i.arrival_time).evidence_id]
    subs = [sub(switch, [*alerts, unknown]), sub(repair, [unknown]), sub(monitor, []),
            sub(escalate, alerts, escalated=True)]
    records = gen.replay_submissions(env, twin_pb, subs)
    assert [r["evidence_ids"] for r in records] == [list(alerts), expected_fallback, [], list(alerts)]
    assert [r["citation_changed"] for r in records] == [True, True, False, False]
    assert all(r["matched"] and r["time"] == env.now for r in records)
    statuses = {r.idempotency_key: r.status.value for r in env.executor.receipts}
    assert [statuses[f"ep3d9:{s['action_id']}"] for s in subs] == ["accepted"] * 4
    assert {(s["tool_id"], s["target_id"]) for s in subs} <= twin_pb.done
    assert {s["action_id"] for s in subs} <= set(twin_pb.submitted)
    assert twin_pb.escalated
    gen.replay_submissions(env, twin_pb, [sub(monitor, [], escalated=False)])
    assert not twin_pb.escalated
    assert gen.replay_submissions(env, twin_pb, []) == [] and not twin_pb.escalated


def test_twin_action_falls_back_to_the_registry_instance_and_advance_twin_lands_on_the_control_step(
        gen, playbook_states):
    env, pb, _, _ = _state(playbook_states, "planning")
    ids = {a.action_id for a in env.candidates()}
    record = {"action_id": f"restore_primary(sensor:stripper)@{env.now:.3f}", "tool_id": "restore_primary",
              "target_id": "sensor:stripper", "args": {}}
    assert record["action_id"] not in ids
    action, matched = gen.twin_action(env, record)
    assert not matched and (action.action_id, action.tool_id, action.target_id) == (
        record["action_id"], "restore_primary", "sensor:stripper")
    cand = next(a for a in env.candidates() if a.tool_id == "trace_link")
    assert gen.twin_action(env, {"action_id": cand.action_id, "tool_id": "x", "target_id": "y", "args": {}}) == \
        (cand, True)
    snap = env.snapshot()
    times = []
    for _ in range(3):
        env.advance_to_decision()
        times.append(env.now)
    env.restore(snap)
    states = []
    assert gen.advance_twin(env, copy.deepcopy(pb), times[-1], states, "stripper")
    assert env.now == times[-1] and states[-1][0] == times[-1] and len(states) >= 1
    assert states[-1][1:] == (env.hidden_state()["link_mode"]["sensor:stripper"], True)
    assert gen.advance_twin(env, copy.deepcopy(pb), env.now, states, "stripper")
    assert env.now == times[-1]


def test_receipt_statuses_stop_at_the_earlier_end_of_the_two_episodes(gen, playbook_states):
    env, _, _, cited = _state(playbook_states, "planning")
    trace = next(a for a in env.candidates() if a.tool_id == "trace_link")
    t0 = env.now
    env.submit(env.make_bundle(trace, cited, "d1"))
    key = f"d1:{trace.action_id}"
    while not env.done and not any(r.idempotency_key == key and r.status.value == "succeeded"
                                   for r in env.executor.receipts):
        env.advance_to_decision()
    started = next(r.started_at for r in env.executor.receipts if r.idempotency_key == key and r.started_at)
    assert gen.receipt_statuses(env.executor, [key]) == {key: [("accepted", None), ("running", None),
                                                               ("succeeded", None)]}
    assert gen.receipt_statuses(env.executor, [key], until=t0) == {key: [("accepted", None)]}
    assert gen.receipt_statuses(env.executor, [key], until=started) == {key: [("accepted", None), ("running", None)]}
    assert gen.receipt_statuses(env.executor, ["other"]) == {"other": []}


def test_playbook_memory_holds_what_the_rules_act_on():
    pb = Playbook("investigate_first")
    pb.traces = {"sensor:stripper": (False, "tool:trace_link:1"), "actuator:reactor": (True, "tool:trace_link:2")}
    pb.link_loss = {"sensor:stripper": "link:sensor:stripper:3", "actuator:reactor": "link:actuator:reactor:3"}
    gen = _script("generate_experience")
    assert gen.playbook_memory(pb) == {"traces": {"sensor:stripper": False, "actuator:reactor": True},
                                       "link_loss": ["actuator:reactor"]}

TWIN_CFG = {"horizon_s": 60.0, "replicates": 1, "follow_mode": "act_first"}


def _submission_records(env, plan, submitted, decision_id, behaviour):
    return [{"time": env.now, "decision_id": decision_id, "action_id": aid, "tool_id": a.tool_id,
             "target_id": a.target_id, "args": dict(a.bounded_arguments), "evidence_ids": list(ev),
             "escalated_after": bool(behaviour.escalated)} for (a, ev), aid in zip(plan, submitted)]


def _hand_primary(gen, seed, mode, stop, onset=700.0):
    from cpsdefense.sim.incidents import Incident, Scenario
    from cpsdefense.sim.tep_env import TEPEnv

    assert split_of(seed) == "train"
    sc = Scenario(seed=seed, incident=Incident("link_fault", "sensor:stripper", onset), background=(1,),
                  horizon_s=1200.0)
    env = TEPEnv(sc)
    env.reset()
    behaviour = Playbook(mode)
    primary = {"states": [gen.world_state(env, "stripper")], "log_sizes": [], "follow": []}
    submissions, n = [], 0
    while True:
        obs = env.advance_to_decision()
        assert not env.done
        primary["states"].append(gen.world_state(env, "stripper"))
        behaviour.update(obs, env)
        if env.now > env.world.onset_time and stop(env, behaviour):
            return sc, env, behaviour, primary, submissions, n
        plan = behaviour.plan(env)
        submissions += _submission_records(env, plan, gen.submit_plan(env, plan, f"ep{seed}d{n}"), f"ep{seed}d{n}",
                                           behaviour)
        n += 1


def _label_point(gen, env, behaviour, primary, seed, n, tools):
    import dataclasses

    from cpsdefense.experience.rollout import relevant_candidates, relevant_evidence

    cands = [a for a in relevant_candidates(env) if (a.tool_id, a.target_id) in tools]
    follow = dataclasses.replace(behaviour, mode=TWIN_CFG["follow_mode"]) \
        if behaviour.mode != TWIN_CFG["follow_mode"] else behaviour
    primary["log_sizes"].append(len(env.log))
    primary["follow"].append(copy.deepcopy(follow))
    record = {"tau": env.now, "decision_id": f"ep{seed}d{n}", "reseed_tag": [n],
              "cited_evidence": relevant_evidence(env), "candidates": [gen.action_record(a) for a in cands]}
    return record, cands, follow


def test_twin_branches_follow_the_defenders_own_history(gen, monkeypatch):
    pytest.importorskip("tep_studio")
    link, seed = "sensor:stripper", 204
    sc, env, behaviour, primary, subs, n = _hand_primary(gen, seed, "investigate_first",
                                                         lambda env, pb: link in pb.traces)
    ok, trace_id = behaviour.traces[link]
    assert not ok
    d, _, _ = _label_point(gen, env, behaviour, primary, seed, n,
                           {("switch_to_backup", link), ("replace_io_module", "source:stripper")})
    assert len(d["candidates"]) == 2
    plan = behaviour.plan(env)
    subs += _submission_records(env, plan, gen.submit_plan(env, plan, d["decision_id"]), d["decision_id"], behaviour)
    env.advance(60.0)
    seen, real = [], gen.branch_labels

    def spy(tenv, policy, *args, **kwargs):
        seen.append((tenv.log.get(trace_id), copy.deepcopy(policy)))
        return real(tenv, policy, *args, **kwargs)

    monkeypatch.setattr(gen, "branch_labels", spy)
    before = copy.deepcopy(primary["follow"][0])
    gen.run_twin(env, sc, [d], subs, primary, behaviour.mode, TWIN_CFG, load_config("sim/incidents.yaml"))
    ((twin_trace, policy),) = seen
    assert float(twin_trace.payload["values"][0]) == 1.0
    assert policy.mode == "act_first" and policy.traces[link] == (False, trace_id)
    assert policy == before and primary["follow"][0] == before
    t = d["twin"]
    assert t["labelled"] and t["pi_after_memory_differs"] and len(t["results"]) == 2


def test_a_replay_citing_only_a_primary_only_report_cites_world_evidence_both_worlds_share(gen):
    pytest.importorskip("tep_studio")
    from cpsdefense.experience.rollout import branch_labels, relevant_evidence

    link, seed = "sensor:stripper", 204
    sc, env, behaviour, primary, subs, n = _hand_primary(gen, seed, "act_first", lambda env, pb: bool(
        relevant_evidence(env)))
    cited = relevant_evidence(env)
    assert [(env.log.get(e).kind, env.log.get(e).subject_id) for e in cited] == [("link_health", link)]
    assert not any(env.log.get(e).kind == "alert" for e in env.valid_evidence())
    assert env.now - env.world.onset_time == pytest.approx(201.8, abs=0.05) and subs == []
    d, cands, follow = _label_point(gen, env, behaviour, primary, seed, n,
                                    {("repair_link", link), ("trace_link", link)})
    _, results = branch_labels(env, follow, TWIN_CFG["horizon_s"], TWIN_CFG["replicates"], tag=tuple(d["reseed_tag"]),
                               candidates=cands, evidence_ids=cited)
    repair = next(a for a in cands if a.tool_id == "repair_link")
    plan = [(repair, cited)]
    subs += _submission_records(env, plan, gen.submit_plan(env, plan, d["decision_id"]), d["decision_id"], behaviour)
    env.advance(60.0)
    key = f"{d['decision_id']}:{repair.action_id}"
    assert gen.receipt_statuses(env.executor, [key])[key] == [("accepted", None), ("running", None)]
    tw = gen.run_twin(env, sc, [d], subs, primary, behaviour.mode, TWIN_CFG, load_config("sim/incidents.yaml"))
    assert tw["divergence"]["receipt_differences"] == [] and not tw["divergence"]["receipt_status_differs"]
    assert tw["divergence"]["first_deterministic_difference"] is None
    (replay,) = tw["replayed"]
    assert replay["citation_changed"] and len(replay["evidence_ids"]) == 1
    fallback = env.log.get(replay["evidence_ids"][0])
    assert (fallback.kind, fallback.subject_id) == ("window_summary", "source:stripper")
    assert d["twin"]["belief"]["deterministic_identical"] and d["twin"]["cited_evidence"] == replay["evidence_ids"]
    assert [(r["tool_id"], r["first_status"]) for r in d["twin"]["results"]] == \
        [(r.tool_id, r.first_status) for r in results]
    assert ("repair_link", "running") in [(r.tool_id, r.first_status) for r in results]


def test_stored_items_keep_everything_valid_at_a_labelled_decision(gen):
    def item(eid, kind, t, lifetime):
        return EvidenceItem(eid, "tool", kind, "sensor:reactor", t, t, t + lifetime,
                            {"values": np.array([1.0, 0.0], np.float32), "event_start": t}, "test")
    log = EvidenceLog([item("old-expired", "trace_link", 100.0, 300.0), item("receipt:1", "receipt", 1000.0, 3600.0),
                       item("recent", "trace_link", 3000.0, 900.0), item("late", "trace_link", 4401.8, 900.0)])
    kept = [d["evidence_id"] for d in gen.stored_items(log, [4000.0, 4100.0], (1800.0, 300.0))]
    assert kept == ["receipt:1", "recent"]
    assert gen.stored_items(log, [], (1800.0, 300.0)) == []


def test_the_progress_line_and_the_run_record_seal_the_labels_of_test_episodes(gen, monkeypatch, tmp_path, capsys):
    pytest.importorskip("tep_studio")
    from cpsdefense.utils.runs import start_run

    class Pool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, fn, jobs):
            return map(fn, list(jobs)[::-1])

    def twin(seed):
        return {"labelled": seed + 1, "receipt_status_differs": False, "first_difference_since_onset_s": 1000.0 + seed}

    def seconds(seed):
        return 777.25 + seed

    def run_episode(job):
        seed = job[0]
        return seed, seed + 1, seconds(seed), f"made-up-cause-{seed}", seed % 2 == 0, seed != 2, twin(seed)

    seeds = range(SEED["test"] + 1)
    monkeypatch.setattr(gen, "Pool", Pool)
    monkeypatch.setattr(gen, "run_episode", run_episode)
    monkeypatch.setattr(gen, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "start_run", lambda kind, cfg, seed=None, name=None: start_run(
        kind, cfg, seed, name=name, root=tmp_path / "runs"))
    monkeypatch.setattr("sys.argv", ["generate_experience.py", "--set", f"seeds=[0,{len(seeds)}]", "workers=1",
                                     "--name", "unit"])
    gen.main()
    out = capsys.readouterr().out
    run_dir = next((tmp_path / "runs" / "experience").iterdir())
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert record["metrics"]["per_episode"] == metrics["per_episode"]
    assert [r["seed"] for r in metrics["per_episode"]] == list(seeds)
    rows = {r["seed"]: r for r in metrics["per_episode"]}
    assert {split_of(s) for s in seeds} == set(SPLITS)
    everything = out + json.dumps(record) + json.dumps(metrics)
    progress = [x for x in out.splitlines() if x.startswith("[")]
    assert len(progress) == len(seeds)
    tests = [s for s in seeds if split_of(s) == "test"]
    assert sum(x.endswith("] held-out test episode: sealed") for x in progress) == len(tests) > 0
    for s in seeds:
        kept = {"seed": s, "decisions": s + 1, "backup_available": s != 2}
        if split_of(s) == "test":
            assert rows[s] == {**kept, "seconds": "sealed", "cause": "sealed", "tripped": "sealed", "twin": "sealed"}
            assert f"made-up-cause-{s}" not in everything and f"{1000.0 + s}" not in everything
            assert f"{seconds(s)}" not in everything and f", {seconds(s):.0f}s," not in out
            assert not any(f" seed {s}:" in x for x in progress)
        else:
            line = next(x for x in progress if f" seed {s}:" in x)
            assert rows[s] == {**kept, "seconds": seconds(s), "cause": f"made-up-cause-{s}", "tripped": s % 2 == 0,
                               "twin": twin(s)}
            assert f"made-up-cause-{s}" in line and line.endswith(f"trip={s % 2 == 0}")
            assert f", {seconds(s):.0f}s, " in line
            assert f"twin={s + 1} labelled, receipts differ=False, first difference={1000.0 + s}," in line
    assert gen.episode_row(SEED["train"], 1, 1.0, "c", False, True)["twin"] is None
    assert gen.episode_row(SEED["test"], 1, 1.0, "c", False, True, None)["twin"] == "sealed"
    assert gen.episode_row(SEED["train"], 1, 5.0, "c", False, True)["seconds"] == 5.0
    assert gen.episode_row(SEED["test"], 1, 5.0, "c", False, True)["seconds"] == "sealed"


def _receipt(n, t, tool, status, target, tools, action):
    values = np.zeros(len(tools) + 8, np.float32)
    values[tools.index(tool)] = 1.0
    values[len(tools) + ["accepted", "rejected", "running", "succeeded", "failed", "timed_out", "duplicate",
                         "compensated"].index(status)] = 1.0
    payload = {"values": values, "tool_id": tool, "status": status, "event_start": t}
    if action is not None:
        payload.update(action_id=action, idempotency_key=f"d:{action}")
    return {"evidence_id": f"receipt:{n}", "source": "tool", "kind": "receipt", "subject_id": target,
            "event_time": t, "arrival_time": t, "valid_until": np.inf if status == "running" else t + 3600.0,
            "provenance": "executor", "payload": payload}


def _tool_result(n, kind, t, subject, values, action, lifetime=900.0, **extra):
    payload = {"values": np.asarray(values, np.float32), "event_start": t, **extra}
    if action is not None:
        payload.update(action_id=action, idempotency_key=f"d:{action}")
    return {"evidence_id": f"tool:{kind}:{n}", "source": "tool", "kind": kind, "subject_id": subject,
            "event_time": t, "arrival_time": t, "valid_until": t + lifetime, "provenance": f"tool:{kind}",
            "payload": payload}


def _build_episode(seed, statuses, backup=True, named=True, named_results=True, incident=None):
    tools = [s.tool_id for s in load_registry()]
    causes = load_config("sim/tep_env.yaml")["causes"]
    cands = [{"action_id": "repair@5000", "tool_id": "repair_link", "target_id": "sensor:reactor", "args": {},
              "family": "respond"},
             {"action_id": "wait@5000", "tool_id": "monitor", "target_id": "plant", "args": {"review_after_s": 300},
              "family": "monitor"},
             {"action_id": "trace@5000", "tool_id": "trace_link", "target_id": "sensor:reactor", "args": {},
              "family": "observe"}]
    results = [_result(i, k, 10.0 + i, status=statuses[i][k], dur=None if statuses[i][k] == "running" else 60.0)
               for i in range(3) for k in range(2)]
    decision = {"tau": 5000.0, "candidates": cands, "results": results,
                "valid_evidence": ["alert:1", "receipt:1", "tool:trace_link:1"],
                "hidden": {"kind": "link_fault", "onset_happened": True, "oracle_time_to_shutdown": None},
                "behaviour": {"explored": False, "actions": ["repair@5000", "trace@5000"], "probabilities": [0.8, 0.1],
                              "empty_plan_probability": 0.0},
                "baselines": {"act_first": ["repair@5000"], "investigate_first": ["trace@5000"]},
                "submitted_action_ids": ["repair@5000", "trace@5000"]}
    own = "repair@5000" if named else None
    result_of = {a: a if named_results else None for a in ("trace@4700", "trace@5000", "escalate@5003.6")}
    items = [{"evidence_id": "alert:1", "source": "detector", "kind": "alert", "subject_id": "source:reactor",
              "event_time": 4900.0, "arrival_time": 4910.0, "valid_until": 5500.0, "provenance": "detector",
              "payload": {"values": np.array([1.0, 0.0, 3.0]), "category": "stale",
                          "measurement": "reactor_pressure", "event_start": 4860.0}},
             _receipt(1, 2000.0, "trace_link", "succeeded", "sensor:reactor", tools, "trace@1990" if named else None),
             _receipt(2, 5000.0, "repair_link", "accepted", "sensor:reactor", tools, own),
             _receipt(3, 5005.4, "repair_link", "running", "sensor:reactor", tools, own),
             _receipt(4, 5003.6, "escalate", "accepted", "incident:1", tools, "escalate@5003.6" if named else None),
             _tool_result(1, "trace_link", 4830.0, "sensor:reactor", [0.0, 0.0], result_of["trace@4700"]),
             _tool_result(2, "trace_link", 5105.4, "sensor:reactor", [0.0, 0.0], result_of["trace@5000"]),
             _tool_result(3, "operator_report", 5090.0, "incident:1", [float(c == "link_fault") for c in causes],
                          result_of["escalate@5003.6"], lifetime=3600.0, cause="link_fault")]
    submissions = [{"time": 5000.0, "decision_id": "d7", "action_id": "repair@5000"},
                   {"time": 5000.0, "decision_id": "d7", "action_id": "trace@5000"},
                   {"time": 5003.6, "decision_id": "d8", "action_id": "escalate@5003.6"}]
    scenario = {"seed": seed, "backup_available": backup, **({"incident": incident} if incident else {})}
    return {"seed": seed, "decisions": [decision], "items": items, "backup_available": backup,
            "scenario": scenario, "submissions": submissions}


def _write_episodes(root, seeds, backup=(True,) * 3):
    root.mkdir(parents=True, exist_ok=True)
    for s, b in zip(seeds, backup):
        ep = _build_episode(s, [["succeeded", "running"], ["succeeded"] * 2, ["running"] * 2], backup=b)
        (root / f"ep_{s:06d}.pkl").write_bytes(pickle.dumps(ep))


def test_build_writes_censoring_labels_and_the_receipt_source(tmp_path):
    root = tmp_path / "episodes"
    _write_episodes(root, [SEED["train"], SEED["val"]])
    meta = build(root, tmp_path / "arrays", provenance={"note": "test"})
    assert meta["label_keys"] == list(LABEL_KEYS) and meta["provenance"] == {"note": "test"}
    a = dict(np.load(tmp_path / "arrays" / "train.npz", allow_pickle=False))
    assert a["y_first_censored"].tolist() == [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
    assert a["y_duration_censored"].tolist() == a["y_first_censored"].tolist()
    np.testing.assert_allclose(a["y_duration_time"], [[60.0, 3600.0], [60.0, 60.0], [3600.0, 3600.0]])
    assert a["y_p_success"][0] == 1.0 and a["y_p_success"][1] == 1.0 and np.isnan(a["y_p_success"][2])
    receipt = meta["vocab"]["sources"].index("receipt")
    valid = a["present"] & (a["arrival_time"] <= 5000.0) & (5000.0 < a["valid_until"])
    assert (valid & (a["source"] == receipt)).sum() == 1


def test_no_receipt_of_the_decisions_own_action_is_in_its_row_valid_or_not(tmp_path):
    root = tmp_path / "episodes"
    _write_episodes(root, [SEED["train"], SEED["val"]], backup=(True, False))
    meta = build(root, tmp_path / "arrays")
    a = dict(np.load(tmp_path / "arrays" / "train.npz", allow_pickle=False))
    ids = json.loads((tmp_path / "arrays" / "train_item_ids.json").read_text(encoding="utf-8"))["keys"]
    shown = [ids[i][1] for i in a["item_id"][0] if i >= 0]
    assert sorted(shown) == ["alert:1", "receipt:1", "tool:trace_link:1"]
    valid = a["present"] & (a["arrival_time"] <= a["decision_time"][:, None]) & (a["decision_time"][:, None]
                                                                                 < a["valid_until"])
    assert sorted(ids[i][1] for i in a["item_id"][valid]) == ["alert:1", "receipt:1", "tool:trace_link:1"]
    assert meta["splits"]["train"]["own_action_receipts_excluded"] == 2
    v = dict(np.load(tmp_path / "arrays" / "val.npz", allow_pickle=False))
    assert a["dec_backup_available"].tolist() == [True] and v["dec_backup_available"].tolist() == [False]


def test_build_refuses_receipts_that_do_not_name_their_action(tmp_path):
    root = tmp_path / "episodes"
    root.mkdir()
    ep = _build_episode(SEED["train"], [["succeeded"] * 2] * 3, named=False)
    (root / f"ep_{SEED['train']:06d}.pkl").write_bytes(pickle.dumps(ep))
    with pytest.raises(ValueError, match="action_id"):
        build(root, tmp_path / "arrays")
    ep["decisions"][0]["submitted_action_ids"] = []
    (root / f"ep_{SEED['train']:06d}.pkl").write_bytes(pickle.dumps(ep))
    with pytest.raises(ValueError, match="action_id"):
        build(root, tmp_path / "arrays")
    ep["submissions"] = []
    (root / f"ep_{SEED['train']:06d}.pkl").write_bytes(pickle.dumps(ep))
    assert build(root, tmp_path / "arrays")["splits"]["train"]["own_action_receipts_excluded"] == 0


def test_no_tool_result_of_the_decisions_own_or_later_actions_is_in_its_row(tmp_path):
    root = tmp_path / "episodes"
    _write_episodes(root, [SEED["train"]])
    meta = build(root, tmp_path / "arrays")
    a = dict(np.load(tmp_path / "arrays" / "train.npz", allow_pickle=False))
    ids = json.loads((tmp_path / "arrays" / "train_item_ids.json").read_text(encoding="utf-8"))["keys"]
    slot = {ids[i][1]: c for c, i in enumerate(a["item_id"][0]) if i >= 0}
    assert not {"tool:trace_link:2", "tool:operator_report:3"} & set(slot)
    c = slot["tool:trace_link:1"]
    assert a["arrival_time"][0, c] <= a["decision_time"][0] < a["valid_until"][0, c]
    split = meta["splits"]["train"]
    assert split["own_action_tool_results_excluded"] == 1 and split["own_action_receipts_excluded"] == 2
    assert split["later_action_items_excluded"] == 2


def test_build_refuses_tool_results_that_do_not_name_their_action(tmp_path):
    root = tmp_path / "episodes"
    root.mkdir()
    path = root / f"ep_{SEED['train']:06d}.pkl"
    ep = _build_episode(SEED["train"], [["succeeded"] * 2] * 3, named_results=False)
    path.write_bytes(pickle.dumps(ep))
    with pytest.raises(ValueError, match="action_id"):
        build(root, tmp_path / "arrays")
    ep["decisions"][0]["submitted_action_ids"], ep["submissions"] = [], []
    path.write_bytes(pickle.dumps(ep))
    assert build(root, tmp_path / "arrays")["splits"]["train"]["own_action_tool_results_excluded"] == 0

LINK_INCIDENT = {"kind": "link_fault", "target": "sensor:reactor", "onset_s": 600.0}


def _twin(decision, totals, trips=None, prior=(0.6, 0.4), identical=True, labelled=True, windows=()):
    results = [_result(i, k, float(t), shutdown=bool(trips and trips[i][k]), tts=600.0 if trips and trips[i][k]
                       else None) for i, row in enumerate(totals) for k, t in enumerate(row)]
    evidence = {"group": "reactor", "causes": ["link_fault", "io_module_fault"], "prior": list(prior),
                "tau": decision["tau"], "params": belief.load_params(), "deterministic_identical": identical,
                "first_difference": None if identical else [4990.0, "win:reactor:1"], "windows": list(windows),
                "traces": [], "operator": []}
    return {"cause": "io_module_fault:reactor", "labelled": labelled, "belief": evidence,
            "results": results if labelled else None, "cited_evidence": [], "candidates_unmatched": [],
            "hidden": None, "pi_after_memory_differs": False}


def _family_episode(seed, **twin):
    ep = _build_episode(seed, [["succeeded"] * 2] * 3, incident=LINK_INCIDENT)
    d = ep["decisions"][0]
    d["twin"] = _twin(d, [[30.0, 30.0], [25.0, 25.0], [12.0, 12.0]], trips=[[1, 1], [0, 0], [0, 0]], **twin)
    return ep


def test_belief_labels_mix_hindsight_and_twin_labels_by_the_posterior():
    ep = _family_episode(SEED["train"])
    b = belief_labels(ep["decisions"][0], in_family(ep))
    assert b["family"] and b["twin_ok"] and not b["belief_fallback"]
    assert b["posterior"] == pytest.approx(0.6) and b["twin_weight"] == pytest.approx(0.4)
    np.testing.assert_allclose(b["J_belief"], [0.6 * 10 + 0.4 * 30, 0.6 * 11 + 0.4 * 25, 12.0])
    np.testing.assert_allclose(b["p_trip_belief"], [0.4, 0.0, 0.0])
    assert candidate_labels(ep["decisions"][0])["voi"][2] == pytest.approx(-2.0)
    assert b["voi_belief"][2] == pytest.approx(16.6 - 12.0) and np.isnan(b["voi_belief"][:2]).all()
    report = {"w": 1, "end": 4000.0, "modes": ["down", "up"], "report": [1.0, 0.0], "age": 1000.0}
    w = belief.posterior(_family_episode(SEED["train"], windows=[report])["decisions"][0]["twin"]["belief"])
    b = belief_labels(_family_episode(SEED["train"], windows=[report])["decisions"][0], True)
    assert w[0] > 0.99 and b["posterior"] == pytest.approx(w[0])
    np.testing.assert_allclose(b["J_belief"], w[0] * np.array([10.0, 11.0, 12.0]) + w[1] * np.array([30, 25, 12]))


def test_belief_labels_fall_back_to_hindsight_only_where_the_belief_is_unavailable():
    hindsight = candidate_labels(_family_episode(SEED["train"])["decisions"][0])
    cases = {
        "excluded": (_family_episode(SEED["train"], identical=False), True, True, False, 1.0),
        "excluded, twin ended": (_family_episode(SEED["train"], identical=False, labelled=False), True, False, False,
                                 1.0),
        "twin ended": (_family_episode(SEED["train"], labelled=False), True, False, True, 0.6),
        "no twin record": (_build_episode(SEED["train"], [["succeeded"] * 2] * 3, incident=LINK_INCIDENT), True,
                           False, True, np.nan),
        "outside the family": (_build_episode(SEED["train"], [["succeeded"] * 2] * 3,
                                              incident={"kind": "disturbance", "target": "IDV6", "onset_s": 600.0}),
                               False, False, True, 1.0),
        "impossible evidence": (_family_episode(SEED["train"], windows=[
            {"w": 1, "end": 4000.0, "modes": ["backup", "backup"], "report": [1.0, 0.0], "age": 1000.0}]),
            True, True, True, np.nan),
    }
    for name, (ep, family, twin_ok, fallback, posterior) in cases.items():
        b = belief_labels(ep["decisions"][0], in_family(ep))
        assert (b["family"], b["twin_ok"], b["belief_fallback"]) == (family, twin_ok, fallback), name
        np.testing.assert_allclose(b["posterior"], posterior, err_msg=name)
        for k, h in (("J_belief", "J"), ("p_trip_belief", "p_trip"), ("voi_belief", "voi")):
            np.testing.assert_array_equal(b[k], hindsight[h], err_msg=name)


def test_build_writes_the_belief_labels_and_flags(tmp_path):
    root = tmp_path / "episodes"
    root.mkdir()
    eps = [_family_episode(SEED["train"]), _family_episode(SEED["val"], labelled=False)]
    other = _build_episode(next(k for k in range(SEED["train"] + 1, 2000) if split_of(k) == "train"),
                           [["succeeded"] * 2] * 3, incident={"kind": "sensor_fault", "target": "reactor_level",
                                                              "onset_s": 600.0})
    for ep in [*eps, other]:
        (root / f"ep_{ep['seed']:06d}.pkl").write_bytes(pickle.dumps(ep))
    meta = build(root, tmp_path / "arrays")
    assert meta["label_format"] == LABEL_FORMAT == "tep-decision-labels/3"
    assert meta["belief_label_keys"] == list(BELIEF_KEYS) and meta["belief_decision_keys"] == list(BELIEF_DECISION_KEYS)
    a = dict(np.load(tmp_path / "arrays" / "train.npz", allow_pickle=False))
    order = sorted([SEED["train"], other["seed"]])
    fam = order.index(SEED["train"])
    assert a["dec_family"].tolist() == [i == fam for i in range(2)]
    assert a["dec_twin_ok"].tolist() == [i == fam for i in range(2)]
    assert a["dec_belief_fallback"].tolist() == [i != fam for i in range(2)]
    np.testing.assert_allclose(a["dec_posterior"], [0.6 if i == fam else 1.0 for i in range(2)])
    J = a["y_J_belief"].reshape(2, 3)
    np.testing.assert_allclose(J[fam], [18.0, 16.6, 12.0])
    np.testing.assert_allclose(J[1 - fam], a["y_J"].reshape(2, 3)[1 - fam])
    np.testing.assert_allclose(a["y_p_trip_belief"].reshape(2, 3)[fam], [0.4, 0.0, 0.0])
    assert a["y_voi_belief"].reshape(2, 3)[fam, 2] == pytest.approx(4.6)
    assert meta["splits"]["train"]["family_decisions"] == 1 and meta["splits"]["train"]["twin_labelled_decisions"] == 1
    assert meta["splits"]["train"]["belief_fallback_decisions"] == 1
    v = dict(np.load(tmp_path / "arrays" / "val.npz", allow_pickle=False))
    assert v["dec_family"].tolist() == [True] and v["dec_twin_ok"].tolist() == [False]
    assert v["dec_belief_fallback"].tolist() == [True] and v["dec_posterior"] == pytest.approx([0.6])
    np.testing.assert_array_equal(v["y_J_belief"], v["y_J"])


@pytest.fixture(scope="module")
def generated(gen, tmp_path_factory):
    pytest.importorskip("tep_studio")
    from cpsdefense.sim.tep_plant import TEPPlant

    out = tmp_path_factory.mktemp("generated")
    incidents = load_config("sim/incidents.yaml")
    incidents.update(onset_s=[600, 600], horizon_s=1500, types={"link_fault": 1.0},
                     link_fault={"links": {"sensor:stripper": 1}}, background={"choices": [[]], "weights": [1.0]})
    cfg = load_config("experience/generate.yaml")
    cfg.update(max_labels_per_episode=2, max_labels_pre_onset=0, horizon_s=300, replicates=1, twins=False)
    plant = TEPPlant()
    plant.reset(seed=0)
    nominal = dict(zip(plant.measurement_names, plant.measured.tolist()))
    seeds = {"with": SEED["train"], "without": SEED["val"]}
    gen.run_episode((seeds["with"], cfg, nominal, incidents, str(out)))
    gen.run_episode((seeds["without"], cfg, nominal, {**incidents, "backup_unavailable_prob": 1.0}, str(out)))
    return out, {k: pickle.loads((out / f"ep_{s:06d}.pkl").read_bytes()) for k, s in seeds.items()}


@pytest.fixture(scope="module")
def twinned(gen, tmp_path_factory):
    pytest.importorskip("tep_studio")
    from cpsdefense.sim.incidents import sample_scenario
    from cpsdefense.sim.tep_plant import TEPPlant

    out = tmp_path_factory.mktemp("twinned")
    incidents = load_config("sim/incidents.yaml")
    incidents.update(onset_s=[600, 600], horizon_s=1500, types={"link_fault": 1.0, "io_module_fault": 1.0},
                     link_fault={"links": {"sensor:stripper": 2, "actuator:reactor": 1}},
                     io_module_fault={"groups": {"stripper": 1, "reactor": 1}},
                     background={"choices": [[1]], "weights": [1.0]})
    cfg = load_config("experience/generate.yaml")
    assert cfg["twins"] is True
    cfg.update(max_labels_per_episode=2, max_labels_pre_onset=0, horizon_s=300, replicates=1)
    plant = TEPPlant()
    plant.reset(seed=0)
    nominal = dict(zip(plant.measurement_names, plant.measured.tolist()))
    configs = {"link": incidents, "module": {**incidents, "backup_unavailable_prob": 1.0}}
    wanted = {"link": ("train", "link_fault:sensor:stripper"), "module": ("val", "io_module_fault:stripper")}
    seeds = {k: next(s for s in range(3000) if split_of(s) == split
                     and sample_scenario(s, nominal, configs[k]).incident.cause == cause)
             for k, (split, cause) in wanted.items()}
    for k, s in seeds.items():
        gen.run_episode((s, cfg, nominal, configs[k], str(out)))
    return out, incidents, {k: pickle.loads((out / f"ep_{s:06d}.pkl").read_bytes()) for k, s in seeds.items()}


def test_the_twin_replays_every_submission_at_its_control_step_and_labels_the_same_candidates(twinned):
    _, incidents, eps = twinned
    for key, ep in eps.items():
        tw = ep["twin"]
        sc, ts = ep["scenario"], tw["scenario"]
        assert ts["incident"]["kind"] == ("io_module_fault" if key == "link" else "link_fault")
        assert tw["onset_shift_s"] == pytest.approx(-1.8 if key == "link" else 1.8)
        assert ts["incident"]["onset_s"] - sc["incident"]["onset_s"] == pytest.approx(tw["onset_shift_s"])
        assert (ts["seed"], ts["background"], ts["backup_available"]) == (sc["seed"], sc["background"],
                                                                           sc["backup_available"])
        assert ts["warmup_s"] + ts["incident"]["onset_s"] + ts["horizon_s"] == pytest.approx(
            sc["warmup_s"] + sc["incident"]["onset_s"] + sc["horizon_s"])
        assert ep["submissions"] and tw["not_replayed"] == []
        assert [(r["time"], r["decision_id"], r["action_id"]) for r in tw["replayed"]] == \
            [(s["time"], s["decision_id"], s["action_id"]) for s in ep["submissions"]]
        assert all(r["matched"] for r in tw["replayed"])
        assert not tw["divergence"]["receipt_status_differs"] and tw["divergence"]["receipt_differences"] == []
        assert tw["prior"] == belief.pair_prior(incidents, "stripper")
        for d in ep["decisions"]:
            t = d["twin"]
            assert t["labelled"] and t["candidates_unmatched"] == [] and t["cause"] == tw["cause"]
            assert t["hidden"]["cause"] == tw["cause"] and t["hidden"]["onset_happened"]
            pairs = lambda rs: sorted((r["candidate"], r["replicate"], r["tool_id"], r["target_id"]) for r in rs)
            assert pairs(t["results"]) == pairs(d["results"])
            ev = t["belief"]
            assert ev["causes"] == [sc["incident"]["kind"], ts["incident"]["kind"]]
            assert ev["prior"] == [tw["prior"][c] for c in ev["causes"]]
            w = belief.posterior(ev)
            assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)
        assert ep["decisions"][0]["twin"]["belief"]["deterministic_identical"]
        first = tw["divergence"]["first_deterministic_difference"]
        assert first is None or first["time"] > ep["submissions"][0]["time"]


def test_twin_records_hold_only_the_primarys_evidence_and_each_worlds_states(twinned):
    _, _, eps = twinned
    for key, ep in eps.items():
        tw = ep["twin"]
        tau0 = ep["decisions"][0]["tau"]
        primary, twin = (next(s for s in states if s[0] == tau0) for states in (tw["primary_states"], tw["states"]))
        link_state, module_state = (primary, twin) if key == "link" else (twin, primary)
        assert tuple(link_state[1:]) == ("down", True) and tuple(module_state[1:]) == ("up", False)
        assert tw["states"][0][0] == tw["primary_states"][0][0] == pytest.approx(ep["scenario"]["warmup_s"], abs=1.0)
        for d in ep["decisions"]:
            ev = d["twin"]["belief"]
            assert set(ev) == {"group", "causes", "prior", "tau", "params", "deterministic_identical",
                               "first_difference", "windows", "traces", "operator"}
            assert all(w["end"] <= d["tau"] and w["modes"][0] != w["modes"][1] for w in ev["windows"])
            assert ev["params"]["p_detect_down"] == load_config("sim/telemetry.yaml")["link_health"]["p_detect_down"]


def test_build_mixes_the_real_twin_labels(twinned, tmp_path):
    root, _, eps = twinned
    build(root, tmp_path / "arrays")
    for split, key in (("train", "link"), ("val", "module")):
        a = dict(np.load(tmp_path / "arrays" / f"{split}.npz", allow_pickle=False))
        assert a["dec_family"].all() and a["dec_twin_ok"].all() and not a["dec_belief_fallback"].any()
        start = 0
        for r, d in enumerate(eps[key]["decisions"]):
            n = len(d["candidates"])
            w = belief.posterior(d["twin"]["belief"])
            J = candidate_labels(d)["J"]
            Jt = candidate_labels({"candidates": d["candidates"], "results": d["twin"]["results"]})["J"]
            np.testing.assert_allclose(a["y_J_belief"][start:start + n], w[0] * J + w[1] * Jt)
            assert a["dec_posterior"][r] == pytest.approx(w[0])
            start += n


def test_the_generator_records_the_submitted_actions_and_the_backup_availability(generated):
    _, eps = generated
    n_submitted = 0
    for key, ep in eps.items():
        assert ep["decisions"]
        assert ep["backup_available"] is (key == "with") and ep["scenario"]["backup_available"] is (key == "with")
        named = {i["payload"]["action_id"] for i in ep["items"] if i["kind"] == "receipt"}
        for d in ep["decisions"]:
            assert d["submitted_action_ids"] == d["behaviour"]["actions"]
            assert set(d["submitted_action_ids"]) <= named
            assert [s["action_id"] for s in ep["submissions"] if s["time"] == d["tau"]] == d["submitted_action_ids"]
            n_submitted += len(d["submitted_action_ids"])
        times = [s["time"] for s in ep["submissions"]]
        assert times == sorted(times) and len({s["action_id"] for s in ep["submissions"]}) == len(times)
        assert all(s["decision_id"].startswith(f"ep{ep['seed']}d") for s in ep["submissions"])
    assert n_submitted > 0
    tools = {c["tool_id"] for d in eps["without"]["decisions"] for c in d["candidates"]}
    assert "switch_to_backup" not in tools and "switch_to_backup" in {c["tool_id"] for d in eps["with"]["decisions"]
                                                                        for c in d["candidates"]}


def test_arrays_of_generated_episodes_hold_no_receipt_or_tool_result_of_a_rows_own_actions(generated, tmp_path):
    root, eps = generated
    build(root, tmp_path / "arrays")
    lookahead = 0
    for split, key in (("train", "with"), ("val", "without")):
        a = dict(np.load(tmp_path / "arrays" / f"{split}.npz", allow_pickle=False))
        keys = json.loads((tmp_path / "arrays" / f"{split}_item_ids.json").read_text(encoding="utf-8"))["keys"]
        items = {i["evidence_id"]: i for i in eps[key]["items"]}
        results = [i for i in eps[key]["items"] if i["kind"] in TOOL_RESULT_KINDS]
        assert {i["payload"]["action_id"] for i in results} <= {s["action_id"] for s in eps[key]["submissions"]}
        for r, d in enumerate(eps[key]["decisions"]):
            assert a["decision_time"][r] == d["tau"]
            own = set(d["behaviour"]["actions"])
            own |= {s["action_id"] for s in eps[key].get("submissions", ()) if s["time"] > d["tau"]}
            shown = [items[keys[i][1]] for i in a["item_id"][r] if i >= 0]
            assert not [i["evidence_id"] for i in shown
                        if i["kind"] in ACTION_ITEM_KINDS and i["payload"]["action_id"] in own]
            lookahead += sum(i["kind"] == "receipt" and i["payload"]["action_id"] in own
                             and d["tau"] < i["arrival_time"] <= d["tau"] + 120.0 for i in eps[key]["items"])
        assert a["dec_backup_available"].tolist() == [key == "with"] * len(eps[key]["decisions"])
        check = _script("dataset_stats").own_action_items_in_rows([eps[key]], a, keys)
        assert check == {"rows": len(eps[key]["decisions"]), "own_action_receipt_slots": 0,
                         "rows_with_own_action_receipts": 0, "later_action_receipt_slots": 0,
                         "receipt_slots_without_action_id": 0, "own_action_tool_result_slots": 0,
                         "rows_with_own_action_tool_results": 0, "later_action_tool_result_slots": 0,
                         "tool_result_slots_without_action_id": 0}
    assert lookahead > 0


def test_build_dataset_records_the_run_and_the_provenance_in_meta(bd, tmp_path):
    root = tmp_path / "episodes"
    _write_episodes(root, [SEED["train"], SEED["val"], SEED["test"]])
    (root / "generation.json").write_text(json.dumps({
        "run": "runs/experience/nonexistent", "config_file": "experience/generate_stress.yaml",
        "config": {"incidents": "sim/incidents_stress.yaml", "epsilon": 0.3}, "config_sha256": "c" * 64,
        "incidents_sha256": "d" * 64, "source_combined": "e" * 64}), encoding="utf-8")
    meta, run = bd.build_with_record(root, tmp_path / "arrays", "unit", runs_root=tmp_path / "runs")
    prov = json.loads((tmp_path / "arrays" / "meta.json").read_text(encoding="utf-8"))["provenance"]
    assert prov == json.loads(json.dumps(meta["provenance"]))
    assert Path(prov["build_run"]) == run.dir.resolve() and (run.dir / "run.json").exists()
    assert json.loads((run.dir / "run.json").read_text(encoding="utf-8"))["kind"] == "data"
    assert prov["build_source_combined"] == bd.SOURCE_AT_IMPORT["combined"]
    assert prov["episode_manifest"]["files"] == 3 and len(prov["episode_manifest"]["sha256"]) == 64
    g = prov["generation"]
    assert (g["config_sha256"], g["incidents_file"], g["incidents_sha256"]) == ("c" * 64, "sim/incidents_stress.yaml",
                                                                                "d" * 64)
    configs = prov["config_files"]
    assert {"configs/sim/incidents_stress.yaml", "configs/experience/generate_stress.yaml",
            "configs/experience/tensors.yaml", "configs/sim/tep.yaml"} <= set(configs)
    assert all(len(v["now"]) == 64 for v in configs.values())
    assert set(meta["splits"]) == {"train", "val", "test"}
    with pytest.raises(FileExistsError):
        bd.build_with_record(root, tmp_path / "arrays", "again", runs_root=tmp_path / "runs")
