import copy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tep_studio")

from cpsdefense.agents import playbook
from cpsdefense.agents.playbook import Playbook, run_episode
from cpsdefense.core.types import EvidenceItem, ExecStatus, ObservationBundle, Receipt
from cpsdefense.defense.evidence import EvidenceLog
from cpsdefense.defense.registry import candidate_id, load_registry
from cpsdefense.experience import rollout
from cpsdefense.experience.rollout import branch_labels, relevant_candidates, relevant_evidence, summarize
from cpsdefense.sim.incidents import Incident, Scenario
from cpsdefense.sim.tep_env import TEPEnv
from cpsdefense.utils.config import load_config

RELEVANT = ("alert", "link_health", "trace_link", "field_check", "operator_report")


def make_env(kind, target, onset=600.0, horizon=2 * 3600, trace_accuracy=1.0):
    config = copy.deepcopy(load_config("sim/tep_env.yaml"))
    config["tools"]["trace_accuracy"] = trace_accuracy
    env = TEPEnv(Scenario(seed=3, incident=Incident(kind, target, onset), background=(), horizon_s=horizon),
                 config=config)
    env.reset()
    return env


def to_stale_decision(env, group, policy):
    while not env.done:
        obs = env.advance_to_decision()
        policy.update(obs, env)
        if any(a.category == "stale" and a.subject_id == f"source:{group}" for a in obs.alerts):
            return obs
    raise AssertionError("no staleness alert")


def to_expiring_evidence(env, policy, within=4.0):
    while not env.done:
        items = [i for i in (env.log.get(e) for e in env.valid_evidence()) if i.kind in RELEVANT]
        left = [i.valid_until - env.now for i in items]
        if left and min(left) < within and max(left) > 300:
            return min(items, key=lambda i: i.valid_until)
        policy.update(env.advance(env.plant.dt_s), env)
    raise AssertionError("no expiring evidence")


def pick(env, tool, target, **args):
    return next(a for a in env.candidates() if a.tool_id == tool and a.target_id == target
                and dict(a.bounded_arguments) == args)


def tools_submitted(log):
    return [(b.action.tool_id, b.action.target_id) for _, b, _ in log]


class ScriptedEnv:
    GROUP = [f"stripper_{i}" for i in range(5)]
    PAIRS = [("trace_link", "sensor:stripper"), ("trace_link", "actuator:stripper"),
             ("switch_to_backup", "sensor:stripper"), ("repair_link", "sensor:stripper"),
             ("repair_link", "actuator:stripper"), ("replace_io_module", "source:stripper"),
             ("escalate", "incident:1")]

    def __init__(self):
        self.now = 0.0
        self.log = EvidenceLog()
        self.config = {"queue_latency_s": 5.0}
        self.plant = SimpleNamespace(sensor_links={"stripper": list(range(len(self.GROUP)))})
        self.registry = load_registry()

    def valid_evidence(self, t=None):
        return self.log.valid_ids(self.now if t is None else t)

    def candidates(self):
        pairs = self.PAIRS + [("recalibrate_sensor", m) for m in self.GROUP]
        return [self.registry.make_action(tool, target, {}, self.now, candidate_id(tool, target, {}, self.now))
                for tool, target in pairs]

    def observe(self, t, *items, receipts=(), active=()):
        self.now = float(t)
        for item in items:
            self.log.add(item)
        return ObservationBundle(time=self.now, version=0, new_evidence=items, receipts=tuple(receipts),
                                 active_actions=tuple(active))


def item(eid, kind, subject, t, lifetime, values, **payload):
    return EvidenceItem(evidence_id=eid, source="detector", kind=kind, subject_id=subject, event_time=float(t),
                        arrival_time=float(t), valid_until=float(t + lifetime),
                        payload={"values": np.asarray(values, np.float32), **payload}, provenance="test")


def stale(i, t, lifetime=600.0):
    return item(f"alert:stale:stripper_{i}:{t:g}", "alert", "source:stripper", t, lifetime, [1, 0, 5],
                category="stale", measurement=f"stripper_{i}")


def deviation(t, lifetime=600.0):
    return item(f"alert:deviation:stripper_0:{t:g}", "alert", "source:stripper", t, lifetime, [0, 1, 9],
                category="deviation", measurement="stripper_0")


def link_loss(link, t, lifetime=240.0):
    return item(f"link:{link}:{t:g}", "link_health", link, t, lifetime, [1, 0])


def trace(link, t, primary_ok, lifetime=900.0):
    return item(f"tool:trace_link:{t:g}", "trace_link", link, t, lifetime, [float(primary_ok), 0])


def refused(action, t):
    return Receipt(decision_id="d", action_id=action.action_id, idempotency_key=f"d:{action.action_id}",
                   status=ExecStatus.REJECTED, reason="stale_evidence", submitted_at=t)


def succeeded(action, t):
    return Receipt(decision_id="d", action_id=action.action_id, idempotency_key=f"d:{action.action_id}",
                   status=ExecStatus.SUCCEEDED, reason=None, submitted_at=t - 60.0, started_at=t - 55.0,
                   finished_at=t)


def planned(plan):
    return [(a.tool_id, a.target_id, list(ev)) for a, ev in plan]


def test_trace_and_link_loss_memory_expire_with_their_evidence():
    env, pb = ScriptedEnv(), Playbook("act_first")
    loss, healthy = link_loss("actuator:stripper", 100), trace("actuator:stripper", 100, True)
    assert pb.decide(env, env.observe(100, loss, healthy)) == []
    pb.decide(env, env.observe(400))
    assert pb.link_loss == {} and "actuator:stripper" in pb.traces
    again = link_loss("actuator:stripper", 1100)
    plan = pb.decide(env, env.observe(1100, again))
    assert pb.traces == {}
    assert planned(plan) == [("repair_link", "actuator:stripper", [again.evidence_id])]


def test_a_failed_trace_alone_still_justifies_the_actuator_repair():
    env, pb = ScriptedEnv(), Playbook("investigate_first")
    pb.decide(env, env.observe(0, link_loss("actuator:stripper", 0)))
    failed = trace("actuator:stripper", 300, False)
    assert planned(pb.decide(env, env.observe(300, failed))) == [("repair_link", "actuator:stripper",
                                                                   [failed.evidence_id])]


def test_an_expired_trace_can_be_repeated_but_not_while_its_answer_or_the_fix_is_pending():
    env, pb = ScriptedEnv(), Playbook("investigate_first")
    plan = pb.decide(env, env.observe(0, *(stale(i, 0) for i in range(3))))
    assert [a.tool_id for a, _ in plan] == ["trace_link"]
    trace_id = plan[0][0].action_id
    assert pb.decide(env, env.observe(60, active=[trace_id])) == []
    plan = pb.decide(env, env.observe(130, trace("sensor:stripper", 130, True)))
    assert [a.tool_id for a, _ in plan] == ["replace_io_module"]
    replace_id = plan[0][0].action_id
    assert pb.decide(env, env.observe(1100, *(stale(i, 1000) for i in range(3)), active=[replace_id])) == []
    assert pb.traces == {}
    plan = pb.decide(env, env.observe(1200))
    assert [(a.tool_id, a.target_id) for a, _ in plan] == [("trace_link", "sensor:stripper")]


def test_act_first_does_not_pile_on_while_a_trace_or_a_module_replacement_runs():
    for tool, target in (("trace_link", "sensor:stripper"), ("replace_io_module", "source:stripper")):
        env, pb = ScriptedEnv(), Playbook("act_first")
        running = next(a for a in env.candidates() if (a.tool_id, a.target_id) == (tool, target))
        pb.note_submitted(running)
        assert pb.decide(env, env.observe(10, *(stale(i, 10) for i in range(3)), active=[running.action_id])) == []
        plan = pb.decide(env, env.observe(20))
        assert [a.tool_id for a, _ in plan] == ["switch_to_backup", "repair_link"]


def test_deviation_counts_expire_with_their_alerts():
    env, pb = ScriptedEnv(), Playbook("act_first")
    assert pb.decide(env, env.observe(0, deviation(0))) == []
    assert pb.decide(env, env.observe(700, deviation(700))) == []
    plan = pb.decide(env, env.observe(800, deviation(800)))
    assert planned(plan) == [("escalate", "incident:1", ["alert:deviation:stripper_0:700",
                                                         "alert:deviation:stripper_0:800"])]


def test_the_playbook_cites_only_evidence_that_outlives_the_executor_queue():
    env, pb = ScriptedEnv(), Playbook("act_first")
    short = stale(0, 0, lifetime=30.0)
    long = [stale(i, 0) for i in (1, 2)]
    plan = pb.decide(env, env.observe(0, short, *long))
    assert [a.tool_id for a, _ in plan] == ["switch_to_backup", "repair_link"]
    assert all(ev == [e.evidence_id for e in long] for _, ev in plan)
    env2, pb2 = ScriptedEnv(), Playbook("act_first")
    assert pb2.decide(env2, env2.observe(0, *(stale(i, 0, lifetime=30.0) for i in range(3)))) == []
    assert pb2.done == set()


def test_citable_evidence_needs_every_parent_to_last():
    env = ScriptedEnv()
    parent = stale(0, 0, lifetime=30.0)
    child = EvidenceItem(evidence_id="derived", source="detector", kind="alert", subject_id="source:stripper",
                         event_time=0.0, arrival_time=0.0, valid_until=600.0,
                         payload={"values": np.zeros(3, np.float32), "derived_from": [parent.evidence_id]},
                         provenance="test")
    lasting = stale(1, 0)
    env.observe(0, parent, child, lasting)
    assert playbook.citable(env, [parent.evidence_id, child.evidence_id, lasting.evidence_id]) == [
        lasting.evidence_id]
    assert playbook.citable(env, [parent.evidence_id], margin_s=0.0) == [parent.evidence_id]
    env.now = 590.0
    assert playbook.citable(env, [lasting.evidence_id]) == []


def test_refused_actions_can_be_submitted_again():
    env, pb = ScriptedEnv(), Playbook("act_first")
    plan = pb.decide(env, env.observe(0, *(stale(i, 0) for i in range(3))))
    switch = plan[0][0]
    assert switch.tool_id == "switch_to_backup"
    plan = pb.decide(env, env.observe(6, receipts=[refused(switch, 0)]))
    assert [(a.tool_id, a.target_id) for a, _ in plan] == [("switch_to_backup", "sensor:stripper")]


def test_a_restore_of_the_primary_path_undoes_the_backup_switch_in_memory():
    back = [("switch_to_backup", "sensor:stripper")]
    for lifetime in (600.0, 150.0):
        env, pb = ScriptedEnv(), Playbook("act_first")
        switch, repair = (a for a, _ in pb.decide(env, env.observe(0, *(stale(i, 0, lifetime) for i in range(3)))))
        pb.decide(env, env.observe(70, receipts=[succeeded(switch, 70)], active=[repair.action_id]))
        assert pb.backup_since == {"sensor:stripper": 70}
        restore = env.registry.make_action("restore_primary", "sensor:stripper", {}, 100.0,
                                           candidate_id("restore_primary", "sensor:stripper", {}, 100.0))
        pb.note_submitted(restore)
        plan = pb.decide(env, env.observe(170, receipts=[succeeded(restore, 170)], active=[repair.action_id]))
        assert pb.backup_since == {}
        if lifetime < 170:
            assert plan == []
            plan = pb.decide(env, env.observe(300, *(stale(i, 300) for i in range(3)), active=[repair.action_id]))
        assert [(a.tool_id, a.target_id) for a, _ in plan] == back


def test_a_group_stale_again_after_its_link_repair_gets_the_module_replaced():
    env, pb = ScriptedEnv(), Playbook("act_first")
    env.PAIRS = [p for p in ScriptedEnv.PAIRS if p[0] != "switch_to_backup"]
    plan = pb.decide(env, env.observe(0, *(stale(i, 0) for i in range(3))))
    assert [(a.tool_id, a.target_id) for a, _ in plan] == [("repair_link", "sensor:stripper")]
    repair = plan[0][0]
    old = [stale(i, 2000) for i in range(3)]
    assert pb.decide(env, env.observe(2400, *old, receipts=[succeeded(repair, 2400)])) == []
    assert pb.repaired_since == {"sensor:stripper": 2400}
    fresh = [stale(i, 2500) for i in range(3)]
    assert planned(pb.decide(env, env.observe(2500, *fresh))) == [
        ("replace_io_module", "source:stripper", [f.evidence_id for f in fresh])]


def test_without_a_backup_act_first_recovers_from_a_repair_that_did_not_help():
    for kind, target, replaced in (("io_module_fault", "stripper", True), ("link_fault", "sensor:stripper", False)):
        env = TEPEnv(Scenario(seed=3, incident=Incident(kind, target, 600.0), background=(), horizon_s=3 * 3600,
                              backup_available=False))
        env.reset()
        submitted = tools_submitted(run_episode(env, Playbook("act_first")))
        assert submitted[0] == ("repair_link", "sensor:stripper")
        assert ("switch_to_backup", "sensor:stripper") not in submitted
        assert (("replace_io_module", "source:stripper") in submitted) == replaced, (kind, submitted)
        if replaced:
            repair = next(r for r in env.executor.receipts if r.action_id.startswith("repair_link")
                          and r.status == ExecStatus.SUCCEEDED)
            replace = next(r for r in env.executor.receipts if r.action_id.startswith("replace_io_module"))
            assert replace.submitted_at > repair.finished_at


def test_act_first_switches_and_repairs_while_investigate_first_traces_then_fixes():
    act_env = make_env("link_fault", "sensor:stripper")
    act = run_episode(act_env, Playbook("act_first"))
    assert ("switch_to_backup", "sensor:stripper") in tools_submitted(act)
    assert ("repair_link", "sensor:stripper") in tools_submitted(act)
    inv_env = make_env("io_module_fault", "stripper")
    inv = run_episode(inv_env, Playbook("investigate_first"))
    submitted = tools_submitted(inv)
    assert submitted.index(("trace_link", "sensor:stripper")) < submitted.index(("replace_io_module", "source:stripper"))
    assert ("switch_to_backup", "sensor:stripper") not in submitted
    for _, bundle, receipt in act + inv:
        assert receipt.status == ExecStatus.ACCEPTED, (bundle.action.tool_id, receipt.reason)
        if bundle.action.family.value == "respond":
            assert bundle.evidence_ids
    for env in (act_env, inv_env):
        assert not [r for r in env.executor.receipts if r.status == ExecStatus.REJECTED]
    replace = [r for r in inv_env.executor.receipts if r.action_id.startswith("replace_io_module")]
    assert ExecStatus.RUNNING in {r.status for r in replace}


def test_playbook_stays_quiet_in_normal_operation():
    env = make_env("link_fault", "sensor:stripper", onset=3600.0, horizon=60.0)
    log = run_episode(env, Playbook("act_first"))
    assert log == []


def test_labelling_leaves_the_environment_and_the_policy_exactly_as_they_were():
    env = make_env("io_module_fault", "stripper")
    policy = Playbook("act_first")
    to_stale_decision(env, "stripper", policy)
    before, policy_before = env.snapshot(), copy.deepcopy(policy)
    cands = [pick(env, "monitor", "plant", review_after_s=900), pick(env, "trace_link", "sensor:stripper"),
             pick(env, "switch_to_backup", "sensor:stripper"), pick(env, "replace_io_module", "source:stripper")]
    _, results = branch_labels(env, policy, horizon_s=1800, replicates=2, tag=(0,), candidates=cands,
                               evidence_ids=[])
    assert {r.first_status for r in results} == {"succeeded", "rejected"}
    assert policy == policy_before

    def trajectory():
        times = [env.advance(600).time for _ in range(3)]
        log = [(e, env.log.get(e).arrival_time, env.log.get(e).valid_until) for e in env.log]
        receipts = [(r.idempotency_key, r.status, r.reason) for r in env.executor.receipts]
        return times, env.cost().total(), env.plant.measured.copy(), log, receipts, env.telemetry.rng.random()

    labelled = trajectory()
    env.restore(before)
    plain = trajectory()
    assert labelled[0] == plain[0] and labelled[1] == plain[1] and np.array_equal(labelled[2], plain[2])
    assert labelled[3:] == plain[3:]


def test_labels_are_reproducible_and_complete():
    env = make_env("io_module_fault", "stripper")
    policy = Playbook("act_first")
    to_stale_decision(env, "stripper", policy)
    cands = [a for a in relevant_candidates(env) if a.tool_id in ("trace_link", "switch_to_backup", "monitor")]
    evidence = relevant_evidence(env)
    c1, r1 = branch_labels(env, policy, horizon_s=1800, replicates=2, tag=(5,), candidates=cands,
                           evidence_ids=evidence)
    c2, r2 = branch_labels(env, policy, horizon_s=1800, replicates=2, tag=(5,), candidates=cands,
                           evidence_ids=evidence)
    assert r1 == r2 and len(r1) == 2 * len(cands)
    assert {(r.candidate, r.replicate) for r in r1} == {(i, k) for i in range(len(cands)) for k in range(2)}
    rows = summarize(c1, r1)
    act = min(row["J"] for row in rows if row["family"] != "observe")
    for row in rows:
        if row["family"] == "observe":
            assert row["voi"] == pytest.approx(act - row["J"])
        else:
            assert row["voi"] is None
    first = {r.tool_id: r.first_status for r in r1}
    assert first["trace_link"] == "succeeded"


def test_branch_results_record_when_the_first_action_started():
    env = make_env("io_module_fault", "stripper")
    policy = Playbook("act_first")
    to_stale_decision(env, "stripper", policy)
    t0 = env.now
    cands = [pick(env, "trace_link", "sensor:stripper"), pick(env, "switch_to_backup", "sensor:stripper")]
    _, results = branch_labels(env, policy, horizon_s=600, replicates=1, tag=(0,), candidates=cands,
                               evidence_ids=[])
    trace, switch = results
    latency = env.config["queue_latency_s"]
    assert trace.first_status == "succeeded" and latency <= trace.first_started < latency + env.plant.dt_s
    assert switch.first_status == "rejected" and switch.first_started is None
    assert env.now == t0


def test_branches_of_one_replicate_share_random_streams():
    env = make_env("io_module_fault", "stripper")
    policy = Playbook("act_first")
    to_stale_decision(env, "stripper", policy)
    root, t0, evidence = env.snapshot(), env.now, relevant_evidence(env)
    traces = [pick(env, "trace_link", "sensor:stripper"), pick(env, "trace_link", "actuator:stripper")]
    _, results = branch_labels(env, policy, horizon_s=600, replicates=2, tag=(1,), candidates=traces)
    duration = {(r.candidate, r.replicate): r.first_duration for r in results}
    assert duration[0, 0] == duration[1, 0] and duration[0, 1] == duration[1, 1]
    assert duration[0, 0] != duration[0, 1]

    def windows(action, k):
        env.restore(root)
        env.reseed(1, k)
        rollout.run_branch(env, copy.deepcopy(policy), action, evidence, t0 + 1800, "crn")
        return {e: env.log.get(e).arrival_time for e in env.log
                if e.startswith("win:") and env.log.get(e).event_time > t0}

    switch, wait = pick(env, "switch_to_backup", "sensor:stripper"), pick(env, "monitor", "plant", review_after_s=900)
    assert windows(switch, 0) == windows(wait, 0) != windows(switch, 1)
    env.restore(root)


def test_first_actions_cite_evidence_that_survives_the_executor_queue():
    env = make_env("io_module_fault", "stripper")
    policy = Playbook("act_first")
    to_stale_decision(env, "stripper", policy)
    expiring = to_expiring_evidence(env, policy)
    horizon = env.now + env.config["queue_latency_s"] + playbook.CITE_MARGIN_S
    cited = relevant_evidence(env)
    assert cited and expiring.evidence_id not in cited
    assert all(env.log.get(e).valid_until > horizon for e in cited)
    cands = [pick(env, "trace_link", "sensor:stripper"), pick(env, "monitor", "plant", review_after_s=300),
             pick(env, "switch_to_backup", "sensor:stripper")]
    _, results = branch_labels(env, policy, horizon_s=600, replicates=1, tag=(0,), candidates=cands)
    assert [r.first_reason for r in results] == [None, None, None]
    assert {r.first_status for r in results} == {"succeeded"}


@pytest.mark.parametrize("cite", ["expiring", "nothing"])
def test_a_refused_first_action_stays_available_to_pi_after(cite):
    env = make_env("io_module_fault", "stripper", horizon=4 * 3600)
    policy = Playbook("act_first")
    to_stale_decision(env, "stripper", policy)
    expiring = to_expiring_evidence(env, policy)
    replace = pick(env, "replace_io_module", "source:stripper")
    evidence = [expiring.evidence_id] if cite == "expiring" else []
    _, results = branch_labels(env, policy, horizon_s=3 * 3600, replicates=1, tag=(0,), candidates=[replace],
                               evidence_ids=evidence)
    (r,) = results
    assert (r.first_status, r.first_reason) == (("rejected", "stale_evidence") if cite == "expiring"
                                                else ("rejected", "policy_forbidden"))
    assert r.resolved and not r.shutdown


def test_a_healthy_trace_stops_vetoing_the_repair_once_it_expires():
    env = make_env("link_fault", "actuator:stripper", horizon=4 * 3600, trace_accuracy=0.0)
    policy = Playbook("act_first")
    while not any(e.startswith("link:actuator:stripper") for e in relevant_evidence(env)):
        policy.update(env.advance_to_decision(), env)
    assert env.now > env.world.onset_time
    tr = pick(env, "trace_link", "actuator:stripper")
    _, results = branch_labels(env, policy, horizon_s=3 * 3600, replicates=1, tag=(0,), candidates=[tr])
    assert results[0].first_status == "succeeded" and results[0].resolved


def test_a_monitor_first_action_holds_pi_after_back_for_its_review_time():
    env = make_env("io_module_fault", "stripper", horizon=4 * 3600)
    policy = Playbook("act_first")
    to_stale_decision(env, "stripper", policy)
    assert copy.deepcopy(policy).plan(env)
    root, t0 = env.snapshot(), env.now
    monitors = [pick(env, "monitor", "plant", review_after_s=s) for s in (300, 900)]
    _, results = branch_labels(env, policy, horizon_s=3600, replicates=1, tag=(0,), candidates=monitors)
    assert results[0].total != results[1].total
    for action in monitors:
        env.restore(root)
        env.reseed(0, 0)
        first = rollout.run_branch(env, copy.deepcopy(policy), action, relevant_evidence(env), t0 + 3600, "m")
        later = [r.submitted_at for r in env.executor.receipts if r.decision_id.startswith(f"{first.decision_id}d")]
        wait = action.bounded_arguments["review_after_s"]
        assert later and min(later) >= t0 + wait and min(later) < t0 + wait + 2 * env.plant.dt_s
    env.restore(root)
