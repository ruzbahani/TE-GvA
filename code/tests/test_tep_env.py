import copy
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("tep_studio")

from cpsdefense.core.types import ExecStatus
from cpsdefense.experience import belief
from cpsdefense.experience.dataset import split_of
from cpsdefense.sim.incidents import Incident, Scenario
from cpsdefense.sim.tep_env import RECEIPT_LIFETIME_S, RECEIPT_STATUSES, TICKET, TEPEnv
from cpsdefense.utils.config import load_config
from cpsdefense.utils.seeding import spawn_generators


def make_env(kind="link_fault", target="sensor:stripper", seed=3, onset=600.0, horizon=3 * 3600, fault_kind=None,
             magnitude=0.0, **kw):
    config = copy.deepcopy(load_config("sim/tep_env.yaml"))
    config["tools"]["trace_accuracy"] = 1.0
    incident = Incident(kind, target, onset, fault_kind, magnitude)
    sc = Scenario(seed=seed, incident=incident, background=(), horizon_s=horizon, **kw)
    env = TEPEnv(sc, config=config)
    return env, env.reset()


def first_alert(env):
    while not env.done:
        obs = env.advance_to_decision()
        if obs.alerts:
            return obs
    raise AssertionError("no alert before the end of the episode")


def pick(env, tool, target):
    return next(a for a in env.candidates() if a.tool_id == tool and a.target_id == target)


def run_to_end(env):
    while not env.done:
        env.advance_to_decision()
    return env.cost().total(), env.plant.shutdown


def wait_for(env, kind):
    while not env.done:
        obs = env.advance_to_decision()
        found = [e for e in obs.new_evidence if e.kind == kind]
        if found:
            return found[0]
    raise AssertionError(f"no {kind} evidence")


def receipt_items(env):
    return [env.log.get(e) for e in env.log if env.log.get(e).kind == "receipt"]


def step_until(env, key, status):
    while not env.done:
        if any(r.idempotency_key == key and r.status == status for r in env.executor.receipts):
            return
        env.advance(env.plant.dt_s)
    raise AssertionError(f"{key} never reached {status}")


def test_reset_observation_and_candidates():
    env, obs = make_env()
    assert obs.time == pytest.approx(env.scenario.warmup_s, abs=env.plant.dt_s)
    assert obs.new_evidence == () and obs.alerts == ()
    assert env.log.valid_ids(env.now)
    tools = {a.tool_id for a in env.candidates()}
    assert {"trace_link", "query_historian", "verify_sensor", "switch_to_backup", "monitor"} <= tools
    assert "escalate" not in tools


def test_evidence_is_shown_only_after_arrival_and_alerts_open_a_ticket():
    env, _ = make_env()
    obs = first_alert(env)
    assert all(e.arrival_time <= obs.time for e in obs.new_evidence)
    assert all(a.category == "stale" and a.subject_id == "source:stripper" for a in obs.alerts)
    assert "escalate" in {a.tool_id for a in env.candidates()}
    assert "cause" not in vars(obs)


def test_branches_replay_exactly_from_a_snapshot():
    env, _ = make_env("io_module_fault", "stripper")
    first_alert(env)
    root = env.snapshot()
    a = run_to_end(env)
    env.restore(root)
    b = run_to_end(env)
    assert a == b and a[1] is not None
    env.restore(root)
    obs_alert_ids = list(env.valid_evidence())[:1]
    env.submit(env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"), obs_alert_ids, "d1"))
    c = run_to_end(env)
    assert c != a


def test_trace_link_tells_a_failed_link_from_a_frozen_module():
    results = {}
    for kind, target in (("link_fault", "sensor:stripper"), ("io_module_fault", "stripper")):
        env, _ = make_env(kind, target)
        obs = first_alert(env)
        env.submit(env.make_bundle(pick(env, "trace_link", "sensor:stripper"), [obs.alerts[0].evidence_ids[0]], "d1"))
        results[kind] = float(wait_for(env, "trace_link").payload["values"][0])
    assert results == {"link_fault": 0.0, "io_module_fault": 1.0}


def test_backup_fixes_a_failed_link_but_not_a_frozen_module():
    for kind, target, fixed in (("link_fault", "sensor:stripper", True), ("io_module_fault", "stripper", False)):
        env, _ = make_env(kind, target)
        obs = first_alert(env)
        env.submit(env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"),
                                   [obs.alerts[0].evidence_ids[0]], "d1"))
        env.advance(600)
        idx = env.plant.sensor_links["stripper"]
        seen = []
        for _ in range(20):
            env.advance(env.plant.dt_s)
            seen.append(env.plant.delivered[idx].copy())
        moving = np.ptp(np.array(seen), axis=0).max() > 0
        assert env.plant.link_mode["sensor:stripper"] == "backup" and moving == fixed, kind


def test_decisions_on_a_changed_state_are_refused_before_any_effect():
    env, _ = make_env(onset=600.0)
    obs = first_alert(env)
    alert = obs.alerts[0].evidence_ids[0]
    action = pick(env, "repair_link", "sensor:stripper")
    stale = env.make_bundle(action, [alert], "d0")
    switch = env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"), [alert], "d1")
    env.submit(switch)
    step_until(env, switch.idempotency_key, ExecStatus.SUCCEEDED)
    compute = env.cost().compute
    late = replace(stale, not_after=env.now + 600.0, action=replace(action, expiry=env.now + 300.0),
                   idempotency_key="d0-late")
    fresh = env.make_bundle(pick(env, "repair_link", "sensor:stripper"), [alert], "d2")
    assert late.observation_version == obs.version < fresh.observation_version
    assert env.submit(late).status == env.submit(fresh).status == ExecStatus.ACCEPTED
    env.advance(30)
    last = {r.idempotency_key: r for r in env.executor.receipts}
    assert (last["d0-late"].status, last["d0-late"].reason) == (ExecStatus.REJECTED, "stale_state")
    assert last[fresh.idempotency_key].status == ExecStatus.RUNNING
    assert "sensor:stripper" in env.world.faulted_links and env.plant.link_mode["sensor:stripper"] == "backup"
    unit = env.costs["compute"]["per_resource_unit"]
    assert env.cost().compute == pytest.approx(compute + 10 * unit)


def test_costs_charge_tools_once_and_a_trip_once():
    env, _ = make_env("io_module_fault", "stripper")
    obs = first_alert(env)
    env.submit(env.make_bundle(pick(env, "trace_link", "sensor:stripper"), [obs.alerts[0].evidence_ids[0]], "d1"))
    total, shutdown = run_to_end(env)
    c = env.cost()
    assert shutdown is not None
    assert c.physical >= env.costs["physical"]["shutdown"]
    assert c.physical < 2 * env.costs["physical"]["shutdown"]
    assert c.compute == pytest.approx(1 * env.costs["compute"]["per_resource_unit"])


def test_tool_effort_is_charged_when_the_tool_starts():
    env, _ = make_env()
    obs = first_alert(env)
    action = pick(env, "repair_link", "sensor:stripper")
    bundle = env.make_bundle(action, [obs.alerts[0].evidence_ids[0]], "d1")
    env.submit(bundle)
    unit = env.costs["compute"]["per_resource_unit"]
    step_until(env, bundle.idempotency_key, ExecStatus.RUNNING)
    assert env.cost().compute == pytest.approx(10 * unit)
    step_until(env, bundle.idempotency_key, ExecStatus.SUCCEEDED)
    assert env.cost().compute == pytest.approx(10 * unit)


def test_every_receipt_becomes_a_receipt_item_in_the_shared_format():
    env, obs0 = make_env()
    tools = [spec.tool_id for spec in env.registry]
    assert len(tools) == 11 and tools[0] == "trace_link"
    action = pick(env, "trace_link", "sensor:stripper")
    bundle = env.make_bundle(action, [next(iter(env.valid_evidence()))], "d1")
    env.submit(bundle)
    env.submit(bundle)
    shown = {}
    while not env.done and len(env.executor.receipts) < 4:
        obs = env.advance(env.plant.dt_s)
        for item in obs.new_evidence:
            if item.kind == "receipt":
                assert item.evidence_id not in shown
                shown[item.evidence_id] = obs.time
    obs = env.advance(env.plant.dt_s)
    assert not [e for e in obs.new_evidence if e.kind == "receipt"]
    receipts, items = env.executor.receipts, receipt_items(env)
    assert [r.status.value for r in receipts] == ["accepted", "duplicate", "running", "succeeded"]
    assert [i.evidence_id for i in items] == ["receipt:1", "receipt:2", "receipt:3", "receipt:4"]
    for receipt, item in zip(receipts, items):
        t = receipt.finished_at if receipt.finished_at is not None else (
            receipt.started_at if receipt.started_at is not None else receipt.submitted_at)
        assert (item.source, item.kind, item.provenance, item.subject_id) == ("tool", "receipt", "executor",
                                                                              "sensor:stripper")
        assert item.event_time == item.arrival_time == t
        running = receipt.status == ExecStatus.RUNNING
        assert item.valid_until == (math.inf if running else t + 3600.0) and RECEIPT_LIFETIME_S == 3600.0
        values = item.payload["values"]
        assert values.dtype == np.float32 and values.shape == (11 + 8,)
        expected = np.zeros(19, np.float32)
        expected[tools.index("trace_link")] = expected[11 + RECEIPT_STATUSES.index(receipt.status.value)] = 1
        assert np.array_equal(values, expected)
        assert item.payload["tool_id"] == "trace_link" and item.payload["status"] == receipt.status.value
        assert item.payload["event_start"] == t
        assert item.payload["action_id"] == receipt.action_id == action.action_id
        assert item.payload["idempotency_key"] == receipt.idempotency_key == bundle.idempotency_key
        assert isinstance(item.payload["action_id"], str) and isinstance(item.payload["idempotency_key"], str)
    assert [i.supersedes for i in items] == [(), (), (), ("receipt:3",)]
    assert shown["receipt:1"] == shown["receipt:2"] == pytest.approx(obs0.time + env.plant.dt_s)
    assert shown["receipt:3"] == receipts[2].started_at and shown["receipt:4"] == receipts[3].finished_at
    assert RECEIPT_STATUSES == tuple(s.value for s in ExecStatus)


def test_receipts_are_ordinary_evidence_for_the_executor():
    env, obs0 = make_env(onset=600.0)
    trace = env.make_bundle(pick(env, "trace_link", "sensor:stripper"), [next(iter(env.valid_evidence()))], "d0")
    env.submit(trace)
    obs = first_alert(env)
    before = obs.version
    switch = env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"), [obs.alerts[0].evidence_ids[0]], "sw")
    env.submit(switch)
    step_until(env, switch.idempotency_key, ExecStatus.SUCCEEDED)
    assert env.log.valid("receipt:1", env.now)
    monitor = pick(env, "monitor", "plant")
    old = replace(env.make_bundle(monitor, ["receipt:1"], "d1"), observation_version=before)
    fresh = env.make_bundle(monitor, ["receipt:1"], "d2")
    assert fresh.observation_version > before > obs0.version
    env.submit(old)
    env.submit(fresh)
    env.advance(30)
    last = {r.idempotency_key: r for r in env.executor.receipts}
    assert last[old.idempotency_key].status == ExecStatus.REJECTED
    assert last[old.idempotency_key].reason == "stale_state"
    assert last[fresh.idempotency_key].status in (ExecStatus.RUNNING, ExecStatus.SUCCEEDED)
    expiry = env.log.get("receipt:1").valid_until
    env.advance(expiry - env.now + env.plant.dt_s)
    assert not env.log.valid("receipt:1", env.now)
    late = env.make_bundle(pick(env, "monitor", "plant"), ["receipt:1"], "d3")
    env.submit(late)
    env.advance(30)
    last = {r.idempotency_key: r for r in env.executor.receipts}
    assert last[late.idempotency_key].status == ExecStatus.REJECTED
    assert last[late.idempotency_key].reason == "stale_evidence"


def test_snapshot_replays_receipt_items_exactly():
    env, _ = make_env("io_module_fault", "stripper")
    obs = first_alert(env)
    env.submit(env.make_bundle(pick(env, "trace_link", "sensor:stripper"), [obs.alerts[0].evidence_ids[0]], "d1"))
    root = env.snapshot()

    def replay():
        env.restore(root)
        env.advance(1800)
        return [(i.evidence_id, i.arrival_time, i.payload["status"], i.payload["values"].tolist())
                for i in receipt_items(env)], env.cost()

    first, second = replay(), replay()
    assert first == second and len(first[0]) == 3


def test_a_forged_action_is_refused_with_a_receipt_item_and_no_effect():
    env, _ = make_env()
    action = replace(pick(env, "trace_link", "sensor:stripper"), tool_id="rm_rf")
    receipt = env.submit(env.make_bundle(action, [next(iter(env.valid_evidence()))], "d1"))
    assert receipt.status == ExecStatus.REJECTED and receipt.reason == "not_in_registry"
    env.advance(300)
    (item,) = receipt_items(env)
    assert item.payload["status"] == "rejected" and item.payload["tool_id"] == "rm_rf"
    assert item.payload["values"][:11].sum() == 0 and item.payload["values"][11 + 1] == 1
    assert env.cost().compute == 0.0 and env.executor.side_effects == [] and len(env.executor.receipts) == 1


def test_a_running_action_stays_in_the_valid_evidence_until_its_terminal_receipt():
    env, _ = make_env(tool_latency_scale=4.0, horizon=6 * 3600)
    obs = first_alert(env)
    bundle = env.make_bundle(pick(env, "repair_link", "sensor:stripper"), [obs.alerts[0].evidence_ids[0]], "d1")
    env.submit(bundle)
    step_until(env, bundle.idempotency_key, ExecStatus.RUNNING)
    env.submit(bundle)
    accepted, running, duplicate = receipt_items(env)
    assert [i.payload["status"] for i in (accepted, running, duplicate)] == ["accepted", "running", "duplicate"]
    assert running.valid_until == math.inf and running.supersedes == () == duplicate.supersedes
    env.advance(running.event_time + RECEIPT_LIFETIME_S + 60.0 - env.now)
    root = env.snapshot()
    for _ in range(2):
        env.restore(root)
        assert bundle.action.action_id in env.executor.active_actions()
        assert not env.log.valid(accepted.evidence_id, env.now)
        assert running.evidence_id in env.valid_evidence()
        step_until(env, bundle.idempotency_key, ExecStatus.SUCCEEDED)
        final = receipt_items(env)[-1]
        assert final.payload["status"] == "succeeded" and final.supersedes == (running.evidence_id,)
        t = final.arrival_time
        assert env.log.valid(running.evidence_id, t - 1e-6) and not env.log.valid(running.evidence_id, t)
        assert final.evidence_id in env.log.valid_ids(t)


def test_recalibrate_sensor_reports_the_same_status_for_a_frozen_module_a_failed_link_and_a_healthy_sensor():
    outcomes = {}
    for kind, target in (("io_module_fault", "stripper"), ("link_fault", "sensor:stripper"),
                         ("sensor_fault", "stripper_level")):
        fault = "frozen" if kind == "sensor_fault" else None
        env, _ = make_env(kind, target, fault_kind=fault, tool_latency_scale=0.05)
        obs = first_alert(env)
        eid = next(a.evidence_ids[0] for a in obs.alerts)
        bundle = env.make_bundle(pick(env, "recalibrate_sensor", "stripper_level"), [eid], "d1")
        env.submit(bundle)
        step_until(env, bundle.idempotency_key, ExecStatus.SUCCEEDED)
        item = receipt_items(env)[-1]
        assert item.payload["status"] == "succeeded" and item.payload["tool_id"] == "recalibrate_sensor"
        i = env.plant.index["stripper_level"]
        readings = []
        for _ in range(20):
            env.advance(env.plant.dt_s)
            readings.append(env.plant.measured[i])
        outcomes[kind] = (np.ptp(readings) > 0, "stripper_level" in env.plant.faults, sorted(env.world.io_faulted))
    assert outcomes["io_module_fault"] == (False, True, ["stripper"])
    assert outcomes["link_fault"] == (True, False, [])
    assert outcomes["sensor_fault"] == (True, False, [])


def test_instrument_faults_are_kept_apart_from_io_module_freezes():
    env, _ = make_env("sensor_fault", "stripper_level", onset=0.0, fault_kind="drift", magnitude=0.01)
    env.advance(10.0)
    world, plant = env.world, env.plant
    onset_fault = dict(plant.faults["stripper_level"])
    assert onset_fault["kind"] == "drift"
    assert world.instrument_faults == {"stripper_level": onset_fault}
    world.apply_incident(Incident("io_module_fault", "stripper", 0.0))
    group = [plant.measurement_names[i] for i in plant.sensor_links["stripper"]]
    other = next(m for m in group if m != "stripper_level")
    recal = {m: pick(env, "recalibrate_sensor", m) for m in group}
    assert all(plant.faults[m]["kind"] == "frozen" for m in group)
    assert world.predicate("sensor_calibrated", recal[other])
    assert not world.predicate("sensor_calibrated", recal["stripper_level"])
    world.apply(pick(env, "replace_io_module", "source:stripper"), {}, env.now)
    assert plant.faults == {"stripper_level": onset_fault}
    world.apply_incident(Incident("io_module_fault", "stripper", 0.0))
    world.apply(recal["stripper_level"], {}, env.now)
    assert world.predicate("sensor_calibrated", recal["stripper_level"]) and world.instrument_faults == {}
    assert plant.faults["stripper_level"]["kind"] == "frozen"
    world.apply(pick(env, "replace_io_module", "source:stripper"), {}, env.now)
    assert plant.faults == {} and world.io_faulted == set()
    assert env.hidden_state()["instrument_faults"] == [] and env.hidden_state()["sensor_faults"] == []


def test_a_scenario_without_backup_offers_no_backup_switch():
    env, _ = make_env(backup_available=False)
    first_alert(env)
    tools = {(a.tool_id, a.target_id) for a in env.candidates()}
    assert ("repair_link", "sensor:stripper") in tools and ("trace_link", "sensor:stripper") in tools
    assert not any(tool == "switch_to_backup" for tool, _ in tools)


def test_replicates_get_their_own_plant_noise_and_branches_share_it():
    env, _ = make_env()
    first_alert(env)
    root = env.snapshot()

    def branch(k, action=None):
        env.restore(root)
        env.reseed(0, k)
        ref = spawn_generators([env.scenario.seed, 0, k], ["telemetry", "tools", "operator"])
        assert env.telemetry.rng.bit_generator.state == ref["telemetry"].bit_generator.state
        if action is not None:
            env.submit(env.make_bundle(pick(env, action, "sensor:stripper"), list(env.valid_evidence())[-1:], "b"))
        env.advance(600)
        return env.plant.true_measured.copy(), env.plant.noise_state()

    a, again, other, acted = branch(0), branch(0), branch(1), branch(0, "switch_to_backup")
    assert np.array_equal(a[0], again[0]) and np.array_equal(a[1], again[1])
    assert not np.array_equal(a[0], other[0])
    assert np.array_equal(a[1], acted[1])
    assert not np.array_equal(a[0], acted[0])


def test_restore_primary_receipts_are_the_same_whether_the_primary_path_is_broken_or_not():
    shown, modes = {}, {}
    for kind, target in (("link_fault", "sensor:stripper"), ("io_module_fault", "stripper")):
        env, _ = make_env(kind, target)
        obs = first_alert(env)
        alert = obs.alerts[0].evidence_ids[0]
        switch = env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"), [alert], "d1")
        env.submit(switch)
        step_until(env, switch.idempotency_key, ExecStatus.SUCCEEDED)
        restore = env.make_bundle(pick(env, "restore_primary", "sensor:stripper"), [alert], "d2")
        receipts = [env.submit(restore)]
        while not env.done and receipts[-1].status in (ExecStatus.ACCEPTED, ExecStatus.RUNNING):
            obs = env.advance(env.plant.dt_s)
            receipts += [r for r in obs.receipts if r.idempotency_key == restore.idempotency_key]
        shown[kind] = [(r.status, r.reason, r.side_effects, r.postconditions_met) for r in receipts]
        modes[kind] = env.plant.link_mode["sensor:stripper"]
    assert modes == {"link_fault": "down", "io_module_fault": "up"}
    assert shown["link_fault"] == shown["io_module_fault"]
    assert [s[0] for s in shown["link_fault"]] == [ExecStatus.ACCEPTED, ExecStatus.RUNNING, ExecStatus.SUCCEEDED]
    assert shown["link_fault"][-1][2:] == (("sensor:stripper:primary",), {"link_not_in_backup": True})


def test_no_tool_reports_the_hidden_state_it_met_in_its_side_effects():
    effects = {}
    for kind, target, fault in (("link_fault", "sensor:stripper", None), ("io_module_fault", "stripper", None),
                                ("sensor_fault", "stripper_level", "frozen"), ("disturbance", "IDV6", None)):
        env, _ = make_env(kind, target, fault_kind=fault)
        env.advance(700.0)
        link, source = "sensor:stripper", "source:stripper"
        steps = [("trace_link", link, {}), ("query_historian", source, {}), ("verify_sensor", source, {}),
                 ("switch_to_backup", link, {}), ("restore_primary", link, {}), ("switch_to_backup", link, {}),
                 ("repair_link", link, {}), ("restore_primary", link, {}), ("recalibrate_sensor", "stripper_level", {}),
                 ("replace_io_module", source, {}), ("set_production", "plant", {"fraction": 0.9}),
                 ("monitor", "plant", {"review_after_s": 300}), ("escalate", TICKET, {})]
        actions = [env.registry.make_action(tool, t, args, env.now, f"{tool}({t})") for tool, t, args in steps]
        out = [env.world.apply(a, dict(a.bounded_arguments), env.now) for a in actions]
        out += [env.world.compensate(actions[3], env.now), env.world.compensate(actions[6], env.now)]
        effects[kind] = [(r["ok"], r["side_effects"]) for r in out]
    assert all(e == effects["link_fault"] for e in effects.values()), effects
    assert effects["link_fault"][4] == (True, ("sensor:stripper:primary",))


def test_the_observation_token_carries_nothing_about_the_onset():
    tokens, world = {}, {}
    for onset in (600.0, 1e6):
        env, obs = make_env(onset=onset)
        tokens[onset], world[onset] = [obs.version], [env.world.version()]
        while len(tokens[onset]) < (400 if onset == 600.0 else len(tokens[600.0])):
            obs = env.advance(10.0)
            tokens[onset].append(obs.version)
            world[onset].append(env.world.version())
            if onset == 600.0 and obs.alerts:
                break
    n = len(tokens[600.0])
    assert tokens[600.0] == tokens[1e6] == list(range(1, n + 1))
    assert world[600.0][:-1] == world[1e6][:-1] == [0] * (n - 1) and world[600.0][-1] == 1


def test_a_bundle_carries_the_observation_token_and_the_executor_the_version_seen_then():
    env, obs0 = make_env(onset=600.0)
    b0 = env.make_bundle(pick(env, "trace_link", "sensor:stripper"), [], "d0")
    assert b0.observation_version == obs0.version == 1
    obs = first_alert(env)
    assert env.world.last_changed_version(TICKET) > 0 and env.world.last_changed_version("sensor:stripper") == 0
    pre_onset = replace(env.make_bundle(pick(env, "trace_link", "sensor:stripper"), [], "pre"),
                        observation_version=obs0.version)
    switch = env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"), [obs.alerts[0].evidence_ids[0]], "sw")
    assert env.submit(pre_onset).status == env.submit(switch).status == ExecStatus.ACCEPTED
    step_until(env, switch.idempotency_key, ExecStatus.SUCCEEDED)
    fresh = env.make_bundle(pick(env, "trace_link", "sensor:stripper"), [], "fresh")
    assert fresh.observation_version > obs.version > obs0.version
    old = replace(fresh, observation_version=obs.version, idempotency_key="old")
    forged = [replace(fresh, observation_version=v, idempotency_key=f"forged{i}")
              for i, v in enumerate((fresh.observation_version + 1, 0, -3, 1.0, "1", True, None))]
    for bundle in (fresh, old, *forged):
        assert env.submit(bundle).status == ExecStatus.ACCEPTED
    env.advance(30)
    last = {r.idempotency_key: r for r in env.executor.receipts}
    started = [r.started_at for r in env.executor.receipts
               if r.idempotency_key == pre_onset.idempotency_key and r.status == ExecStatus.RUNNING]
    assert len(started) == 1 and started[0] > env.world.onset_time
    assert last[fresh.idempotency_key].status in (ExecStatus.RUNNING, ExecStatus.SUCCEEDED)
    for key in ["old"] + [f"forged{i}" for i in range(len(forged))]:
        assert (last[key].status, last[key].reason) == (ExecStatus.REJECTED, "stale_state"), key
    assert old.observation_version == obs.version


def test_the_hidden_onset_is_recorded_apart_and_moves_no_version():
    for kind, target, fault in (("link_fault", "sensor:stripper", None), ("io_module_fault", "stripper", None),
                                ("sensor_fault", "stripper_level", "drift"), ("disturbance", "IDV6", None)):
        env, _ = make_env(kind, target, fault_kind=fault, magnitude=0.01, onset=60.0)
        stripper = [env.plant.measurement_names[i] for i in env.plant.sensor_links["stripper"]]
        assert env.hidden_state()["hidden_changes"] == {} and env.world.version() == 0
        env.advance(90.0)
        h, world = env.hidden_state(), env.world
        expected = {"link_fault": {"sensor:stripper"}, "io_module_fault": {"source:stripper", *stripper},
                    "sensor_fault": {"stripper_level", "source:stripper"}, "disturbance": {"plant"}}[kind]
        assert h["onset_happened"] and set(h["hidden_changes"]) == expected, kind
        assert all(world.onset_time <= t < world.onset_time + env.plant.dt_s for t in h["hidden_changes"].values())
        assert all(world.last_changed_version(s) == 0 for s in expected)
        assert set(world.changed) <= {TICKET}
        world.apply(pick(env, "switch_to_backup", "sensor:stripper"), {}, env.now)
        assert world.last_changed_version("sensor:stripper") == world.version() > 0

PROBE_ONSET_S = 60.0
PROBE_WINDOW_S = 300.0


def mark_every_subject(env):
    tools = {"link": "repair_link", "source": "replace_io_module", "sensor": "recalibrate_sensor",
             "plant": "set_production"}
    candidates = {}
    for action in env.candidates():
        candidates.setdefault((action.tool_id, action.target_id), action)
    marks = {}
    for kind, tool in tools.items():
        for target in env.targets()[kind]:
            receipt = env.submit(env.make_bundle(candidates[tool, target], [], f"mark:{target}"))
            assert (receipt.status, receipt.reason) == (ExecStatus.REJECTED, "policy_forbidden")
            item = receipt_items(env)[-1]
            assert item.subject_id == target and item.payload["idempotency_key"] == receipt.idempotency_key
            marks.setdefault(kind, []).append(item.evidence_id)
    return marks


def probe_view(kind, target, fault=None, onset=PROBE_ONSET_S):
    env, obs0 = make_env(kind, target, fault_kind=fault, onset=onset, magnitude=0.01)
    marks = mark_every_subject(env)
    links, sources = env.targets()["link"], env.targets()["source"]
    end, rows, previous = env.now + PROBE_WINDOW_S, [], obs0.version
    while not env.done and env.now < end:
        obs = env.advance_to_decision(max_s=end - env.now)
        if obs.alerts:
            break
        shown = [(r.idempotency_key, r.status.value if hasattr(r, "status") else r.kind, getattr(r, "reason", None))
                 for r in obs.receipts]
        n = len(rows)

        def action(tool, target, **args):
            return env.registry.make_action(tool, target, args, env.now, f"{tool}({target})@{n}")

        monitor = action("monitor", "plant", review_after_s=300)
        probes = [(monitor, [], "none")] + [(monitor, ids, f"cites-{group}") for group, ids in marks.items()]
        probes += [(action("trace_link", links[n % len(links)]), [], "target-link"),
                   (action("query_historian", sources[n % len(sources)]), [], "target-source")]
        returned = []
        for name, token in (("latest", obs.version), ("previous", previous), ("first", obs0.version)):
            for act, cited, what in probes:
                bundle = replace(env.make_bundle(act, cited, f"p{n}:{name}:{what}"), observation_version=token)
                returned.append(env.submit(bundle).status.value)
        rows.append((obs.time, obs.version, shown, returned))
        previous = obs.version
    return rows, env


def test_no_bundle_sequence_reads_the_hidden_cause_or_onset_from_refusals():
    base, _ = probe_view("link_fault", "sensor:stripper", onset=1e6)
    for kind, target, fault in (("link_fault", "sensor:stripper", None), ("link_fault", "actuator:reactor", None),
                                ("io_module_fault", "stripper", None), ("sensor_fault", "stripper_level", "frozen"),
                                ("sensor_fault", "reactor_pressure", "drift"), ("disturbance", "IDV6", None)):
        rows, env = probe_view(kind, target, fault)
        n = min(len(rows), len(base))
        assert rows[:n] == base[:n], (kind, target)
        onset, latency = env.world.onset_time, env.config["queue_latency_s"]
        probed = [t for t, *_ in rows[:n] if t > onset + latency]
        assert len(probed) >= 10 and probed[-1] > onset + 60.0, (kind, target)
        assert env.hidden_state()["onset_happened"] and env.hidden_state()["hidden_changes"]
        refused = [r for r in env.executor.receipts if r.status == ExecStatus.REJECTED and r.reason == "stale_state"]
        assert refused == []
    assert len(base) >= 40 and all(returned.count("accepted") == len(returned) for *_, returned in base)

BACKUP_ONSET_S = 600.0
BACKUP_SEED = next(s for s in range(100, 200) if split_of(s) == "train")


def latest_window(env, group="stripper"):
    return max((e for e in env.valid_evidence() if e.startswith(f"win:{group}:")),
               key=lambda e: int(e.rsplit(":", 1)[1]))


def backup_before_onset_view(kind, target, onset=BACKUP_ONSET_S):
    env, _ = make_env(kind, target, seed=BACKUP_SEED, onset=onset)
    env.advance(200.0)
    switch = env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"), [latest_window(env)], "sw")
    assert env.submit(switch).status == ExecStatus.ACCEPTED
    link_onset = env.scenario.warmup_s + BACKUP_ONSET_S
    restore_at, end = link_onset + 60.0, link_onset + 400.0
    names = [env.plant.measurement_names[i] for i in env.plant.sensor_links["stripper"]]
    rows, readings, states, submitted = [], [], [], None
    while not env.done and env.now < end - 1e-9:
        stop = end if submitted is not None else restore_at
        obs = env.advance_to_decision(max_s=max(min(30.0, stop - env.now), env.plant.dt_s))
        if obs.alerts:
            break
        cands = sorted((a.tool_id, a.target_id, tuple(sorted(a.bounded_arguments.items()))) for a in env.candidates())
        shown = [(r.idempotency_key, r.status.value if hasattr(r, "status") else r.kind, getattr(r, "reason", None))
                 for r in obs.receipts]
        rows.append((obs.time, cands, shown))
        readings.append((obs.time, [obs.physical[m] for m in names]))
        h = env.hidden_state()
        states.append((env.now, h["link_mode"]["sensor:stripper"], "sensor:stripper" in h["faulted_links"]))
        if submitted is None and env.now >= restore_at - 1e-9:
            restore = env.registry.make_action("restore_primary", "sensor:stripper", {}, env.now, "restore@probe")
            submitted = env.submit(env.make_bundle(restore, [latest_window(env)], "rp")).status.value
    return rows, submitted, readings, states, env


def test_a_link_fault_meeting_the_backup_channel_breaks_only_the_primary_path():
    runs = {"link": backup_before_onset_view("link_fault", "sensor:stripper"),
            "module": backup_before_onset_view("io_module_fault", "stripper", onset=BACKUP_ONSET_S - 1.8),
            "none": backup_before_onset_view("link_fault", "sensor:stripper", onset=1e6)}
    env = runs["link"][-1]
    link_onset, dt = env.world.onset_time, env.plant.dt_s
    compared = {}
    for key in ("link", "module"):
        rows, base = runs[key][0], runs["none"][0]
        n = min(len(rows), len(base))
        assert rows[:n] == base[:n], key
        compared[key] = rows[n - 1][0]
    assert {key: run[1] for key, run in runs.items()} == {"link": "accepted", "module": "accepted", "none": "accepted"}
    rows = runs["link"][0]
    shown = [s for _, _, receipts in rows for s in receipts if s[0].startswith("rp:")]
    assert [s[1] for s in shown] == ["running", "succeeded"]
    succeeded = next(t for t, _, receipts in rows if any(s[0].startswith("rp:") and s[1] == "succeeded"
                                                         for s in receipts))
    assert min(compared.values()) > succeeded + 60.0
    readings, states = runs["link"][2], runs["link"][3]
    flowing = [v for t, v in readings if link_onset + dt < t < succeeded]
    held = [v for t, v in readings if t > succeeded]
    assert len(flowing) >= 2 and np.ptp(np.array(flowing), axis=0).max() > 0
    assert len(held) >= 2 and np.ptp(np.array(held), axis=0).max() == 0
    assert {s[1:] for s in states if link_onset + dt < s[0] < succeeded} == {("backup", True)}
    assert {s[1:] for s in states if s[0] > succeeded} == {("down", True)}
    assert runs["none"][3][-1][1:] == ("up", False) and runs["module"][3][-1][1:] == ("up", False)
    pre = [s for s in states if s[0] < link_onset]
    t = next(s[0] for s in states if link_onset + dt < s[0] < succeeded)
    assert belief.state_at(belief.world(belief.LINK, link_onset, dt, pre), t) == ("backup", True)


def test_tool_results_name_the_action_they_are_the_result_of():
    env, _ = make_env(tool_latency_scale=0.1)
    obs = first_alert(env)
    alert = obs.alerts[0].evidence_ids[0]
    tools = {"trace_link": ("trace_link", "sensor:stripper"), "historian": ("query_historian", "source:stripper"),
             "field_check": ("verify_sensor", "source:stripper"), "operator_report": ("escalate", TICKET)}
    bundles = {kind: env.make_bundle(pick(env, *tt), [alert], f"d-{kind}") for kind, tt in tools.items()}
    for bundle in bundles.values():
        assert env.submit(bundle).status == ExecStatus.ACCEPTED
    shown = {}
    while not env.done and len(shown) < len(tools):
        obs = env.advance_to_decision()
        shown.update({i.kind: i for i in obs.new_evidence if i.kind in tools})
    for kind, bundle in bundles.items():
        item = shown[kind]
        assert item.payload["action_id"] == bundle.action.action_id and isinstance(item.payload["action_id"], str)
        assert item.payload["idempotency_key"] == bundle.idempotency_key
        assert env.log.get(item.evidence_id) is item
        assert item.provenance == f"tool:{kind}" and "values" in item.payload


def test_a_respond_action_cannot_justify_itself_with_the_defenders_own_receipts():
    env, _ = make_env()
    obs = first_alert(env)
    alert = obs.alerts[0].evidence_ids[0]
    monitor = env.make_bundle(pick(env, "monitor", "plant"), [], "d1")
    env.submit(monitor)
    step_until(env, monitor.idempotency_key, ExecStatus.SUCCEEDED)
    own = [e for e in env.valid_evidence() if env.log.get(e).kind == "receipt"]
    assert len(own) == 2 and all(env.log.get(e).provenance == "executor" for e in own)
    backup = pick(env, "switch_to_backup", "sensor:stripper")
    for i in range(3):
        predicted = f"receipt:{len(receipt_items(env)) + 1}"
        cited = (own, own[:1], [predicted])[i]
        receipt = env.submit(env.make_bundle(backup, cited, f"self{i}"))
        assert (receipt.status, receipt.reason) == (ExecStatus.REJECTED, "policy_forbidden"), cited
    assert env.log.get(predicted).payload["idempotency_key"].startswith("self2:")
    env.advance(60)
    assert env.plant.link_mode["sensor:stripper"] == "down" and env.cost().compute == 0.0
    mixed = env.make_bundle(pick(env, "switch_to_backup", "sensor:stripper"), own + [alert], "mixed")
    assert env.submit(mixed).status == ExecStatus.ACCEPTED
    step_until(env, mixed.idempotency_key, ExecStatus.SUCCEEDED)
    assert env.plant.link_mode["sensor:stripper"] == "backup"
