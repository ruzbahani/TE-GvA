import copy
import itertools
import math
import pickle
from collections import Counter
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from cpsdefense.core.types import (REJECT_REASONS, ActionFamily, EvidenceItem, ExecStatus, ExecutionBundle, Receipt,
                                   SideEffectClass)
from cpsdefense.defense.evidence import EvidenceLog
from cpsdefense.defense.executor import Executor, ReviewItem, canonical_json, verify_receipt
from cpsdefense.defense.policy import EXECUTOR_PROVENANCE, POLICY_CONFIG, Policy
from cpsdefense.defense.registry import arg_grid, load_registry
from cpsdefense.utils.config import load_config

KEY = b"executor-test-key"
LATENCY = 5.0
SENSOR_LINKS = ("sensor:reactor", "sensor:stripper", "sensor:separator")
ACTUATOR_LINKS = ("actuator:feeds", "actuator:reactor")
SOURCES = ("reactor", "stripper", "separator")
SENSORS = ("reactor_pressure", "stripper_level")
INCIDENTS = ("inc-1",)
REGISTRY = load_registry()
POLICY_CFG = load_config(POLICY_CONFIG)
REJ, TIMED_OUT, OK, DUP = ExecStatus.REJECTED, ExecStatus.TIMED_OUT, ExecStatus.SUCCEEDED, ExecStatus.DUPLICATE


class FakeWorld:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.v = 0
        self.changed = {}
        self.mode = {link: "up" for link in SENSOR_LINKS + ACTUATOR_LINKS}
        self.healthy = {link: True for link in self.mode}
        self.backup_ok = True
        self.production = 1.0
        self.calibrated = {s: True for s in SENSORS}
        self.broken = set()
        self.no_effect = set()
        self.apply_ok = True
        self.compensate_ok = True
        self.raise_on_apply = False
        self.durations = {}
        self.applied = []
        self.compensated = []
        self.sampled = []

    def targets_by_kind(self):
        return {"link": list(self.mode), "source": list(SOURCES), "sensor": list(SENSORS), "plant": ["plant"],
                "incident": list(INCIDENTS)}

    def touch(self, subject):
        self.v += 1
        self.changed[subject] = self.v

    def version(self):
        return self.v

    def last_changed_version(self, subject_id):
        return self.changed.get(subject_id, 0)

    def exists(self, target_id, kind):
        return target_id in self.targets_by_kind().get(kind, ())

    def predicate(self, name, action):
        if name in self.broken:
            return False
        t, fraction = action.target_id, action.bounded_arguments.get("fraction")
        return {
            "link_exists": t in self.mode,
            "sensor_link": t in self.mode and t.startswith("sensor:"),
            "link_in_backup": self.mode.get(t) == "backup",
            "link_not_in_backup": t in self.mode and self.mode[t] != "backup",
            "backup_available": self.backup_ok,
            "link_healthy": self.healthy.get(t, False),
            "source_exists": t in SOURCES,
            "sensor_exists": t in SENSORS,
            "sensor_calibrated": self.calibrated.get(t, False),
            "incident_exists": t in INCIDENTS,
            "production_differs": fraction != self.production,
            "production_at_target": fraction == self.production,
        }[name]

    def sample_duration(self, spec, action):
        self.sampled.append(action.action_id)
        draw = float(spec.duration_median * np.exp(spec.duration_sigma * self.rng.standard_normal()))
        return self.durations.get(action.tool_id, draw)

    def apply(self, action, params, now):
        self.applied.append((action.action_id, action.tool_id, action.target_id, dict(params), now))
        if self.raise_on_apply:
            raise RuntimeError("effector failure")
        tool, t = action.tool_id, action.target_id
        if tool not in self.no_effect:
            if tool == "switch_to_backup":
                self.mode[t] = "backup"
            elif tool == "restore_primary":
                self.mode[t] = "up"
            elif tool == "repair_link":
                self.healthy[t] = True
            elif tool == "recalibrate_sensor":
                self.calibrated[t] = True
            elif tool == "set_production":
                self.production = params["fraction"]
                t = "plant"
            if action.family == ActionFamily.RESPOND:
                self.touch(t)
        evidence = ()
        if action.family == ActionFamily.OBSERVE:
            evidence = (EvidenceItem(f"result:{action.action_id}", "tool", "tool_result", action.target_id, now, now,
                                     now + 600.0, {"tool": tool}, tool),)
        effects = (f"{tool}:{action.target_id}",) if action.family == ActionFamily.RESPOND else ()
        return {"ok": self.apply_ok, "side_effects": effects, "evidence": evidence}

    def compensate(self, action, now):
        self.compensated.append((action.action_id, now))
        if action.tool_id == "switch_to_backup":
            self.mode[action.target_id] = "up"
            self.touch(action.target_id)
        return {"ok": self.compensate_ok, "side_effects": (f"compensate:{action.tool_id}:{action.target_id}",),
                "evidence": ()}


def setup(world=None, policy_config=None, latency=LATENCY):
    world = world or FakeWorld()
    policy = Policy(policy_config or POLICY_CFG, REGISTRY)
    log = EvidenceLog()
    ex = Executor(REGISTRY, policy, log, world, world, KEY, latency)
    return SimpleNamespace(world=world, policy=policy, log=log, ex=ex)


def add_evidence(log, eid, subject, now, arrival=None, valid_until=None, supersedes=(), payload=None,
                 provenance="test"):
    arrival = now - 1.0 if arrival is None else arrival
    log.add(EvidenceItem(eid, "network", "link_health", subject, arrival - 1.0, arrival,
                         now + 10_000.0 if valid_until is None else valid_until, payload or {}, provenance,
                         supersedes=tuple(supersedes)))
    return eid


def make_bundle(ctx, action, now, key, evidence=None):
    if evidence is None:
        evidence = (add_evidence(ctx.log, f"{key}-ev", action.target_id, now),)
    return ExecutionBundle(decision_id=f"dec-{key}", action=action, parameter_values=dict(action.bounded_arguments),
                           evidence_ids=tuple(evidence), observation_version=ctx.world.version(),
                           policy_version=ctx.policy.version, not_after=now + 10_000.0,
                           expected_postconditions=action.postconditions, monitor_rule="review on receipt",
                           recovery_or_compensation_id=action.compensation_or_recovery_id, idempotency_key=key)


def drain(ex):
    out = []
    while ex.active_actions():
        out += ex.tick(ex.next_event_time())
    return out


def run(ctx, bundle, now, hook=None, start_at=None):
    out = [ctx.ex.submit(bundle, now)]
    if out[0].status != ExecStatus.ACCEPTED:
        return out
    if hook is not None:
        hook()
    out += ctx.ex.tick(now + ctx.ex.queue_latency_s if start_at is None else start_at)
    return out + drain(ctx.ex)


def receipts_of(records):
    return [r for r in records if isinstance(r, Receipt)]


def with_action(bundle, **changes):
    return replace(bundle, action=replace(bundle.action, **changes))


def spec_of(bundle):
    return REGISTRY.get(bundle.action.tool_id)


def random_world(rng):
    world = FakeWorld(int(rng.integers(1 << 30)))
    for link in SENSOR_LINKS:
        if rng.random() < 0.4:
            world.mode[link] = "backup"
    world.backup_ok = rng.random() < 0.8
    world.production = float(rng.choice([0.8, 0.9, 1.0]))
    for link in world.mode:
        world.healthy[link] = rng.random() < 0.7
    for sensor in SENSORS:
        world.calibrated[sensor] = rng.random() < 0.7
    for _ in range(int(rng.integers(0, 6))):
        world.touch(str(rng.choice(list(world.mode) + list(SOURCES) + ["plant"])))
    return world


def base_case(rng, key):
    ctx = setup(random_world(rng))
    now = float(rng.uniform(500.0, 5000.0))
    cands = REGISTRY.candidates(ctx.world.targets_by_kind(), now - float(rng.uniform(0.0, 100.0)), ctx.policy,
                                ctx.world)
    action = cands[int(rng.integers(len(cands)))]
    respond = spec_of(SimpleNamespace(action=action)).family == ActionFamily.RESPOND
    pool = [action.target_id, "reactor", "plant", "sensor:reactor", f"subject-{key}"]
    ids = tuple(add_evidence(ctx.log, f"{key}-ev{i}", pool[int(rng.integers(len(pool)))], now)
                for i in range(int(rng.integers(1 if respond else 0, 4))))
    return ctx, make_bundle(ctx, action, now, key, ids), now


@dataclass
class Case:
    bundle: ExecutionBundle
    status: ExecStatus = OK
    reason: str | None = None
    hook: object = None
    start_at: float | None = None
    policy_config: dict | None = None


def pick(rng, values):
    return values[int(rng.integers(len(values)))]


def some_evidence(ctx, b, now, key):
    if b.evidence_ids:
        return pick(np.random.default_rng(len(key)), list(b.evidence_ids)), b
    eid = add_evidence(ctx.log, f"{key}-extra", b.action.target_id, now)
    return eid, replace(b, evidence_ids=(eid,))


def m_valid(ctx, rng, b, now):
    return Case(b)


def m_valid_late_tick(ctx, rng, b, now):
    return Case(b, start_at=now + LATENCY + float(rng.uniform(0.0, 200.0)))


def m_unknown_tool(ctx, rng, b, now):
    return Case(with_action(b, tool_id=pick(rng, ["open_all_valves", "", "TRACE_LINK", "trace_link "])),
                REJ, "not_in_registry")


def m_tool_version(ctx, rng, b, now):
    return Case(with_action(b, tool_version=pick(rng, ["0", "2", "", "1.0"])), REJ, "tool_version_mismatch")


def m_forged_field(ctx, rng, b, now):
    a = b.action
    options = [
        dict(family=ActionFamily.OBSERVE if a.family != ActionFamily.OBSERVE else ActionFamily.RESPOND),
        dict(preconditions=a.preconditions[1:] if a.preconditions else ("backup_available",)),
        dict(preconditions=list(a.preconditions)),
        dict(postconditions=a.postconditions + ("link_healthy",)),
        dict(side_effect_class=SideEffectClass.NONE if a.side_effect_class != SideEffectClass.NONE
             else SideEffectClass.IRREVERSIBLE),
        dict(resource_limit=a.resource_limit + 1.0),
        dict(expected_duration=a.expected_duration / 10.0),
        dict(duration_q90=a.duration_q90 * 2.0),
        dict(compensation_or_recovery_id="escalate"),
    ]
    if rng.random() < 0.15:
        return Case(replace(b, recovery_or_compensation_id="escalate"), REJ, "not_in_registry")
    return Case(with_action(b, **pick(rng, options)), REJ, "not_in_registry")


def m_bad_arg(ctx, rng, b, now):
    a = b.action
    if a.bounded_arguments:
        value = pick(rng, [0.85, 301, "0.9", True, None, math.nan, -1, [0.9], 1e9])
        args = {**a.bounded_arguments, next(iter(a.bounded_arguments)): value}
    else:
        args = {"force": True}
    return Case(replace(b, action=replace(a, bounded_arguments=args), parameter_values=dict(args)),
                REJ, "arg_out_of_bounds")


def m_unknown_arg(ctx, rng, b, now):
    args = {**b.action.bounded_arguments, pick(rng, ["force", "ttl", "fraction2"]): 1}
    return Case(replace(b, action=replace(b.action, bounded_arguments=args), parameter_values=dict(args)),
                REJ, "arg_out_of_bounds")


def m_missing_arg(ctx, rng, b, now):
    if not b.action.bounded_arguments:
        return None
    return Case(replace(b, action=replace(b.action, bounded_arguments={}), parameter_values={}),
                REJ, "arg_out_of_bounds")


def m_param_mismatch(ctx, rng, b, now):
    spec = spec_of(b)
    if not spec.args:
        return Case(replace(b, parameter_values={"extra": 1}), REJ, "arg_out_of_bounds")
    arg = spec.args[0]
    other = pick(rng, [c for c in arg.choices if c != b.action.bounded_arguments[arg.name]])
    return Case(replace(b, parameter_values={**b.action.bounded_arguments, arg.name: other}),
                REJ, "arg_out_of_bounds")


def m_unknown_target(ctx, rng, b, now):
    wrong_kind = {"link": "reactor", "source": "sensor:reactor", "sensor": "reactor", "plant": "inc-1",
                  "incident": "plant"}[spec_of(b).target_kinds[0]]
    target = pick(rng, ["sensor:nowhere", "", "plant2", wrong_kind, wrong_kind])
    return Case(with_action(b, target_id=target), REJ, "arg_out_of_bounds")


def m_no_authority(ctx, rng, b, now):
    scopes = [s for s in POLICY_CFG["authority"] if s != spec_of(b).required_authority]
    return Case(b, REJ, "unauthorized", policy_config={**POLICY_CFG, "authority": scopes})


def m_deep_production_cut(ctx, rng, b, now):
    a = REGISTRY.make_action("set_production", "plant", {"fraction": 0.8}, now, f"deep-{b.idempotency_key}")
    return Case(replace(b, action=a, parameter_values=dict(a.bounded_arguments), recovery_or_compensation_id=None),
                REJ, "unauthorized")


def m_stale_policy_version(ctx, rng, b, now):
    version = pick(rng, ["", "0" * 64, ctx.policy.version[:-1] + "x", ctx.policy.version.upper()])
    return Case(replace(b, policy_version=version), REJ, "policy_version_mismatch")


def m_respond_without_evidence(ctx, rng, b, now):
    if spec_of(b).family != ActionFamily.RESPOND:
        return None
    return Case(replace(b, evidence_ids=()), REJ, "policy_forbidden")


def m_respond_citing_only_receipts(ctx, rng, b, now):
    if spec_of(b).family != ActionFamily.RESPOND:
        return None
    key = b.idempotency_key
    ids = tuple(add_evidence(ctx.log, f"{key}-receipt{i}", pick(rng, [b.action.target_id, "plant"]), now,
                             provenance=EXECUTOR_PROVENANCE) for i in range(int(rng.integers(1, 3))))
    return Case(replace(b, evidence_ids=ids), REJ, "policy_forbidden")


def m_backup_on_actuator_link(ctx, rng, b, now):
    a = REGISTRY.make_action("switch_to_backup", pick(rng, list(ACTUATOR_LINKS)), {}, now, f"act-{b.idempotency_key}")
    return Case(replace(b, action=a, parameter_values={}, recovery_or_compensation_id=a.compensation_or_recovery_id),
                REJ, "policy_forbidden")


def m_expired_candidate(ctx, rng, b, now):
    return Case(with_action(b, expiry=pick(rng, [now, now - 1.0, now - 300.0, math.nan])), REJ, "candidate_expired")


def m_deadline_at_submit(ctx, rng, b, now):
    return Case(replace(b, not_after=pick(rng, [now - 10.0, now, now + LATENCY, math.nan])), REJ, "deadline_passed")


def m_deadline_in_queue(ctx, rng, b, now):
    not_after = now + LATENCY + float(rng.uniform(0.1, 50.0))
    return Case(replace(b, not_after=not_after), TIMED_OUT, "deadline_passed",
                start_at=not_after + pick(rng, [0.0, 1.0, 100.0]))


def m_unknown_evidence(ctx, rng, b, now):
    return Case(replace(b, evidence_ids=b.evidence_ids + ("ev-never-seen",)), REJ, "unknown_evidence")


def m_future_evidence(ctx, rng, b, now):
    eid = add_evidence(ctx.log, f"{b.idempotency_key}-future", b.action.target_id, now,
                       arrival=now + LATENCY + float(rng.uniform(0.001, 100.0)))
    return Case(replace(b, evidence_ids=b.evidence_ids + (eid,)), REJ, "unknown_evidence")


def m_expired_evidence(ctx, rng, b, now):
    until = now + pick(rng, [-0.5, LATENCY / 2, LATENCY])
    eid = add_evidence(ctx.log, f"{b.idempotency_key}-old", b.action.target_id, now, valid_until=until)
    return Case(replace(b, evidence_ids=b.evidence_ids + (eid,)), REJ, "stale_evidence")


def m_retracted_before_submit(ctx, rng, b, now):
    eid, b = some_evidence(ctx, b, now, b.idempotency_key)
    ctx.log.retract(eid, now - float(rng.uniform(0.0, 0.9)))
    return Case(b, REJ, "stale_evidence")


def m_retracted_in_queue(ctx, rng, b, now):
    eid, b = some_evidence(ctx, b, now, b.idempotency_key)
    at = now + pick(rng, [1.0, LATENCY])
    return Case(b, REJ, "stale_evidence", hook=lambda: ctx.log.retract(eid, at))


def m_superseded(ctx, rng, b, now):
    eid, b = some_evidence(ctx, b, now, b.idempotency_key)
    at = now + pick(rng, [0.5, LATENCY])
    key = b.idempotency_key
    return Case(b, REJ, "stale_evidence",
                hook=lambda: add_evidence(ctx.log, f"{key}-newer", "reactor", now, arrival=at, supersedes=(eid,)))


def m_derived_parent_retracted(ctx, rng, b, now):
    key = b.idempotency_key
    parent = add_evidence(ctx.log, f"{key}-parent", "reactor", now)
    child = add_evidence(ctx.log, f"{key}-child", b.action.target_id, now, payload={"derived_from": [parent]})
    return Case(replace(b, evidence_ids=b.evidence_ids + (child,)), REJ, "stale_evidence",
                hook=lambda: ctx.log.retract(parent, now + 1.0))


def m_target_changed(ctx, rng, b, now):
    target = b.action.target_id
    if rng.random() < 0.5:
        ctx.world.touch(target)
        return Case(b, REJ, "stale_state")
    return Case(b, REJ, "stale_state", hook=lambda: ctx.world.touch(target))


def m_evidence_subject_changed(ctx, rng, b, now):
    subject = f"subject-{b.idempotency_key}-other"
    eid = add_evidence(ctx.log, f"{b.idempotency_key}-subj", subject, now)
    return Case(replace(b, evidence_ids=b.evidence_ids + (eid,)), REJ, "stale_state",
                hook=lambda: ctx.world.touch(subject))


def m_future_observation_version(ctx, rng, b, now):
    v = ctx.world.version()
    return Case(replace(b, observation_version=pick(rng, [v + 1, v + 7, float(v), str(v), True])), REJ, "stale_state")


def m_precondition_fails(ctx, rng, b, now):
    pre = spec_of(b).preconditions
    if not pre:
        return None
    name = pick(rng, list(pre))
    return Case(b, REJ, "precondition_failed", hook=lambda: ctx.world.broken.add(name))

MUTATIONS = [m_valid, m_valid, m_valid, m_valid_late_tick, m_unknown_tool, m_tool_version, m_forged_field,
             m_bad_arg, m_unknown_arg, m_missing_arg, m_param_mismatch, m_unknown_target, m_no_authority,
             m_deep_production_cut, m_stale_policy_version, m_respond_without_evidence,
             m_respond_citing_only_receipts, m_backup_on_actuator_link,
             m_expired_candidate, m_deadline_at_submit, m_deadline_in_queue, m_unknown_evidence, m_future_evidence,
             m_expired_evidence, m_retracted_before_submit, m_retracted_in_queue, m_superseded,
             m_derived_parent_retracted, m_target_changed, m_evidence_subject_changed, m_future_observation_version,
             m_precondition_fails]


def run_case(ctx, case, now):
    if case.policy_config is not None:
        policy = Policy(case.policy_config, REGISTRY)
        ctx.policy = policy
        ctx.ex = Executor(REGISTRY, policy, ctx.log, ctx.world, ctx.world, KEY, LATENCY)
        case.bundle = replace(case.bundle, policy_version=policy.version)
    return run(ctx, case.bundle, now, case.hook, case.start_at)


@pytest.mark.parametrize("seed", range(20))
def test_random_bundles_take_effect_exactly_when_valid(seed):
    rng = np.random.default_rng(seed)
    seen = Counter()
    for i in range(3 * len(MUTATIONS)):
        mutation = MUTATIONS[i % len(MUTATIONS)]
        for attempt in range(50):
            ctx, b, now = base_case(rng, f"s{seed}-c{i}-{attempt}")
            case = mutation(ctx, rng, b, now)
            if case is not None:
                break
        seen[mutation.__name__] += 1
        records = run_case(ctx, case, now)
        receipts = receipts_of(records)
        final = receipts[-1]
        action_id = case.bundle.action.action_id
        assert (final.status, final.reason) == (case.status, case.reason), (mutation.__name__, receipts)
        assert case.reason is None or case.reason in REJECT_REASONS
        valid = case.status == OK
        assert len(ctx.world.applied) == int(valid) and ctx.world.sampled == ([action_id] if valid else [])
        assert not ctx.world.compensated
        statuses = [r.status for r in receipts]
        if valid:
            assert statuses == [ExecStatus.ACCEPTED, ExecStatus.RUNNING, OK]
            assert ctx.world.applied[0][0] == action_id
            start = now + LATENCY if case.start_at is None else case.start_at
            assert receipts[1].started_at == final.started_at == start and final.finished_at > start
            assert final.postconditions_met == {p: True for p in spec_of(case.bundle).postconditions}
        elif statuses[0] == ExecStatus.ACCEPTED:
            assert statuses == [ExecStatus.ACCEPTED, case.status] and final.started_at is None
        else:
            assert statuses == [case.status]
        assert all(r.idempotency_key == case.bundle.idempotency_key for r in receipts)
        assert all(verify_receipt(r, KEY) for r in ctx.ex.receipts + ctx.ex.reviews)
        assert not ctx.ex.active_actions()
    assert set(seen) == {m.__name__ for m in MUTATIONS}


@pytest.mark.parametrize("seed", range(10))
def test_duplicate_keys_never_cause_a_second_effect(seed):
    rng = np.random.default_rng(100 + seed)
    ctx, b, now = base_case(rng, f"dup{seed}")
    first = ctx.ex.submit(b, now)
    assert first.status == ExecStatus.ACCEPTED
    again = ctx.ex.submit(b, now + 1.0)
    assert (again.status, again.reason) == (DUP, None)
    other = REGISTRY.make_action("monitor", "plant", {"review_after_s": 300}, now, "other-action")
    hijack = replace(b, action=other, parameter_values=dict(other.bounded_arguments), recovery_or_compensation_id=None)
    assert ctx.ex.submit(hijack, now + 2.0).status == DUP
    drain_all = ctx.ex.tick(now + LATENCY) + drain(ctx.ex)
    assert [r.status for r in receipts_of(drain_all)] == [ExecStatus.RUNNING, OK]
    later = ctx.ex.submit(b, ctx.ex.now + 1.0)
    assert later.status in (DUP, REJ)
    drain(ctx.ex)
    assert [x[0] for x in ctx.world.applied] == [b.action.action_id]


def test_static_checks_follow_the_documented_order():
    order = [("tool", "not_in_registry"), ("version", "tool_version_mismatch"), ("forged", "not_in_registry"),
             ("args", "arg_out_of_bounds"), ("authority", "unauthorized"),
             ("policy_version", "policy_version_mismatch"), ("forbidden", "policy_forbidden"),
             ("expired", "candidate_expired"), ("deadline", "deadline_passed"), ("duplicate", None)]
    ctx = setup(policy_config={**POLICY_CFG, "rate_limits": []})
    now = 1000.0
    ev = add_evidence(ctx.log, "ev", "plant", now)
    used = REGISTRY.make_action("set_production", "plant", {"fraction": 0.9}, now, "used")
    assert ctx.ex.submit(make_bundle(ctx, used, now, "used", (ev,)), now).status == ExecStatus.ACCEPTED
    counter = itertools.count()

    def build(active):
        action = REGISTRY.make_action("set_production", "plant", {"fraction": 0.8 if "authority" in active else 0.9},
                                      now - 10.0, f"peel-{next(counter)}")
        if "expired" in active:
            action = replace(action, expiry=now)
        if "forged" in active:
            action = replace(action, preconditions=())
        if "version" in active:
            action = replace(action, tool_version="0")
        if "tool" in active:
            action = replace(action, tool_id="nuke")
        params = dict(action.bounded_arguments)
        if "args" in active:
            params["force"] = 1
        return replace(make_bundle(ctx, action, now, "used" if "duplicate" in active else action.action_id,
                                   () if "forbidden" in active else (ev,)),
                       parameter_values=params, policy_version="stale" if "policy_version" in active
                       else ctx.policy.version, not_after=now if "deadline" in active else now + 1000.0)

    for i, (_, reason) in enumerate(order):
        r = ctx.ex.submit(build({d for d, _ in order[i:]}), now)
        assert (r.status, r.reason) == ((DUP, None) if reason is None else (REJ, reason))
    rng = np.random.default_rng(0)
    for _ in range(300):
        active = {d for d, _ in order if rng.random() < 0.4}
        r = ctx.ex.submit(build(active), now)
        first = next((reason for d, reason in order if d in active), "accepted")
        if first == "accepted":
            assert r.status == ExecStatus.ACCEPTED
        else:
            assert (r.status, r.reason) == ((DUP, None) if first is None else (REJ, first))
    assert len(ctx.world.applied) == 0


@pytest.mark.parametrize("active", [set(c) for n in range(6)
                                    for c in itertools.combinations(("late", "unknown", "stale", "changed",
                                                                     "precondition"), n)])


def test_revalidation_follows_the_documented_order(active):
    order = [("late", TIMED_OUT, "deadline_passed"), ("unknown", REJ, "unknown_evidence"),
             ("stale", REJ, "stale_evidence"), ("changed", REJ, "stale_state"),
             ("precondition", REJ, "precondition_failed")]
    ctx = setup()
    now = 100.0
    action = REGISTRY.make_action("switch_to_backup", "sensor:stripper", {}, now, "act")
    ev = add_evidence(ctx.log, "ev", "sensor:stripper", now)
    b = replace(make_bundle(ctx, action, now, "k", (ev,) + (("ghost",) if "unknown" in active else ())),
                not_after=now + LATENCY + 1.0)
    assert ctx.ex.submit(b, now).status == ExecStatus.ACCEPTED
    if "stale" in active:
        ctx.log.retract(ev, now + 1.0)
    if "changed" in active:
        ctx.world.touch("sensor:stripper")
    if "precondition" in active:
        ctx.world.broken.add("backup_available")
    records = ctx.ex.tick(now + LATENCY + (2.0 if "late" in active else 0.0)) + drain(ctx.ex)
    final = receipts_of(records)[-1]
    want = next(((status, reason) for d, status, reason in order if d in active), (OK, None))
    assert (final.status, final.reason) == want
    assert len(ctx.world.applied) == int(want[0] == OK)


class Audited:
    def __init__(self, world, log, policy, bundles):
        self.world, self.log, self.policy, self.bundles = world, log, policy, bundles
        self.ex = None
        self.verdicts = {}

    def version(self):
        return self.world.version()

    def last_changed_version(self, subject_id):
        return self.world.last_changed_version(subject_id)

    def exists(self, target_id, kind):
        return self.world.exists(target_id, kind)

    def predicate(self, name, action):
        return self.world.predicate(name, action)

    def sample_duration(self, spec, action):
        b, t = self.bundles[action.action_id], self.ex.now
        subjects = lambda: [b.action.target_id] + [self.log.get(e).subject_id for e in b.evidence_ids]
        ok = (REGISTRY.matches(action) and b.policy_version == self.policy.version
              and self.policy.authorized(action) and t < b.not_after
              and all(self.log.valid(e, t) for e in b.evidence_ids)
              and all(self.world.last_changed_version(s) <= b.observation_version for s in subjects())
              and all(self.world.predicate(p, action) for p in spec.preconditions))
        self.verdicts.setdefault(action.action_id, []).append(ok)
        return self.world.sample_duration(spec, action)

    def apply(self, action, params, now):
        return self.world.apply(action, params, now)

    def compensate(self, action, now):
        return self.world.compensate(action, now)


def interleaved_script(seed, steps=120):
    rng = np.random.default_rng(seed)
    world = random_world(rng)
    log = EvidenceLog()
    policy = Policy(POLICY_CFG, REGISTRY)
    bundles = {}
    audited = Audited(world, log, policy, bundles)
    ex = Executor(REGISTRY, policy, log, audited, audited, KEY, LATENCY)
    audited.ex = ex
    subjects = list(world.mode) + list(SOURCES) + list(SENSORS) + ["plant", "inc-1"]
    accepted_keys, invalid = [], set()
    records, t, n = [], 0.0, 0
    for _ in range(steps):
        t += float(rng.exponential(40.0))
        while ex.next_event_time() is not None and ex.next_event_time() <= t:
            records += ex.tick(ex.next_event_time())
        r = rng.random()
        valid_now = log.valid_ids(t)
        if r < 0.15 and valid_now:
            log.retract(pick(rng, list(valid_now)), t + float(rng.uniform(0.0, 20.0)))
        elif r < 0.35:
            world.touch(pick(rng, subjects))
        elif r < 0.4:
            world.backup_ok = not world.backup_ok
        for _ in range(int(rng.integers(0, 3))):
            n += 1
            add_evidence(log, f"e{n}", pick(rng, subjects), t, arrival=t, valid_until=t + float(rng.exponential(300.0)),
                         supersedes=(f"e{int(rng.integers(1, n))}",) if n > 1 and rng.random() < 0.1 else ())
        for _ in range(int(rng.integers(0, 3))):
            cands = REGISTRY.candidates(world.targets_by_kind(), t, policy, world)
            valid_now = list(log.valid_ids(t))
            if not cands or not valid_now:
                continue
            n += 1
            base = pick(rng, cands)
            action = REGISTRY.make_action(base.tool_id, base.target_id, base.bounded_arguments, t, f"a{n}")
            ids = tuple(pick(rng, valid_now) for _ in range(int(rng.integers(1, 4))))
            b = ExecutionBundle(f"d{n}", action, dict(action.bounded_arguments), ids,
                                world.version() - int(rng.choice([0, 0, 0, 1, 2])), policy.version,
                                t + float(rng.uniform(1.0, 600.0)), action.postconditions, "",
                                action.compensation_or_recovery_id, f"k{n}")
            corrupt = rng.random()
            if corrupt < 0.05:
                b = with_action(b, tool_id="open_all_valves")
            elif corrupt < 0.1:
                b = replace(b, parameter_values={**b.parameter_values, "force": True})
            elif corrupt < 0.15:
                b = replace(b, policy_version="stale")
            elif corrupt < 0.2 and accepted_keys:
                b = replace(b, idempotency_key=pick(rng, accepted_keys))
            bundles[action.action_id] = b
            receipt = ex.submit(b, t)
            if corrupt < 0.2 and (corrupt < 0.15 or accepted_keys):
                invalid.add(action.action_id)
                assert receipt.status in (REJ, DUP)
            if receipt.status == ExecStatus.ACCEPTED:
                accepted_keys.append(b.idempotency_key)
            records.append(receipt)
    records += drain(ex)
    return world, ex, audited, bundles, invalid, records


@pytest.mark.parametrize("seed", range(8))
def test_interleaved_script_never_applies_an_unjustified_or_repeated_effect(seed):
    world, ex, audited, bundles, invalid, records = interleaved_script(seed)
    applied = [x[0] for x in world.applied]
    assert len(applied) == len(set(applied))
    keys = [bundles[a].idempotency_key for a in applied]
    assert len(keys) == len(set(keys))
    assert not invalid & set(applied) and not invalid & set(world.sampled)
    assert set(applied) == set(world.sampled)
    assert all(v == [True] for v in audited.verdicts.values())
    assert applied and all(verify_receipt(r, KEY) for r in records)
    reasons = Counter(r.reason for r in receipts_of(records) if r.status in (REJ, TIMED_OUT))
    assert set(reasons) <= set(REJECT_REASONS)


def test_executor_is_deterministic_given_its_inputs():
    first = [canonical_json(r) + r.signature for r in interleaved_script(3, steps=80)[-1]]
    second = [canonical_json(r) + r.signature for r in interleaved_script(3, steps=80)[-1]]
    assert first == second and len(first) > 20


def test_pickled_executor_continues_identically():
    ctx = setup()
    now = 100.0
    a = REGISTRY.make_action("switch_to_backup", "sensor:reactor", {}, now, "a")
    ctx.ex.submit(make_bundle(ctx, a, now, "a"), now)
    ctx.ex.tick(now + LATENCY)
    clone = pickle.loads(pickle.dumps(ctx))

    def cont(c):
        m = REGISTRY.make_action("monitor", "plant", {"review_after_s": 300}, now + 10, "m")
        out = [c.ex.submit(make_bundle(c, m, now + 10, "m", ()), now + 10)] + drain(c.ex)
        out += c.ex.tick(now + 10_000)
        return [(type(r).__name__, r.signature) for r in out], [x[0] for x in c.world.applied]

    assert cont(ctx) == cont(clone)
    assert clone.world.mode["sensor:reactor"] == "backup"


def test_compensation_runs_when_a_postcondition_fails():
    world = FakeWorld()
    world.no_effect.add("switch_to_backup")
    ctx = setup(world)
    a = REGISTRY.make_action("switch_to_backup", "sensor:reactor", {}, 10.0, "a")
    records = run(ctx, make_bundle(ctx, a, 10.0, "k"), 10.0)
    final = receipts_of(records)[-1]
    assert (final.status, final.reason) == (ExecStatus.COMPENSATED, "postcondition_failed")
    assert final.postconditions_met == {"link_in_backup": False}
    assert final.side_effects == ("switch_to_backup:sensor:reactor", "compensate:switch_to_backup:sensor:reactor")
    assert [x[0] for x in world.applied] == ["a"] and [c[0] for c in world.compensated] == ["a"]
    assert [e[2] for e in ctx.ex.side_effects] == list(final.side_effects)
    assert ctx.ex.next_event_time() is None and not ctx.ex.tick(1e6)


def test_tool_error_and_failed_compensation():
    world = FakeWorld()
    world.apply_ok = False
    ctx = setup(world)
    a = REGISTRY.make_action("switch_to_backup", "sensor:reactor", {}, 10.0, "a")
    final = receipts_of(run(ctx, make_bundle(ctx, a, 10.0, "k"), 10.0))[-1]
    assert (final.status, final.reason) == (ExecStatus.COMPENSATED, "tool_error")
    world = FakeWorld()
    world.no_effect.add("switch_to_backup")
    world.compensate_ok = False
    ctx = setup(world)
    final = receipts_of(run(ctx, make_bundle(ctx, a, 10.0, "k"), 10.0))[-1]
    assert (final.status, final.reason) == (ExecStatus.FAILED, "compensation_failed")
    assert len(world.compensated) == 1


def test_failure_without_compensation_is_reported_and_not_compensated():
    world = FakeWorld()
    world.no_effect.add("set_production")
    ctx = setup(world)
    a = REGISTRY.make_action("set_production", "plant", {"fraction": 0.9}, 10.0, "a")
    final = receipts_of(run(ctx, make_bundle(ctx, a, 10.0, "k"), 10.0))[-1]
    assert (final.status, final.reason) == (ExecStatus.FAILED, "postcondition_failed")
    assert final.postconditions_met == {"production_at_target": False}
    assert not world.compensated and not ctx.ex.tick(1e6)


@pytest.mark.parametrize("caps,ttl", [(None, 3600.0), ({"switch_to_backup": 1800, "respond": 900}, 1800.0),
                                      ({"respond": 900}, 900.0)])


def test_ttl_expiry_raises_a_review_and_undoes_nothing(caps, ttl):
    config = POLICY_CFG if caps is None else {**POLICY_CFG, "ttl_caps": caps}
    ctx = setup(policy_config=config)
    a = REGISTRY.make_action("switch_to_backup", "sensor:reactor", {}, 10.0, "a")
    records = run(ctx, make_bundle(ctx, a, 10.0, "k"), 10.0)
    done = receipts_of(records)[-1]
    assert done.status == OK and not [r for r in records if isinstance(r, ReviewItem)]
    assert ctx.ex.next_event_time() == done.finished_at + ttl
    assert ctx.ex.tick(done.finished_at + ttl - 1.0) == []
    (review,) = ctx.ex.tick(done.finished_at + ttl + 5.0)
    assert isinstance(review, ReviewItem) and review.kind == "ttl_expired"
    assert (review.action_id, review.tool_id, review.target_id) == ("a", "switch_to_backup", "sensor:reactor")
    assert review.due_at == done.finished_at + ttl and review.raised_at == done.finished_at + ttl + 5.0
    assert verify_receipt(review, KEY) and not verify_receipt(replace(review, kind="review_due"), KEY)
    assert ctx.world.mode["sensor:reactor"] == "backup"
    assert len(ctx.world.applied) == 1 and not ctx.world.compensated
    assert ctx.ex.tick(1e7) == [] and ctx.ex.next_event_time() is None and ctx.ex.reviews == [review]


def test_monitor_schedules_a_review_after_completion():
    ctx = setup()
    a = REGISTRY.make_action("monitor", "plant", {"review_after_s": 900}, 10.0, "m")
    records = run(ctx, make_bundle(ctx, a, 10.0, "k", ()), 10.0)
    done = receipts_of(records)[-1]
    assert done.status == OK
    (review,) = ctx.ex.tick(done.finished_at + 900.0)
    assert review.kind == "review_due" and review.due_at == done.finished_at + 900.0


def test_tool_results_enter_the_evidence_log_and_can_justify_later_actions():
    ctx = setup()
    trace = REGISTRY.make_action("trace_link", "sensor:stripper", {}, 10.0, "t")
    done = receipts_of(run(ctx, make_bundle(ctx, trace, 10.0, "k1", ()), 10.0))[-1]
    assert done.status == OK and ctx.log.valid("result:t", done.finished_at)
    now = done.finished_at + 1.0
    switch = REGISTRY.make_action("switch_to_backup", "sensor:stripper", {}, now, "s")
    final = receipts_of(run(ctx, make_bundle(ctx, switch, now, "k2", ("result:t",)), now))[-1]
    assert final.status == OK


class ResultWorld(FakeWorld):
    def _result(self, eid, action, now):
        return EvidenceItem(eid, "tool", "tool_result", action.target_id, now, now, now + 600.0,
                            {"values": [1.0], "action_id": "someone-else"}, "tool")

    def apply(self, action, params, now):
        out = super().apply(action, params, now)
        return {**out, "evidence": out["evidence"] + (self._result(f"apply:{action.action_id}", action, now),)}

    def compensate(self, action, now):
        return {**super().compensate(action, now), "evidence": (self._result(f"comp:{action.action_id}", action, now),)}


def test_tool_results_are_stamped_with_the_action_they_are_the_result_of():
    world = ResultWorld()
    world.no_effect.add("switch_to_backup")
    ctx = setup(world)
    trace = REGISTRY.make_action("trace_link", "sensor:stripper", {}, 10.0, "t")
    run(ctx, make_bundle(ctx, trace, 10.0, "k1", ()), 10.0)
    assert ctx.log.get("result:t").payload == {"tool": "trace_link", "action_id": "t", "idempotency_key": "k1"}
    assert ctx.log.get("apply:t").payload == {"values": [1.0], "action_id": "t", "idempotency_key": "k1"}
    now = ctx.ex.now + 1.0
    switch = REGISTRY.make_action("switch_to_backup", "sensor:reactor", {}, now, "s")
    final = receipts_of(run(ctx, make_bundle(ctx, switch, now, "k2"), now))[-1]
    assert final.status == ExecStatus.COMPENSATED
    for eid in ("apply:s", "comp:s"):
        assert ctx.log.get(eid).payload == {"values": [1.0], "action_id": "s", "idempotency_key": "k2"}


def test_respond_actions_need_world_evidence_the_log_holds():
    ctx = setup()
    now = 100.0
    receipt = add_evidence(ctx.log, "receipt:1", "plant", now, provenance=EXECUTOR_PROVENANCE)
    later = add_evidence(ctx.log, "later", "sensor:reactor", now, arrival=now + 1.0)
    world = add_evidence(ctx.log, "world", "sensor:reactor", now)
    a = REGISTRY.make_action("repair_link", "sensor:reactor", {}, now, "a")
    for i, ids in enumerate([(receipt,), ("receipt:2",), (later,), (receipt, "receipt:2", later)]):
        r = ctx.ex.submit(make_bundle(ctx, a, now, f"k{i}", ids), now)
        assert (r.status, r.reason) == (REJ, "policy_forbidden"), ids
    ok = run(ctx, make_bundle(ctx, a, now, "ok", (receipt, world)), now)
    assert [r.status for r in receipts_of(ok)] == [ExecStatus.ACCEPTED, ExecStatus.RUNNING, OK]
    assert [x[0] for x in ctx.world.applied] == ["a"]


def test_own_effects_make_older_observations_stale():
    ctx = setup()
    v0 = ctx.world.version()
    a = REGISTRY.make_action("switch_to_backup", "sensor:reactor", {}, 10.0, "a")
    assert receipts_of(run(ctx, make_bundle(ctx, a, 10.0, "k1"), 10.0))[-1].status == OK
    now = ctx.ex.now + 1.0
    repair = REGISTRY.make_action("repair_link", "sensor:reactor", {}, now, "r")
    stale = replace(make_bundle(ctx, repair, now, "k2"), observation_version=v0)
    assert receipts_of(run(ctx, stale, now))[-1].reason == "stale_state"
    fresh = make_bundle(ctx, repair, ctx.ex.now, "k3")
    assert receipts_of(run(ctx, fresh, ctx.ex.now))[-1].status == OK
    assert [x[0] for x in ctx.world.applied] == ["a", "r"]


def test_rate_limit_through_the_executor():
    ctx = setup()
    now = 10.0
    bundles = [make_bundle(ctx, REGISTRY.make_action("repair_link", "sensor:reactor", {}, now, f"r{i}"), now, f"k{i}")
               for i in range(4)]
    statuses = [ctx.ex.submit(b, now) for b in bundles]
    assert [r.status for r in statuses[:3]] == [ExecStatus.ACCEPTED] * 3
    assert (statuses[3].status, statuses[3].reason) == (REJ, "resource_limit")
    assert ctx.ex.submit(bundles[0], now).status == DUP
    other = REGISTRY.make_action("repair_link", "sensor:stripper", {}, now, "o")
    assert ctx.ex.submit(make_bundle(ctx, other, now, "ko"), now).status == ExecStatus.ACCEPTED


def test_effector_error_never_leads_to_a_second_apply():
    world = FakeWorld()
    world.raise_on_apply = True
    ctx = setup(world)
    a = REGISTRY.make_action("repair_link", "sensor:reactor", {}, 10.0, "a")
    ctx.ex.submit(make_bundle(ctx, a, 10.0, "k"), 10.0)
    ctx.ex.tick(10.0 + LATENCY)
    with pytest.raises(RuntimeError):
        ctx.ex.tick(1e6)
    world.raise_on_apply = False
    assert ctx.ex.tick(2e6) == [] and not ctx.ex.active_actions()
    assert len(world.applied) == 1


def test_event_bookkeeping():
    ctx = setup()
    a = REGISTRY.make_action("set_production", "plant", {"fraction": 0.9}, 10.0, "a")
    accepted = ctx.ex.submit(make_bundle(ctx, a, 10.0, "k"), 10.0)
    assert accepted.status == ExecStatus.ACCEPTED and accepted.started_at is None
    assert ctx.ex.active_actions() == ("a",) and ctx.ex.next_event_time() == 10.0 + LATENCY
    assert ctx.ex.tick(10.0 + LATENCY - 0.001) == []
    (running,) = ctx.ex.tick(10.0 + LATENCY)
    assert running.status == ExecStatus.RUNNING and running.started_at == 10.0 + LATENCY
    finish = ctx.ex.next_event_time()
    assert finish > running.started_at and ctx.world.sampled == ["a"] and not ctx.world.applied
    (done,) = ctx.ex.tick(finish)
    assert done.status == OK and done.finished_at == finish and done.submitted_at == 10.0
    assert done.side_effects == ("set_production:plant",) and ctx.world.production == 0.9
    assert ctx.ex.active_actions() == () and ctx.ex.next_event_time() == finish + 3600.0
    assert [r.status for r in ctx.ex.receipts] == [ExecStatus.ACCEPTED, ExecStatus.RUNNING, OK]
    assert all(ctx.ex.verify(r) for r in ctx.ex.receipts) and not ctx.ex.verify(replace(done, reason="x"))


def test_same_time_events_complete_before_they_start():
    ctx = setup(latency=0.0)
    p = REGISTRY.make_action("set_production", "plant", {"fraction": 0.9}, 0.0, "p")
    ctx.ex.submit(make_bundle(ctx, p, 0.0, "k1"), 0.0)
    ctx.ex.tick(0.0)
    finish = ctx.ex.next_event_time()
    m = REGISTRY.make_action("monitor", "plant", {"review_after_s": 300}, finish, "m")
    ctx.ex.submit(make_bundle(ctx, m, finish, "k2", ()), finish)
    order = [f"{r.action_id}:{r.status.value}:{r.reason}" for r in receipts_of(ctx.ex.tick(finish))]
    assert order == ["p:succeeded:None", "m:rejected:stale_state"]


def test_reviews_come_after_completions_due_at_the_same_time():
    world = FakeWorld()
    world.durations = {"monitor": 1.0, "set_production": 300.0}
    ctx = setup(world, latency=0.0)
    m = REGISTRY.make_action("monitor", "plant", {"review_after_s": 300}, 0.0, "m")
    ctx.ex.submit(make_bundle(ctx, m, 0.0, "k1", ()), 0.0)
    ctx.ex.tick(0.0)
    ctx.ex.tick(1.0)
    p = REGISTRY.make_action("set_production", "plant", {"fraction": 0.9}, 1.0, "p")
    ctx.ex.submit(make_bundle(ctx, p, 1.0, "k2"), 1.0)
    ctx.ex.tick(1.0)
    records = ctx.ex.tick(301.0)
    assert [(type(r).__name__, r.action_id) for r in records] == [("Receipt", "p"), ("ReviewItem", "m")]


@pytest.mark.parametrize("seed", range(6))
def test_candidates_are_exactly_the_instances_the_executor_starts(seed):
    rng = np.random.default_rng(200 + seed)
    world = random_world(rng)
    now = 1000.0
    targets = world.targets_by_kind()
    policy = Policy(POLICY_CFG, REGISTRY)
    probe_targets = {kind: ids + ["ghost"] for kind, ids in targets.items()}
    cands = {(a.tool_id, a.target_id, tuple(a.bounded_arguments.items()))
             for a in REGISTRY.candidates(probe_targets, now, policy, world)}
    started = set()
    for spec in REGISTRY:
        for kind in spec.target_kinds:
            for target in probe_targets[kind]:
                for args in arg_grid(spec):
                    ctx = setup(copy.deepcopy(world), latency=0.0)
                    action = REGISTRY.make_action(spec.tool_id, target, args, now, "probe")
                    b = make_bundle(ctx, action, now, "probe")
                    records = [ctx.ex.submit(b, now)] + ctx.ex.tick(now)
                    if any(r.status == ExecStatus.RUNNING for r in receipts_of(records)):
                        started.add((spec.tool_id, target, tuple(action.bounded_arguments.items())))
    assert started == cands and len(cands) > 5


def _sample_records():
    world = FakeWorld()
    world.no_effect.add("switch_to_backup")
    ctx = setup(world)
    a = REGISTRY.make_action("switch_to_backup", "sensor:reactor", {}, 10.0, "a")
    run(ctx, make_bundle(ctx, a, 10.0, "k"), 10.0)
    m = REGISTRY.make_action("monitor", "plant", {"review_after_s": 300}, 200.0, "m")
    run(ctx, make_bundle(ctx, m, 200.0, "km", ()), 200.0)
    ctx.ex.tick(5000.0)
    ctx.ex.submit(with_action(make_bundle(ctx, m, 5000.0, "bad", ()), tool_id="nuke"), 5000.0)
    return ctx.ex.receipts + ctx.ex.reviews


def _tampered(value):
    if isinstance(value, ExecStatus):
        return ExecStatus.SUCCEEDED if value != ExecStatus.SUCCEEDED else ExecStatus.FAILED
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1.0
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, tuple):
        return value + ("extra",)
    if isinstance(value, dict):
        return {**value, "forged": True}
    return "forged"


def test_receipts_verify_and_any_tampering_is_detected():
    records = _sample_records()
    kinds = {r.status for r in records if isinstance(r, Receipt)}
    assert {ExecStatus.ACCEPTED, ExecStatus.RUNNING, ExecStatus.COMPENSATED, OK, REJ} <= kinds
    assert any(isinstance(r, ReviewItem) for r in records)
    for record in records:
        assert verify_receipt(record, KEY) and verify_receipt(record, KEY.decode())
        assert not verify_receipt(record, b"another-key")
        assert not verify_receipt(replace(record, signature=""), KEY)
        assert not verify_receipt(replace(record, signature=record.signature[:-1] + "0"), KEY) or \
            record.signature.endswith("0")
        for f in fields(record):
            if f.name != "signature":
                forged = replace(record, **{f.name: _tampered(getattr(record, f.name))})
                assert not verify_receipt(forged, KEY), f.name
        if isinstance(record, Receipt) and record.postconditions_met:
            flipped = {k: not v for k, v in record.postconditions_met.items()}
            assert not verify_receipt(replace(record, postconditions_met=flipped), KEY)


def test_canonical_json_is_stable_and_typed():
    record = _sample_records()[0]
    assert canonical_json(record) == canonical_json(copy.deepcopy(record))
    assert '"record_type":"Receipt"' in canonical_json(record) and '"signature"' not in canonical_json(record)
    review = ReviewItem("r", "ttl_expired", "d", "a", "k", "t", "x", 1.0, 2.0)
    assert '"record_type":"ReviewItem"' in canonical_json(review)


def test_time_must_not_go_backwards():
    ctx = setup()
    ctx.ex.tick(10.0)
    with pytest.raises(ValueError):
        ctx.ex.tick(9.0)
    with pytest.raises(ValueError):
        ctx.ex.tick(math.nan)
    a = REGISTRY.make_action("monitor", "plant", {"review_after_s": 300}, 10.0, "m")
    with pytest.raises(ValueError):
        ctx.ex.submit(make_bundle(ctx, a, 10.0, "k", ()), 5.0)


def test_malformed_submissions():
    ctx = setup()
    a = REGISTRY.make_action("monitor", "plant", {"review_after_s": 300}, 10.0, "m")
    b = make_bundle(ctx, a, 10.0, "k", ())
    with pytest.raises(TypeError):
        ctx.ex.submit({"action": a}, 10.0)
    with pytest.raises(ValueError):
        ctx.ex.submit(replace(b, idempotency_key=""), 10.0)
    r = ctx.ex.submit(replace(b, action=None), 10.0)
    assert (r.status, r.reason) == (REJ, "not_in_registry")
    r = ctx.ex.submit(replace(b, parameter_values=None), 10.0)
    assert (r.status, r.reason) == (REJ, "arg_out_of_bounds")
    add_evidence(ctx.log, "a", "plant", 10.0)
    add_evidence(ctx.log, "b", "plant", 10.0)
    queued = ctx.ex.submit(replace(b, evidence_ids="ab"), 10.0)
    assert queued.status == ExecStatus.ACCEPTED
    (final,) = ctx.ex.tick(10.0 + LATENCY)
    assert (final.status, final.reason) == (REJ, "unknown_evidence")
    with pytest.raises(ValueError):
        Executor(REGISTRY, ctx.policy, ctx.log, ctx.world, ctx.world, b"", LATENCY)
    with pytest.raises(ValueError):
        Executor(REGISTRY, ctx.policy, ctx.log, ctx.world, ctx.world, KEY, -1.0)
    assert not ctx.world.applied
