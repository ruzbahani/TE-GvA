import copy
from dataclasses import asdict, dataclass

import numpy as np

from cpsdefense.agents.playbook import CITE_MARGIN_S, citable
from cpsdefense.core.types import ExecStatus
from cpsdefense.defense.registry import REVIEW_ARG

GLOBAL_TOOLS = ("monitor", "escalate", "set_production")
RELEVANT_KINDS = ("alert", "link_health", "trace_link", "field_check", "operator_report")


@dataclass(frozen=True)
class BranchResult:
    candidate: int
    replicate: int
    tool_id: str
    target_id: str
    args: dict
    family: str
    total: float
    cost: dict
    shutdown: bool
    time_to_shutdown: float | None
    resolved: bool | None
    mitigated: bool | None
    first_status: str
    first_reason: str | None
    first_duration: float | None
    first_started: float | None
    horizon_s: float


def relevant_evidence(env, limit=8, margin_s=CITE_MARGIN_S):
    ids = env.valid_evidence()
    if margin_s is not None:
        ids = citable(env, ids, margin_s)
    items = [i for i in (env.log.get(e) for e in ids) if i.kind in RELEVANT_KINDS]
    items.sort(key=lambda i: i.arrival_time, reverse=True)
    return [i.evidence_id for i in items[:limit]]


def relevant_candidates(env):
    groups, links, sensors = set(), set(), set()
    for eid in relevant_evidence(env, limit=64, margin_s=None):
        item = env.log.get(eid)
        if item.subject_id.startswith("source:"):
            groups.add(item.subject_id[7:])
            if item.kind == "alert":
                sensors.add(item.payload["measurement"])
        elif item.subject_id in env.plant.link_mode:
            links.add(item.subject_id)
    for g in groups:
        links.update({f"sensor:{g}", f"actuator:{g}"})
    targets = links | {f"source:{g}" for g in groups} | sensors
    return [a for a in env.candidates() if a.target_id in targets or a.tool_id in GLOBAL_TOOLS]


def _outcome(env):
    h = env.hidden_state()
    if h["kind"] == "disturbance":
        return None, None
    cleared = not (h["faulted_links"] or h["io_faulted"] or h["sensor_faults"])
    flowing = all(h["link_mode"][l] == "backup" for l in h["faulted_links"]) and not h["io_faulted"] \
        and not h["sensor_faults"]
    return cleared, cleared or flowing


def run_branch(env, policy, action, evidence_ids, end, name):
    t0 = env.now
    bundle = env.make_bundle(action, evidence_ids, name)
    if env.submit(bundle).status == ExecStatus.ACCEPTED:
        policy.note_submitted(action)
    wait_until = t0 + float(action.bounded_arguments[REVIEW_ARG]) if action.family.value == "monitor" else t0
    n = 0
    while not env.done and env.now < end - 1e-9:
        stop = min(end, wait_until) if env.now < wait_until - 1e-9 else end
        obs = env.advance_to_decision(max_s=stop - env.now)
        if env.now < wait_until - 1e-9:
            policy.update(obs, env)
        else:
            for act, ev in policy.decide(env, obs):
                env.submit(env.make_bundle(act, ev, f"{name}d{n}"))
        n += 1
    return bundle


def branch_labels(env, policy, horizon_s, replicates=4, tag=(0,), candidates=None, evidence_ids=None,
                  margin_s=CITE_MARGIN_S):
    root, t0 = env.snapshot(), env.now
    c0 = env.cost()
    cands = list(candidates) if candidates is not None else relevant_candidates(env)
    evidence = list(evidence_ids) if evidence_ids is not None else relevant_evidence(env, margin_s=margin_s)
    end = min(env.scenario.end_s, t0 + horizon_s)
    results = []
    for i, action in enumerate(cands):
        for k in range(replicates):
            env.restore(root)
            env.reseed(*tag, k)
            bundle = run_branch(env, copy.deepcopy(policy), action, evidence, end, f"branch{i}r{k}")
            c = env.cost()
            cost = {k2: getattr(c, k2) - getattr(c0, k2) for k2 in ("attack", "service", "physical", "compute")}
            first = [r for r in env.executor.receipts if r.idempotency_key == bundle.idempotency_key]
            last = first[-1] if first else None
            resolved, mitigated = _outcome(env)
            shutdown = env.plant.shutdown
            results.append(BranchResult(
                candidate=i, replicate=k, tool_id=action.tool_id, target_id=action.target_id,
                args=dict(action.bounded_arguments), family=action.family.value, total=float(sum(cost.values())),
                cost=cost, shutdown=shutdown is not None,
                time_to_shutdown=None if shutdown is None else float(shutdown[0] - t0),
                resolved=resolved, mitigated=mitigated,
                first_status=last.status.value if last else "none", first_reason=last.reason if last else None,
                first_duration=(last.finished_at - last.started_at) if last is not None and last.finished_at
                is not None and last.started_at is not None else None,
                first_started=None if last is None or last.started_at is None else float(last.started_at - t0),
                horizon_s=float(end - t0)))
    env.restore(root)
    return cands, results


def summarize(cands, results):
    by = {}
    for r in results:
        by.setdefault(r.candidate, []).append(r)
    mean = {i: float(np.mean([r.total for r in rs])) for i, rs in by.items()}
    act = [mean[i] for i, a in enumerate(cands) if a.family.value != "observe" and i in mean]
    best_act = min(act) if act else None
    rows = []
    for i, a in enumerate(cands):
        rs = by.get(i, [])
        rows.append({
            "tool": a.tool_id, "target": a.target_id, "args": dict(a.bounded_arguments), "family": a.family.value,
            "J": mean.get(i), "p_shutdown": float(np.mean([r.shutdown for r in rs])) if rs else None,
            "p_resolved": _mean_or_none([r.resolved for r in rs]),
            "voi": (best_act - mean[i]) if a.family.value == "observe" and best_act is not None else None})
    return rows


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def as_record(result):
    return asdict(result)
