from dataclasses import dataclass, field

from cpsdefense.core.types import ExecStatus
from cpsdefense.defense.evidence import parents_of

MODES = ("act_first", "investigate_first")
CITE_MARGIN_S = 60.0
REFUSED = (ExecStatus.REJECTED, ExecStatus.TIMED_OUT)


def citable(env, evidence_ids, margin_s=CITE_MARGIN_S):
    until = env.now + float(env.config["queue_latency_s"]) + float(margin_s)
    valid = set(env.valid_evidence())
    return [e for e in evidence_ids if e in valid and _lasts(env.log, e, until)]


def _lasts(log, evidence_id, t):
    item = log.get(evidence_id)
    return item is not None and t < item.valid_until and all(_lasts(log, p, t) for p in parents_of(item))


@dataclass
class Playbook:
    mode: str = "act_first"
    stale_fraction: float = 0.5
    deviation_decisions: int = 2
    cite_margin_s: float = CITE_MARGIN_S
    stale: dict = field(default_factory=dict)
    stale_time: dict = field(default_factory=dict)
    deviating: dict = field(default_factory=dict)
    link_loss: dict = field(default_factory=dict)
    traces: dict = field(default_factory=dict)
    done: set = field(default_factory=set)
    submitted: dict = field(default_factory=dict)
    backup_since: dict = field(default_factory=dict)
    repaired_since: dict = field(default_factory=dict)
    active: set = field(default_factory=set)
    escalated: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

    def update(self, obs, env):
        deviations = {}
        for item in obs.new_evidence:
            if item.kind == "alert":
                group = item.subject_id.split(":", 1)[1]
                if item.payload["category"] == "stale":
                    self.stale.setdefault(group, {})[item.payload["measurement"]] = item.evidence_id
                    self.stale_time.setdefault(group, {})[item.payload["measurement"]] = item.event_time
                else:
                    deviations.setdefault(group, []).append(item.evidence_id)
            elif item.kind == "link_health" and item.payload["values"][0] > 0:
                self.link_loss[item.subject_id] = item.evidence_id
            elif item.kind == "trace_link":
                self.traces[item.subject_id] = (bool(item.payload["values"][0] > 0.5), item.evidence_id)
        for g, ids in deviations.items():
            self.deviating.setdefault(g, []).append(tuple(ids))
        for receipt in obs.receipts:
            status = getattr(receipt, "status", None)
            tool_target = self.submitted.get(getattr(receipt, "action_id", None))
            if status is None or tool_target is None:
                continue
            if status == ExecStatus.SUCCEEDED and tool_target[0] == "switch_to_backup":
                self.backup_since[tool_target[1]] = receipt.finished_at
            elif status == ExecStatus.SUCCEEDED and tool_target[0] == "restore_primary":
                self.backup_since.pop(tool_target[1], None)
                self.done.discard(("switch_to_backup", tool_target[1]))
            elif status == ExecStatus.SUCCEEDED and tool_target[0] == "repair_link":
                self.repaired_since[tool_target[1]] = receipt.finished_at
            elif status in REFUSED:
                self.done.discard(tool_target)
                if tool_target[0] == "escalate":
                    self.escalated = False
        self.active = set(obs.active_actions)
        self.forget_expired(env)

    def forget_expired(self, env):
        valid = set(env.valid_evidence())
        for g in list(self.stale):
            self.stale[g] = {m: e for m, e in self.stale[g].items() if e in valid}
            self.stale_time[g] = {m: t for m, t in self.stale_time.get(g, {}).items() if m in self.stale[g]}
            if not self.stale[g]:
                del self.stale[g]
                self.stale_time.pop(g, None)
        self.link_loss = {link: e for link, e in self.link_loss.items() if e in valid}
        for link in [link for link, (_, e) in self.traces.items() if e not in valid]:
            del self.traces[link]
            self.done.discard(("trace_link", link))
        for g in list(self.deviating):
            self.deviating[g] = [ids for ids in self.deviating[g] if any(e in valid for e in ids)]
            if not self.deviating[g]:
                del self.deviating[g]

    def decide(self, env, obs):
        self.update(obs, env)
        return self.plan(env)

    def plan(self, env):
        cands = {(a.tool_id, a.target_id): a for a in env.candidates() if not a.bounded_arguments}
        cite = citable(env, env.valid_evidence(), self.cite_margin_s)
        cite_set = set(cite)
        running = {self.submitted[a] for a in self.active if a in self.submitted}
        plan = []

        def want(tool, target, evidence):
            key = (tool, target)
            ev = [e for e in evidence if e in cite_set]
            if key in cands and key not in self.done and ev:
                plan.append((cands[key], ev))
                self.note_submitted(cands[key])

        for g, channels in self.stale.items():
            link, source = f"sensor:{g}", f"source:{g}"
            evidence = list(channels.values())
            size = len(env.plant.sensor_links[g])
            if len(channels) >= max(2, self.stale_fraction * size):
                trace = self.traces.get(link)
                repaired = self.stale_after(g, self.repaired_since.get(link))
                if trace is not None:
                    ok, trace_id = trace
                    if ok:
                        want("replace_io_module", source, [trace_id, *evidence])
                    else:
                        want("switch_to_backup", link, [trace_id, *evidence])
                        want("repair_link", link, [trace_id, *evidence])
                elif link in self.backup_since:
                    since = self.backup_since[link]
                    fresh = [channels[m] for m, t in self.stale_time.get(g, {}).items() if t > since and m in channels]
                    if len(fresh) >= max(2, self.stale_fraction * size):
                        want("replace_io_module", source, fresh)
                    else:
                        want("repair_link", link, evidence)
                elif len(repaired) >= max(2, self.stale_fraction * size):
                    want("replace_io_module", source, repaired)
                elif ("trace_link", link) in running or ("replace_io_module", source) in running:
                    pass
                elif self.mode == "act_first":
                    want("switch_to_backup", link, evidence)
                    want("repair_link", link, evidence)
                else:
                    want("trace_link", link, evidence)
            else:
                for measurement, eid in channels.items():
                    want("recalibrate_sensor", measurement, [eid])

        failed = [link for link, (ok, _) in self.traces.items() if not ok and link not in self.link_loss]
        for link in [*self.link_loss, *failed]:
            if not link.startswith("actuator:"):
                continue
            loss, trace = self.link_loss.get(link), self.traces.get(link)
            if trace is not None:
                if not trace[0]:
                    want("repair_link", link, [trace[1], *([loss] if loss else [])])
            elif ("trace_link", link) in running:
                continue
            elif self.mode == "act_first":
                want("repair_link", link, [loss])
            else:
                want("trace_link", link, [loss])

        if not self.escalated and any(len(runs) >= self.deviation_decisions for runs in self.deviating.values()):
            incident = next((a for (tool, _), a in cands.items() if tool == "escalate"), None)
            recent = [e for e in cite if e.startswith("alert:deviation")]
            if incident is not None and recent:
                plan.append((incident, recent[-3:]))
                self.note_submitted(incident)
                self.escalated = True
        return plan

    def stale_after(self, group, since):
        if since is None:
            return []
        channels = self.stale.get(group, {})
        return [channels[m] for m, t in self.stale_time.get(group, {}).items() if t > since and m in channels]

    def note_submitted(self, action):
        self.done.add((action.tool_id, action.target_id))
        self.submitted[action.action_id] = (action.tool_id, action.target_id)


def run_episode(env, policy, decision_prefix="pb", max_decisions=None):
    log, n = [], 0
    obs = env.advance_to_decision()
    while not env.done:
        for action, evidence in policy.decide(env, obs):
            bundle = env.make_bundle(action, evidence, f"{decision_prefix}{n}")
            log.append((env.now, bundle, env.submit(bundle)))
        n += 1
        if max_decisions is not None and n >= max_decisions:
            break
        obs = env.advance_to_decision()
    return log
