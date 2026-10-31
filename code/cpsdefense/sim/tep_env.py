import bisect
import copy
import math
import numbers
from dataclasses import dataclass, replace

import numpy as np

from cpsdefense.core.types import (CostBreakdown, EvidenceItem, EvidenceSource, ExecStatus, ExecutionBundle,
                                   ObservationBundle)
from cpsdefense.defense.evidence import EvidenceLog
from cpsdefense.defense.executor import Executor, ReviewItem
from cpsdefense.defense.policy import EXECUTOR_PROVENANCE, load_policy
from cpsdefense.defense.registry import load_registry
from cpsdefense.sim.tep_plant import TEPPlant
from cpsdefense.sim.tep_telemetry import Telemetry, as_alert
from cpsdefense.sim.tep_world import TEPWorld
from cpsdefense.utils.config import load_config
from cpsdefense.utils.seeding import spawn_generators

TERMINAL = {ExecStatus.SUCCEEDED, ExecStatus.FAILED, ExecStatus.COMPENSATED, ExecStatus.TIMED_OUT,
            ExecStatus.REJECTED}
TICKET = "incident:1"
RECEIPT_STATUSES = ("accepted", "rejected", "running", "succeeded", "failed", "timed_out", "duplicate", "compensated")
RECEIPT_LIFETIME_S = 3600.0


@dataclass(frozen=True)
class EnvSnapshot:
    plant: object
    state: dict


def receipt_item(receipt, n, tool_id, target_id, tools, supersedes=()):
    t = float(next(x for x in (receipt.finished_at, receipt.started_at, receipt.submitted_at) if x is not None))
    status = ExecStatus(receipt.status).value
    tools = list(tools)
    values = np.zeros(len(tools) + len(RECEIPT_STATUSES), np.float32)
    if tool_id in tools:
        values[tools.index(tool_id)] = 1.0
    values[len(tools) + RECEIPT_STATUSES.index(status)] = 1.0
    until = math.inf if status == ExecStatus.RUNNING.value else t + RECEIPT_LIFETIME_S
    return EvidenceItem(evidence_id=f"receipt:{n}", source=EvidenceSource.TOOL.value, kind="receipt",
                        subject_id=str(target_id), event_time=t, arrival_time=t, valid_until=until,
                        payload={"values": values, "tool_id": str(tool_id), "status": status, "event_start": t,
                                 "action_id": str(receipt.action_id), "idempotency_key": str(receipt.idempotency_key)},
                        provenance=EXECUTOR_PROVENANCE, supersedes=tuple(supersedes))


class TEPEnv:
    _STATE = ("telemetry", "world", "log", "executor", "_cost", "_charged_shutdown", "_pending", "_new_items",
              "_new_alerts", "_new_receipts", "_onset_done", "_next_tick", "_n_obs", "_obs_versions",
              "nominal_margins", "_jobs", "_n_receipts", "_running_items")

    def __init__(self, scenario, config=None, plant=None, registry=None, policy=None, costs=None):
        self.config = config or load_config("sim/tep_env.yaml")
        self.costs = costs or load_config("sim/costs.yaml")
        self.scenario = scenario
        self.plant = plant or TEPPlant()
        self.registry = registry or load_registry()
        self.policy = policy or load_policy(self.registry)
        self.tools = [spec.tool_id for spec in self.registry]

    def reset(self):
        sc, p = self.scenario, self.plant
        streams = spawn_generators(sc.seed, ["telemetry", "tools", "operator"])
        p.reset(seed=sc.seed, disturbances=sc.background)
        self.telemetry = Telemetry(p, streams["telemetry"], evidence_scale=sc.evidence_scale)
        self.world = TEPWorld(p, sc, streams["tools"], streams["operator"], self.config)
        self.log = EvidenceLog()
        self.executor = Executor(self.registry, self.policy, self.log, self.world, self.world,
                                 self.config["signing_key"], self.config["queue_latency_s"])
        self._cost = CostBreakdown()
        self._charged_shutdown = False
        self._pending, self._new_items, self._new_alerts, self._new_receipts = [], [], [], []
        self._jobs = {}
        self._n_receipts = 0
        self._running_items = {}
        self._n_obs = 0
        self._obs_versions = []
        self._onset_done = False
        self._next_tick = sc.warmup_s + float(self.config["decision_tick_s"])
        steps = int(round(sc.warmup_s / p.dt_s))
        for _ in range(steps):
            p.step()
            self._ingest(self.telemetry.observe_step())
        self.telemetry.freeze_baseline()
        self.nominal_margins = dict(p.margins)
        self._arrivals()
        self._new_items, self._new_alerts = [], []
        return self.observe()

    @property
    def now(self):
        return self.plant.time_s

    @property
    def done(self):
        return self.plant.shutdown is not None or self.now >= self.scenario.end_s

    def _ingest(self, items):
        for item in items:
            self.log.add(item)
            self._pending.append(item)

    def _arrivals(self):
        now, keep = self.now, []
        for item in self._pending:
            if item.arrival_time <= now:
                self._new_items.append(item)
                if item.kind == "alert" and now >= self.scenario.warmup_s:
                    self._new_alerts.append(as_alert(item))
                    self.world.open_ticket(TICKET)
            else:
                keep.append(item)
        self._pending = keep

    def _step(self):
        p, sc = self.plant, self.scenario
        if not self._onset_done and self.now >= self.world.onset_time:
            self.world.apply_incident(sc.incident)
            self._onset_done = True
        p.step()
        self._ingest(self.telemetry.observe_step())
        nxt = self.executor.next_event_time()
        if nxt is not None and nxt <= self.now:
            for record in self.executor.tick(self.now):
                self._new_receipts.append(record)
                if isinstance(record, ReviewItem):
                    continue
                key = record.idempotency_key
                tool, target = self._jobs[key]
                ended = record.status in TERMINAL and record.started_at is not None and key in self._running_items
                item = self._log_receipt(record, tool, target, (self._running_items.pop(key),) if ended else ())
                if record.status == ExecStatus.RUNNING:
                    self._running_items[key] = item.evidence_id
                    spec = self.registry.get(tool)
                    self._cost.compute += spec.resource_cost * self.costs["compute"]["per_resource_unit"]
            self._pending.extend(self._logged(item) for item in self.world.outbox)
            self.world.outbox.clear()
        self._arrivals()
        self._accrue(p.dt_s)

    def _logged(self, item):
        logged = self.log.get(item.evidence_id)
        if logged is None:
            raise RuntimeError(f"tool result {item.evidence_id} is not in the evidence log")
        return logged

    def _log_receipt(self, receipt, tool, target, supersedes=()):
        self._n_receipts += 1
        item = receipt_item(receipt, self._n_receipts, tool, target, self.tools, supersedes)
        self._ingest([item])
        return item

    def _accrue(self, dt):
        p, c = self.plant, self.costs
        if p.shutdown is not None:
            if not self._charged_shutdown:
                self._cost.physical += c["physical"]["shutdown"]
                self._charged_shutdown = True
            return
        frac = c["physical"]["near_limit_fraction"]
        near = any(p.margins[k] < frac * m for k, m in self.nominal_margins.items() if m > 0)
        if near:
            self._cost.physical += c["physical"]["near_limit_per_h"] * dt / 3600.0
        prod, nominal = p.production_rate()
        shortfall = max(0.0, 1.0 - prod / nominal) if nominal > 0 else 0.0
        self._cost.service += c["service"]["production_shortfall_per_h"] * shortfall * dt / 3600.0
        if any(m == "backup" for m in p.link_mode.values()):
            self._cost.service += c["service"]["degraded_per_h"] * dt / 3600.0

    def advance(self, seconds):
        end = self.now + seconds
        while not self.done and self.now + 1e-9 < end:
            self._step()
        return self.observe()

    def advance_to_decision(self, max_s=None):
        end = None if max_s is None else self.now + max_s
        while not self.done:
            self._step()
            if self._new_alerts or any(self._is_trigger(r) for r in self._new_receipts):
                break
            if self.now >= self._next_tick:
                break
            if end is not None and self.now >= end:
                break
        while self._next_tick <= self.now:
            self._next_tick += float(self.config["decision_tick_s"])
        return self.observe()

    @staticmethod
    def _is_trigger(record):
        return isinstance(record, ReviewItem) or record.status in TERMINAL

    def observe(self):
        p = self.plant
        self._n_obs += 1
        world_version = self.world.version()
        if not self._obs_versions or self._obs_versions[-1][1] != world_version:
            self._obs_versions.append((self._n_obs, world_version))
        obs = ObservationBundle(
            time=self.now, version=self._n_obs, new_evidence=tuple(self._new_items),
            alerts=tuple(self._new_alerts), physical=dict(zip(p.measurement_names, p.delivered.tolist())),
            active_actions=self.executor.active_actions(), receipts=tuple(self._new_receipts),
            policy_version=self.policy.version)
        self._new_items, self._new_alerts, self._new_receipts = [], [], []
        return obs

    def targets(self):
        return {"link": list(self.plant.links), "source": [f"source:{g}" for g in self.plant.sensor_links],
                "sensor": list(self.plant.measurement_names), "plant": ["plant"],
                "incident": [self.world.open_incident] if self.world.open_incident else []}

    def candidates(self):
        return self.registry.candidates(self.targets(), self.now, self.policy, self.world,
                                        history=tuple(self.executor.history))

    def make_bundle(self, action, evidence_ids, decision_id, not_after_s=600.0, monitor_rule=""):
        return ExecutionBundle(
            decision_id=decision_id, action=action, parameter_values=dict(action.bounded_arguments),
            evidence_ids=tuple(evidence_ids), observation_version=self._n_obs,
            policy_version=self.policy.version, not_after=self.now + not_after_s,
            expected_postconditions=tuple(action.postconditions), monitor_rule=monitor_rule,
            recovery_or_compensation_id=action.compensation_or_recovery_id,
            idempotency_key=f"{decision_id}:{action.action_id}")

    def _world_version_at(self, token):
        if not isinstance(token, numbers.Integral) or isinstance(token, bool) or not 1 <= token <= self._n_obs:
            return None
        return self._obs_versions[bisect.bisect_right(self._obs_versions, (int(token), math.inf)) - 1][1]

    def submit(self, bundle):
        if isinstance(bundle, ExecutionBundle):
            bundle = replace(bundle, observation_version=self._world_version_at(bundle.observation_version))
        receipt = self.executor.submit(bundle, self.now)
        tool, target = getattr(bundle.action, "tool_id", ""), getattr(bundle.action, "target_id", "")
        if receipt.status == ExecStatus.ACCEPTED:
            self._jobs[bundle.idempotency_key] = (tool, target)
        if receipt.status in TERMINAL or receipt.status == ExecStatus.DUPLICATE:
            self._new_receipts.append(receipt)
        self._log_receipt(receipt, tool, target)
        return receipt

    def valid_evidence(self, t=None):
        return self.log.valid_ids(self.now if t is None else t)

    def cost(self):
        return copy.copy(self._cost)

    def hidden_state(self):
        p, w = self.plant, self.world
        return {"cause": self.scenario.incident.cause, "kind": self.scenario.incident.kind,
                "onset_time": w.onset_time, "onset_happened": self._onset_done, "shutdown": p.shutdown,
                "hidden_changes": dict(sorted(w.hidden_changes.items())),
                "faulted_links": sorted(w.faulted_links), "io_faulted": sorted(w.io_faulted),
                "sensor_faults": sorted(p.faults), "instrument_faults": sorted(w.instrument_faults),
                "link_mode": dict(p.link_mode),
                "production": p.production_fraction, "disturbances": p.active_disturbances()}

    def oracle_time_to_shutdown(self, horizon_s):
        return self.plant.time_to_shutdown(horizon_s)

    def reseed(self, *tag):
        names = ["telemetry", "tools", "operator", "process"]
        streams = spawn_generators([self.scenario.seed, *(int(t) for t in tag)], names)
        self.telemetry.rng = streams["telemetry"]
        self.world.rng_tools, self.world.rng_operator = streams["tools"], streams["operator"]
        self.world.reset_tool_streams()
        self.plant.reseed_noise(int(streams["process"].integers(2**63 - 1)))

    def _shared_memo(self):
        return {id(self.registry): self.registry, id(self.policy): self.policy, id(self.costs): self.costs,
                id(self.config): self.config}

    def snapshot(self):
        state = copy.deepcopy({k: getattr(self, k) for k in self._STATE}, self._shared_memo())
        return EnvSnapshot(self.plant.snapshot(), state)

    def restore(self, snap):
        self.plant.restore(snap.plant)
        for key, value in copy.deepcopy(snap.state, self._shared_memo()).items():
            setattr(self, key, value)
        self.telemetry.plant = self.world.plant = self.plant
