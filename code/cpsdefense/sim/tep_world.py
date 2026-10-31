import copy
import zlib
from dataclasses import replace

import numpy as np

from cpsdefense.core.types import EvidenceItem, EvidenceSource


class TEPWorld:
    def __init__(self, plant, scenario, rng_tools, rng_operator, config):
        self.plant = plant
        self.scenario = scenario
        self.rng_tools = rng_tools
        self.rng_operator = rng_operator
        self.config = config
        self.groups = list(plant.sensor_links)
        self.group_of = {plant.measurement_names[i]: g for g, idx in plant.sensor_links.items() for i in idx}
        self.faulted_links = set()
        self.io_faulted = set()
        self.instrument_faults = {}
        self.open_incident = None
        self._version = 0
        self.changed = {}
        self.hidden_changes = {}
        self.n_evidence = 0
        self.outbox = []
        self.reset_tool_streams()

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        for key, value in self.__dict__.items():
            if key != "plant":
                setattr(clone, key, copy.deepcopy(value, memo))
        return clone

    def touch(self, *subjects):
        self._version += 1
        for s in subjects:
            self.changed[s] = self._version

    def version(self):
        return self._version

    def last_changed_version(self, subject_id):
        return self.changed.get(subject_id, 0)

    def apply_incident(self, incident):
        incident.apply(self.plant)
        if incident.kind == "link_fault":
            self.faulted_links.add(incident.target)
            subjects = (incident.target,)
        elif incident.kind == "io_module_fault":
            self.io_faulted.add(incident.target)
            idx = self.plant.sensor_links[incident.target]
            subjects = (f"source:{incident.target}", *(self.plant.measurement_names[i] for i in idx))
        elif incident.kind == "sensor_fault":
            self.instrument_faults[incident.target] = dict(self.plant.faults[incident.target])
            subjects = (incident.target, f"source:{self.group_of[incident.target]}")
        else:
            subjects = ("plant",)
        for s in subjects:
            self.hidden_changes[s] = float(self.plant.time_s)

    def open_ticket(self, ticket_id):
        if self.open_incident is None:
            self.open_incident = ticket_id
            self.touch(ticket_id)

    def exists(self, target_id, kind):
        if kind == "link":
            return target_id in self.plant.link_mode
        if kind == "source":
            return target_id.startswith("source:") and target_id[7:] in self.plant.sensor_links
        if kind == "sensor":
            return target_id in self.plant.index
        if kind == "plant":
            return target_id == "plant"
        if kind == "incident":
            return target_id is not None and target_id == self.open_incident
        return False

    def predicate(self, name, action):
        t = action.target_id
        mode = self.plant.link_mode.get(t)
        if name == "link_exists":
            return mode is not None
        if name == "sensor_link":
            return mode is not None and t.startswith("sensor:")
        if name == "link_in_backup":
            return mode == "backup"
        if name == "link_not_in_backup":
            return mode is not None and mode != "backup"
        if name == "backup_available":
            return t.startswith("sensor:") and bool(self.scenario.backup_available)
        if name == "link_healthy":
            return mode is not None and t not in self.faulted_links
        if name == "source_exists":
            return self.exists(t, "source")
        if name == "sensor_exists":
            return t in self.plant.index
        if name == "sensor_calibrated":
            return t in self.plant.index and t not in self.instrument_faults
        if name == "incident_exists":
            return self.exists(t, "incident")
        if name == "production_differs":
            return float(action.bounded_arguments["fraction"]) != self.plant.production_fraction
        if name == "production_at_target":
            return float(action.bounded_arguments["fraction"]) == self.plant.production_fraction
        raise KeyError(f"unknown predicate {name!r}")

    def reset_tool_streams(self):
        self.tool_seed = int(self.rng_tools.integers(2**63 - 1))
        self.tool_rngs = {}

    def tool_stream(self, tool_id):
        if tool_id not in self.tool_rngs:
            seq = np.random.SeedSequence([self.tool_seed, zlib.crc32(tool_id.encode("utf-8"))])
            self.tool_rngs[tool_id] = np.random.default_rng(seq)
        return self.tool_rngs[tool_id]

    def sample_duration(self, spec, action):
        z = self.tool_stream(action.tool_id).standard_normal()
        return float(spec.duration_median * self.scenario.tool_latency_scale * np.exp(spec.duration_sigma * z))

    def _evidence(self, kind, subject, now, lifetime, values, **extra):
        self.n_evidence += 1
        item = EvidenceItem(evidence_id=f"tool:{kind}:{self.n_evidence}", source=EvidenceSource.TOOL.value,
                            kind=kind, subject_id=subject, event_time=float(now), arrival_time=float(now),
                            valid_until=float(now + lifetime),
                            payload={"values": np.asarray(values, np.float32), "event_start": float(now), **extra},
                            provenance=f"tool:{kind}")
        self.outbox.append(item)
        return item

    def apply(self, action, params, now):
        tool, t, cfg = action.tool_id, action.target_id, self.config["tools"]
        p = self.plant
        if tool == "trace_link":
            healthy = t not in self.faulted_links
            reported = healthy if self.tool_stream("trace_link:accuracy").random() < cfg["trace_accuracy"] \
                else not healthy
            item = self._evidence("trace_link", t, now, cfg["trace_lifetime_s"],
                                  [float(reported), float(p.link_mode[t] == "backup")])
            return {"ok": True, "side_effects": (), "evidence": (item,)}
        if tool == "query_historian":
            idx = p.sensor_links[t[7:]]
            item = self._evidence("historian", t, now, cfg["historian_lifetime_s"], p.delivered[idx])
            return {"ok": True, "side_effects": (), "evidence": (item,)}
        if tool == "verify_sensor":
            idx = p.sensor_links[t[7:]]
            values = np.concatenate([p.true_measured[idx], p.measured[idx]])
            item = self._evidence("field_check", t, now, cfg["verify_lifetime_s"], values)
            return {"ok": True, "side_effects": (), "evidence": (item,)}
        if tool == "switch_to_backup":
            if not self.scenario.backup_available:
                return {"ok": False, "side_effects": ("backup_unavailable",), "evidence": ()}
            p.set_link_mode(t, "backup")
            self.touch(t)
            return {"ok": True, "side_effects": (f"{t}:backup",), "evidence": ()}
        if tool == "restore_primary":
            p.set_link_mode(t, "down" if t in self.faulted_links else "up")
            self.touch(t)
            return {"ok": True, "side_effects": (f"{t}:primary",), "evidence": ()}
        if tool == "repair_link":
            self.faulted_links.discard(t)
            if p.link_mode[t] == "down":
                p.set_link_mode(t, "up")
            self.touch(t)
            return {"ok": True, "side_effects": (f"{t}:repaired",), "evidence": ()}
        if tool == "recalibrate_sensor":
            self.instrument_faults.pop(t, None)
            if self.group_of.get(t) not in self.io_faulted:
                p.clear_sensor_fault(t)
            self.touch(t, f"source:{self.group_of[t]}")
            return {"ok": True, "side_effects": (f"{t}:recalibrated",), "evidence": ()}
        if tool == "replace_io_module":
            group = t[7:]
            if group in self.io_faulted:
                self.io_faulted.discard(group)
                for i in p.sensor_links[group]:
                    name = p.measurement_names[i]
                    p.clear_sensor_fault(name)
                    if name in self.instrument_faults:
                        p.faults[name] = dict(self.instrument_faults[name])
            self.touch(t, *(p.measurement_names[i] for i in p.sensor_links[group]))
            return {"ok": True, "side_effects": (f"{t}:module_replaced",), "evidence": ()}
        if tool == "set_production":
            p.set_production(float(params["fraction"]))
            self.touch("plant")
            return {"ok": True, "side_effects": (f"production:{params['fraction']}",), "evidence": ()}
        if tool == "monitor":
            return {"ok": True, "side_effects": (), "evidence": ()}
        if tool == "escalate":
            return {"ok": True, "side_effects": ("operator_engaged",), "evidence": (self._diagnosis(t, now),)}
        raise KeyError(f"no effect defined for tool {tool!r}")

    def _diagnosis(self, ticket, now):
        causes, cfg = self.config["causes"], self.config["tools"]
        true = self.scenario.incident.kind if now >= self.onset_time else "none"
        u = self.rng_operator.random()
        wrong = [c for c in causes if c != true]
        named = true if u < cfg["operator_accuracy"] else wrong[int(self.rng_operator.integers(len(wrong)))]
        values = [float(c == named) for c in causes]
        return self._evidence("operator_report", ticket, now, cfg["operator_lifetime_s"], values, cause=named)

    def compensate(self, action, now):
        if action.tool_id == "switch_to_backup":
            return self.apply(replace(action, tool_id="restore_primary"), {}, now)
        return {"ok": False, "side_effects": ("no_compensation",), "evidence": ()}

    @property
    def onset_time(self):
        return self.scenario.warmup_s + self.scenario.incident.onset_s
