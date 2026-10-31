import copy

import numpy as np

from cpsdefense.core.types import Alert, EvidenceItem, EvidenceSource
from cpsdefense.utils.config import load_config

STATS = ("last", "mean", "std", "slope", "frac_unchanged")


class Telemetry:
    def __init__(self, plant, rng, config=None, evidence_scale=1.0):
        self.config = config or load_config("sim/telemetry.yaml")
        self.plant = plant
        self.rng = rng
        self.scale = float(evidence_scale)
        self.window_steps = int(self.config["window_steps"])
        self.groups = list(plant.sensor_links)
        n = len(plant.measurement_names)
        self.buffer = []
        self.window = 0
        self.outage_left = {g: 0 for g in self.groups}
        self.baseline_rows = []
        self.baseline = None
        self.constant_run = np.zeros(n, np.int64)
        self.max_constant_run = np.zeros(n, np.int64)
        self.deviation_run = np.zeros(n, np.int64)
        self.active = set()
        self.last_alert = {}

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        for key, value in self.__dict__.items():
            if key != "plant":
                setattr(clone, key, copy.deepcopy(value, memo))
        return clone

    def _delay(self, median, sigma):
        if sigma == 0:
            return float(median * self.scale)
        return float(median * self.scale * np.exp(sigma * self.rng.standard_normal()))

    def _item(self, eid, source, kind, subject, start, end, delay, lifetime, payload, provenance):
        arrival = end + delay
        return EvidenceItem(evidence_id=eid, source=source, kind=kind, subject_id=subject, event_time=float(end),
                            arrival_time=float(arrival), valid_until=float(arrival + lifetime / self.scale),
                            payload={**payload, "event_start": float(start)}, provenance=provenance)

    def observe_step(self):
        p = self.plant
        self.buffer.append((p.time_s, p.delivered.copy(), dict(p.link_mode)))
        if len(self.buffer) < self.window_steps:
            return []
        items = self._end_window()
        self.buffer = []
        self.window += 1
        return items

    def freeze_baseline(self):
        rows = np.array(self.baseline_rows)
        if len(rows) < 5:
            raise RuntimeError("not enough warm-up windows to fit the detector baseline")
        cfg = self.config["detectors"]
        mean = rows.mean(0)
        std = np.maximum(rows.std(0), cfg["deviation_rel_floor"] * np.abs(mean))
        self.baseline = {
            "mean": mean,
            "std": np.where(std > 0, std, np.inf),
            "stale_windows": np.maximum(cfg["stale_min_windows"],
                                        np.ceil(cfg["stale_factor"] * np.maximum(self.max_constant_run, 1))),
        }

    def _end_window(self):
        times = np.array([b[0] for b in self.buffer])
        values = np.stack([b[1] for b in self.buffer])
        modes = self.buffer[-1][2]
        start, end = float(times[0] - self.plant.dt_s), float(times[-1])
        stats = self._stats(times, values)
        self.constant_run = np.where(np.all(values == values[0], axis=0), self.constant_run + 1, 0)
        if self.baseline is None and end >= self.config["detectors"]["baseline_from_s"]:
            self.baseline_rows.append(stats[:, 1].copy())
            self.max_constant_run = np.maximum(self.max_constant_run, self.constant_run)
        return self._source_items(start, end, stats) + self._link_items(start, end, modes) + \
            self._alerts(start, end, stats)

    @staticmethod
    def _stats(times, values):
        t = times - times.mean()
        slope = (t[:, None] * (values - values.mean(0))).sum(0) / max(float((t ** 2).sum()), 1e-12)
        frac_unchanged = (np.diff(values, axis=0) == 0).mean(0)
        return np.stack([values[-1], values.mean(0), values.std(0), slope, frac_unchanged], 1)

    def _source_items(self, start, end, stats):
        items, out = [], self.config["outage"]
        for g in self.groups:
            if self.outage_left[g] > 0:
                self.outage_left[g] -= 1
                continue
            if self.rng.random() < out["start_prob"]:
                self.outage_left[g] = int(self.rng.geometric(1.0 / out["mean_windows"])) - 1
                continue
            spec = self.config["sources"][g]
            items.append(self._item(
                f"win:{g}:{self.window}", EvidenceSource.PHYSICAL.value, "window_summary", f"source:{g}", start,
                end, self._delay(spec["delay_s"], spec["sigma"]), spec["lifetime_s"],
                {"values": stats[self.plant.sensor_links[g]].astype(np.float32).ravel()}, f"historian:{g}"))
        return items

    def _link_items(self, start, end, modes):
        cfg, items = self.config["link_health"], []
        for link, mode in modes.items():
            u = self.rng.random()
            if mode == "down":
                report = u < cfg["p_detect_down"]
            elif mode == "backup":
                report = bool(cfg["report_backup"])
            else:
                report = u < cfg["p_false"]
            delay = self._delay(cfg["delay_s"], cfg["sigma"])
            if report:
                values = np.array([float(mode != "backup"), float(mode == "backup")], np.float32)
                items.append(self._item(f"link:{link}:{self.window}", EvidenceSource.NETWORK.value, "link_health",
                                        link, start, end, delay, cfg["lifetime_s"], {"values": values}, "netmon"))
        return items

    def _alerts(self, start, end, stats):
        if self.baseline is None:
            return []
        cfg, items = self.config["detectors"], []
        z = np.abs(stats[:, 1] - self.baseline["mean"]) / self.baseline["std"]
        self.deviation_run = np.where(z > cfg["deviation_z"], self.deviation_run + 1, 0)
        deviating = self.deviation_run >= cfg["deviation_persist_windows"]
        stale = self.constant_run >= self.baseline["stale_windows"]
        for g in self.groups:
            for i in self.plant.sensor_links[g]:
                for category, hit, score in (("stale", stale[i], float(self.constant_run[i])),
                                             ("deviation", deviating[i], float(z[i]))):
                    key = (category, int(i))
                    if not hit:
                        self.active.discard(key)
                        continue
                    due = key not in self.active or self.window - self.last_alert[key] >= cfg["realert_windows"]
                    self.active.add(key)
                    if not due:
                        continue
                    self.last_alert[key] = self.window
                    name = self.plant.measurement_names[i]
                    items.append(self._item(
                        f"alert:{category}:{name}:{self.window}", EvidenceSource.DETECTOR.value, "alert",
                        f"source:{g}", start, end, self._delay(cfg["delay_s"], 0.0), cfg["lifetime_s"],
                        {"values": np.array([category == "stale", category == "deviation", score], np.float32),
                         "category": category, "measurement": name}, f"detector:{category}"))
        return items


def as_alert(item):
    return Alert(alert_id=item.evidence_id, detector_id=item.provenance, subject_id=item.subject_id,
                 time=item.arrival_time, score=float(item.payload["values"][-1]),
                 category=item.payload["category"], evidence_ids=(item.evidence_id,))
