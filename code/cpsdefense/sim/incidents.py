from dataclasses import dataclass, field, replace

import numpy as np

from cpsdefense.utils.config import load_config

FACTORS = ("background", "link", "evidence_validity", "tool_latency", "backup")


@dataclass(frozen=True)
class Incident:
    kind: str
    target: str
    onset_s: float
    fault_kind: str | None = None
    magnitude: float = 0.0

    def apply(self, plant):
        if self.kind == "link_fault":
            if plant.link_mode[self.target] != "backup":
                plant.set_link_mode(self.target, "down")
        elif self.kind == "io_module_fault":
            for i in plant.sensor_links[self.target]:
                plant.set_sensor_fault(plant.measurement_names[i], "frozen")
        elif self.kind == "sensor_fault":
            plant.set_sensor_fault(self.target, self.fault_kind, self.magnitude)
        elif self.kind == "disturbance":
            plant.set_disturbances(sorted({*plant.active_disturbances(), int(self.target[3:])}))
        else:
            raise ValueError(f"unknown incident kind {self.kind!r}")

    @property
    def cause(self):
        return f"{self.kind}:{self.target}" + (f":{self.fault_kind}" if self.fault_kind else "")


@dataclass(frozen=True)
class Scenario:
    seed: int
    incident: Incident
    background: tuple = ()
    warmup_s: float = 3600.0
    horizon_s: float = 21600.0
    evidence_scale: float = 1.0
    tool_latency_scale: float = 1.0
    backup_available: bool = True
    pair_id: str | None = None
    factor: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def end_s(self):
        return self.warmup_s + self.incident.onset_s + self.horizon_s


def _choice(rng, options, weights=None):
    options = list(options)
    p = None if weights is None else np.asarray(weights, np.float64) / np.sum(weights)
    return options[int(rng.choice(len(options), p=p))]


def sample_incident(rng, config, nominal, background=()):
    kind = _choice(rng, config["types"], list(config["types"].values()))
    onset = float(rng.uniform(*config["onset_s"]))
    if kind == "link_fault":
        links = config["link_fault"]["links"]
        return Incident("link_fault", _choice(rng, links, list(links.values())), onset), 0
    if kind == "io_module_fault":
        groups = config["io_module_fault"]["groups"]
        return Incident("io_module_fault", _choice(rng, groups, list(groups.values())), onset), 0
    if kind == "sensor_fault":
        sf = config["sensor_fault"]
        sensor = _choice(rng, sf["sensors"])
        fault = _choice(rng, sf["kinds"], [v["weight"] for v in sf["kinds"].values()])
        spec, scale = sf["kinds"][fault], abs(float(nominal[sensor])) or 1.0
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if fault == "drift":
            magnitude = sign * float(rng.uniform(*spec["magnitude_rel_per_h"])) * scale / 3600.0
        elif fault == "bias":
            magnitude = sign * float(rng.uniform(*spec["magnitude_rel"])) * scale
        else:
            magnitude = 0.0
        return Incident("sensor_fault", sensor, onset, fault, magnitude), 0
    options = config["disturbance"]["idv"]
    idv, redraws = int(_choice(rng, options)), 0
    if config.get("redraw_active_disturbance", False):
        active = {int(i) for i in background}
        if not {int(i) for i in options} - active:
            raise ValueError(f"every incident disturbance {sorted(options)} is already active in {sorted(active)}")
        while idv in active:
            idv, redraws = int(_choice(rng, options)), redraws + 1
    return Incident("disturbance", f"IDV{idv}", onset), redraws


def sample_scenario(seed, nominal, config=None):
    config = config or load_config("sim/incidents.yaml")
    rng = np.random.default_rng([int(seed), 104729])
    bg = config["background"]
    background = tuple(_choice(rng, bg["choices"], bg["weights"]))
    incident, redraws = sample_incident(rng, config, nominal, background)
    p_unavailable = float(config.get("backup_unavailable_prob", 0.0))
    if not 0.0 <= p_unavailable <= 1.0:
        raise ValueError(f"backup_unavailable_prob must lie in [0, 1], got {p_unavailable}")
    backup = bool(np.random.default_rng([int(seed), 104729, 1]).random() >= p_unavailable)
    extra = {"redraw_active_disturbance": bool(config.get("redraw_active_disturbance", False)),
             "backup_unavailable_prob": p_unavailable, "disturbance_redraws": redraws}
    return Scenario(int(seed), incident, background, float(config["warmup_s"]), float(config["horizon_s"]),
                    backup_available=backup, extra=extra)


def paired(base, factor, value):
    if factor not in FACTORS:
        raise ValueError(f"factor must be one of {FACTORS}")
    pair_id = base.pair_id or f"s{base.seed}"
    if factor == "background":
        changed = replace(base, background=tuple(value))
    elif factor == "link":
        if base.incident.kind != "link_fault":
            raise ValueError("the link factor needs a link-fault incident")
        changed = replace(base, incident=replace(base.incident, target=value))
    elif factor == "evidence_validity":
        changed = replace(base, evidence_scale=float(value))
    elif factor == "tool_latency":
        changed = replace(base, tool_latency_scale=float(value))
    else:
        changed = replace(base, backup_available=bool(value))
    return replace(base, pair_id=pair_id, factor=factor), replace(changed, pair_id=pair_id, factor=factor)
