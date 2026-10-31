import copy
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from tep_studio.control import RickerMultiLoopController
from tep_studio.simulation.core import TennesseeEastmanProcess
from tep_studio.simulation.schema import TEP_SCHEMA

from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_file

N_IDV = 28
SENSOR_MODES = ("up", "down", "backup")
FAULT_KINDS = ("frozen", "bias", "drift", "noise")
RAND_STATE_OFFSET = 8 * (41 + 32 + 21 + 62 + 96 + 12 + 29)


@dataclass(frozen=True)
class PlantSnapshot:
    kernel: object
    controller: object
    state: dict


@dataclass(frozen=True)
class StepResult:
    time_s: float
    measured: np.ndarray
    true_measured: np.ndarray
    delivered: np.ndarray
    applied: np.ndarray
    margins: dict
    shutdown: tuple | None


def check_kernel(config):
    import tep_studio

    expected = config.get("kernel", {}).get("temexd_mod_c_sha256")
    source = Path(tep_studio.__file__).resolve().parents[2] / "temexd_mod" / "temexd_mod.c"
    if expected is None or not source.exists():
        return None
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(f"TE kernel {source} has sha256 {actual}, configs/sim/tep.yaml records {expected}")
    return actual


class TEPPlant:
    def __init__(self, config=None):
        self.config = config or load_config("sim/tep.yaml")
        check_kernel(self.config)
        self.dt_s = float(self.config["control_interval_s"])
        self.sim = TennesseeEastmanProcess(fixed_step=self.dt_s / 3600.0)
        self.measurement_names = list(TEP_SCHEMA.names("measurements"))
        self.valve_names = list(TEP_SCHEMA.names("manipulated_variables"))
        self.index = {name: i for i, name in enumerate(self.measurement_names)}
        self.sensor_links = self._index_links("sensor_links", self.measurement_names)
        self.actuator_links = self._index_links("actuator_links", self.valve_names)
        backup = self.config.get("backup_channel", {})
        self.backup_delay_steps = int(round(float(backup.get("delay_s", 0.0)) / self.dt_s))
        self.controller = None
        self.shutdown = None

    def _index_links(self, key, names):
        position = {name: i for i, name in enumerate(names)}
        links = {}
        for link, members in self.config[key].items():
            unknown = [m for m in members if m not in position]
            if unknown:
                raise ValueError(f"{key}.{link}: unknown names {unknown}")
            links[link] = np.array([position[m] for m in members], dtype=np.int64)
        return links

    @property
    def links(self):
        return [f"sensor:{k}" for k in self.sensor_links] + [f"actuator:{k}" for k in self.actuator_links]

    @property
    def links_down(self):
        return frozenset(k for k, mode in self.link_mode.items() if mode == "down")

    @property
    def time_s(self):
        return self.sim.time * 3600.0

    def reset(self, seed=0, disturbances=()):
        kernel_measured, info = self.sim.reset(mode=self.config["mode"], seed=float(seed))
        model = self.sim.snapshot().model_bytes
        self.noise_layout_ok = bool(np.array_equal(np.frombuffer(model, np.float64, 41), kernel_measured)
                                    and np.all(np.frombuffer(model, np.float64, 3, RAND_STATE_OFFSET) == float(seed)))
        self.controller = RickerMultiLoopController()
        self.controller.reset(kernel_measured, time=self.sim.time)
        self.nominal_production = float(self.controller.setpoints.production_rate)
        self.production_fraction = 1.0
        self.rng = np.random.default_rng([int(seed), 7919])
        self.faults = {}
        self.link_mode = {link: "up" for link in self.links}
        self.true_measured = kernel_measured.copy()
        self.measured = kernel_measured.copy()
        self.delivered = kernel_measured.copy()
        self.history = deque([kernel_measured.copy()] * (self.backup_delay_steps + 1),
                             maxlen=self.backup_delay_steps + 1)
        self.applied = np.array(info["implemented_action"], dtype=np.float64)
        self.idv = self._idv_vector(disturbances)
        self.shutdown = None
        self.margins = info["constraint_margins"]
        return self._result()

    def _idv_vector(self, disturbances):
        vec = np.zeros(N_IDV, dtype=np.float64)
        for number in disturbances:
            if not 1 <= number <= N_IDV:
                raise ValueError(f"IDV number {number} outside 1..{N_IDV}")
            vec[number - 1] = 1.0
        return vec

    def set_disturbances(self, disturbances):
        self.idv = self._idv_vector(disturbances)

    def active_disturbances(self):
        return [i + 1 for i in np.flatnonzero(self.idv)]

    def set_link(self, link, up):
        self.set_link_mode(link, "up" if up else "down")

    def set_link_mode(self, link, mode):
        if link not in self.links:
            raise KeyError(f"unknown link {link!r}; known: {self.links}")
        allowed = SENSOR_MODES if link.startswith("sensor:") else ("up", "down")
        if mode not in allowed:
            raise ValueError(f"{link}: mode {mode!r} not in {allowed}")
        self.link_mode[link] = mode

    def set_sensor_fault(self, name, kind, magnitude=0.0):
        if name not in self.index:
            raise KeyError(f"unknown measurement {name!r}")
        if kind not in FAULT_KINDS:
            raise ValueError(f"fault kind {kind!r} not in {FAULT_KINDS}")
        i = self.index[name]
        self.faults[name] = {"kind": kind, "magnitude": float(magnitude), "onset_s": self.time_s,
                             "frozen": float(self.measured[i])}

    def clear_sensor_fault(self, name):
        self.faults.pop(name, None)

    def set_production(self, fraction):
        if not 0.5 <= fraction <= 1.05:
            raise ValueError("production fraction must lie in [0.5, 1.05]")
        self.production_fraction = float(fraction)
        self.controller.setpoints = replace(self.controller.setpoints,
                                            production_rate=self.nominal_production * fraction)

    def _publish(self, kernel_measured):
        published = kernel_measured.copy()
        for name, f in self.faults.items():
            i = self.index[name]
            if f["kind"] == "frozen":
                published[i] = f["frozen"]
            elif f["kind"] == "bias":
                published[i] += f["magnitude"]
            elif f["kind"] == "drift":
                published[i] += f["magnitude"] * (self.time_s - f["onset_s"])
            else:
                published[i] += self.rng.normal(0.0, f["magnitude"])
        return published

    def step(self, n=1):
        for _ in range(n):
            if self.shutdown is not None:
                break
            for link, idx in self.sensor_links.items():
                mode = self.link_mode[f"sensor:{link}"]
                if mode == "up":
                    self.delivered[idx] = self.measured[idx]
                elif mode == "backup":
                    self.delivered[idx] = self.history[0][idx]
            action, _ = self.controller.compute_action(self.delivered.copy(), time=self.sim.time)
            for link, idx in self.actuator_links.items():
                if self.link_mode[f"actuator:{link}"] == "up":
                    self.applied[idx] = action[idx]
            res = self.sim.advance(self.applied, control_interval=self.dt_s / 3600.0, disturbances=self.idv)
            self.true_measured = res.measurements.copy()
            self.measured = self._publish(self.true_measured)
            self.history.append(self.measured.copy())
            self.applied = res.implemented_action.copy()
            self.margins = res.constraint_margins
            if res.shutdown_status["terminated"]:
                self.shutdown = (self.time_s, res.shutdown_status["message"].strip())
        return self._result()

    def _result(self):
        return StepResult(self.time_s, self.measured.copy(), self.true_measured.copy(), self.delivered.copy(),
                          self.applied.copy(), dict(self.margins), self.shutdown)

    def production_rate(self):
        return float(self.true_measured[self.index["stripper_underflow"]]), self.nominal_production

    _STATE = ("measured", "true_measured", "delivered", "applied", "idv", "shutdown", "link_mode", "faults",
              "history", "rng", "production_fraction", "nominal_production")

    def snapshot(self):
        state = {k: copy.deepcopy(getattr(self, k)) for k in self._STATE}
        return PlantSnapshot(self.sim.snapshot(), copy.deepcopy(self.controller), state)

    def restore(self, snap):
        self.sim.restore(snap.kernel)
        self.controller = copy.deepcopy(snap.controller)
        for k, v in snap.state.items():
            setattr(self, k, copy.deepcopy(v))
        self.margins = TEP_SCHEMA.constraint_margins(self.true_measured)

    def noise_state(self):
        return np.frombuffer(self.sim.snapshot().model_bytes, np.float64, 3, RAND_STATE_OFFSET).copy()

    def reseed_noise(self, seed):
        if not getattr(self, "noise_layout_ok", False):
            raise RuntimeError(f"the TE kernel's generator state was not found at byte {RAND_STATE_OFFSET} of its "
                               "model struct at reset; refusing to reseed")
        seq = np.random.SeedSequence(seed)
        value = float(int(seq.generate_state(1, np.uint32)[0]) | 1)
        snap = self.sim.snapshot()
        model = bytearray(snap.model_bytes)
        model[RAND_STATE_OFFSET:RAND_STATE_OFFSET + 24] = np.full(3, value, np.float64).tobytes()
        self.sim.restore(replace(snap, model_bytes=bytes(model)))
        self.rng = np.random.default_rng(seq.spawn(1)[0])
        return value

    def time_to_shutdown(self, horizon_s, links_down=None, disturbances=None, hypothesis=None):
        snap = self.snapshot()
        try:
            if links_down is not None:
                for link in links_down:
                    if link not in self.links:
                        raise KeyError(f"unknown link {link!r}")
                for link in self.links:
                    self.link_mode[link] = "down" if link in links_down else "up"
            if disturbances is not None:
                self.idv = self._idv_vector(disturbances)
            if hypothesis is not None:
                hypothesis(self)
            start = self.time_s
            self.step(int(np.ceil(horizon_s / self.dt_s)))
            if self.shutdown is None:
                return None, None
            return self.shutdown[0] - start, self.shutdown[1]
        finally:
            self.restore(snap)
