import copy
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip("tep_studio")

from cpsdefense.sim.incidents import Incident, paired, sample_scenario
from cpsdefense.sim.tep_plant import TEPPlant
from cpsdefense.sim.tep_telemetry import Telemetry, as_alert
from cpsdefense.utils.config import load_config

WARMUP_STEPS = 2000


@pytest.fixture(scope="module")
def warm():
    plant = TEPPlant()
    plant.reset(seed=3)
    tel = Telemetry(plant, np.random.default_rng(3))
    items = []
    for _ in range(WARMUP_STEPS):
        plant.step()
        items += tel.observe_step()
    tel.freeze_baseline()
    return plant, tel, plant.snapshot(), items


def run_after(warm, incident=None, steps=400):
    plant, tel0, snap, _ = warm
    plant.restore(snap)
    tel = copy.deepcopy(tel0)
    assert tel.plant is plant and tel.rng is not tel0.rng
    if incident is not None:
        incident.apply(plant)
    items = []
    for _ in range(steps):
        plant.step()
        items += tel.observe_step()
    return items


def stale_measurements(items):
    return {i.payload["measurement"] for i in items if i.kind == "alert" and i.payload["category"] == "stale"}


def test_items_follow_the_evidence_contract(warm):
    plant, _, _, items = warm
    ids = [i.evidence_id for i in items]
    assert len(ids) == len(set(ids))
    for i in items:
        assert i.arrival_time >= i.event_time >= i.payload["event_start"]
        assert i.valid_until > i.arrival_time
    windows = [i for i in items if i.kind == "window_summary"]
    for w in windows[:20]:
        group = w.subject_id.split(":", 1)[1]
        assert w.payload["values"].shape == (5 * len(plant.sensor_links[group]),)


def test_link_and_io_module_faults_freeze_the_whole_group_alike(warm):
    plant = warm[0]
    group = {plant.measurement_names[i] for i in plant.sensor_links["stripper"]}
    link = stale_measurements(run_after(warm, Incident("link_fault", "sensor:stripper", 0.0)))
    io = stale_measurements(run_after(warm, Incident("io_module_fault", "stripper", 0.0)))
    single = stale_measurements(run_after(warm, Incident("sensor_fault", "stripper_level", 0.0, "frozen")))
    assert group <= link and group <= io
    assert "stripper_level" in single and not (group - {"stripper_level"}) & single


def test_no_staleness_alert_in_normal_operation(warm):
    assert stale_measurements(run_after(warm, None, steps=4000)) == set()


def test_io_module_fault_is_not_fixed_by_the_backup_path(warm):
    plant, _, snap, _ = warm
    plant.restore(snap)

    def io_with_backup(p):
        p.set_disturbances([1])
        Incident("io_module_fault", "stripper", 0.0).apply(p)
        p.set_link_mode("sensor:stripper", "backup")

    def link_with_backup(p):
        p.set_disturbances([1])
        Incident("link_fault", "sensor:stripper", 0.0).apply(p)
        p.set_link_mode("sensor:stripper", "backup")

    assert plant.time_to_shutdown(6 * 3600, hypothesis=io_with_backup)[0] is not None
    assert plant.time_to_shutdown(6 * 3600, hypothesis=link_with_backup)[0] is None


def test_alert_records_and_scenarios():
    item_values = np.array([1.0, 0.0, 7.0], np.float32)
    from cpsdefense.core.types import EvidenceItem

    item = EvidenceItem("alert:stale:x:3", "detector", "alert", "source:reactor", 10.0, 15.0, 600.0,
                        {"values": item_values, "category": "stale", "measurement": "x"}, "detector:stale")
    alert = as_alert(item)
    assert alert.category == "stale" and alert.score == 7.0 and alert.evidence_ids == (item.evidence_id,)
    nominal = dict.fromkeys(TEPPlant().measurement_names, 100.0)
    a, b = sample_scenario(11, nominal), sample_scenario(11, nominal)
    assert a == b
    kinds = {sample_scenario(s, nominal).incident.kind for s in range(200)}
    assert kinds == {"link_fault", "io_module_fault", "sensor_fault", "disturbance"}
    base = next(sample_scenario(s, nominal) for s in range(200) if sample_scenario(s, nominal).incident.kind == "link_fault")
    x, y = paired(base, "tool_latency", 2.0)
    assert x.pair_id == y.pair_id and x.tool_latency_scale == 1.0 and y.tool_latency_scale == 2.0
    assert x.incident == y.incident and x.background == y.background
    with pytest.raises(ValueError):
        paired(base, "nonsense", 1)


def _noop(scenario):
    inc = scenario.incident
    return inc.kind == "disturbance" and int(inc.target[3:]) in scenario.background


@pytest.fixture(scope="module")
def nominal():
    return dict.fromkeys(TEPPlant().measurement_names, 100.0)


def test_an_incident_disturbance_already_active_in_the_background_is_drawn_again(nominal):
    config = copy.deepcopy(load_config("sim/incidents.yaml"))
    config.update(types={"disturbance": 1.0}, background={"choices": [[1]], "weights": [1.0]},
                  disturbance={"idv": [1, 1, 2]})
    off = {**config, "redraw_active_disturbance": False}
    before = [sample_scenario(s, nominal, off) for s in range(60)]
    after = [sample_scenario(s, nominal, config) for s in range(60)]
    assert sum(map(_noop, before)) > 20
    assert not any(map(_noop, after)) and {s.incident.target for s in after} == {"IDV2"}
    for b, a in zip(before, after):
        assert (a.background, a.incident.onset_s) == (b.background, b.incident.onset_s)
        assert (a.incident == b.incident) == (a.extra["disturbance_redraws"] == 0) == (not _noop(b))
    config["disturbance"] = {"idv": [1, 1]}
    with pytest.raises(ValueError, match="already active"):
        sample_scenario(0, nominal, config)


@pytest.mark.parametrize("name,backup_share", [("incidents", 0.0), ("incidents_stress", 0.5)])
def test_both_incidents_configs_redraw_active_disturbances_and_record_the_sampler(nominal, name, backup_share):
    config = load_config(f"sim/{name}.yaml")
    assert config["redraw_active_disturbance"] is True and config["backup_unavailable_prob"] == backup_share
    off = {**config, "redraw_active_disturbance": False, "backup_unavailable_prob": 0.0}
    seeds = range(3000)
    scenarios = [sample_scenario(s, nominal, config) for s in seeds]
    plain = [sample_scenario(s, nominal, off) for s in seeds]
    assert not any(map(_noop, scenarios)) and sum(map(_noop, plain)) > 0
    for sc, p in zip(scenarios, plain):
        assert (sc.incident == p.incident and sc.background == p.background) == (not _noop(p))
        assert asdict(sc)["extra"] == {"redraw_active_disturbance": True, "backup_unavailable_prob": backup_share,
                                       "disturbance_redraws": sc.extra["disturbance_redraws"]}
        assert (sc.extra["disturbance_redraws"] > 0) == _noop(p) and p.backup_available
    unavailable = np.mean([not sc.backup_available for sc in scenarios])
    assert unavailable == backup_share if backup_share == 0.0 else abs(unavailable - backup_share) < 0.03
    assert asdict(scenarios[0])["backup_available"] == scenarios[0].backup_available


def test_backup_availability_has_its_own_stream(nominal):
    config = copy.deepcopy(load_config("sim/incidents_stress.yaml"))
    draws = {p: [sample_scenario(s, nominal, {**config, "backup_unavailable_prob": p}) for s in range(400)]
             for p in (0.0, 0.3, 1.0)}
    for a, b, c in zip(*draws.values()):
        assert a.incident == b.incident == c.incident and a.background == b.background == c.background
    assert all(s.backup_available for s in draws[0.0]) and not any(s.backup_available for s in draws[1.0])
    assert 0.25 < np.mean([not s.backup_available for s in draws[0.3]]) < 0.35
    by_kind = {k: np.mean([not s.backup_available for s in draws[0.3] if s.incident.kind == k])
               for k in ("link_fault", "io_module_fault")}
    assert all(0.15 < v < 0.45 for v in by_kind.values())
    with pytest.raises(ValueError, match="backup_unavailable_prob"):
        sample_scenario(0, nominal, {**config, "backup_unavailable_prob": 1.5})
