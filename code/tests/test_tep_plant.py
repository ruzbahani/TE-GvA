import pickle

import numpy as np
import pytest

pytest.importorskip("tep_studio")

from cpsdefense.sim.tep_plant import RAND_STATE_OFFSET, TEPPlant

STEPS_PER_MIN = 1 / 0.03


@pytest.fixture(scope="module")
def warmed():
    plant = TEPPlant()
    plant.reset(seed=3)
    plant.step(1000)
    assert plant.shutdown is None
    return plant, plant.snapshot()


def trajectory(plant, n):
    return np.array([plant.step().measured for _ in range(n)])


def test_links_cover_every_signal_once():
    plant = TEPPlant()
    sensors = np.concatenate(list(plant.sensor_links.values()))
    valves = np.concatenate(list(plant.actuator_links.values()))
    assert sorted(sensors) == list(range(41))
    assert sorted(valves) == list(range(12))


def test_same_seed_is_bitwise_reproducible_and_seeds_differ():
    a, b, c = TEPPlant(), TEPPlant(), TEPPlant()
    a.reset(seed=5)
    b.reset(seed=5)
    c.reset(seed=6)
    ta, tb, tc = trajectory(a, 200), trajectory(b, 200), trajectory(c, 200)
    assert np.array_equal(ta, tb)
    assert not np.array_equal(ta, tc)


def test_restore_replays_bitwise_with_links_and_disturbances(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_disturbances([1])
    plant.set_link("sensor:stripper", up=False)
    branch = plant.snapshot()
    first = trajectory(plant, 300)
    plant.restore(pickle.loads(pickle.dumps(branch)))
    second = trajectory(plant, 300)
    assert np.array_equal(first, second)
    plant.restore(branch)
    plant.set_link("sensor:stripper", up=True)
    third = trajectory(plant, 300)
    assert not np.array_equal(first, third)


def test_sensor_link_down_holds_delivered_values(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_disturbances([8])
    plant.set_link("sensor:reactor", up=False)
    idx = plant.sensor_links["reactor"]
    other = plant.sensor_links["stripper"]
    held = plant.delivered[idx].copy()
    for _ in range(50):
        previous = plant.measured.copy()
        res = plant.step()
    assert plant.shutdown is None
    assert np.array_equal(res.delivered[idx], held)
    assert not np.array_equal(res.measured[idx], held)
    assert np.array_equal(res.delivered[other], previous[other])


def test_actuator_link_down_holds_valves(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_disturbances([1])
    plant.set_link("actuator:stripper", up=False)
    idx = plant.actuator_links["stripper"]
    held = plant.applied[idx].copy()
    applied = np.array([plant.step().applied for _ in range(200)])
    assert np.array_equal(applied[:, idx], np.tile(held, (200, 1)))
    others = np.setdiff1d(np.arange(12), idx)
    assert np.ptp(applied[:, others], axis=0).max() > 0


def test_time_to_shutdown_matches_a_direct_run_and_leaves_state_untouched(warmed):
    plant, snap = warmed
    plant.restore(snap)
    before = plant.snapshot()
    seconds, message = plant.time_to_shutdown(8 * 3600, links_down=["sensor:stripper"], disturbances=[1])
    assert seconds is not None and 60 < seconds < 8 * 3600
    assert "Stripper" in message
    reference = trajectory(plant, 100)
    plant.restore(before)
    assert np.array_equal(reference, trajectory(plant, 100))
    plant.restore(before)
    plant.set_disturbances([1])
    plant.set_link("sensor:stripper", up=False)
    start = plant.time_s
    plant.step(int(np.ceil(8 * 3600 / plant.dt_s)))
    assert plant.shutdown is not None
    assert plant.shutdown[0] - start == pytest.approx(seconds)


def test_nominal_plant_does_not_trip_and_tripped_plant_stays_put(warmed):
    plant, snap = warmed
    plant.restore(snap)
    assert plant.time_to_shutdown(3600)[0] is None
    plant.restore(snap)
    plant.set_disturbances([1])
    plant.set_link("sensor:stripper", up=False)
    plant.step(int(8 * 3600 / plant.dt_s))
    tripped = plant.shutdown
    assert tripped is not None
    t = plant.time_s
    plant.step(10)
    assert plant.time_s == t and plant.shutdown == tripped


def test_unknown_link_and_disturbance_are_rejected():
    plant = TEPPlant()
    plant.reset(seed=1)
    with pytest.raises(KeyError):
        plant.set_link("sensor:nope", up=False)
    with pytest.raises(ValueError):
        plant.set_disturbances([29])


def test_sensor_fault_changes_published_not_true_reading(warmed):
    plant, snap = warmed
    plant.restore(snap)
    i = plant.index["reactor_pressure"]
    plant.set_sensor_fault("reactor_pressure", "bias", -50.0)
    res = plant.step()
    assert res.measured[i] == pytest.approx(res.true_measured[i] - 50.0)
    others = np.setdiff1d(np.arange(41), [i])
    assert np.array_equal(res.measured[others], res.true_measured[others])
    plant.set_sensor_fault("reactor_pressure", "frozen")
    held = plant.measured[i]
    readings = [plant.step().measured[i] for _ in range(20)]
    assert all(r == held for r in readings)
    plant.clear_sensor_fault("reactor_pressure")
    res = plant.step()
    assert res.measured[i] == res.true_measured[i]


def test_backup_channel_delivers_delayed_readings(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_disturbances([8])
    plant.set_link_mode("sensor:stripper", "backup")
    idx = plant.sensor_links["stripper"]
    lag = plant.backup_delay_steps
    assert lag >= 1
    published = []
    for _ in range(lag + 30):
        published.append(plant.measured.copy())
        res = plant.step()
    assert np.array_equal(res.delivered[idx], published[-1 - lag][idx])
    with pytest.raises(ValueError):
        plant.set_link_mode("actuator:stripper", "backup")


def test_backup_restores_a_lost_link_but_not_a_faulty_sensor(warmed):
    plant, snap = warmed
    plant.restore(snap)
    down, _ = plant.time_to_shutdown(8 * 3600, hypothesis=lambda p: (p.set_disturbances([1]),
                                                                     p.set_link_mode("sensor:stripper", "down")))
    backup, _ = plant.time_to_shutdown(8 * 3600, hypothesis=lambda p: (p.set_disturbances([1]),
                                                                       p.set_link_mode("sensor:stripper", "backup")))
    assert down is not None and backup is None
    frozen, _ = plant.time_to_shutdown(8 * 3600, hypothesis=lambda p: (
        p.set_disturbances([1]), p.set_sensor_fault("stripper_level", "frozen"),
        p.set_link_mode("sensor:stripper", "backup")))
    assert frozen is not None


def test_reset_finds_the_kernel_generator_state():
    plant = TEPPlant()
    plant.reset(seed=5)
    assert plant.noise_layout_ok and plant.noise_state().tolist() == [5.0, 5.0, 5.0]
    plant.step(3)
    assert plant.noise_state()[0] != 5.0


def test_reseed_noise_changes_only_the_generator_state(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_disturbances([8])
    plant.step(50)
    kernel = plant.sim.snapshot()
    state, t = plant.sim.state.copy(), plant.sim.time
    fields = pickle.dumps({k: getattr(plant, k) for k in plant._STATE if k != "rng"})
    controller = pickle.dumps(plant.controller)
    seed = plant.reseed_noise(11)
    after = plant.sim.snapshot()
    before_bytes = np.frombuffer(kernel.model_bytes, np.uint8)
    after_bytes = np.frombuffer(after.model_bytes, np.uint8)
    changed = np.flatnonzero(before_bytes != after_bytes)
    assert changed.size and changed.min() >= RAND_STATE_OFFSET and changed.max() < RAND_STATE_OFFSET + 24
    assert plant.noise_state().tolist() == [seed] * 3 and seed % 2 == 1 and 0 < seed < 2**32
    assert np.array_equal(plant.sim.state, state) and plant.sim.time == t
    for name in ("state", "xmv", "idv"):
        assert np.array_equal(getattr(after, name), getattr(kernel, name))
    assert pickle.dumps({k: getattr(plant, k) for k in plant._STATE if k != "rng"}) == fields
    assert pickle.dumps(plant.controller) == controller


def test_reseed_noise_is_deterministic_and_seeds_give_different_noise(warmed):
    plant, snap = warmed
    runs = {}
    for key, seed in (("a", 11), ("b", 11), ("c", 12)):
        plant.restore(snap)
        plant.reseed_noise(seed)
        first = plant.step()
        runs[key] = (first.measured, plant.sim.state.copy(), trajectory(plant, 300))
    assert np.array_equal(runs["a"][2], runs["b"][2]) and np.array_equal(runs["a"][0], runs["b"][0])
    assert not np.array_equal(runs["a"][2], runs["c"][2])
    assert np.array_equal(runs["a"][1], runs["c"][1])
    assert not np.array_equal(runs["a"][0], runs["c"][0])
    plant.restore(snap)
    plain = trajectory(plant, 300)
    assert not np.array_equal(plain, runs["a"][2])


def test_reseed_noise_replicates_random_disturbances_from_the_next_knot(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_disturbances([8])
    start = plant.snapshot()
    dist = 8 * (41 + 32)
    runs = []
    for seed in (11, 12):
        plant.restore(start)
        plant.reseed_noise(seed)
        rows = []
        for _ in range(2000):
            plant.step()
            rows.append(np.frombuffer(plant.sim.snapshot().model_bytes, np.float64, 3, dist).copy())
        runs.append(np.array(rows))
    differs = np.flatnonzero(np.any(runs[0] != runs[1], axis=1))
    assert differs.size and differs[0] > 0
    assert np.abs(runs[0] - runs[1]).max() > 0.1


def test_reseeded_noise_survives_snapshot_restore(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_sensor_fault("reactor_level", "noise", 0.5)
    plant.reseed_noise(21)
    branch = plant.snapshot()
    first = trajectory(plant, 200)
    plant.restore(pickle.loads(pickle.dumps(branch)))
    assert np.array_equal(first, trajectory(plant, 200))


def test_reseed_noise_refuses_an_unverified_kernel_layout(warmed):
    plant, snap = warmed
    plant.restore(snap)
    before = plant.snapshot()
    plant.noise_layout_ok = False
    try:
        with pytest.raises(RuntimeError):
            plant.reseed_noise(1)
        after = plant.snapshot()
        assert after.kernel.model_bytes == before.kernel.model_bytes
        assert pickle.dumps(after.state) == pickle.dumps(before.state)
    finally:
        plant.noise_layout_ok = True


def test_faults_backup_and_production_survive_snapshot_restore(warmed):
    plant, snap = warmed
    plant.restore(snap)
    plant.set_sensor_fault("reactor_level", "noise", 0.5)
    plant.set_link_mode("sensor:reactor", "backup")
    plant.set_production(0.9)
    branch = plant.snapshot()
    first = trajectory(plant, 200)
    plant.restore(pickle.loads(pickle.dumps(branch)))
    assert np.array_equal(first, trajectory(plant, 200))
    assert plant.production_fraction == 0.9
    with pytest.raises(ValueError):
        plant.set_production(0.2)
