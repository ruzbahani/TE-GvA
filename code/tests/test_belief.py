import math

import numpy as np
import pytest

from cpsdefense.core.types import EvidenceItem
from cpsdefense.experience import belief
from cpsdefense.experience.belief import LINK, MODULE
from cpsdefense.sim.incidents import Incident, Scenario, sample_incident
from cpsdefense.utils.config import load_config

PARAMS = {"p_detect_down": 0.05, "p_false": 0.0003, "report_backup": True, "delay_s": 15.0, "delay_sigma": 0.3,
          "evidence_scale": 1.0, "window_s": 36.0, "trace_accuracy": 0.97, "operator_accuracy": 0.85,
          "causes": ["none", "link_fault", "io_module_fault", "sensor_fault", "disturbance"]}


def _evidence(prior=(0.5, 0.5), windows=(), traces=(), operator=(), identical=True):
    return {"group": "stripper", "causes": [LINK, MODULE], "prior": list(prior), "tau": 5000.0, "params": PARAMS,
            "deterministic_identical": identical, "first_difference": None, "windows": list(windows),
            "traces": list(traces), "operator": list(operator)}


def _normalised(a, b):
    return np.array([a, b]) / (a + b)


def test_the_family_is_link_faults_on_sensor_links_and_module_faults():
    assert belief.family_group(Incident("link_fault", "sensor:stripper", 600.0)) == "stripper"
    assert belief.family_group(Incident("io_module_fault", "reactor", 600.0)) == "reactor"
    assert belief.family_group({"kind": "link_fault", "target": "sensor:feeds", "onset_s": 1.0}) == "feeds"
    for other in (Incident("link_fault", "actuator:reactor", 600.0), Incident("sensor_fault", "reactor_level", 600.0),
                  Incident("disturbance", "IDV6", 600.0)):
        assert belief.family_group(other) is None
        with pytest.raises(ValueError):
            belief.twin_incident(other, 1.8)
    assert belief.family_group(None) is None


def test_the_twin_scenario_swaps_the_cause_and_shifts_the_module_onset_one_step_earlier():
    base = Scenario(seed=7, incident=Incident("link_fault", "sensor:stripper", 1000.0), background=(1,),
                    backup_available=False, evidence_scale=1.5, tool_latency_scale=0.5, extra={"x": 1})
    twin = belief.twin_scenario(base, 1.8)
    assert (twin.incident.kind, twin.incident.target) == (MODULE, "stripper")
    assert twin.incident.onset_s == pytest.approx(1000.0 - 1.8)
    assert (twin.seed, twin.background, twin.backup_available, twin.evidence_scale, twin.tool_latency_scale) == \
        (7, (1,), False, 1.5, 0.5)
    assert twin.end_s == pytest.approx(base.end_s, abs=1e-9)
    assert twin.extra == {"x": 1, "twin_of": "link_fault:sensor:stripper",
                          "twin_onset_shift_s": pytest.approx(-1.8)}
    back = belief.twin_scenario(twin, 1.8)
    assert (back.incident.kind, back.incident.target) == (LINK, "sensor:stripper")
    assert back.incident.onset_s == pytest.approx(1000.0) and back.end_s == pytest.approx(base.end_s, abs=1e-9)


def test_the_prior_is_the_samplers_probability_of_the_two_incidents_normalised_within_the_pair():
    natural, stress = load_config("sim/incidents.yaml"), load_config("sim/incidents_stress.yaml")
    link, module = 0.35 * 2 / 13, 0.20 * 2 / 8
    assert belief.pair_prior(natural, "stripper") == pytest.approx({LINK: link / (link + module),
                                                                    MODULE: module / (link + module)})
    assert belief.pair_prior(natural, "stripper")[LINK] == pytest.approx(0.5185, abs=1e-4)
    link, module = 0.35 * 0.5 / 15, 0.35 * 0.5 / 9.5
    assert belief.pair_prior(stress, "analyzers")[LINK] == pytest.approx(link / (link + module))
    rng = np.random.default_rng(0)
    nominal = {s: 1.0 for s in natural["sensor_fault"]["sensors"]}
    draws = [sample_incident(rng, natural, nominal)[0] for _ in range(20000)]
    pair = [d.kind == LINK for d in draws if belief.family_group(d) == "stripper"]
    assert np.mean(pair) == pytest.approx(belief.pair_prior(natural, "stripper")[LINK], abs=0.04)
    with pytest.raises(ValueError):
        belief.pair_prior({**natural, "types": {"disturbance": 1.0}}, "stripper")


def test_the_likelihood_parameters_come_from_the_configs():
    telemetry, env = load_config("sim/telemetry.yaml"), load_config("sim/tep_env.yaml")
    p = belief.load_params(evidence_scale=2.0)
    lh = telemetry["link_health"]
    assert (p["p_detect_down"], p["p_false"], p["report_backup"], p["delay_s"], p["delay_sigma"]) == \
        (lh["p_detect_down"], lh["p_false"], lh["report_backup"], lh["delay_s"], lh["sigma"])
    assert (p["trace_accuracy"], p["operator_accuracy"], p["causes"]) == \
        (env["tools"]["trace_accuracy"], env["tools"]["operator_accuracy"], env["causes"])
    assert p["window_s"] == pytest.approx(telemetry["window_steps"] * load_config("sim/tep.yaml")["control_interval_s"])
    assert p["evidence_scale"] == 2.0


def test_the_report_delay_cdf_is_the_telemetry_lognormal():
    assert belief.delay_cdf(15.0, PARAMS) == pytest.approx(0.5)
    assert belief.delay_cdf(0.0, PARAMS) == 0.0 and belief.delay_cdf(-3.0, PARAMS) == 0.0
    z = math.log(20.0 / 15.0) / 0.3
    assert belief.delay_cdf(20.0, PARAMS) == pytest.approx(0.5 * math.erfc(-z / math.sqrt(2.0)))
    assert belief.delay_cdf(30.0, {**PARAMS, "evidence_scale": 2.0}) == pytest.approx(0.5)
    fixed = {**PARAMS, "delay_sigma": 0.0}
    assert belief.delay_cdf(14.9, fixed) == 0.0 and belief.delay_cdf(15.0, fixed) == 1.0


def test_state_at_uses_the_last_decision_point_before_the_step_and_the_onset():
    states = [(900.0, "up", False), (1500.0, "backup", True), (2000.0, "down", True)]
    link = belief.world(LINK, 1000.0, 1.8, states)
    module = belief.world(MODULE, 998.2, 1.8, [(t, m if t < 1500 else "backup", False) for t, m, _ in states[:2]])
    assert belief.state_at(link, 1001.0) == ("up", False)
    assert belief.state_at(link, 1002.0) == ("down", True)
    assert belief.state_at(link, 1500.0) == ("down", True)
    assert belief.state_at(link, 1501.8) == ("backup", True)
    assert belief.state_at(link, 2500.0) == ("down", True)
    assert belief.state_at(module, 1200.0) == ("up", False) and belief.state_at(module, 1501.8) == ("backup", False)
    assert belief.state_at(belief.world(LINK, 1000.0, 1.8, []), 500.0) == ("up", False)
    assert belief.true_kind(link, 999.9) == "none" and belief.true_kind(link, 1000.0) == LINK
    on_backup = belief.world(LINK, 1000.0, 1.8, [(900.0, "up", False), (950.0, "backup", False)])
    assert belief.state_at(on_backup, 1001.0) == ("backup", False)
    assert belief.state_at(on_backup, 1002.0) == ("backup", True)
    assert belief.state_at(on_backup, 3000.0) == ("backup", True)


def test_without_evidence_the_posterior_is_the_prior():
    np.testing.assert_allclose(belief.posterior(_evidence(prior=(0.52, 0.48))), [0.52, 0.48])


def test_link_health_windows_have_the_analytic_likelihood():
    report = {"w": 1, "end": 4000.0, "modes": ["down", "up"], "report": [1.0, 0.0], "age": 1000.0}
    np.testing.assert_allclose(belief.posterior(_evidence(windows=[report])), _normalised(0.05, 0.0003))
    quiet = [{"w": w, "end": 4000.0, "modes": ["down", "up"], "report": None, "age": 1000.0} for w in range(10)]
    np.testing.assert_allclose(belief.posterior(_evidence(prior=(0.3, 0.7), windows=quiet)),
                               _normalised(0.3 * 0.95 ** 10, 0.7 * (1 - 0.0003) ** 10))
    pending = {"w": 2, "end": 4985.0, "modes": ["down", "up"], "report": None, "age": 15.0}
    np.testing.assert_allclose(belief.posterior(_evidence(windows=[pending])),
                               _normalised(1 - 0.05 * 0.5, 1 - 0.0003 * 0.5))
    np.testing.assert_allclose(belief.posterior(_evidence(windows=[{**report, "modes": ["up", "down"]}])),
                               _normalised(0.0003, 0.05))
    backup = {"w": 3, "end": 4000.0, "modes": ["backup", "down"], "report": [0.0, 1.0], "age": 1000.0}
    np.testing.assert_allclose(belief.posterior(_evidence(windows=[backup])), [1.0, 0.0])
    np.testing.assert_allclose(belief.posterior(_evidence(windows=[{**backup, "report": [1.0, 0.0]}])), [0.0, 1.0])
    assert belief.window_probability("backup", None, 1000.0, {**PARAMS, "report_backup": False}) == 1.0


def test_traces_and_operator_reports_have_their_accuracy_likelihoods():
    failed = {"time": 4500.0, "reported_healthy": False, "healthy": [False, True]}
    np.testing.assert_allclose(belief.posterior(_evidence(traces=[failed])), _normalised(0.97, 0.03))
    wrong = {**failed, "reported_healthy": True}
    np.testing.assert_allclose(belief.posterior(_evidence(traces=[wrong])), _normalised(0.03, 0.97))
    named = {"time": 4600.0, "named": MODULE, "true": [LINK, MODULE]}
    np.testing.assert_allclose(belief.posterior(_evidence(operator=[named])), _normalised(0.15 / 4, 0.85))
    before = {"time": 900.0, "named": "none", "true": ["none", "none"]}
    np.testing.assert_allclose(belief.posterior(_evidence(prior=(0.6, 0.4), operator=[before])), [0.6, 0.4])
    report = {"w": 1, "end": 4000.0, "modes": ["down", "up"], "report": [1.0, 0.0], "age": 1000.0}
    np.testing.assert_allclose(belief.posterior(_evidence(prior=(0.4, 0.6), windows=[report], traces=[failed],
                                                          operator=[named])),
                               _normalised(0.4 * 0.05 * 0.97 * 0.15 / 4, 0.6 * 0.0003 * 0.03 * 0.85))


def test_different_deterministic_evidence_excludes_the_twin_and_impossible_evidence_is_undefined():
    failed = {"time": 4500.0, "reported_healthy": True, "healthy": [False, True]}
    np.testing.assert_allclose(belief.posterior(_evidence(traces=[failed], identical=False)), [1.0, 0.0])
    impossible = {"w": 3, "end": 4000.0, "modes": ["backup", "backup"], "report": [1.0, 0.0], "age": 1000.0}
    assert np.isnan(belief.posterior(_evidence(windows=[impossible]))).all()


def _item(eid, kind, subject, t, values, arrival=None, lifetime=600.0, **payload):
    arrival = t if arrival is None else arrival
    return EvidenceItem(eid, "test", kind, subject, float(t), float(arrival), float(arrival + lifetime),
                        {"values": np.asarray(values, np.float32), "event_start": float(t), **payload}, "test")


def _worlds():
    link = belief.world(LINK, 4200.0, 1.8, [(3600.0, "up", False), (4201.8, "down", True), (4700.0, "backup", True)])
    module = belief.world(MODULE, 4198.2, 1.8, [(3600.0, "up", False), (4201.8, "up", False),
                                               (4700.0, "backup", False)])
    return [link, module]


def _primary_items():
    items = [_item(f"win:reactor:{w}", "window_summary", "source:reactor", 36.0 * w, [1.0, 2.0], 36.0 * w + 10.0)
             for w in range(110, 133)]
    items += [_item("link:sensor:stripper:118", "link_health", "sensor:stripper", 4248.0, [1, 0], 4263.0),
              _item("link:sensor:stripper:125", "link_health", "sensor:stripper", 4500.0, [1, 0], 4515.0),
              _item("alert:stale:stripper_level:119", "alert", "source:stripper", 4284.0, [1, 0, 5], 4289.0,
                    category="stale", measurement="stripper_level"),
              _item("tool:trace_link:1", "trace_link", "sensor:stripper", 4450.0, [0.0, 0.0], action_id="t@4330"),
              _item("tool:operator_report:2", "operator_report", "incident:1", 4600.0,
                    [0, 0, 1, 0, 0], cause=MODULE, action_id="e@3700")]
    return items


def test_belief_evidence_lists_the_windows_where_the_worlds_modes_differ_and_the_primarys_observations():
    ev = belief.belief_evidence(_primary_items(), 4650.0, "stripper", _worlds(), PARAMS, [0.5, 0.5])
    ends = [w["end"] for w in ev["windows"]]
    assert ends == pytest.approx([36.0 * w for w in range(117, 130)])
    assert all(w["modes"] == ["down", "up"] for w in ev["windows"])
    reported = {w["w"]: w["report"] for w in ev["windows"] if w["report"] is not None}
    assert reported == {118: [1.0, 0.0], 125: [1.0, 0.0]}
    assert ev["windows"][-1]["age"] == pytest.approx(4650.0 - 36.0 * 129)
    assert ev["traces"] == [{"time": 4450.0, "reported_healthy": False, "healthy": [False, True]}]
    assert ev["operator"] == [{"time": 4600.0, "named": MODULE, "true": [LINK, MODULE]}]
    assert ev["deterministic_identical"] and ev["first_difference"] is None
    expected = [0.5, 0.5]
    for w in ev["windows"]:
        for c in (0, 1):
            p = belief.report_probability(w["modes"][c], PARAMS)
            expected[c] *= p if w["report"] else 1 - p * belief.delay_cdf(w["age"], PARAMS)
    expected[0] *= 0.97 * 0.15 / 4
    expected[1] *= 0.03 * 0.85
    np.testing.assert_allclose(belief.posterior(ev), _normalised(*expected))
    early = belief.belief_evidence(_primary_items(), 4510.0, "stripper", _worlds(), PARAMS, [0.5, 0.5])
    assert {w["w"] for w in early["windows"] if w["report"] is not None} == {118}
    assert early["traces"] and not early["operator"]
    with pytest.raises(ValueError):
        belief.belief_evidence([], 4650.0, "stripper", _worlds(), PARAMS, [0.5, 0.5])


def _twin_items(primary):
    out = []
    for i in primary:
        if i.kind == "trace_link":
            out.append(_item(i.evidence_id, i.kind, i.subject_id, i.event_time, [1.0, 0.0], action_id="t@4330"))
        elif i.kind == "operator_report":
            out.append(_item(i.evidence_id, i.kind, i.subject_id, i.event_time, [0, 1, 0, 0, 0], cause=LINK,
                             action_id="e@3700"))
        elif i.evidence_id != "link:sensor:stripper:125":
            out.append(i)
    return out


def test_the_twins_realised_stochastic_draws_are_not_used():
    primary = _primary_items()
    twin = _twin_items(primary)
    assert belief.first_difference(primary, twin) is None
    ev = belief.belief_evidence(primary, 4650.0, "stripper", _worlds(), PARAMS, [0.5, 0.5],
                                belief.first_difference(primary, twin))
    one = belief.belief_evidence([i for i in primary if i.kind in ("window_summary", "link_health")
                                  and i.evidence_id != "link:sensor:stripper:125"], 4400.0, "stripper", _worlds(),
                                 PARAMS, [0.5, 0.5])
    got = belief.posterior(one)
    assert got[0] > 0.9
    expected = [0.5, 0.5]
    for w in one["windows"]:
        for c in (0, 1):
            p = belief.report_probability(w["modes"][c], PARAMS)
            expected[c] *= p if w["report"] else 1 - p * belief.delay_cdf(w["age"], PARAMS)
    np.testing.assert_allclose(got, _normalised(*expected))
    variants = [_twin_items(primary) + [_item("link:sensor:stripper:125", "link_health", "sensor:stripper", 4500.0,
                                              [1, 0], 4515.0)],
                [i for i in twin if i.kind != "link_health"],
                [i if i.kind != "trace_link" else _item(i.evidence_id, "trace_link", i.subject_id, i.event_time,
                                                          [0.0, 0.0], action_id="t@4330") for i in twin]]
    for items in variants:
        first = belief.first_difference(primary, items)
        assert first is None
        assert belief.belief_evidence(primary, 4650.0, "stripper", _worlds(), PARAMS, [0.5, 0.5], first) == ev
    changed = [i if i.evidence_id != "win:reactor:125" else _item(i.evidence_id, i.kind, i.subject_id, i.event_time,
                                                                   [1.0, 2.5], i.arrival_time) for i in twin]
    first = belief.first_difference(primary, changed)
    assert first == (36.0 * 125 + 10.0, "win:reactor:125")
    excluded = belief.belief_evidence(primary, 4650.0, "stripper", _worlds(), PARAMS, [0.5, 0.5], first)
    np.testing.assert_allclose(belief.posterior(excluded), [1.0, 0.0])
    assert excluded["first_difference"] == [36.0 * 125 + 10.0, "win:reactor:125"]


def test_first_difference_compares_deterministic_content_only():
    a = [_item("win:reactor:1", "window_summary", "source:reactor", 36.0, [1.0], 40.0),
         _item("tool:trace_link:1", "trace_link", "sensor:stripper", 50.0, [0.0, 1.0]),
         _item("receipt:1", "receipt", "sensor:stripper", 60.0, [0, 1], tool_id="trace_link", status="running")]
    assert belief.first_difference(a, list(a)) is None
    missing = belief.first_difference(a, a[1:])
    assert missing == (40.0, "win:reactor:1")
    backup_bit = [a[0], _item("tool:trace_link:1", "trace_link", "sensor:stripper", 50.0, [0.0, 0.0]), a[2]]
    assert belief.first_difference(a, backup_bit) == (50.0, "tool:trace_link:1")
    health_bit = [a[0], _item("tool:trace_link:1", "trace_link", "sensor:stripper", 50.0, [1.0, 1.0]), a[2]]
    assert belief.first_difference(a, health_bit) is None
    status = [*a[:2], _item("receipt:1", "receipt", "sensor:stripper", 60.0, [0, 1], tool_id="trace_link",
                            status="rejected")]
    assert belief.first_difference(a, status) == (60.0, "receipt:1")
    assert belief.first_difference(a, status, until=59.0) is None
    later = [*a, _item("win:reactor:2", "window_summary", "source:reactor", 72.0, [1.0], 80.0)]
    assert belief.first_difference(a, later) == (80.0, "win:reactor:2")
    assert belief.first_difference(a, [*a, _item("link:sensor:stripper:1", "link_health", "sensor:stripper", 36.0,
                                                 [1, 0], 51.0)]) is None


def test_a_link_fault_and_its_module_twin_give_bit_identical_telemetry_only_with_the_one_step_shift():
    pytest.importorskip("tep_studio")
    from cpsdefense.sim.tep_env import TEPEnv

    base = Scenario(seed=12, incident=Incident("link_fault", "sensor:stripper", 600.7), background=(1,),
                    horizon_s=1500.0)
    twin = belief.twin_scenario(base, 1.8)
    unshifted = Scenario(seed=12, incident=Incident("io_module_fault", "stripper", 600.7), background=(1,),
                         horizon_s=1500.0)

    def run(scenario):
        env = TEPEnv(scenario)
        env.reset()
        while not env.done:
            env.advance(300.0)
        return [env.log.get(e) for e in env.log], env

    link_items, link_env = run(base)
    twin_items, twin_env = run(twin)
    assert belief.first_difference(link_items, twin_items) is None
    assert np.array_equal(link_env.plant.delivered, twin_env.plant.delivered)
    assert link_env.hidden_state()["link_mode"]["sensor:stripper"] == "down"
    assert twin_env.hidden_state()["io_faulted"] == ["stripper"]
    unshifted_items, _ = run(unshifted)
    first = belief.first_difference(link_items, unshifted_items)
    assert first is not None and first[0] > 3600.0 + 600.7
