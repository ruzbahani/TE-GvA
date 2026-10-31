import copy
from dataclasses import replace

import pytest
import yaml

from cpsdefense.core.types import REJECT_REASONS, EvidenceItem, ExecutionBundle
from cpsdefense.defense.evidence import EvidenceLog
from cpsdefense.defense.policy import EXECUTOR_PROVENANCE, POLICY_CONFIG, Attempt, Policy, load_policy
from cpsdefense.defense.registry import TOOLS_CONFIG, Registry, load_registry
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_json


class State:
    def __init__(self, false=()):
        self.false = set(false)
        self.calls = []

    def predicate(self, name, action):
        self.calls.append(name)
        if name in self.false:
            return False
        return action.target_id.startswith("sensor:") if name == "sensor_link" else True


def item(eid, provenance="netmon", arrival=-10.0):
    return EvidenceItem(eid, "network", "link_health", "sensor:reactor", arrival - 1.0, arrival, 1e9, {}, provenance)


@pytest.fixture(scope="module")
def log():
    return EvidenceLog([item("e1"), item("e2"), item("receipt:1", EXECUTOR_PROVENANCE)])


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def config():
    return load_config(POLICY_CONFIG)


@pytest.fixture(scope="module")
def policy(registry, config):
    return Policy(config, registry)


def act(registry, tool, target, args=None, now=0.0):
    return registry.make_action(tool, target, args or {}, now, f"{tool}-{target}")


def bundle(action, policy, evidence=("e1",), key="k", version=None):
    return ExecutionBundle(decision_id="d", action=action, parameter_values=dict(action.bounded_arguments),
                           evidence_ids=tuple(evidence), observation_version=0,
                           policy_version=policy.version if version is None else version, not_after=1e9,
                           expected_postconditions=action.postconditions, monitor_rule="",
                           recovery_or_compensation_id=action.compensation_or_recovery_id, idempotency_key=key)


def test_version_is_the_hash_of_canonical_content(tmp_path, registry, config, policy):
    assert policy.version == sha256_json(config) == load_policy(registry).version
    reordered = {k: config[k] for k in reversed(list(config))}
    path = tmp_path / "policy.yaml"
    path.write_text("# a different comment\n" + yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")
    assert load_policy(registry, path).version == policy.version
    changed = copy.deepcopy(config)
    changed["rate_limits"][0]["max"] = 4
    assert Policy(changed, registry).version != policy.version
    assert policy.config == config and policy.config is not config


def test_default_policy_contents(policy):
    assert policy.scopes == {"observe", "contain", "escalate"}
    assert {r["id"] for r in policy.forbid} == {"backup_only_for_sensor_links", "respond_needs_evidence"}
    assert policy.rate_limits[0]["max"] == 3 and policy.rate_limits[0]["window_s"] == 1800


def test_authority_and_scope_rules(registry, policy, config, log):
    state = State()
    for fraction, ok in ((0.8, False), (0.9, True), (1.0, True)):
        a = act(registry, "set_production", "plant", {"fraction": fraction})
        assert policy.authorized(a) is ok
        assert policy.check(bundle(a, policy), state, 0.0, (), log) == \
            ((True, None) if ok else (False, "unauthorized"))
    deep = act(registry, "set_production", "plant", {"fraction": 0.8})
    assert policy.required_scopes(deep) == {"contain", "process"}
    process = Policy({**config, "authority": ["observe", "contain", "process"]}, registry)
    assert process.check(bundle(deep, process), state, 0.0, (), log) == (True, None)
    escalate = act(registry, "escalate", "inc-1")
    assert not process.authorized(escalate) and policy.authorized(escalate)
    observe = Policy({"authority": ["observe"]}, registry)
    assert observe.authorized(act(registry, "trace_link", "sensor:reactor"))
    assert not observe.authorized(act(registry, "repair_link", "sensor:reactor"))


def test_missing_or_incomparable_arguments_match_scope_rules(registry, policy):
    deep = act(registry, "set_production", "plant", {"fraction": 0.9})
    assert not policy.authorized(replace(deep, bounded_arguments={}))
    assert not policy.authorized(replace(deep, bounded_arguments={"fraction": "0.9"}))
    assert policy.authorized(deep)


def test_check_order_and_reasons(registry, policy, log):
    state = State()
    ok = act(registry, "switch_to_backup", "sensor:reactor")
    assert policy.check(bundle(ok, policy), state, 0.0, (), log) == (True, None)
    deep = act(registry, "set_production", "plant", {"fraction": 0.8})
    assert policy.check(bundle(deep, policy, evidence=(), version="stale"), state, 0.0, (), log)[1] == "unauthorized"
    wrong_link = act(registry, "switch_to_backup", "actuator:feeds")
    assert policy.check(bundle(wrong_link, policy, version="stale"), state, 0.0, (), log)[1] == \
        "policy_version_mismatch"
    assert policy.check(bundle(wrong_link, policy), state, 0.0, (), log) == (False, "policy_forbidden")
    history = [Attempt(0.0, ok, f"h{i}") for i in range(3)]
    assert policy.check(bundle(wrong_link, policy), state, 10.0, history, log)[1] == "policy_forbidden"
    assert policy.check(bundle(ok, policy), state, 10.0, history, log) == (False, "resource_limit")
    unknown = replace(ok, tool_id="nuke")
    assert policy.check(bundle(unknown, policy), state, 0.0, (), log) == (False, "not_in_registry")
    for b in (bundle(deep, policy), bundle(wrong_link, policy), bundle(ok, policy)):
        _, reason = policy.check(b, state, 10.0, history, log)
        assert reason in REJECT_REASONS


def test_forbid_rules(registry, policy, config, log):
    state = State()
    assert policy.forbidden_by(act(registry, "switch_to_backup", "actuator:feeds"), state) == \
        "backup_only_for_sensor_links"
    assert policy.forbidden_by(act(registry, "switch_to_backup", "sensor:feeds"), state) is None
    repair = act(registry, "repair_link", "actuator:feeds")
    assert policy.forbidden_by(repair, state) is None
    assert policy.forbidden_by(repair, state, (), log) == "respond_needs_evidence"
    assert policy.check(bundle(repair, policy, evidence=()), state, 0.0, (), log) == (False, "policy_forbidden")
    assert policy.check(bundle(act(registry, "trace_link", "x"), policy, evidence=()), state, 0.0, (), log) == \
        (True, None)
    strict = copy.deepcopy(config)
    strict["forbid"].append({"id": "two_sources", "tool": ["repair_link"], "min_evidence": 2})
    strict["forbid"].append({"id": "no_escalation", "family": "escalate"})
    strict["forbid"].append({"id": "healthy_only", "tool": "recalibrate_sensor", "unless": ["sensor_calibrated"]})
    strict = Policy(strict, registry)
    assert strict.forbidden_by(repair, state, ("e1", "e1"), log) == "two_sources"
    assert strict.forbidden_by(repair, state, ("e1", "e2"), log) is None
    assert strict.forbidden_by(act(registry, "escalate", "inc-1"), state) == "no_escalation"
    recal = act(registry, "recalibrate_sensor", "reactor_pressure")
    assert strict.forbidden_by(recal, State(false={"sensor_calibrated"})) == "healthy_only"
    assert strict.forbidden_by(recal, state) is None
    assert policy.check(bundle(recal, policy, evidence=["e1"]), state, 0.0, (), log) == (True, None)


def test_the_defenders_own_receipts_never_satisfy_min_evidence(registry, policy, config, log):
    state = State()
    repair = act(registry, "repair_link", "sensor:reactor")
    forbidden = (False, "policy_forbidden")
    assert policy.check(bundle(repair, policy, evidence=("receipt:1",)), state, 0.0, (), log) == forbidden
    assert policy.check(bundle(repair, policy, evidence=("receipt:1", "e1")), state, 0.0, (), log) == (True, None)
    assert policy.check(bundle(repair, policy, evidence=("receipt:2",)), state, 0.0, (), log) == forbidden
    assert policy.check(bundle(repair, policy), state, 0.0, ()) == forbidden
    late = EvidenceLog([item("e3", arrival=5.0)])
    assert policy.forbidden_by(repair, state, ("e3",), late, now=4.9) == "respond_needs_evidence"
    assert policy.forbidden_by(repair, state, ("e3",), late, now=5.0) is None
    assert Policy.counted_evidence(["e1", "e1", "e2", "receipt:1", "ghost", 7, None], log, 0.0) == {"e1", "e2"}
    strict = copy.deepcopy(config)
    strict["forbid"].append({"id": "two_sources", "tool": ["repair_link"], "min_evidence": 2})
    strict = Policy(strict, registry)
    assert strict.forbidden_by(repair, state, ("e1", "receipt:1"), log) == "two_sources"
    assert strict.check(bundle(repair, strict, evidence=("e1", "receipt:1", "e2")), state, 0.0, (), log) == \
        (True, None)


def test_rate_limit_window_boundaries(registry, policy, config):
    a = act(registry, "repair_link", "sensor:reactor")
    other = act(registry, "repair_link", "sensor:stripper")
    trace = act(registry, "trace_link", "sensor:reactor")
    history = [Attempt(100.0, a, "k1"), Attempt(200.0, a, "k2"), Attempt(300.0, a, "k3"),
               Attempt(300.0, other, "k4"), Attempt(300.0, trace, "k5")]
    assert policy.rate_limited_by(a, 300.0, history) == "respond_per_target"
    assert policy.rate_limited_by(a, 1899.9, history) == "respond_per_target"
    assert policy.rate_limited_by(a, 1900.0, history) is None
    assert policy.rate_limited_by(a, 1900.0, history + [Attempt(1900.0, a, "k6")]) == "respond_per_target"
    assert policy.rate_limited_by(a, 300.0, history, idempotency_key="k3") is None
    assert policy.rate_limited_by(other, 300.0, history) is None
    assert policy.rate_limited_by(trace, 300.0, history * 10) is None
    overall = copy.deepcopy(config)
    overall["rate_limits"] = [{"id": "global", "family": ["respond", "escalate"], "per": "all", "max": 4,
                               "window_s": 1000}]
    overall = Policy(overall, registry)
    assert overall.rate_limited_by(other, 300.0, history) == "global"
    assert overall.rate_limited_by(other, 300.0, history[1:]) is None


def _registry_with_ttl_tool():
    config = load_config(TOOLS_CONFIG)
    config["tools"]["block"] = {
        "version": "1", "family": "respond", "target_kinds": ["link"], "duration": {"median": 5, "sigma": 0.3},
        "side_effect_class": "reversible", "required_authority": "contain",
        "args": [{"name": "ttl_s", "kind": "float", "low": 60, "high": 86400, "choices": [600, 3600]}],
    }
    return Registry.from_config(config)


def test_ttl_caps(registry, policy, config):
    assert policy.ttl(act(registry, "switch_to_backup", "sensor:reactor")) == 3600
    assert policy.ttl(act(registry, "trace_link", "sensor:reactor")) is None
    tight = Policy({**config, "ttl_caps": {"switch_to_backup": 1800, "respond": 900}}, registry)
    assert tight.ttl(act(registry, "switch_to_backup", "sensor:reactor")) == 1800
    assert tight.ttl(act(registry, "set_production", "plant", {"fraction": 0.9})) == 900
    assert tight.forbidden_by(act(registry, "switch_to_backup", "sensor:reactor"), State()) is None
    reg = _registry_with_ttl_tool()
    capped = Policy({**config, "ttl_caps": {"respond": 7200}}, reg)
    short = reg.make_action("block", "sensor:reactor", {"ttl_s": 3600}, 0.0, "b")
    long = reg.make_action("block", "sensor:reactor", {"ttl_s": 7200.5}, 0.0, "b")
    assert capped.ttl(short) == 3600 and capped.forbidden_by(short, State()) is None
    assert capped.forbidden_by(long, State()) == "ttl_cap"
    assert capped.check(bundle(long, capped), State(), 0.0, ()) == (False, "policy_forbidden")
    assert not capped.allows(long, State())


def test_allows_is_the_candidate_filter(registry, policy):
    state = State()
    assert policy.allows(act(registry, "switch_to_backup", "sensor:reactor"), state)
    assert not policy.allows(act(registry, "switch_to_backup", "actuator:feeds"), state)
    assert not policy.allows(act(registry, "set_production", "plant", {"fraction": 0.8}), state)
    a = act(registry, "repair_link", "sensor:reactor")
    history = [Attempt(0.0, a, f"k{i}") for i in range(3)]
    assert policy.allows(a, state, None, history)
    assert not policy.allows(a, state, 10.0, history)


@pytest.mark.parametrize("edit", [
    lambda c: c.update(authorities=["observe"]),
    lambda c: c.update(authority=["observe", "root"]),
    lambda c: c["scope_rules"][0].update(scope="root"),
    lambda c: c["scope_rules"][0].update(tool="nuke"),
    lambda c: c["scope_rules"][0].update(args={"fraction": {"below": 0.9}}),
    lambda c: c["scope_rules"][0].update(args={"fraction": {"in": 0.9}}),
    lambda c: c["forbid"][0].update(unles=["sensor_link"]),
    lambda c: c["forbid"][0].update(unless=["sensor_is_nice"]),
    lambda c: c["forbid"][0].update(family="attack"),
    lambda c: c["forbid"][0].pop("id"),
    lambda c: c["forbid"][1].update(min_evidence=-1),
    lambda c: c["forbid"][1].update(id="backup_only_for_sensor_links"),
    lambda c: c["rate_limits"][0].update(per="host"),
    lambda c: c["rate_limits"][0].update(max=1.5),
    lambda c: c["rate_limits"][0].update(window_s=0),
    lambda c: c["ttl_caps"].update(nuke=10),
    lambda c: c["ttl_caps"].update(respond=0),
])


def test_malformed_policy_is_refused(registry, config, edit):
    bad = copy.deepcopy(config)
    edit(bad)
    with pytest.raises(ValueError):
        Policy(bad, registry)
