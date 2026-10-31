import copy
import math

import numpy as np
import pytest

from cpsdefense.core.types import ActionFamily, ArgSpec, SideEffectClass
from cpsdefense.defense.policy import POLICY_CONFIG, Attempt, Policy
from cpsdefense.defense.registry import (PREDICATES, TOOLS_CONFIG, Registry, arg_grid, candidate_id, canonical_args,
                                         check_arg, load_registry)
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_json

LINKS = ("sensor:reactor", "sensor:stripper", "actuator:feeds")


class LinkWorld:
    def __init__(self, backup=(), production=1.0, backup_ok=True):
        self.mode = {link: ("backup" if link in backup else "up") for link in LINKS}
        self.production = production
        self.backup_ok = backup_ok
        self.targets = {"link": set(LINKS), "source": {"reactor", "stripper"}, "sensor": {"reactor_pressure"},
                        "plant": {"plant"}, "incident": {"inc-1"}}

    def exists(self, target, kind):
        return target in self.targets.get(kind, ())

    def predicate(self, name, action):
        t, fraction = action.target_id, action.bounded_arguments.get("fraction")
        return {
            "link_exists": t in self.mode,
            "sensor_link": t in self.mode and t.startswith("sensor:"),
            "link_in_backup": self.mode.get(t) == "backup",
            "link_not_in_backup": t in self.mode and self.mode[t] != "backup",
            "backup_available": self.backup_ok,
            "link_healthy": True,
            "source_exists": t in self.targets["source"],
            "sensor_exists": t in self.targets["sensor"],
            "sensor_calibrated": True,
            "incident_exists": t in self.targets["incident"],
            "production_differs": fraction != self.production,
            "production_at_target": fraction == self.production,
        }[name]

TARGETS = {"link": list(LINKS) + ["sensor:ghost"], "source": ["reactor", "stripper"], "sensor": ["reactor_pressure"],
           "plant": ["plant", "plant-ghost"], "incident": ["inc-1"]}


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def policy(registry):
    return Policy(load_config(POLICY_CONFIG), registry)


def test_registry_holds_the_te_tools(registry):
    expected = {
        "trace_link": ("observe", "link", 120, 1, "observe"),
        "query_historian": ("observe", "source", 10, 0.2, "observe"),
        "verify_sensor": ("observe", "source", 1200, 5, "observe"),
        "switch_to_backup": ("respond", "link", 60, 1, "contain"),
        "restore_primary": ("respond", "link", 60, 1, "contain"),
        "repair_link": ("respond", "link", 2400, 10, "contain"),
        "recalibrate_sensor": ("respond", "sensor", 3600, 10, "contain"),
        "replace_io_module": ("respond", "source", 3600, 10, "contain"),
        "set_production": ("respond", "plant", 30, 1, "contain"),
        "monitor": ("monitor", "plant", 1, 0, "observe"),
        "escalate": ("escalate", "incident", 900, 20, "escalate"),
    }
    assert [s.tool_id for s in registry] == list(expected)
    for tool_id, (family, kind, median, resource, authority) in expected.items():
        spec = registry.get(tool_id)
        assert spec.family == ActionFamily(family) and spec.target_kinds == (kind,)
        assert spec.duration_median == median and 0.3 <= spec.duration_sigma <= 0.5
        assert spec.resource_cost == resource and spec.required_authority == authority
        assert set(spec.preconditions + spec.postconditions) <= set(PREDICATES)
    backup = registry.get("switch_to_backup")
    assert backup.side_effect_class == SideEffectClass.REVERSIBLE
    assert backup.compensation_tool_id == "restore_primary" and backup.default_ttl == 3600
    assert set(backup.preconditions) == {"link_exists", "sensor_link", "link_not_in_backup", "backup_available"}
    assert backup.postconditions == ("link_in_backup",)
    production = registry.get("set_production")
    assert production.side_effect_class == SideEffectClass.PARTIALLY_REVERSIBLE
    assert production.args == (ArgSpec("fraction", "enum", choices=(0.8, 0.9, 1.0)),)
    assert registry.get("monitor").args == (ArgSpec("review_after_s", "enum", choices=(300, 900)),)
    assert registry.get("verify_sensor").side_effect_class == SideEffectClass.OBSERVATIONAL
    assert registry.get("nuke") is None and "nuke" not in registry and 5 not in registry and len(registry) == 11


def test_calibration_docs_describe_the_instrument_not_the_published_reading(registry):
    for text in (PREDICATES["sensor_calibrated"], registry.get("recalibrate_sensor").description):
        assert "instrument itself is calibrated" in text and "frozen I/O module may still hold" in text, text
    assert "does not say whether the primary path delivers" in registry.get("restore_primary").description


def test_registry_version_is_the_content_hash(registry):
    config = load_config(TOOLS_CONFIG)
    assert registry.version == sha256_json(config) == load_registry().version
    changed = copy.deepcopy(config)
    changed["tools"]["trace_link"]["duration"]["median"] = 121
    assert Registry.from_config(changed).version != registry.version


def test_make_action_builds_a_complete_instance(registry):
    a = registry.make_action("switch_to_backup", "sensor:reactor", {}, now=100.0, action_id="a1", expiry_s=60)
    assert a.action_id == "a1" and a.tool_version == "1" and a.family == ActionFamily.RESPOND
    assert a.expected_duration == 60 and a.duration_q90 == pytest.approx(60 * math.exp(1.2816 * 0.3))
    assert a.expiry == 160.0 and a.resource_limit == 1 and a.compensation_or_recovery_id == "restore_primary"
    assert a.preconditions == registry.get("switch_to_backup").preconditions
    assert a.side_effect_class == SideEffectClass.REVERSIBLE and dict(a.bounded_arguments) == {}
    b = registry.make_action("set_production", "plant", {"fraction": np.float64(0.9)}, 5.0, "b")
    assert b.bounded_arguments == {"fraction": 0.9} and type(b.bounded_arguments["fraction"]) is float
    assert b.expiry == 5.0 + registry.candidate_expiry_s
    assert registry.matches(a) and registry.matches(b)
    with pytest.raises(ValueError):
        registry.make_action("nuke", "plant", {}, 0.0, "x")


@pytest.mark.parametrize("args", [
    {"fraction": 0.85}, {"fraction": 0.5}, {"fraction": "0.9"}, {"fraction": True}, {"fraction": None},
    {"fraction": math.nan}, {"fraction": [0.9]}, {}, {"fraction": 0.9, "force": True}, {"force": 1},
])


def test_make_action_rejects_bad_arguments(registry, args):
    with pytest.raises(ValueError):
        registry.make_action("set_production", "plant", args, 0.0, "x")


def test_check_arg_kinds():
    i = ArgSpec("n", "int", low=1, high=5)
    f = ArgSpec("x", "float", low=0.0, high=1.0)
    b = ArgSpec("on", "bool")
    e = ArgSpec("mode", "enum", choices=("auto", 2))
    assert check_arg(i, 3) == 3 and check_arg(i, np.int64(5)) == 5
    assert check_arg(f, 1) == 1.0 and check_arg(f, np.float32(0.5)) == 0.5
    assert check_arg(b, np.bool_(True)) is True
    assert check_arg(e, "auto") == "auto" and check_arg(e, 2.0) == 2
    for spec, bad in [(i, 0), (i, 6), (i, 2.0), (i, True), (f, 1.5), (f, math.inf), (f, "0.5"), (b, 1),
                      (e, "AUTO"), (e, 3), (e, False)]:
        with pytest.raises(ValueError):
            check_arg(spec, bad)
    spec = load_registry().get("monitor")
    with pytest.raises(ValueError):
        canonical_args(spec, [("review_after_s", 300)])


def test_argument_grid_and_numeric_grids():
    config = load_config(TOOLS_CONFIG)
    config["tools"]["block"] = {
        "version": "1", "family": "respond", "target_kinds": ["link"], "duration": {"median": 5, "sigma": 0.3},
        "side_effect_class": "reversible", "required_authority": "contain",
        "args": [{"name": "ttl_s", "kind": "float", "low": 60, "high": 7200, "choices": [600, 3600]},
                 {"name": "burst", "kind": "int", "low": 1, "high": 3},
                 {"name": "log", "kind": "bool"}],
    }
    registry = Registry.from_config(config)
    grid = arg_grid(registry.get("block"))
    assert len(grid) == 2 * 3 * 2 and grid[0] == {"ttl_s": 600, "burst": 1, "log": False}
    assert len({tuple(g.items()) for g in grid}) == len(grid)
    assert registry.make_action("block", "sensor:reactor", {"ttl_s": 1234.5, "burst": 2, "log": True}, 0, "x")
    for g in grid:
        registry.make_action("block", "sensor:reactor", g, 0, "x")
    assert arg_grid(registry.get("trace_link")) == [{}]
    assert arg_grid(registry.get("set_production")) == [{"fraction": 0.8}, {"fraction": 0.9}, {"fraction": 1.0}]


@pytest.mark.parametrize("path,value", [
    (("trace_link", "family"), "attack"),
    (("trace_link", "side_effect_class"), "harmless"),
    (("trace_link", "target_kinds"), ["host"]),
    (("trace_link", "target_kinds"), []),
    (("trace_link", "preconditions"), ["link_exists", "vibes_ok"]),
    (("trace_link", "required_authority"), "root"),
    (("trace_link", "duration"), {"median": 0, "sigma": 0.3}),
    (("trace_link", "duration"), {"median": 10, "sigma": -1}),
    (("trace_link", "resource_cost"), -1),
    (("trace_link", "default_ttl"), 0),
    (("trace_link", "compensation"), "unknown_tool"),
    (("trace_link", "compensation"), "trace_link"),
    (("trace_link", "colour"), "red"),
    (("monitor", "args"), [{"name": "review_after_s", "kind": "enum"}]),
    (("monitor", "args"), [{"name": "x", "kind": "float", "low": 0, "high": 1}]),
    (("monitor", "args"), [{"name": "x", "kind": "int", "low": 0, "high": 100}]),
    (("monitor", "args"), [{"name": "x", "kind": "int", "low": 3, "high": 1}]),
    (("monitor", "args"), [{"name": "x", "kind": "float", "low": 0, "high": 1, "choices": [2.0]}]),
    (("monitor", "args"), [{"name": "x", "kind": "text", "choices": ["a"]}]),
    (("monitor", "args"), [{"name": "x", "kind": "bool"}, {"name": "x", "kind": "bool"}]),
])


def test_malformed_registry_is_refused(path, value):
    config = load_config(TOOLS_CONFIG)
    config["tools"][path[0]][path[1]] = value
    with pytest.raises(ValueError):
        Registry.from_config(config)


def test_unknown_top_level_key_is_refused():
    config = load_config(TOOLS_CONFIG)
    config["toolz"] = {}
    with pytest.raises(ValueError):
        Registry.from_config(config)


def _keys(actions):
    return {(a.tool_id, a.target_id, tuple(a.bounded_arguments.items())) for a in actions}


def test_candidates_enumerate_policy_allowed_instances_whose_preconditions_hold(registry, policy):
    world = LinkWorld(backup=("sensor:reactor",), production=1.0)
    cands = registry.candidates(TARGETS, 50.0, policy, world)
    keys = _keys(cands)
    want = {("trace_link", link, ()) for link in LINKS}
    want |= {(tool, src, ()) for tool in ("query_historian", "verify_sensor", "replace_io_module")
             for src in ("reactor", "stripper")}
    want |= {("switch_to_backup", "sensor:stripper", ()),
             ("restore_primary", "sensor:reactor", ()),
             ("recalibrate_sensor", "reactor_pressure", ()),
             ("set_production", "plant", (("fraction", 0.9),)),
             ("monitor", "plant", (("review_after_s", 300),)), ("monitor", "plant", (("review_after_s", 900),)),
             ("escalate", "inc-1", ())}
    want |= {("repair_link", link, ()) for link in LINKS}
    assert keys == want and len(cands) == len(want)
    for a in cands:
        assert registry.matches(a) and a.expiry == 50.0 + registry.candidate_expiry_s
        assert a.action_id == candidate_id(a.tool_id, a.target_id, a.bounded_arguments, 50.0)
    assert [a.action_id for a in cands] == [a.action_id for a in registry.candidates(TARGETS, 50.0, policy, world)]
    assert len({a.action_id for a in cands}) == len(cands)


def test_candidates_follow_policy_and_state(registry, policy):
    process = Policy({**load_config(POLICY_CONFIG), "authority": ["observe", "contain", "process", "escalate"]},
                     registry)
    world = LinkWorld(production=0.9)
    keys = _keys(registry.candidates(TARGETS, 0.0, process, world))
    assert {("set_production", "plant", (("fraction", f),)) for f in (0.8, 1.0)} <= keys
    assert ("set_production", "plant", (("fraction", 0.9),)) not in keys
    no_backup = _keys(registry.candidates(TARGETS, 0.0, policy, LinkWorld(backup_ok=False)))
    assert not any(k[0] == "switch_to_backup" for k in no_backup)
    observe_only = Policy({"authority": ["observe"]}, registry)
    tools = {k[0] for k in _keys(registry.candidates(TARGETS, 0.0, observe_only, LinkWorld()))}
    assert tools == {"trace_link", "query_historian", "verify_sensor", "monitor"}
    assert registry.candidates({}, 0.0, policy, LinkWorld()) == []


def test_candidates_apply_rate_limits_when_history_is_given(registry, policy):
    world = LinkWorld()
    repair = registry.make_action("repair_link", "sensor:stripper", {}, 0.0, "r")
    history = [Attempt(10.0 * i, repair, f"k{i}") for i in range(3)]
    keys = _keys(registry.candidates(TARGETS, 100.0, policy, world, history=history))
    assert not any(k[1] == "sensor:stripper" and k[0] in ("repair_link", "switch_to_backup") for k in keys)
    assert ("trace_link", "sensor:stripper", ()) in keys
    assert ("repair_link", "sensor:reactor", ()) in keys
    later = _keys(registry.candidates(TARGETS, 1800.0 + 20.0 + 1.0, policy, world, history=history))
    assert ("repair_link", "sensor:stripper", ()) in later
