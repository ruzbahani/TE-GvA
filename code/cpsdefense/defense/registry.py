import itertools
import math
import numbers
from typing import Mapping

import numpy as np

from cpsdefense.core.types import Action, ActionFamily, ArgSpec, SideEffectClass, ToolSpec
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_json

TOOLS_CONFIG = "defense/tools.yaml"
TARGET_KINDS = ("link", "source", "sensor", "plant", "incident")
AUTHORITY_SCOPES = ("observe", "contain", "process", "escalate")
ARG_KINDS = ("int", "float", "enum", "bool")
Q90_Z = 1.2816
TTL_ARG = "ttl_s"
REVIEW_ARG = "review_after_s"
MAX_INT_GRID = 64
TOOL_KEYS = {"version", "family", "target_kinds", "args", "duration", "side_effect_class", "resource_cost",
             "required_authority", "preconditions", "postconditions", "compensation", "default_ttl", "description"}

PREDICATES = {
    "link_exists": "the target link exists",
    "sensor_link": "the target is a sensor link (sensor:<group>)",
    "link_in_backup": "the target sensor link is on its backup channel",
    "link_not_in_backup": "the target link is not on a backup channel",
    "backup_available": "the backup channel of the target link is available",
    "link_healthy": "the primary path of the target link delivers",
    "source_exists": "the target telemetry source exists",
    "sensor_exists": "the target sensor exists",
    "sensor_calibrated": "the target instrument itself is calibrated (a frozen I/O module may still hold its "
                         "published reading)",
    "incident_exists": "the target incident is open",
    "production_differs": "the requested production fraction differs from the current target",
    "production_at_target": "the production target equals the requested fraction",
}


def _is_bool(value):
    return isinstance(value, (bool, np.bool_))


def check_arg(arg, value):
    where = f"argument {arg.name!r}"
    if arg.kind == "bool":
        if not _is_bool(value):
            raise ValueError(f"{where}: expected a bool, got {value!r}")
        return bool(value)
    if _is_bool(value):
        raise ValueError(f"{where}: a bool is not a {arg.kind}")
    if arg.kind == "enum":
        if isinstance(value, (str, numbers.Real)):
            for choice in arg.choices:
                if value == choice:
                    return choice
        raise ValueError(f"{where}: {value!r} not in {arg.choices}")
    if arg.kind == "int":
        if not isinstance(value, numbers.Integral):
            raise ValueError(f"{where}: expected an int, got {value!r}")
        value = int(value)
    else:
        if not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{where}: expected a finite number, got {value!r}")
        value = float(value)
    if not arg.low <= value <= arg.high:
        raise ValueError(f"{where}: {value} outside [{arg.low}, {arg.high}]")
    return value


def canonical_args(spec, args):
    if not isinstance(args, Mapping):
        raise ValueError(f"{spec.tool_id}: arguments must be a mapping")
    names = [a.name for a in spec.args]
    unknown = [k for k in args if k not in names]
    if unknown:
        raise ValueError(f"{spec.tool_id}: unknown arguments {unknown}")
    missing = [k for k in names if k not in args]
    if missing:
        raise ValueError(f"{spec.tool_id}: missing arguments {missing}")
    return {a.name: check_arg(a, args[a.name]) for a in spec.args}


def arg_grid(spec):
    axes = []
    for a in spec.args:
        if a.kind == "bool":
            axes.append((False, True))
        elif a.choices:
            axes.append(tuple(a.choices))
        else:
            axes.append(tuple(range(int(a.low), int(a.high) + 1)))
    return [dict(zip([a.name for a in spec.args], combo)) for combo in itertools.product(*axes)]


def candidate_id(tool_id, target_id, args, now):
    text = ",".join(f"{k}={args[k]}" for k in sorted(args))
    return f"{tool_id}({target_id}{';' + text if text else ''})@{float(now):.3f}"


def arg_from_config(entry):
    return ArgSpec(name=entry["name"], kind=entry["kind"], low=entry.get("low"), high=entry.get("high"),
                   choices=tuple(entry.get("choices", ())))


def spec_from_config(tool_id, entry):
    unknown = set(entry) - TOOL_KEYS
    if unknown:
        raise ValueError(f"{tool_id}: unknown keys {sorted(unknown)}")
    duration = entry["duration"]
    ttl = entry.get("default_ttl")
    return ToolSpec(
        tool_id=tool_id,
        tool_version=str(entry["version"]),
        family=ActionFamily(entry["family"]),
        target_kinds=tuple(entry["target_kinds"]),
        args=tuple(arg_from_config(a) for a in entry.get("args", ())),
        duration_median=float(duration["median"]),
        duration_sigma=float(duration["sigma"]),
        side_effect_class=SideEffectClass(entry["side_effect_class"]),
        resource_cost=float(entry.get("resource_cost", 0.0)),
        preconditions=tuple(entry.get("preconditions", ())),
        postconditions=tuple(entry.get("postconditions", ())),
        compensation_tool_id=entry.get("compensation"),
        default_ttl=None if ttl is None else float(ttl),
        required_authority=entry["required_authority"],
        description=entry.get("description", ""),
    )


def _check_arg_spec(tool_id, a):
    where = f"{tool_id}.{a.name}"
    if a.kind not in ARG_KINDS:
        raise ValueError(f"{where}: kind {a.kind!r} not in {ARG_KINDS}")
    if a.kind == "enum" and not a.choices:
        raise ValueError(f"{where}: an enum needs choices")
    if a.kind in ("int", "float"):
        if a.low is None or a.high is None or not a.low <= a.high:
            raise ValueError(f"{where}: numeric arguments need bounds low <= high")
        if a.choices:
            for c in a.choices:
                check_arg(a, c)
        elif a.kind == "float" or a.high - a.low >= MAX_INT_GRID:
            raise ValueError(f"{where}: give choices as the candidate grid")


def check_spec(spec, tool_ids):
    where = spec.tool_id
    if not spec.target_kinds or not set(spec.target_kinds) <= set(TARGET_KINDS):
        raise ValueError(f"{where}: target kinds {spec.target_kinds} not within {TARGET_KINDS}")
    for value in (spec.duration_median, spec.duration_sigma):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{where}: duration median and sigma must be positive")
    if not (math.isfinite(spec.resource_cost) and spec.resource_cost >= 0):
        raise ValueError(f"{where}: resource cost must be >= 0")
    if spec.required_authority not in AUTHORITY_SCOPES:
        raise ValueError(f"{where}: authority {spec.required_authority!r} not in {AUTHORITY_SCOPES}")
    unknown = [p for p in spec.preconditions + spec.postconditions if p not in PREDICATES]
    if unknown:
        raise ValueError(f"{where}: unknown predicates {unknown}")
    if spec.compensation_tool_id is not None and (spec.compensation_tool_id not in tool_ids
                                                  or spec.compensation_tool_id == spec.tool_id):
        raise ValueError(f"{where}: compensation {spec.compensation_tool_id!r} is not another registered tool")
    if spec.default_ttl is not None and not spec.default_ttl > 0:
        raise ValueError(f"{where}: default_ttl must be positive")
    names = [a.name for a in spec.args]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate argument names")
    for a in spec.args:
        _check_arg_spec(where, a)


def duration_q90(spec):
    return spec.duration_median * math.exp(Q90_Z * spec.duration_sigma)


def spec_fields(spec):
    return {
        "tool_version": spec.tool_version,
        "family": spec.family,
        "expected_duration": spec.duration_median,
        "duration_q90": duration_q90(spec),
        "preconditions": spec.preconditions,
        "postconditions": spec.postconditions,
        "resource_limit": spec.resource_cost,
        "side_effect_class": spec.side_effect_class,
        "compensation_or_recovery_id": spec.compensation_tool_id,
    }


class Registry:
    def __init__(self, specs, candidate_expiry_s=300.0, version=""):
        self.specs = {}
        for spec in specs:
            if spec.tool_id in self.specs:
                raise ValueError(f"duplicate tool {spec.tool_id!r}")
            self.specs[spec.tool_id] = spec
        for spec in self.specs.values():
            check_spec(spec, self.specs)
        if not candidate_expiry_s > 0:
            raise ValueError("candidate_expiry_s must be positive")
        self.candidate_expiry_s = float(candidate_expiry_s)
        self.version = version

    @classmethod
    def from_config(cls, config):
        unknown = set(config) - {"candidate_expiry_s", "tools"}
        if unknown:
            raise ValueError(f"tool registry: unknown keys {sorted(unknown)}")
        specs = [spec_from_config(tool_id, entry) for tool_id, entry in config["tools"].items()]
        return cls(specs, config.get("candidate_expiry_s", 300.0), sha256_json(config))

    def __contains__(self, tool_id):
        return self.get(tool_id) is not None

    def __iter__(self):
        return iter(self.specs.values())

    def __len__(self):
        return len(self.specs)

    def get(self, tool_id):
        return self.specs.get(tool_id) if isinstance(tool_id, str) else None

    def make_action(self, tool_id, target_id, args, now, action_id, expiry_s=None):
        spec = self.get(tool_id)
        if spec is None:
            raise ValueError(f"unknown tool {tool_id!r}")
        expiry_s = self.candidate_expiry_s if expiry_s is None else float(expiry_s)
        return Action(action_id=action_id, tool_id=tool_id, target_id=target_id,
                      bounded_arguments=canonical_args(spec, args), expiry=float(now) + expiry_s,
                      **spec_fields(spec))

    def matches(self, action):
        spec = self.get(action.tool_id)
        if spec is None:
            return False
        return all(getattr(action, k) == v for k, v in spec_fields(spec).items())

    def candidates(self, targets_by_kind, now, policy, state_view, expiry_s=None, history=()):
        out, seen = [], set()
        for spec in self.specs.values():
            grid = arg_grid(spec)
            for kind in spec.target_kinds:
                for target in targets_by_kind.get(kind, ()):
                    if not state_view.exists(target, kind):
                        continue
                    for args in grid:
                        key = (spec.tool_id, target, tuple(args.items()))
                        if key in seen:
                            continue
                        seen.add(key)
                        action = self.make_action(spec.tool_id, target, args, now,
                                                  candidate_id(spec.tool_id, target, args, now), expiry_s)
                        if (policy.allows(action, state_view, now, history)
                                and all(state_view.predicate(p, action) for p in spec.preconditions)):
                            out.append(action)
        return out


def load_registry(path=TOOLS_CONFIG, overrides=None):
    return Registry.from_config(load_config(path, overrides))
