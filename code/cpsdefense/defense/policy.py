import copy
import math
import operator
from dataclasses import dataclass

from cpsdefense.core.types import Action, ActionFamily
from cpsdefense.defense.registry import AUTHORITY_SCOPES, PREDICATES, TTL_ARG
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_json

POLICY_CONFIG = "defense/policy.yaml"
EXECUTOR_PROVENANCE = "executor"
ARG_OPS = {
    "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "eq": operator.eq,
    "ne": operator.ne, "in": lambda v, ref: v in ref, "not_in": lambda v, ref: v not in ref,
}
POLICY_KEYS = {"authority", "scope_rules", "forbid", "rate_limits", "ttl_caps"}
MATCH_KEYS = {"id", "tool", "family", "args"}
RULE_KEYS = {
    "scope_rules": MATCH_KEYS | {"scope"},
    "forbid": MATCH_KEYS | {"unless", "min_evidence"},
    "rate_limits": MATCH_KEYS | {"per", "max", "window_s"},
}
FAMILIES = tuple(f.value for f in ActionFamily)


@dataclass(frozen=True)
class Attempt:
    time: float
    action: Action
    idempotency_key: str


def _names(value):
    return (value,) if isinstance(value, str) else tuple(value)


def _arg_holds(value, op, ref):
    try:
        return bool(ARG_OPS[op](value, ref))
    except TypeError:
        return True


class Policy:
    def __init__(self, config, registry):
        self.config = copy.deepcopy(config)
        self.version = sha256_json(self.config)
        self.registry = registry
        self.scopes = frozenset(self.config.get("authority", ()))
        self.scope_rules = tuple(self.config.get("scope_rules", ()))
        self.forbid = tuple(self.config.get("forbid", ()))
        self.rate_limits = tuple(self.config.get("rate_limits", ()))
        self.ttl_caps = dict(self.config.get("ttl_caps", {}))
        self._check()

    def _check(self):
        unknown = set(self.config) - POLICY_KEYS
        if unknown:
            raise ValueError(f"policy: unknown keys {sorted(unknown)}")
        if not self.scopes <= set(AUTHORITY_SCOPES):
            raise ValueError(f"policy: authority {sorted(self.scopes)} not within {AUTHORITY_SCOPES}")
        ids = []
        for section in ("scope_rules", "forbid", "rate_limits"):
            for rule in getattr(self, section):
                self._check_rule(section, rule)
                ids.append(rule["id"])
        if len(set(ids)) != len(ids):
            raise ValueError("policy: duplicate rule ids")
        for key, cap in self.ttl_caps.items():
            if key not in self.registry and key not in FAMILIES:
                raise ValueError(f"policy: ttl cap for unknown tool or family {key!r}")
            if not (isinstance(cap, (int, float)) and cap > 0):
                raise ValueError(f"policy: ttl cap {key!r} must be positive")

    def _check_rule(self, section, rule):
        where = f"policy.{section}.{rule.get('id')}"
        unknown = set(rule) - RULE_KEYS[section]
        if unknown or not isinstance(rule.get("id"), str):
            raise ValueError(f"{where}: needs a string id; unknown keys {sorted(unknown)}")
        for tool in _names(rule.get("tool", ())):
            if tool not in self.registry:
                raise ValueError(f"{where}: unknown tool {tool!r}")
        for family in _names(rule.get("family", ())):
            if family not in FAMILIES:
                raise ValueError(f"{where}: unknown family {family!r}")
        for name, conds in rule.get("args", {}).items():
            for op, ref in conds.items():
                if op not in ARG_OPS or (op in ("in", "not_in") and not isinstance(ref, list)):
                    raise ValueError(f"{where}: bad condition {name}.{op}")
        if section == "scope_rules" and rule.get("scope") not in AUTHORITY_SCOPES:
            raise ValueError(f"{where}: scope must be one of {AUTHORITY_SCOPES}")
        if section == "forbid":
            unknown = [p for p in rule.get("unless", ()) if p not in PREDICATES]
            if unknown:
                raise ValueError(f"{where}: unknown predicates {unknown}")
            if "min_evidence" in rule and not (isinstance(rule["min_evidence"], int) and rule["min_evidence"] >= 0):
                raise ValueError(f"{where}: min_evidence must be a non-negative int")
        if section == "rate_limits":
            if rule.get("per", "target") not in ("target", "all"):
                raise ValueError(f"{where}: per must be 'target' or 'all'")
            if not (isinstance(rule.get("max"), int) and rule["max"] >= 0):
                raise ValueError(f"{where}: max must be a non-negative int")
            window = rule.get("window_s")
            if not (isinstance(window, (int, float)) and window > 0):
                raise ValueError(f"{where}: window_s must be positive")

    def _spec(self, action):
        return self.registry.get(action.tool_id)

    def _matches(self, rule, action):
        spec = self._spec(action)
        if spec is None:
            return False
        if "tool" in rule and spec.tool_id not in _names(rule["tool"]):
            return False
        if "family" in rule and spec.family.value not in _names(rule["family"]):
            return False
        args = action.bounded_arguments
        for name, conds in rule.get("args", {}).items():
            if name in args and not all(_arg_holds(args[name], op, ref) for op, ref in conds.items()):
                return False
        return True

    def required_scopes(self, action):
        spec = self._spec(action)
        scopes = {spec.required_authority}
        scopes.update(rule["scope"] for rule in self.scope_rules if self._matches(rule, action))
        return frozenset(scopes)

    def authorized(self, action):
        return self._spec(action) is not None and self.required_scopes(action) <= self.scopes

    def ttl_cap(self, spec):
        cap = self.ttl_caps.get(spec.tool_id, self.ttl_caps.get(spec.family.value))
        return None if cap is None else float(cap)

    def ttl(self, action):
        spec = self._spec(action)
        requested = action.bounded_arguments.get(TTL_ARG, spec.default_ttl)
        if requested is None:
            return None
        cap = self.ttl_cap(spec)
        return float(requested) if cap is None else min(float(requested), cap)

    @staticmethod
    def counted_evidence(evidence_ids, evidence_log, now=math.inf):
        if evidence_log is None:
            return frozenset()
        counted = set()
        for eid in evidence_ids:
            item = evidence_log.get(eid)
            if item is not None and evidence_log.known(eid, now) and item.provenance != EXECUTOR_PROVENANCE:
                counted.add(eid)
        return frozenset(counted)

    def forbidden_by(self, action, state_view, evidence_ids=None, evidence_log=None, now=math.inf):
        spec = self._spec(action)
        cap = self.ttl_cap(spec)
        if TTL_ARG in action.bounded_arguments and cap is not None:
            if not float(action.bounded_arguments[TTL_ARG]) <= cap:
                return "ttl_cap"
        for rule in self.forbid:
            if not self._matches(rule, action):
                continue
            if "unless" not in rule and "min_evidence" not in rule:
                return rule["id"]
            if not all(state_view.predicate(p, action) for p in rule.get("unless", ())):
                return rule["id"]
            if "min_evidence" in rule and evidence_ids is not None:
                if len(self.counted_evidence(evidence_ids, evidence_log, now)) < rule["min_evidence"]:
                    return rule["id"]
        return None

    def rate_limited_by(self, action, now, history, idempotency_key=None):
        for rule in self.rate_limits:
            if not self._matches(rule, action):
                continue
            count = 0
            for attempt in history:
                if attempt.idempotency_key == idempotency_key or not self._matches(rule, attempt.action):
                    continue
                if rule.get("per", "target") == "target" and attempt.action.target_id != action.target_id:
                    continue
                if now - rule["window_s"] < attempt.time <= now:
                    count += 1
            if count >= rule["max"]:
                return rule["id"]
        return None

    def allows(self, action, state_view, now=None, history=()):
        return (self.authorized(action) and self.forbidden_by(action, state_view) is None
                and (now is None or self.rate_limited_by(action, now, history) is None))

    def check(self, bundle, state_view, now, history, evidence_log=None):
        action = bundle.action
        if self._spec(action) is None:
            return False, "not_in_registry"
        if not self.authorized(action):
            return False, "unauthorized"
        if bundle.policy_version != self.version:
            return False, "policy_version_mismatch"
        evidence = bundle.evidence_ids if isinstance(bundle.evidence_ids, (tuple, list)) else ()
        if self.forbidden_by(action, state_view, evidence, evidence_log, now) is not None:
            return False, "policy_forbidden"
        if self.rate_limited_by(action, now, history, bundle.idempotency_key) is not None:
            return False, "resource_limit"
        return True, None


def load_policy(registry, path=POLICY_CONFIG, overrides=None):
    return Policy(load_config(path, overrides), registry)
