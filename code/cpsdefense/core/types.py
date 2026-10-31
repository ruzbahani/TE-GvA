from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionFamily(str, Enum):
    OBSERVE = "observe"
    RESPOND = "respond"
    MONITOR = "monitor"
    ESCALATE = "escalate"


class SideEffectClass(str, Enum):
    NONE = "none"
    OBSERVATIONAL = "observational"
    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"


class EvidenceSource(str, Enum):
    NETWORK = "network"
    HOST = "host"
    IDENTITY = "identity"
    PHYSICAL = "physical"
    DETECTOR = "detector"
    TOOL = "tool"
    OPERATOR = "operator"


class ExecStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DUPLICATE = "duplicate"
    COMPENSATED = "compensated"

REJECT_REASONS = (
    "not_in_registry",
    "tool_version_mismatch",
    "arg_out_of_bounds",
    "unauthorized",
    "policy_version_mismatch",
    "policy_forbidden",
    "candidate_expired",
    "deadline_passed",
    "stale_evidence",
    "unknown_evidence",
    "stale_state",
    "precondition_failed",
    "resource_limit",
)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source: str
    kind: str
    subject_id: str
    event_time: float
    arrival_time: float
    valid_until: float
    payload: Mapping[str, Any]
    provenance: str
    quality: float = 1.0
    supersedes: tuple = ()


@dataclass(frozen=True)
class Alert:
    alert_id: str
    detector_id: str
    subject_id: str
    time: float
    score: float
    category: str
    evidence_ids: tuple = ()


@dataclass
class ObservationBundle:
    time: float
    version: int
    new_evidence: tuple = ()
    alerts: tuple = ()
    physical: Mapping[str, float] = field(default_factory=dict)
    active_actions: tuple = ()
    receipts: tuple = ()
    policy_version: str = ""
    deadline_hint: float | None = None
    missing: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ArgSpec:
    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple = ()


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    tool_version: str
    family: ActionFamily
    target_kinds: tuple
    args: tuple = ()
    duration_median: float = 1.0
    duration_sigma: float = 0.3
    side_effect_class: SideEffectClass = SideEffectClass.NONE
    resource_cost: float = 0.0
    preconditions: tuple = ()
    postconditions: tuple = ()
    compensation_tool_id: str | None = None
    default_ttl: float | None = None
    required_authority: str = "observe"
    description: str = ""


@dataclass(frozen=True)
class Action:
    action_id: str
    tool_id: str
    tool_version: str
    family: ActionFamily
    target_id: str
    bounded_arguments: Mapping[str, Any]
    expected_duration: float
    duration_q90: float
    preconditions: tuple
    postconditions: tuple
    resource_limit: float
    side_effect_class: SideEffectClass
    expiry: float
    compensation_or_recovery_id: str | None = None


@dataclass(frozen=True)
class ExecutionBundle:
    decision_id: str
    action: Action
    parameter_values: Mapping[str, Any]
    evidence_ids: tuple
    observation_version: int
    policy_version: str
    not_after: float
    expected_postconditions: tuple
    monitor_rule: str
    recovery_or_compensation_id: str | None
    idempotency_key: str


@dataclass(frozen=True)
class Receipt:
    decision_id: str
    action_id: str
    idempotency_key: str
    status: ExecStatus
    reason: str | None
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    side_effects: tuple = ()
    postconditions_met: Mapping[str, bool] | None = None
    signature: str = ""

COST_COMPONENTS = ("attack", "service", "physical", "compute")


@dataclass
class CostBreakdown:
    attack: float = 0.0
    service: float = 0.0
    physical: float = 0.0
    compute: float = 0.0

    def total(self, weights=None):
        w = {"attack": 1.0, "service": 1.0, "physical": 1.0, "compute": 1.0, **(weights or {})}
        return sum(w[k] * getattr(self, k) for k in COST_COMPONENTS)

    def add(self, other):
        for k in COST_COMPONENTS:
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self

    def as_dict(self):
        return asdict(self)
