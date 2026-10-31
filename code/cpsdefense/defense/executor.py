import hashlib
import hmac
import json
import math
import numbers
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Mapping

import numpy as np

from cpsdefense.core.types import Action, ExecStatus, ExecutionBundle, Receipt, ToolSpec
from cpsdefense.defense.policy import Attempt
from cpsdefense.defense.registry import REVIEW_ARG, canonical_args

DUPLICATE = "duplicate"


@dataclass(frozen=True)
class ReviewItem:
    review_id: str
    kind: str
    decision_id: str
    action_id: str
    idempotency_key: str
    tool_id: str
    target_id: str
    due_at: float
    raised_at: float
    signature: str = ""


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json(record):
    data = {f.name: getattr(record, f.name) for f in fields(record) if f.name != "signature"}
    data["record_type"] = type(record).__name__
    return json.dumps(_plain(data), sort_keys=True, separators=(",", ":"))


def _key_bytes(key):
    key = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    if not key:
        raise ValueError("signing key is empty")
    return key


def signature(record, key):
    return hmac.new(_key_bytes(key), canonical_json(record).encode("utf-8"), hashlib.sha256).hexdigest()


def sign_record(record, key):
    return replace(record, signature=signature(record, key))


def verify_receipt(record, key):
    try:
        expected = signature(record, key)
    except (TypeError, ValueError):
        return False
    return isinstance(record.signature, str) and hmac.compare_digest(record.signature, expected)


def _lt(a, b):
    try:
        return bool(a < b)
    except TypeError:
        return False


@dataclass(eq=False)
class _Job:
    seq: int
    bundle: ExecutionBundle
    spec: ToolSpec
    params: dict
    submitted_at: float
    start_at: float
    submit_version: int
    state: str = "queued"
    started_at: float | None = None
    finish_at: float | None = None


@dataclass(eq=False)
class _Timer:
    seq: int
    kind: str
    due_at: float
    job: _Job


class Executor:
    def __init__(self, registry, policy, evidence_log, state_view, effector, key, queue_latency_s=0.0):
        queue_latency_s = float(queue_latency_s)
        if not (math.isfinite(queue_latency_s) and queue_latency_s >= 0):
            raise ValueError("queue_latency_s must be finite and >= 0")
        self.registry = registry
        self.policy = policy
        self.evidence = evidence_log
        self.state_view = state_view
        self.effector = effector
        self._key = _key_bytes(key)
        self.queue_latency_s = queue_latency_s
        self.now = -math.inf
        self.keys = set()
        self.history = []
        self.receipts = []
        self.reviews = []
        self.side_effects = []
        self._active = []
        self._timers = []
        self._seq = 0

    def submit(self, bundle, now):
        if not isinstance(bundle, ExecutionBundle):
            raise TypeError("submit expects an ExecutionBundle")
        if not isinstance(bundle.idempotency_key, str) or not bundle.idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        now = self._clock(now)
        reason = self._static_reason(bundle, now)
        if reason == DUPLICATE:
            return self._issue(bundle, ExecStatus.DUPLICATE, None, now)
        if reason is not None:
            return self._issue(bundle, ExecStatus.REJECTED, reason, now)
        spec = self.registry.get(bundle.action.tool_id)
        job = _Job(self._next_seq(), bundle, spec, canonical_args(spec, bundle.parameter_values), now,
                   now + self.queue_latency_s, self.state_view.version())
        self._active.append(job)
        self.keys.add(bundle.idempotency_key)
        self.history.append(Attempt(now, bundle.action, bundle.idempotency_key))
        return self._issue(bundle, ExecStatus.ACCEPTED, None, now)

    def tick(self, now):
        now = self._clock(now)
        out = []
        while True:
            event = self._next_event(now)
            if event is None:
                return out
            kind, item = event
            if kind == "finish":
                out.append(self._finish(item, now))
            elif kind == "start":
                out.append(self._start(item, now))
            else:
                out.append(self._raise_review(item, now))

    def next_event_time(self):
        times = [key[0] for key, _, _ in self._events()]
        return min(times) if times else None

    def active_actions(self):
        return tuple(job.bundle.action.action_id for job in self._active)

    def verify(self, record):
        return verify_receipt(record, self._key)

    def _static_reason(self, bundle, now):
        action = bundle.action
        spec = self.registry.get(action.tool_id) if isinstance(action, Action) else None
        if spec is None:
            return "not_in_registry"
        if action.tool_version != spec.tool_version:
            return "tool_version_mismatch"
        forged = not self.registry.matches(action)
        if forged or bundle.recovery_or_compensation_id not in (None, spec.compensation_tool_id):
            return "not_in_registry"
        if not self._args_ok(spec, bundle):
            return "arg_out_of_bounds"
        ok, reason = self.policy.check(bundle, self.state_view, now, tuple(self.history), self.evidence)
        if not ok:
            return reason
        if not _lt(now, action.expiry):
            return "candidate_expired"
        if not _lt(now + self.queue_latency_s, bundle.not_after):
            return "deadline_passed"
        if bundle.idempotency_key in self.keys:
            return DUPLICATE
        return None

    def _args_ok(self, spec, bundle):
        try:
            params = canonical_args(spec, bundle.parameter_values)
            bounded = canonical_args(spec, bundle.action.bounded_arguments)
        except ValueError:
            return False
        target = bundle.action.target_id
        return (params == bounded and isinstance(target, str)
                and any(self.state_view.exists(target, kind) for kind in spec.target_kinds))

    def _revalidate(self, job, now):
        bundle, action = job.bundle, job.bundle.action
        if not _lt(now, bundle.not_after):
            return "deadline_passed"
        ids = bundle.evidence_ids
        if not isinstance(ids, (tuple, list)):
            return "unknown_evidence"
        reason = self.evidence.invalid_reason(ids, now)
        if reason is not None:
            return reason
        version = bundle.observation_version
        if (not isinstance(version, numbers.Integral) or isinstance(version, bool)
                or not version <= job.submit_version):
            return "stale_state"
        subjects = [action.target_id] + [self.evidence.get(e).subject_id for e in ids]
        if not all(self.state_view.last_changed_version(s) <= version for s in subjects):
            return "stale_state"
        if not all(self.state_view.predicate(p, action) for p in job.spec.preconditions):
            return "precondition_failed"
        return None

    def _events(self):
        for job in self._active:
            if job.state == "queued":
                yield (job.start_at, 1, job.seq), "start", job
            else:
                yield (job.finish_at, 0, job.seq), "finish", job
        for timer in self._timers:
            yield (timer.due_at, 2, timer.seq), "review", timer

    def _next_event(self, now):
        due = [(key, kind, item) for key, kind, item in self._events() if key[0] <= now]
        if not due:
            return None
        _, kind, item = min(due, key=lambda event: event[0])
        return kind, item

    def _start(self, job, now):
        reason = self._revalidate(job, now)
        if reason is not None:
            self._active.remove(job)
            status = ExecStatus.TIMED_OUT if reason == "deadline_passed" else ExecStatus.REJECTED
            return self._issue(job.bundle, status, reason, job.submitted_at)
        duration = float(self.effector.sample_duration(job.spec, job.bundle.action))
        if not (math.isfinite(duration) and duration >= 0):
            raise ValueError(f"effector returned an invalid duration {duration}")
        job.state, job.started_at, job.finish_at = "running", now, now + duration
        return self._issue(job.bundle, ExecStatus.RUNNING, None, job.submitted_at, started_at=now)

    def _finish(self, job, now):
        action, spec = job.bundle.action, job.spec
        self._active.remove(job)
        result = self.effector.apply(action, dict(job.params), now)
        effects = self._absorb(result, job.bundle, now)
        post = {name: bool(self.state_view.predicate(name, action)) for name in spec.postconditions}
        applied = bool(result.get("ok", False))
        status, reason = ExecStatus.SUCCEEDED, None
        if applied and all(post.values()):
            self._schedule_reviews(job, now)
        else:
            status, reason = ExecStatus.FAILED, "postcondition_failed" if applied else "tool_error"
            if spec.compensation_tool_id is not None:
                compensation = self.effector.compensate(action, now)
                effects += self._absorb(compensation, job.bundle, now)
                if compensation.get("ok", False):
                    status = ExecStatus.COMPENSATED
                else:
                    reason = "compensation_failed"
        return self._issue(job.bundle, status, reason, job.submitted_at, started_at=job.started_at,
                           finished_at=now, side_effects=effects, postconditions_met=post)

    def _absorb(self, result, bundle, now):
        action = bundle.action
        effects = tuple(str(e) for e in result.get("side_effects", ()))
        self.side_effects.extend((now, action.action_id, e) for e in effects)
        stamp = {"action_id": str(action.action_id), "idempotency_key": str(bundle.idempotency_key)}
        for item in result.get("evidence", ()):
            self.evidence.add(replace(item, payload={**item.payload, **stamp}))
        return effects

    def _schedule_reviews(self, job, now):
        ttl = self.policy.ttl(job.bundle.action)
        if ttl is not None:
            self._timers.append(_Timer(self._next_seq(), "ttl_expired", now + ttl, job))
        review = job.params.get(REVIEW_ARG)
        if review is not None:
            self._timers.append(_Timer(self._next_seq(), "review_due", now + float(review), job))

    def _raise_review(self, timer, now):
        self._timers.remove(timer)
        bundle = timer.job.bundle
        action = bundle.action
        item = sign_record(ReviewItem(review_id=f"{action.action_id}:{timer.kind}", kind=timer.kind,
                                      decision_id=str(bundle.decision_id), action_id=action.action_id,
                                      idempotency_key=bundle.idempotency_key, tool_id=action.tool_id,
                                      target_id=action.target_id, due_at=timer.due_at, raised_at=now), self._key)
        self.reviews.append(item)
        return item

    def _clock(self, now):
        now = float(now)
        if math.isnan(now) or now < self.now:
            raise ValueError(f"time must not go backwards ({now} after {self.now})")
        self.now = now
        return now

    def _next_seq(self):
        self._seq += 1
        return self._seq

    def _issue(self, bundle, status, reason, submitted_at, **extra):
        receipt = Receipt(decision_id=str(bundle.decision_id), action_id=str(getattr(bundle.action, "action_id", "")),
                          idempotency_key=bundle.idempotency_key, status=status, reason=reason,
                          submitted_at=submitted_at, **extra)
        receipt = sign_record(receipt, self._key)
        self.receipts.append(receipt)
        return receipt
