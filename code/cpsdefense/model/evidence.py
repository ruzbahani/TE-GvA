from dataclasses import dataclass, replace

import numpy as np
import torch

TIME_FIELDS = ("event_start", "event_time", "arrival_time", "valid_until", "retracted_at")
SLOT_FIELDS = ("item_id", "present", "source", "kind", *TIME_FIELDS, "payload")
ROW_FIELDS = ("decision_time", "segment")
FIELDS = (*ROW_FIELDS, *SLOT_FIELDS)
NUMPY_DTYPES = {
    "decision_time": np.float64, "segment": np.int64, "item_id": np.int64, "present": np.bool_,
    "source": np.int64, "kind": np.int64, **{name: np.float64 for name in TIME_FIELDS}, "payload": np.float32,
}
TORCH_DTYPES = {name: torch.from_numpy(np.zeros(0, dtype)).dtype for name, dtype in NUMPY_DTYPES.items()}
PAD_VALUES = {
    "item_id": -1, "present": False, "source": 0, "kind": 0, "event_start": np.nan, "event_time": np.nan,
    "arrival_time": np.inf, "valid_until": -np.inf, "retracted_at": -np.inf, "payload": 0.0,
}


def validity(present, arrival_time, valid_until, retracted_at, t):
    return present & (arrival_time <= t) & (t < valid_until) & (t < retracted_at)


def derived_validity(arrival_time, valid_until, retracted_at, axis=-1):
    if isinstance(arrival_time, torch.Tensor):
        return arrival_time.amax(axis), valid_until.amin(axis), retracted_at.amin(axis)
    return np.max(arrival_time, axis), np.min(valid_until, axis), np.min(retracted_at, axis)


def split_rows(values, mask):
    if mask.shape[0] == 0:
        return []
    if isinstance(values, torch.Tensor):
        values, mask = values.cpu(), mask.cpu()
        return list(torch.split(values[mask], mask.sum(1).tolist()))
    return np.split(values[mask], np.cumsum(mask.sum(1))[:-1])


def _where(mask, x, value):
    if isinstance(x, torch.Tensor):
        return torch.where(mask, x, torch.as_tensor(value, dtype=x.dtype, device=x.device))
    return np.where(mask, x, np.asarray(value, dtype=x.dtype))


@dataclass(eq=False)
class EvidenceBatch:
    decision_time: object
    segment: object
    item_id: object
    present: object
    source: object
    kind: object
    event_start: object
    event_time: object
    arrival_time: object
    valid_until: object
    retracted_at: object
    payload: object

    @classmethod
    def from_arrays(cls, decision_time, item_id, source, event_time, arrival_time, valid_until, payload,
                    present=None, segment=None, kind=None, event_start=None, retracted_at=None):
        item_id = np.asarray(item_id, np.int64)
        values = {
            "decision_time": decision_time, "item_id": item_id, "source": source, "event_time": event_time,
            "arrival_time": arrival_time, "valid_until": valid_until, "payload": payload,
            "present": item_id >= 0 if present is None else present,
            "segment": np.zeros(item_id.shape[0]) if segment is None else segment,
            "kind": np.zeros(item_id.shape) if kind is None else kind,
            "event_start": event_time if event_start is None else event_start,
            "retracted_at": np.full(item_id.shape, np.inf) if retracted_at is None else retracted_at,
        }
        values = {name: np.array(v, dtype=NUMPY_DTYPES[name]) for name, v in values.items()}
        batch = cls(**values)
        batch.check()
        return batch.pad_absent()

    @property
    def is_torch(self):
        return isinstance(self.item_id, torch.Tensor)

    @property
    def n(self):
        return self.item_id.shape[0]

    @property
    def m(self):
        return self.item_id.shape[1]

    @property
    def p(self):
        return self.payload.shape[2]

    @property
    def device(self):
        return self.item_id.device if self.is_torch else None

    def fields(self):
        return {name: getattr(self, name) for name in FIELDS}

    def replace(self, **changes):
        return replace(self, **changes)

    def check(self, payload_dim=None):
        if self.item_id.ndim != 2 or self.payload.ndim != 3:
            raise ValueError("item_id must be [N, M] and payload [N, M, P]")
        n, m = self.item_id.shape
        for name, value in self.fields().items():
            shape = (n,) if name in ROW_FIELDS else (n, m, self.payload.shape[2]) if name == "payload" else (n, m)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
            expected = TORCH_DTYPES[name] if self.is_torch else NUMPY_DTYPES[name]
            if value.dtype != expected:
                raise ValueError(f"{name} has dtype {value.dtype}, expected {expected}")
        ids, present = self._numpy("item_id"), self._numpy("present")
        for row in range(n):
            row_ids = ids[row][present[row]]
            if np.unique(row_ids).size != row_ids.size:
                raise ValueError(f"decision {row}: duplicate item ids")
        if payload_dim is not None:
            payload, source = self._numpy("payload"), self._numpy("source")
            for s, dim in enumerate(payload_dim):
                rows = present & (source == s)
                if np.any(payload[rows][:, dim:] != 0):
                    raise ValueError(f"source {s}: payload beyond its prefix of {dim} is not zero")
        return self

    def _numpy(self, name):
        value = getattr(self, name)
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value

    def to(self, device=None):
        out = {}
        for name, value in self.fields().items():
            dtype = TORCH_DTYPES[name]
            out[name] = (value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor)
                         else torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device))
        return EvidenceBatch(**out)

    def numpy(self):
        return EvidenceBatch(**{name: self._numpy(name) for name in FIELDS})

    def time_column(self, t=None):
        t = self.decision_time if t is None else t
        if self.is_torch:
            t = torch.as_tensor(t, dtype=torch.float64, device=self.device)
            t = t.expand(self.n) if t.ndim == 0 else t
        else:
            t = np.asarray(t, dtype=np.float64)
            t = np.full(self.n, t) if t.ndim == 0 else t
        if tuple(t.shape) == (self.n, 1):
            return t
        if tuple(t.shape) != (self.n,):
            raise ValueError(f"time has shape {tuple(t.shape)}, expected a scalar, ({self.n},) or ({self.n}, 1)")
        return t[:, None]

    def validity(self, t=None):
        return validity(self.present, self.arrival_time, self.valid_until, self.retracted_at, self.time_column(t))

    def known(self, t=None):
        return self.present & (self.arrival_time <= self.time_column(t))

    def dependency_set(self, t=None):
        return split_rows(self.item_id, self.validity(t))

    def revalidate(self, dependency_ids, t_exec):
        if len(dependency_ids) != self.n:
            raise ValueError(f"{len(dependency_ids)} dependency sets for {self.n} decisions")
        valid = self.validity(t_exec)
        out = []
        for row, deps in enumerate(dependency_ids):
            ok = self.item_id[row][valid[row]]
            if self.is_torch:
                deps = torch.as_tensor(deps, dtype=torch.int64, device=self.device).reshape(-1)
                out.append(deps[~torch.isin(deps, ok)])
            else:
                deps = np.asarray(deps, dtype=np.int64).reshape(-1)
                out.append(deps[~np.isin(deps, ok)])
        return out

    def pad_absent(self):
        present = self.present
        out = {}
        for name in SLOT_FIELDS:
            value = getattr(self, name)
            mask = present[..., None] if name == "payload" else present
            out[name] = _where(mask, value, PAD_VALUES[name])
        return self.replace(**out)

    def take(self, rows):
        return EvidenceBatch(**{name: value[rows] for name, value in self.fields().items()})

    def permute_slots(self, order):
        rows = (torch.arange(self.n, device=self.device) if self.is_torch else np.arange(self.n))[:, None]
        return self.replace(**{name: getattr(self, name)[rows, order] for name in SLOT_FIELDS})

    def compact(self, keep=None):
        keep = self.present if keep is None else keep
        if self.is_torch:
            order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices
        else:
            order = np.argsort((~keep).astype(np.int8), axis=1, kind="stable")
        m = max(int(keep.sum(1).max()), 1) if self.n else 1
        kept = self.permute_slots(order)
        rows = (torch.arange(self.n, device=self.device) if self.is_torch else np.arange(self.n))[:, None]
        mask = keep[rows, order][:, :m]
        out = {name: getattr(kept, name)[:, :m] for name in SLOT_FIELDS}
        out["present"] = out["present"] & mask
        return self.replace(**out).pad_absent()

    def pad_to(self, m):
        if m < self.m:
            raise ValueError(f"cannot pad {self.m} slots down to {m}")
        extra = m - self.m
        out = {}
        for name in SLOT_FIELDS:
            value = getattr(self, name)
            shape = (self.n, extra, *value.shape[2:])
            if self.is_torch:
                pad = torch.full(shape, PAD_VALUES[name], dtype=value.dtype, device=value.device)
                out[name] = torch.cat([value, pad], 1)
            else:
                out[name] = np.concatenate([value, np.full(shape, PAD_VALUES[name], dtype=value.dtype)], 1)
        return self.replace(**out)


def concat(batches):
    m = max(b.m for b in batches)
    batches = [b.pad_to(m) for b in batches]
    join = torch.cat if batches[0].is_torch else np.concatenate
    return EvidenceBatch(**{name: join([getattr(b, name) for b in batches], 0) for name in FIELDS})
