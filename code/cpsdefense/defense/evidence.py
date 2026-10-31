import math

DERIVED_KEY = "derived_from"


def parents_of(item):
    parents = (item.payload or {}).get(DERIVED_KEY, ())
    if not isinstance(parents, (list, tuple)) or not all(isinstance(p, str) for p in parents):
        raise ValueError(f"{item.evidence_id}: payload[{DERIVED_KEY!r}] must be a list of evidence ids")
    return tuple(parents)


def _time(t):
    t = float(t)
    return -math.inf if math.isnan(t) else t


class EvidenceLog:
    def __init__(self, items=()):
        self._items = {}
        self._parents = {}
        self._retracted = {}
        self._superseded = {}
        for item in items:
            self.add(item)

    def __len__(self):
        return len(self._items)

    def __contains__(self, evidence_id):
        return isinstance(evidence_id, str) and evidence_id in self._items

    def __iter__(self):
        return iter(self._items)

    def add(self, item):
        eid = item.evidence_id
        if not isinstance(eid, str) or not eid:
            raise ValueError(f"evidence id must be a non-empty string, got {eid!r}")
        if eid in self._items:
            raise ValueError(f"duplicate evidence id {eid!r}")
        parents = parents_of(item)
        missing = [p for p in parents if p not in self._items]
        if missing:
            raise ValueError(f"{eid}: derived from unknown items {missing}")
        if eid in item.supersedes:
            raise ValueError(f"{eid}: an item cannot supersede itself")
        self._items[eid] = item
        self._parents[eid] = parents
        for old in item.supersedes:
            self._superseded[old] = min(self._superseded.get(old, math.inf), _time(item.arrival_time))
        return item

    def get(self, evidence_id):
        return self._items.get(evidence_id) if isinstance(evidence_id, str) else None

    def retract(self, evidence_id, t):
        t = float(t)
        if math.isnan(t):
            raise ValueError("retraction time is NaN")
        self._retracted[evidence_id] = min(self._retracted.get(evidence_id, math.inf), t)

    def retracted_at(self, evidence_id):
        t = min(self._retracted.get(evidence_id, math.inf), self._superseded.get(evidence_id, math.inf))
        for parent in self._parents.get(evidence_id, ()):
            t = min(t, self.retracted_at(parent))
        return t

    def known(self, evidence_id, t):
        item = self.get(evidence_id)
        return item is not None and item.arrival_time <= t

    def valid(self, evidence_id, t):
        item = self.get(evidence_id)
        if item is None:
            return False
        if not (item.arrival_time <= t < item.valid_until and t < self.retracted_at(evidence_id)):
            return False
        return all(self.valid(parent, t) for parent in self._parents[evidence_id])

    def valid_ids(self, t):
        return tuple(eid for eid in self._items if self.valid(eid, t))

    def invalid_reason(self, evidence_ids, t):
        ids = tuple(evidence_ids)
        if not all(self.known(eid, t) for eid in ids):
            return "unknown_evidence"
        if not all(self.valid(eid, t) for eid in ids):
            return "stale_evidence"
        return None
