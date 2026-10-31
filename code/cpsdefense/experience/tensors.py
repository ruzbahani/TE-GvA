import json
import math
import numbers
from pathlib import Path

import numpy as np

from cpsdefense.core.types import ActionFamily, SideEffectClass
from cpsdefense.defense.evidence import EvidenceLog, parents_of
from cpsdefense.defense.registry import TARGET_KINDS, canonical_args
from cpsdefense.model.evidence import FIELDS, NUMPY_DTYPES, PAD_VALUES, TIME_FIELDS, EvidenceBatch, validity
from cpsdefense.sim.tep_telemetry import STATS
from cpsdefense.utils.config import load_config

CONFIG = "experience/tensors.yaml"
FORMAT = "tep-evidence-tensors/2"
GROUP_FAMILIES = {"window_summary": "window", "historian": "historian", "field_check": "field_check"}
SHARED_KINDS = ("alert", "link_health", "trace_link", "operator_report", "receipt")
RECEIPT_STATUSES = ("accepted", "rejected", "running", "succeeded", "failed", "timed_out", "duplicate", "compensated")
KINDS = (*GROUP_FAMILIES, *SHARED_KINDS)
UNSCALED_STATS = ("frac_unchanged",)
IQR_TO_STD = 1.349
SCALAR_COLUMNS = ("log1p_duration_median_s", "log1p_duration_q90_s", "resource_cost_rel")


def build_vocab(env, config=None):
    from cpsdefense.sim.tep_env import TICKET

    opts = (config or load_config(CONFIG))["vocab"]
    plant = env.plant
    groups = list(plant.sensor_links)
    names = list(plant.measurement_names)
    members = {g: [names[i] for i in plant.sensor_links[g]] for g in groups}
    links = list(plant.links)
    causes = [str(c) for c in env.config["causes"]]
    vocab = {
        "format": FORMAT, "groups": groups, "group_measurements": members, "measurements": names,
        "links": links, "causes": causes, "stats": list(STATS), "kinds": list(KINDS),
        "options": {k: bool(opts[k]) for k in ("alert_measurement", "field_check_difference")},
        "sources": [], "source_kind": [], "source_group": [], "payload_dim": [], "payload_layout": {},
        "payload_scaled": {}, "subject_blocks": {},
    }

    def add(source, kind, group, layout, scaled, blocks=()):
        layout, scaled, out = list(layout), list(scaled), []
        for key, ids in blocks:
            out.append({"key": key, "offset": len(layout), "ids": list(ids)})
            layout += [f"{key}={i}" for i in ids]
            scaled += [False] * len(ids)
        vocab["sources"].append(source)
        vocab["source_kind"].append(kind)
        vocab["source_group"].append(group)
        vocab["payload_dim"].append(len(layout))
        vocab["payload_layout"][source] = layout
        vocab["payload_scaled"][source] = scaled
        if out:
            vocab["subject_blocks"][source] = out

    for g in groups:
        add(f"window:{g}", "window_summary", g, [f"{m}:{s}" for m in members[g] for s in STATS],
            [s not in UNSCALED_STATS for m in members[g] for s in STATS])
    for g in groups:
        add(f"historian:{g}", "historian", g, [f"{m}:delivered" for m in members[g]], [True] * len(members[g]))
    parts = ["field", "published"] + (["published_minus_field"] if vocab["options"]["field_check_difference"] else [])
    for g in groups:
        add(f"field_check:{g}", "field_check", g, [f"{m}:{p}" for p in parts for m in members[g]],
            [True] * (len(parts) * len(members[g])))
    add("alert", "alert", None, ["stale", "deviation", "score"], [False, False, True],
        [("subject", [f"source:{g}" for g in groups])]
        + ([("measurement", names)] if vocab["options"]["alert_measurement"] else []))
    add("link_health", "link_health", None, ["loss", "backup"], [False, False], [("subject", links)])
    add("trace_link", "trace_link", None, ["healthy", "backup"], [False, False], [("subject", links)])
    add("operator_report", "operator_report", None, [f"cause={c}" for c in causes], [False] * len(causes))
    targets = {"link": links, "source": [f"source:{g}" for g in groups], "sensor": names, "plant": ["plant"],
               "incident": [TICKET]}
    targets = {k: list(targets[k]) for k in TARGET_KINDS}
    tools = [spec.tool_id for spec in env.registry]
    labels = [f"tool={t}" for t in tools] + [f"status={s}" for s in RECEIPT_STATUSES]
    add("receipt", "receipt", None, labels, [False] * len(labels),
        [("subject", [t for k in TARGET_KINDS for t in targets[k]])])
    vocab["receipt_statuses"] = list(RECEIPT_STATUSES)
    vocab["P"] = max(vocab["payload_dim"])
    vocab.update(_action_vocab(env.registry, targets))
    return vocab


def _action_vocab(registry, targets):
    families = [f.value for f in ActionFamily]
    classes = [c.value for c in SideEffectClass]
    tools = [spec.tool_id for spec in registry]
    columns, blocks, args = [], {}, []

    def block(name, labels):
        blocks[name] = [len(columns), len(columns) + len(labels)]
        columns.extend(f"{name}:{x}" for x in labels)

    block("tool", tools)
    block("family", families)
    block("target_kind", TARGET_KINDS)
    block("target", [t for k in TARGET_KINDS for t in targets[k]])
    for spec in registry:
        for a in spec.args:
            numeric = all(isinstance(c, numbers.Real) and not isinstance(c, bool) for c in a.choices)
            if a.kind == "enum" and not numeric:
                for c in a.choices:
                    args.append({"column": len(columns), "tool": spec.tool_id, "name": a.name, "choice": c})
                    columns.append(f"arg:{spec.tool_id}.{a.name}={c}")
                continue
            if a.kind == "bool":
                low, high = 0.0, 1.0
            elif a.kind == "enum":
                low, high = float(min(a.choices)), float(max(a.choices))
            else:
                low, high = float(a.low), float(a.high)
            args.append({"column": len(columns), "tool": spec.tool_id, "name": a.name, "low": low, "high": high})
            columns.append(f"arg:{spec.tool_id}.{a.name}")
    columns.extend(SCALAR_COLUMNS)
    block("side_effect", classes)
    return {
        "registry_version": registry.version, "targets": targets, "tools": tools, "families": families,
        "target_kinds": list(TARGET_KINDS), "side_effect_classes": classes, "action_blocks": blocks,
        "action_args": args, "max_resource_cost": float(max(spec.resource_cost for spec in registry)),
        "action_columns": columns, "F": len(columns),
    }


def save_vocab(vocab, path):
    path = Path(path)
    path.write_text(json.dumps(vocab, indent=1), encoding="utf-8")
    return path


def load_vocab(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ItemIds:
    def __init__(self, keys=()):
        self._keys = []
        self._index = {}
        for segment, evidence_id in keys:
            self.id(segment, evidence_id)

    def __len__(self):
        return len(self._keys)

    def id(self, segment, evidence_id):
        key = (int(segment), str(evidence_id))
        item_id = self._index.get(key)
        if item_id is None:
            item_id = self._index[key] = len(self._keys)
            self._keys.append(key)
        return item_id

    def find(self, segment, evidence_id):
        return self._index.get((int(segment), str(evidence_id)), -1)

    def key(self, item_id):
        item_id = int(item_id)
        if not 0 <= item_id < len(self._keys):
            raise KeyError(f"unknown item id {item_id}")
        return self._keys[item_id]

    def evidence_ids(self, item_ids):
        return [self.key(i)[1] for i in np.asarray(item_ids, np.int64).reshape(-1) if i >= 0]

    def to_dict(self):
        return {"keys": [list(k) for k in self._keys]}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(k) for k in d["keys"])


def item_source(item, vocab):
    if item.kind in GROUP_FAMILIES:
        group = item.subject_id[7:] if item.subject_id.startswith("source:") else None
        if group not in vocab["groups"]:
            raise ValueError(f"{item.evidence_id}: subject {item.subject_id!r} is not a sensor group source")
        return f"{GROUP_FAMILIES[item.kind]}:{group}"
    if item.kind in SHARED_KINDS:
        return item.kind
    raise ValueError(f"{item.evidence_id}: kind {item.kind!r} is not in the vocabulary")


def item_payload(item, vocab, source=None):
    source = source or item_source(item, vocab)
    values = np.asarray(item.payload["values"], np.float64).reshape(-1)
    if item.kind == "field_check" and vocab["options"]["field_check_difference"]:
        n = values.size // 2
        values = np.concatenate([values, values[n:2 * n] - values[:n]])
    if item.kind == "receipt":
        _check_receipt(item, values, vocab)
    parts = [values]
    for block in vocab["subject_blocks"].get(source, ()):
        key = item.subject_id if block["key"] == "subject" else item.payload.get(block["key"])
        if key not in block["ids"]:
            raise ValueError(f"{item.evidence_id}: {block['key']} {key!r} is not in the vocabulary")
        onehot = np.zeros(len(block["ids"]))
        onehot[block["ids"].index(key)] = 1.0
        parts.append(onehot)
    out = np.concatenate(parts)
    dim = vocab["payload_dim"][vocab["sources"].index(source)]
    if out.size != dim:
        raise ValueError(f"{item.evidence_id}: payload width {out.size}, source {source} expects {dim}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{item.evidence_id}: non-finite payload")
    return out


def _check_receipt(item, values, vocab):
    tools, statuses = vocab["tools"], vocab["receipt_statuses"]
    tool, status = item.payload.get("tool_id"), item.payload.get("status")
    if status not in statuses:
        raise ValueError(f"{item.evidence_id}: receipt status {status!r} is not in the vocabulary")
    expected = np.zeros(len(tools) + len(statuses))
    expected[len(tools) + statuses.index(status)] = 1.0
    if tool in tools:
        expected[tools.index(tool)] = 1.0
    if values.shape != expected.shape or not np.array_equal(values, expected):
        raise ValueError(f"{item.evidence_id}: receipt values are not the one-hots of {tool!r} and {status!r}")


def _items(log_items):
    if isinstance(log_items, EvidenceLog):
        return [log_items.get(eid) for eid in log_items], log_items.retracted_at
    return list(log_items), None


def _withdrawal(t):
    t = float(t)
    return -math.inf if math.isnan(t) else t


def item_times(items, retracted_at=None):
    pos = {it.evidence_id: i for i, it in enumerate(items)}
    if len(pos) != len(items):
        raise ValueError("duplicate evidence ids")
    if retracted_at is None:
        superseded = {}
        for it in items:
            for old in it.supersedes:
                superseded[old] = min(superseded.get(old, math.inf), _withdrawal(it.arrival_time))
        own = [superseded.get(it.evidence_id, math.inf) for it in items]
    else:
        own = [retracted_at(it.evidence_id) for it in items]
    out = {
        "event_start": np.array([float(it.payload.get("event_start", it.event_time)) for it in items], np.float64),
        "event_time": np.array([it.event_time for it in items], np.float64),
        "arrival_time": np.array([it.arrival_time for it in items], np.float64),
        "valid_until": np.array([it.valid_until for it in items], np.float64),
        "retracted_at": np.array(own, np.float64),
    }
    arrival, until, retracted = out["arrival_time"], out["valid_until"], out["retracted_at"]
    for i, it in enumerate(items):
        for parent in parents_of(it):
            j = pos.get(parent, len(items))
            if j >= i:
                raise ValueError(f"{it.evidence_id}: parent {parent!r} is not an earlier item")
            arrival[i] = np.maximum(arrival[i], arrival[j])
            until[i] = np.minimum(until[i], until[j])
            retracted[i] = np.minimum(retracted[i], retracted[j])
    return out


def _select(tau, times, source, sel, unknown=None, drop=None):
    arrival, event_time = times["arrival_time"], times["event_time"]
    valid = (arrival <= tau) & (tau < times["valid_until"]) & (tau < times["retracted_at"])
    keep = (arrival <= tau + float(sel["lookahead_s"])) & ((event_time >= tau - float(sel["context_s"])) | valid)
    if unknown is not None:
        keep &= ~(valid & unknown)
    if drop is not None:
        keep &= ~drop
    idx = np.flatnonzero(keep)
    cap = int(sel["max_items"])
    if idx.size > cap:
        s = idx[np.lexsort((-idx, -event_time[idx], -arrival[idx], source[idx]))]
        first = np.r_[True, source[s][1:] != source[s][:-1]]
        rank = np.arange(s.size) - np.maximum.accumulate(np.where(first, np.arange(s.size), 0))
        idx = s[np.lexsort((-s, -arrival[s], rank))[:cap]]
    return idx[np.lexsort((idx, arrival[idx]))]


def _with_derived(items, evidence_ids):
    out = np.zeros(len(items), bool)
    if not evidence_ids:
        return out
    pos = {it.evidence_id: i for i, it in enumerate(items)}
    for i, it in enumerate(items):
        out[i] = it.evidence_id in evidence_ids or any(out[pos[p]] for p in parents_of(it) if p in pos)
    return out


def _iter_items(items):
    for x in [items] if isinstance(items, EvidenceLog) else items:
        if isinstance(x, EvidenceLog):
            yield from (x.get(eid) for eid in x)
        else:
            yield x


def fit_payload_scaler(items, vocab, config=None):
    cfg = (config or load_config(CONFIG))["scaling"]
    sources, cap, min_count = vocab["sources"], int(cfg["max_rows"]), int(cfg["min_count"])
    rows = {s: [] for s in sources}
    stride = {s: 1 for s in sources}
    seen = {s: 0 for s in sources}
    for item in _iter_items(items):
        name = item_source(item, vocab)
        if seen[name] % stride[name] == 0:
            rows[name].append(item_payload(item, vocab, name))
            if len(rows[name]) >= 2 * cap:
                rows[name] = rows[name][::2]
                stride[name] *= 2
        seen[name] += 1
    fitted = {}
    for s, name in enumerate(sources):
        dim = vocab["payload_dim"][s]
        scaled = np.array(vocab["payload_scaled"][name], bool)
        data = np.array(rows[name], np.float64).reshape(-1, dim)
        center, scale = np.zeros(dim), np.ones(dim)
        origin = np.where(scaled, "unfitted", "fixed").astype(object)
        if len(data) >= min_count and scaled.any():
            q25, median, q75 = np.percentile(data[:, scaled], [25, 50, 75], axis=0)
            sc = np.maximum((q75 - q25) / IQR_TO_STD, float(cfg["floor_frac"]) * data[:, scaled].std(0))
            sc[~(sc > float(cfg["min_scale"]))] = 1.0
            center[scaled], scale[scaled], origin[scaled] = median, sc, "fitted"
        fitted[name] = {"count": int(seen[name]), "center": center, "scale": scale, "origin": origin}
    for s, name in enumerate(sources):
        group, entry = vocab["source_group"][s], fitted[name]
        if vocab["source_kind"][s] not in ("historian", "field_check") or entry["count"] >= min_count:
            continue
        ref_name = f"window:{group}"
        ref, ref_layout = fitted[ref_name], vocab["payload_layout"][ref_name]
        if ref["count"] < min_count:
            continue
        for j, label in enumerate(vocab["payload_layout"][name]):
            measurement, part = label.rsplit(":", 1)
            k = ref_layout.index(f"{measurement}:last")
            entry["center"][j] = 0.0 if part == "published_minus_field" else ref["center"][k]
            entry["scale"][j] = ref["scale"][k]
            entry["origin"][j] = f"{ref_name}/{measurement}:last"
    return {
        "format": FORMAT, "clip": float(cfg["clip"]),
        "sources": {name: {"count": e["count"], "center": e["center"].tolist(), "scale": e["scale"].tolist(),
                           "origin": [str(o) for o in e["origin"]]} for name, e in fitted.items()},
    }


def scale_payload(raw, source, vocab, scaler):
    entry = scaler["sources"][source]
    z = (np.asarray(raw, np.float64) - np.asarray(entry["center"])) / np.asarray(entry["scale"])
    scaled = np.array(vocab["payload_scaled"][source], bool)
    clip = float(scaler["clip"])
    z[..., scaled] = np.clip(z[..., scaled], -clip, clip)
    return z


def evidence_tensors(log_items, decision_times, segment, vocab, config=None, scaler=None, ids=None, known=None,
                     exclude=None):
    sel = (config or load_config(CONFIG))["selection"]
    ids = ItemIds() if ids is None else ids
    items, retracted_at = _items(log_items)
    times = item_times(items, retracted_at)
    index = {name: s for s, name in enumerate(vocab["sources"])}
    kind_index = {k: i for i, k in enumerate(vocab["kinds"])}
    names = [item_source(it, vocab) for it in items]
    source = np.array([index[n] for n in names], np.int64)
    kind = np.array([kind_index[it.kind] for it in items], np.int64)
    tau = np.asarray(decision_times, np.float64).reshape(-1)
    n = tau.size
    seg = np.broadcast_to(np.asarray(segment, np.int64), (n,)).copy()
    for name, per_decision in (("known", known), ("exclude", exclude)):
        if per_decision is not None and len(per_decision) != n:
            raise ValueError(f"{name} has {len(per_decision)} entries for {n} decisions")
    eids = [it.evidence_id for it in items]
    unknown = [None] * n if known is None else [np.array([e not in ok for e in eids], bool)
                                                for ok in (set(k) for k in known)]
    drop = [None] * n if exclude is None else [_with_derived(items, set(x)) for x in exclude]
    rows = [_select(t, times, source, sel, u, d) for t, u, d in zip(tau, unknown, drop)]
    m = int(sel["max_items"]) if sel.get("fixed_slots", True) else max([1] + [r.size for r in rows])

    used = np.unique(np.concatenate(rows)) if n else np.zeros(0, np.int64)
    table = np.zeros((used.size, vocab["P"]), np.float32)
    for s, name in enumerate(vocab["sources"]):
        members = used[source[used] == s]
        if members.size == 0:
            continue
        raw = np.stack([item_payload(items[i], vocab, name) for i in members])
        if scaler is not None:
            raw = scale_payload(raw, name, vocab, scaler)
        table[np.searchsorted(used, members), :raw.shape[1]] = raw

    out = {"decision_time": tau, "segment": seg}
    for name in ("item_id", "present", "source", "kind", *TIME_FIELDS):
        out[name] = np.full((n, m), PAD_VALUES[name], NUMPY_DTYPES[name])
    out["payload"] = np.zeros((n, m, vocab["P"]), np.float32)
    for r, row in enumerate(rows):
        k = row.size
        out["item_id"][r, :k] = [ids.id(seg[r], items[i].evidence_id) for i in row]
        out["present"][r, :k] = True
        out["source"][r, :k] = source[row]
        out["kind"][r, :k] = kind[row]
        for name in TIME_FIELDS:
            out[name][r, :k] = times[name][row]
        out["payload"][r, :k] = table[np.searchsorted(used, row)]
    valid = validity(out["present"], out["arrival_time"], out["valid_until"], out["retracted_at"], tau[:, None])
    t = np.broadcast_to(tau[:, None], valid.shape)
    future = ~((out["event_start"] <= t) & (out["event_time"] <= t) & (out["arrival_time"] <= t))
    if np.any(valid & future):
        raise AssertionError("an item valid at its decision time holds information from after it")
    out["payload_dim"] = np.asarray(vocab["payload_dim"], np.int64)
    return out


def evidence_batch(arrays):
    return EvidenceBatch(**{name: arrays[name] for name in FIELDS})


def action_features(actions, registry, vocab):
    columns = list(vocab["action_columns"])
    blocks, targets = vocab["action_blocks"], vocab["targets"]
    tool_pos = {t: i for i, t in enumerate(vocab["tools"])}
    target_pos = {t: i for i, t in enumerate(t for k in vocab["target_kinds"] for t in targets[k])}
    col = {c: i for i, c in enumerate(columns)}
    max_cost = float(vocab["max_resource_cost"])
    families, classes = vocab["families"], vocab["side_effect_classes"]
    x = np.zeros((len(actions), len(columns)), np.float32)
    for r, a in enumerate(actions):
        spec = registry.get(a.tool_id)
        if spec is None or a.tool_id not in tool_pos:
            raise ValueError(f"{a.action_id}: tool {a.tool_id!r} is not in both the registry and the vocabulary")
        args = canonical_args(spec, a.bounded_arguments)
        kind = next((k for k in spec.target_kinds if a.target_id in targets.get(k, ())), None)
        if kind is None:
            raise ValueError(f"{a.action_id}: target {a.target_id!r} is not in the vocabulary")
        x[r, blocks["tool"][0] + tool_pos[a.tool_id]] = 1.0
        x[r, blocks["family"][0] + families.index(ActionFamily(a.family).value)] = 1.0
        x[r, blocks["target_kind"][0] + vocab["target_kinds"].index(kind)] = 1.0
        x[r, blocks["target"][0] + target_pos[a.target_id]] = 1.0
        for arg in vocab["action_args"]:
            if arg["tool"] != a.tool_id:
                continue
            value = args[arg["name"]]
            if "choice" in arg:
                x[r, arg["column"]] = float(value == arg["choice"])
            elif arg["high"] > arg["low"]:
                x[r, arg["column"]] = (float(value) - arg["low"]) / (arg["high"] - arg["low"])
        x[r, col[SCALAR_COLUMNS[0]]] = math.log1p(a.expected_duration)
        x[r, col[SCALAR_COLUMNS[1]]] = math.log1p(a.duration_q90)
        x[r, col[SCALAR_COLUMNS[2]]] = a.resource_limit / max_cost if max_cost > 0 else 0.0
        x[r, blocks["side_effect"][0] + classes.index(SideEffectClass(a.side_effect_class).value)] = 1.0
    return x, columns
