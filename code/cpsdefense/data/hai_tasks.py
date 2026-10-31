import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

from cpsdefense.data import hai
from cpsdefense.data.splits import run_bounds
from cpsdefense.paths import PROCESSED_DIR, ensure_dir
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_file
from cpsdefense.utils.seeding import spawn_generators

CONFIG = "data/hai_tasks.yaml"
DEV_SPLITS = ("train", "val", "dev_test")
KINDS = ("window_summary", "cross_source_summary")
STATS = ("first", "last", "mean", "std", "min", "max", "slope")
STREAMS = ("outage", "delay", "lifetime", "retraction", "replay")
CONTRACT_FIELDS = (
    "decision_time", "segment", "item_id", "present", "source", "kind",
    "event_start", "event_time", "arrival_time", "valid_until", "retracted_at", "payload",
)
TIME_FIELDS = ("event_start", "event_time", "arrival_time", "valid_until", "retracted_at")
SEMI_SYNTHETIC_NOTE = (
    "SEMI-SYNTHETIC: payload content and targets are real HAI data; evidence timing (delivery delay, "
    "validity lifetime, Markov outages, retractions with replayed payloads) is synthetic, drawn from "
    "seeded generators."
)


def check_split(split):
    if split not in DEV_SPLITS:
        raise ValueError(f"split {split!r} refused: HAI tasks use only {DEV_SPLITS}; the held-out test split is never used")


def tasks_dir(version, root=None):
    return Path(root or PROCESSED_DIR) / "hai_tasks" / hai.normalize_version(version)


def seed_entropy(seed, version, split):
    check_split(split)
    return [int(seed), int(hai.normalize_version(version).replace(".", "")), hai.SPLIT_CODE[split]]


def group_sources(columns, prefixes):
    groups = {name: [] for name in prefixes}
    for j, column in enumerate(columns):
        hits = [name for name, pre in prefixes.items() if any(column.startswith(p) for p in pre)]
        if len(hits) != 1:
            raise ValueError(f"column {column!r} matches sources {hits}; needs exactly one")
        groups[hits[0]].append(j)
    empty = [name for name, idx in groups.items() if not idx]
    if empty:
        raise ValueError(f"sources without columns: {empty}")
    return groups


def _timing_params(config, name):
    timing = config["timing"]
    params = {**timing["default"], **(timing.get("sources") or {}).get(name, {})}
    if params["outage_mean_windows"] < 1 or not 0 <= params["outage_p_start"] < 1:
        raise ValueError(f"{name}: outage_mean_windows must be >= 1 and outage_p_start in [0, 1)")
    if not 0 <= params["lifetime_jitter"] < 1:
        raise ValueError(f"{name}: lifetime_jitter must be in [0, 1)")
    return params


def resolve_spec(config, version, columns):
    version = hai.normalize_version(version)
    col = {c: j for j, c in enumerate(columns)}
    groups = group_sources(columns, config["sources"])
    stats = list(config["windows"]["stats"])
    unknown = sorted(set(stats) - set(STATS))
    if unknown:
        raise ValueError(f"unknown stats {unknown}; choose from {STATS}")
    window_s = float(config["windows"]["window_s"])
    period = float(config["windows"].get("sample_period_s", 1))
    length = int(round(window_s / period))
    if length < 1 or abs(length * period - window_s) > 1e-9:
        raise ValueError("window_s must be a positive multiple of sample_period_s")
    stride_s = float(config["decisions"]["stride_s"])
    stride_w = int(round(stride_s / window_s))
    if stride_w < 1 or abs(stride_w * window_s - stride_s) > 1e-9:
        raise ValueError("stride_s must be a positive multiple of window_s")
    k = int(config["decisions"]["recent_windows"])
    stale = sorted(int(x) for x in config["decisions"].get("stale_lags") or [])
    if k < 1 or any(x < k for x in stale) or len(set(stale)) != len(stale):
        raise ValueError("need recent_windows >= 1 and distinct stale_lags >= recent_windows")
    sources = list(groups)
    timing = {name: _timing_params(config, name) for name in sources}
    derived = config.get("derived") or {}
    if derived.get("enabled"):
        missing = sorted(set(derived["stats"]) - set(stats))
        if missing:
            raise ValueError(f"derived stats {missing} are not window stats")
    horizon = float(config["targets"]["horizon_s"])
    questions = {}
    for name, q in config["targets"]["questions"].items():
        qv = q.get(version) or q.get("all")
        if qv is None:
            raise ValueError(f"question {name!r} has no definition for {version} and no 'all'")
        if q["source"] not in groups:
            raise ValueError(f"question {name!r}: unknown source {q['source']!r}")
        entry = {"type": qv["type"], "source": q["source"]}
        if qv["type"] == "discrete_change":
            entry["columns"] = list(qv["columns"])
        elif qv["type"] == "hysteresis":
            entry["columns"] = [qv["column"]]
            if "half_width_iqr" in qv:
                entry["half_width_iqr"] = float(qv["half_width_iqr"])
            else:
                entry["lo"], entry["hi"] = float(qv["lo"]), float(qv["hi"])
                if entry["lo"] > entry["hi"]:
                    raise ValueError(f"question {name!r}: lo > hi")
        else:
            raise ValueError(f"question {name!r}: unknown type {qv['type']!r}")
        absent = [c for c in entry["columns"] if c not in col]
        if absent:
            raise ValueError(f"question {name!r}: columns {absent} not in the {version} feature columns")
        entry["column_index"] = [col[c] for c in entry["columns"]]
        questions[name] = entry
    score = dict(config["targets"].get("score") or {})
    if score:
        if score["column"] not in col:
            raise ValueError(f"score column {score['column']!r} not in the feature columns")
        score["column_index"] = col[score["column"]]
        score["lead_rows"] = int(round(float(score["lead_s"]) / period))
    max_lifetime = max(p["lifetime_s"] * (1 + p["lifetime_jitter"]) for p in timing.values())
    return {
        "version": version,
        "columns": list(columns),
        "sources": sources,
        "source_columns": {name: [columns[j] for j in idx] for name, idx in groups.items()},
        "source_index": groups,
        "stats": stats,
        "window_s": window_s,
        "period_s": period,
        "window_len": length,
        "stride_windows": stride_w,
        "lags": list(range(k)) + stale,
        "timing": timing,
        "retraction": dict(config["timing"]["retraction"]),
        "derived": dict(derived) if derived.get("enabled") else None,
        "horizon_s": horizon,
        "questions": questions,
        "score": score or None,
        "scaling": dict(config["scaling"]),
        "stale_lags_always_stale": bool(not stale or min(stale) * window_s >= max_lifetime),
    }


def source_names(spec):
    return spec["sources"] + ([spec["derived"]["name"]] if spec["derived"] else [])


def payload_dims(spec):
    dims = [len(spec["source_index"][s]) * len(spec["stats"]) for s in spec["sources"]]
    if spec["derived"]:
        dims.append(len(spec["sources"]) * len(spec["derived"]["stats"]))
    return dims


def payload_layout(spec):
    out = {s: [f"{stat}:{c}" for stat in spec["stats"] for c in spec["source_columns"][s]] for s in spec["sources"]}
    if spec["derived"]:
        out[spec["derived"]["name"]] = [f"rms_{stat}:{s}" for stat in spec["derived"]["stats"] for s in spec["sources"]]
    return out


def windows(t, segment, window_len, period_s=1.0):
    t = np.asarray(t, np.float64)
    segment = np.asarray(segment)
    same = segment[1:] == segment[:-1]
    steps = np.diff(t)[same]
    if steps.size and not np.all(steps == period_s):
        bad = np.unique(steps[steps != period_s])[:5]
        raise ValueError(f"cadence inside segments is not {period_s} s (found steps {bad.tolist()})")
    row0, seg, k, first, seg_stop = [], [], [], [], []
    count = 0
    for a, b in zip(*run_bounds(segment)):
        n = (b - a) // window_len
        row0.append(a + np.arange(n) * window_len)
        seg.append(np.full(n, segment[a]))
        k.append(np.arange(n))
        first.append(np.full(n, count))
        seg_stop.append(np.full(n, b))
        count += n
    cat = lambda parts: np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)
    return {"row0": cat(row0), "seg": cat(seg), "k": cat(k), "first": cat(first), "seg_stop": cat(seg_stop)}


def window_stats(X, row0, window_len, stats):
    X = np.asarray(X)
    block = X[row0[:, None] + np.arange(window_len)].astype(np.float64)
    mean = block.mean(axis=1)
    out = []
    for stat in stats:
        if stat == "first":
            out.append(block[:, 0])
        elif stat == "last":
            out.append(block[:, -1])
        elif stat == "mean":
            out.append(mean)
        elif stat == "std":
            out.append(block.std(axis=1))
        elif stat == "min":
            out.append(block.min(axis=1))
        elif stat == "max":
            out.append(block.max(axis=1))
        elif stat == "slope":
            i = np.arange(window_len) - (window_len - 1) / 2
            denom = float(np.sum(i * i))
            out.append(np.einsum("l,wlf->wf", i, block - mean[:, None]) / denom if denom else np.zeros_like(mean))
        else:
            raise ValueError(f"unknown stat {stat!r}")
    return np.concatenate(out, axis=1)


def fit_scaler(raw, floor):
    raw = np.asarray(raw, np.float64)
    scale = np.maximum(raw.std(axis=0), np.asarray(floor, np.float64))
    scale[~(scale > 0)] = 1.0
    return {"mean": raw.mean(axis=0).tolist(), "scale": scale.tolist()}


def apply_scaler(raw, scaler, clip):
    z = (np.asarray(raw, np.float64) - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    n_clipped = int(np.count_nonzero(np.abs(z) > clip))
    return np.clip(z, -clip, clip).astype(np.float32), n_clipped


def derived_raw(blocks, spec):
    feats = []
    for stat in spec["derived"]["stats"]:
        k = spec["stats"].index(stat)
        for s, name in enumerate(spec["sources"]):
            f = len(spec["source_index"][name])
            part = np.asarray(blocks[s][:, k * f:(k + 1) * f], np.float64)
            feats.append(np.sqrt(np.mean(part * part, axis=1)))
    return np.stack(feats, axis=1)


def markov_outage(u, first, p_start, p_end):
    pi_down = p_start / (p_start + p_end) if p_start > 0 else 0.0
    down = np.zeros(u.size, bool)
    state = False
    for i in range(u.size):
        if first[i]:
            state = u[i] < pi_down
        elif state:
            state = u[i] >= p_end
        else:
            state = u[i] < p_start
        down[i] = state
    return down


def source_timing(event_time, win, params, retraction, rngs, prefix):
    n = event_time.size
    first = win["k"] == 0
    down = markov_outage(rngs[f"{prefix}/outage"].random(n), first,
                         params["outage_p_start"], 1.0 / params["outage_mean_windows"])
    delay = rngs[f"{prefix}/delay"].lognormal(np.log(params["delay_median_s"]), params["delay_sigma"], n)
    jitter = rngs[f"{prefix}/lifetime"].uniform(-1.0, 1.0, n)
    lifetime = params["lifetime_s"] * (1.0 + params["lifetime_jitter"] * jitter)
    r = rngs[f"{prefix}/retraction"]
    retract = r.random(n) < retraction["rate"]
    retract_delay = r.lognormal(np.log(retraction["delay_median_s"]), retraction["delay_sigma"], n)
    lo, hi = retraction.get("replay_lag_windows", [1, 1])
    lag = rngs[f"{prefix}/replay"].integers(int(lo), int(hi) + 1, n)
    arrival = event_time + delay
    idx = np.arange(n)
    replay_from = np.maximum(idx - lag, win["first"])
    retract &= ~down
    replay = retract & bool(retraction.get("replay_payload")) & (replay_from < idx)
    return {
        "delivered": ~down,
        "arrival_time": arrival,
        "valid_until": event_time + lifetime,
        "retracted_at": np.where(retract, arrival + retract_delay, np.inf),
        "replay_from": np.where(replay, replay_from, -1),
    }


def hysteresis_state(x, lo, hi, segment):
    x = np.asarray(x, np.float64)
    n = x.size
    mark = np.full(n, -1, np.int8)
    mark[x <= lo] = 0
    mark[x >= hi] = 1
    starts, _ = run_bounds(segment)
    undecided = starts[mark[starts] < 0]
    mark[undecided] = x[undecided] >= (lo + hi) / 2
    pos = np.where(mark >= 0, np.arange(n), 0)
    np.maximum.accumulate(pos, out=pos)
    return mark[pos]


def change_rows(values, segment):
    v = np.asarray(values).reshape(len(values), -1)
    segment = np.asarray(segment)
    diff = np.any(v[1:] != v[:-1], axis=1) & (segment[1:] == segment[:-1])
    return np.flatnonzero(diff) + 1


def fit_question(X, q):
    out = {k: q[k] for k in ("type", "source", "columns")}
    if q["type"] == "hysteresis":
        x = np.asarray(X[:, q["column_index"][0]], np.float64)
        if "half_width_iqr" in q:
            median = float(np.median(x))
            q25, q75 = np.percentile(x, [25, 75])
            half = q["half_width_iqr"] * float(q75 - q25)
            out.update(median=median, iqr=float(q75 - q25), half_width_iqr=q["half_width_iqr"],
                       lo=median - half, hi=median + half)
        else:
            out.update(lo=q["lo"], hi=q["hi"])
    else:
        values = np.unique(np.asarray(X[:, q["column_index"]], np.float64), axis=0)
        out["codebook"] = values.tolist()
    return out


def question_series(X, segment, q, fitted):
    if q["type"] == "hysteresis":
        state = hysteresis_state(X[:, q["column_index"][0]], fitted["lo"], fitted["hi"], segment)
        return state.astype(np.int64), change_rows(state, segment)
    values = np.asarray(X[:, q["column_index"]], np.float64)
    uniq, inverse = np.unique(values, axis=0, return_inverse=True)
    book = {tuple(row): i for i, row in enumerate(fitted["codebook"])}
    codes = np.array([book.get(tuple(row), -1) for row in uniq.tolist()], np.int64)
    return codes[inverse.reshape(-1)], change_rows(values, segment)


def event_targets(t, event_rows, tau_rows, seg_start, seg_stop, horizon):
    t = np.asarray(t, np.float64)
    tau = t[tau_rows]
    censor = np.minimum(horizon, t[seg_stop - 1] - tau)
    time = censor.copy()
    observed = np.zeros(tau_rows.size, bool)
    since = np.full(tau_rows.size, np.nan)
    if event_rows.size:
        pos = np.searchsorted(event_rows, tau_rows, side="right")
        nxt = event_rows[np.minimum(pos, event_rows.size - 1)]
        gap = t[nxt] - tau
        observed = (pos < event_rows.size) & (nxt < seg_stop) & (gap <= horizon)
        time = np.where(observed, gap, censor)
        prev = event_rows[np.maximum(pos - 1, 0)]
        has_prev = (pos > 0) & (prev >= seg_start)
        since = np.where(has_prev, tau - t[prev], np.nan)
    return time, observed, since


def score_bins(x, edges):
    return np.searchsorted(np.asarray(edges, np.float64), np.asarray(x, np.float64), side="right").astype(np.int64)


def fit_params(train, spec):
    X, seg = train["X"], train["segment"]
    t = hai.to_seconds(train["timestamp"])
    win = windows(t, seg, spec["window_len"], spec["period_s"])
    n_stats = len(spec["stats"])
    floor_frac = float(spec["scaling"]["floor_frac"])
    scalers, blocks = {}, []
    for name in spec["sources"]:
        cols = spec["source_index"][name]
        raw = window_stats(X[:, cols], win["row0"], spec["window_len"], spec["stats"])
        feature_std = np.asarray(X[:, cols], np.float64).std(axis=0)
        scalers[name] = fit_scaler(raw, floor_frac * np.tile(feature_std, n_stats))
        blocks.append(apply_scaler(raw, scalers[name], spec["scaling"]["clip"])[0])
    if spec["derived"]:
        raw = derived_raw(blocks, spec)
        scalers[spec["derived"]["name"]] = fit_scaler(raw, np.full(raw.shape[1], floor_frac))
    params = {
        "scalers": scalers,
        "questions": {name: fit_question(X, q) for name, q in spec["questions"].items()},
        "score": None,
    }
    if spec["score"]:
        x = np.asarray(X[:, spec["score"]["column_index"]], np.float64)
        n_bins = int(spec["score"]["n_bins"])
        params["score"] = {"column": spec["score"]["column"],
                           "edges": np.quantile(x, np.arange(1, n_bins) / n_bins).tolist()}
    return params


def decision_windows(win, spec):
    max_lag = max(spec["lags"])
    k = win["k"]
    ok = (k >= max_lag) & ((k - max_lag) % spec["stride_windows"] == 0)
    ok &= win["row0"] + spec["window_len"] < win["seg_stop"]
    return np.flatnonzero(ok)


def build_split(data, spec, params, entropy):
    X, seg, y = data["X"], np.asarray(data["segment"]), np.asarray(data["y"])
    t = hai.to_seconds(data["timestamp"])
    if list(data["columns"]) != spec["columns"]:
        raise ValueError("split columns differ from the spec's columns")
    L = spec["window_len"]
    win = windows(t, seg, L, spec["period_s"])
    W = win["row0"].size
    names = source_names(spec)
    S = len(spec["sources"])
    dims = payload_dims(spec)
    P = max(dims)
    n_items = len(names) * W
    ev_start = t[win["row0"]]
    ev_end = t[win["row0"] + L - 1]
    rngs = spawn_generators(entropy, [f"{s}/{k}" for s in spec["sources"] for k in STREAMS] + ["derived/delay"])

    delivered = np.zeros(n_items, bool)
    arrival = np.full(n_items, np.nan)
    valid_until = np.full(n_items, np.nan)
    retracted = np.full(n_items, np.nan)
    replay_from = np.full(n_items, -1, np.int64)
    parents = np.full((n_items, S), -1, np.int64)
    table = np.zeros((n_items, P), np.float32)
    clipped, blocks = {}, []
    for s, name in enumerate(spec["sources"]):
        cols = spec["source_index"][name]
        raw = window_stats(X[:, cols], win["row0"], L, spec["stats"])
        tim = source_timing(ev_end, win, spec["timing"][name], spec["retraction"], rngs, name)
        rep = tim["replay_from"] >= 0
        raw[rep] = raw[tim["replay_from"][rep]].copy()
        z, clipped[name] = apply_scaler(raw, params["scalers"][name], spec["scaling"]["clip"])
        sl = slice(s * W, (s + 1) * W)
        ok = tim["delivered"]
        delivered[sl] = ok
        arrival[sl] = tim["arrival_time"]
        valid_until[sl] = tim["valid_until"]
        retracted[sl] = tim["retracted_at"]
        replay_from[sl] = np.where(rep, s * W + tim["replay_from"], -1)
        table[sl, :dims[s]] = np.where(ok[:, None], z, 0.0)
        blocks.append(z)
    if spec["derived"]:
        d = spec["derived"]
        sl = slice(S * W, (S + 1) * W)
        inputs = np.arange(S)[None, :] * W + np.arange(W)[:, None]
        ok = delivered[inputs].all(axis=1)
        delay = rngs["derived/delay"].lognormal(np.log(d["delay_median_s"]), d["delay_sigma"], W)
        z, clipped[d["name"]] = apply_scaler(derived_raw(blocks, spec), params["scalers"][d["name"]],
                                             spec["scaling"]["clip"])
        delivered[sl] = ok
        arrival[sl] = arrival[inputs].max(axis=1) + delay
        valid_until[sl] = valid_until[inputs].min(axis=1)
        retracted[sl] = retracted[inputs].min(axis=1)
        parents[sl] = inputs
        table[sl, :dims[S]] = np.where(ok[:, None], z, 0.0)
    item_source = np.repeat(np.arange(len(names)), W)
    item_kind = np.where(item_source < S, 0, 1).astype(np.int64)
    item_start = np.tile(ev_start, len(names))
    item_end = np.tile(ev_end, len(names))
    for field in (arrival, valid_until, retracted):
        field[~delivered] = np.nan

    dec = decision_windows(win, spec)
    lags = np.asarray(spec["lags"], np.int64)
    slot_source = np.repeat(np.arange(len(names)), lags.size)
    slot_lag = np.tile(lags, len(names))
    idx = slot_source[None, :] * W + (dec[:, None] - slot_lag[None, :])
    present = delivered[idx]
    tau_rows = win["row0"][dec] + L - 1
    tau = t[tau_rows]

    def gather(values):
        return np.where(present, values[idx], np.nan)

    arrays = {
        "decision_time": tau,
        "segment": seg[tau_rows].astype(np.int64),
        "item_id": np.where(present, idx, -1).astype(np.int64),
        "present": present,
        "source": np.where(present, item_source[idx], -1).astype(np.int64),
        "kind": np.where(present, item_kind[idx], -1).astype(np.int64),
        "event_start": gather(item_start),
        "event_time": gather(item_end),
        "arrival_time": gather(arrival),
        "valid_until": gather(valid_until),
        "retracted_at": gather(retracted),
        "payload": table[idx],
    }
    if np.any(arrays["event_time"][present] > np.broadcast_to(tau[:, None], present.shape)[present]):
        raise AssertionError("an evidence item contains samples after its decision time")

    starts, stops = run_bounds(seg)
    seg_start = np.repeat(starts, stops - starts)[tau_rows]
    seg_stop = np.repeat(stops, stops - starts)[tau_rows]
    H = spec["horizon_s"]
    q_names = list(spec["questions"])
    tte = np.zeros((dec.size, len(q_names)))
    observed = np.zeros((dec.size, len(q_names)), bool)
    state = np.zeros((dec.size, len(q_names)), np.int64)
    since = np.zeros((dec.size, len(q_names)))
    q_report = {}
    for i, name in enumerate(q_names):
        codes, events = question_series(X, seg, spec["questions"][name], params["questions"][name])
        tte[:, i], observed[:, i], since[:, i] = event_targets(t, events, tau_rows, seg_start, seg_stop, H)
        state[:, i] = codes[tau_rows]
        censored = ~observed[:, i]
        by_end = censored & (tte[:, i] < H)
        q_report[name] = {
            "events_in_split": int(events.size),
            "events_per_hour": float(events.size / (len(t) * spec["period_s"] / 3600)),
            "decisions": int(dec.size),
            "observed_rate": float(observed[:, i].mean()) if dec.size else None,
            "censoring_rate": float(censored.mean()) if dec.size else None,
            "censored_at_horizon": float((censored & ~by_end).mean()) if dec.size else None,
            "censored_at_segment_end": float(by_end.mean()) if dec.size else None,
            "median_time_to_event_observed_s": float(np.median(tte[observed[:, i], i])) if observed[:, i].any() else None,
            "state_unknown_at_tau": int(np.sum(state[:, i] < 0)),
        }
    arrays.update(time_to_event=tte, event_observed=observed, state_at_tau=state, time_since_event=since)
    score_report = None
    if spec["score"]:
        x = X[:, spec["score"]["column_index"]]
        ahead = tau_rows + spec["score"]["lead_rows"]
        ok = ahead < seg_stop
        arrays["score_bin"] = np.where(ok, score_bins(x[np.minimum(ahead, len(x) - 1)], params["score"]["edges"]), -1)
        arrays["score_bin_now"] = score_bins(x[tau_rows], params["score"]["edges"])
        n_bins = len(params["score"]["edges"]) + 1
        counts = np.bincount(arrays["score_bin"][ok], minlength=n_bins)
        score_report = {"missing": int((~ok).sum()), "class_fraction": (counts / max(1, ok.sum())).tolist(),
                        "persistence_accuracy": float(np.mean(arrays["score_bin"][ok] == arrays["score_bin_now"][ok]))
                        if ok.any() else None}
    cum = np.r_[0, np.cumsum(y != 0)]
    end = np.minimum(tau_rows + 1 + int(round(H / spec["period_s"])), seg_stop)
    arrays["tau_row"] = tau_rows.astype(np.int64)
    arrays["attack_at_tau"] = (y[tau_rows] != 0).astype(np.int8)
    arrays["attack_in_horizon"] = (cum[end] - cum[tau_rows + 1] > 0).astype(np.int8)
    arrays.update(
        items_delivered=delivered, items_source=item_source.astype(np.int64), items_kind=item_kind,
        items_event_start=item_start, items_event_time=item_end, items_arrival_time=arrival,
        items_valid_until=valid_until, items_retracted_at=retracted, items_replayed_from=replay_from,
        items_parents=parents,
    )
    report = split_report(arrays, spec, slot_source, slot_lag, W, clipped)
    report["questions"] = q_report
    report["score"] = score_report
    return arrays, report


def valid_at(tensors, t):
    t = np.broadcast_to(np.asarray(t, np.float64), tensors["decision_time"].shape)[:, None]
    return (tensors["present"] & (tensors["arrival_time"] <= t) & (t < tensors["valid_until"])
            & (t < tensors["retracted_at"]))


def split_report(arrays, spec, slot_source, slot_lag, n_windows, clipped):
    names = source_names(spec)
    present = arrays["present"]
    tau = arrays["decision_time"][:, None]
    valid = valid_at(arrays, arrays["decision_time"])
    n_present = max(1, int(present.sum()))
    invalid = present & ~valid

    def frac(mask, where=None):
        where = present if where is None else where
        return float(mask[where].mean()) if where.any() else None

    by_source = {}
    for s, name in enumerate(names):
        cols = slot_source == s
        items = arrays["items_source"] == s
        by_source[name] = {
            "slots_present": frac(present[:, cols], np.ones_like(present[:, cols])),
            "invalid_at_tau": frac(invalid[:, cols], present[:, cols]),
            "valid_per_decision": float(valid[:, cols].sum(axis=1).mean()) if len(valid) else None,
            "items_delivered": int(arrays["items_delivered"][items].sum()),
            "items_dropped": int((~arrays["items_delivered"][items]).sum()),
            "items_retracted": int(np.isfinite(arrays["items_retracted_at"][items]).sum()),
            "items_replayed": int((arrays["items_replayed_from"][items] >= 0).sum()),
            "clipped_values": int(clipped.get(name, 0)),
        }
    by_lag = {int(lag): frac(invalid[:, slot_lag == lag], present[:, slot_lag == lag]) for lag in np.unique(slot_lag)}
    return {
        "n_decisions": int(present.shape[0]),
        "M": int(present.shape[1]),
        "P": int(arrays["payload"].shape[2]),
        "n_windows": int(n_windows),
        "item_ids_used": int(np.unique(arrays["item_id"][present]).size),
        "slots_present": float(present.mean()) if present.size else None,
        "invalid_at_tau": float(invalid.sum() / n_present),
        "not_arrived_at_tau": frac(arrays["arrival_time"] > tau),
        "expired_at_tau": frac(arrays["valid_until"] <= tau),
        "retracted_at_tau": frac(arrays["retracted_at"] <= tau),
        "valid_per_decision": {"mean": float(valid.sum(1).mean()), "min": int(valid.sum(1).min()),
                               "max": int(valid.sum(1).max())} if len(valid) else None,
        "invalid_at_tau_by_lag": by_lag,
        "sources": by_source,
    }


def discrete_columns(X, columns, segment, spec, max_levels):
    hours = len(X) * spec["period_s"] / 3600
    out = {name: [] for name in spec["sources"]}
    for name in spec["sources"]:
        for j in spec["source_index"][name]:
            values = np.unique(X[:, j])
            if 2 <= values.size <= max_levels:
                n = int(change_rows(X[:, j], segment).size)
                out[name].append({"column": columns[j], "levels": int(values.size),
                                  "changes_train": n, "changes_per_hour_train": n / hours})
    return out


def _save(path, arrays, compress):
    tmp = path.with_name(path.stem + ".tmp.npz")
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def prepare_version(version, config=None, out_root=None, hai_root=None, verbose=True):
    version = hai.normalize_version(version)
    config = load_config(CONFIG) if config is None else config
    splits = list(config.get("splits") or DEV_SPLITS)
    for split in splits:
        check_split(split)
    out_dir = ensure_dir(tasks_dir(version, out_root))
    meta_path = out_dir / "meta.json"
    meta_path.unlink(missing_ok=True)
    hai_meta = hai.load_meta(version, hai_root)
    train = hai.load_split(version, "train", root=hai_root)
    spec = resolve_spec(config, version, train["columns"])
    params = fit_params(train, spec)
    discovery = discrete_columns(train["X"], train["columns"], train["segment"], spec,
                                 int((config.get("discovery") or {}).get("max_levels", 20)))
    compress = bool((config.get("save") or {}).get("compress", True))
    reports, entropies = {}, {}
    for split in splits:
        data = train if split == "train" else hai.load_split(version, split, root=hai_root)
        entropies[split] = seed_entropy(config.get("seed", 0), version, split)
        arrays, report = build_split(data, spec, params, entropies[split])
        path = out_dir / f"{split}.npz"
        _save(path, arrays, compress)
        report["file"] = path.name
        report["file_bytes"] = path.stat().st_size
        reports[split] = report
        if verbose:
            print(f"[hai_tasks {version}] {split}: N={report['n_decisions']} M={report['M']} P={report['P']} "
                  f"invalid@tau={report['invalid_at_tau']:.3f} file={report['file_bytes'] / 1e6:.0f} MB", flush=True)
        del arrays, data
    minimum = config.get("min_events") or {}
    warnings = [
        f"{q}: {r['questions'][q]['events_in_split']} events in {split} < {minimum[split]}"
        for split, r in reports.items() if split in minimum
        for q in r["questions"] if r["questions"][q]["events_in_split"] < minimum[split]
    ]
    names = source_names(spec)
    meta = {
        "dataset": "HAI tasks",
        "version": version,
        "semi_synthetic": True,
        "semi_synthetic_note": SEMI_SYNTHETIC_NOTE,
        "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "seed": config.get("seed", 0),
        "seed_entropy": entropies,
        "splits": splits,
        "heldout_test_used": False,
        "hai_meta_prepared_utc": hai_meta.get("prepared_utc"),
        "code_sha256": {"hai_tasks.py": sha256_file(Path(__file__)), "hai.py": sha256_file(Path(hai.__file__))},
        "contract_fields": list(CONTRACT_FIELDS),
        "sources": names,
        "kinds": list(KINDS),
        "payload_dim": payload_dims(spec),
        "payload_layout": payload_layout(spec),
        "source_columns": spec["source_columns"],
        "stats": spec["stats"],
        "window_s": spec["window_s"],
        "stride_s": spec["stride_windows"] * spec["window_s"],
        "lags": spec["lags"],
        "slot_source": np.repeat(np.arange(len(names)), len(spec["lags"])).tolist(),
        "slot_lag": np.tile(spec["lags"], len(names)).tolist(),
        "stale_lags_always_stale": spec["stale_lags_always_stale"],
        "timing": spec["timing"],
        "retraction": spec["retraction"],
        "derived": spec["derived"],
        "horizon_s": spec["horizon_s"],
        "questions": params["questions"],
        "question_names": list(spec["questions"]),
        "score": {**{k: v for k, v in spec["score"].items() if k != "column_index"}, **params["score"]}
        if spec["score"] else None,
        "scalers": params["scalers"],
        "discrete_columns_train": discovery,
        "extra_fields": {
            "time_to_event": "float64 [N, Q] seconds from tau to the next event (> 0), or the censoring time",
            "event_observed": "bool [N, Q]; false = right-censored at horizon_s or at the segment end",
            "state_at_tau": "int64 [N, Q] question state at tau from raw data (hysteresis 0/1, discrete codebook "
                            "index, -1 unseen in train); undelayed baseline covariate, not evidence",
            "time_since_event": "float64 [N, Q] seconds since the last event at or before tau in the segment "
                                "(NaN if none); raw data, baseline covariate, not evidence",
            "score_bin": "int64 [N] train-quantile bin of the score column lead_s after tau (-1 past the segment end)",
            "score_bin_now": "int64 [N] the same bin at tau (persistence baseline)",
            "tau_row": "int64 [N] row of tau in hai.load_split(version, split)",
            "attack_at_tau": "int8 [N] attack label at tau (analysis only)",
            "attack_in_horizon": "int8 [N] any attack label in (tau, tau + horizon] (analysis only)",
            "items_*": "per item id (source-major: id = source * n_windows + window): delivered, source, kind, "
                       "times, replayed_from (item id whose payload a retracted item carries, -1 none), "
                       "parents (derived items' input item ids, -1 for window items)",
        },
        "reports": reports,
        "warnings": warnings,
    }
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, meta_path)
    return meta


def load_meta(version, root=None):
    path = tasks_dir(version, root) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_hai_tasks.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tasks(version, split, root=None, fields=None):
    check_split(split)
    path = tasks_dir(version, root) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_hai_tasks.py")
    with np.load(path) as z:
        return {k: z[k] for k in (fields or z.files)}
