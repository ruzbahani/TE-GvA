import hashlib
import warnings

import numpy as np

from cpsdefense.data.splits import run_bounds
from cpsdefense.eval.metrics import (
    UndefinedMetricWarning,
    alarm_segments,
    average_precision,
    cluster_bootstrap_ci,
    duration_aware_false_alarms,
    event_metrics,
    normal_alarm_time,
    precision_recall_f1,
    roc_auc,
    strict_event_metrics,
    threshold_for_false_alarm_rate,
)

IQR_TO_STD = 1.3489795003921634
NAN = float("nan")
MIN_CI_CLUSTERS = 10


def array_digest(X):
    X = np.ascontiguousarray(X)
    digest = hashlib.sha256(f"{X.dtype.str}{X.shape}".encode())
    digest.update(memoryview(X).cast("B"))
    return digest.hexdigest()


class FeatureScaler:
    def __init__(self, method="standard", std_floor=0.1):
        if method not in ("standard", "robust"):
            raise ValueError(f"unknown scaling method {method!r}")
        if not 0 < std_floor < np.inf:
            raise ValueError(f"std_floor must be positive and finite, got {std_floor}")
        self.method = method
        self.std_floor = float(std_floor)

    def fit(self, X):
        X = np.asarray(X, np.float32)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError(f"need a 2-D array with at least 2 rows, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("training features must be finite")
        std = X.std(axis=0, dtype=np.float64)
        floored = iqr_zero = np.zeros(X.shape[1], bool)
        if self.method == "standard":
            center, scale = X.mean(axis=0, dtype=np.float64), std.copy()
        else:
            q25, center, q75 = np.percentile(X, [25, 50, 75], axis=0).astype(np.float64)
            iqr_scale = (q75 - q25) / IQR_TO_STD
            floor = self.std_floor * std
            iqr_zero, floored = iqr_scale <= 0, iqr_scale < floor
            scale = np.maximum(iqr_scale, floor)
        self.iqr_zero = np.flatnonzero(iqr_zero).tolist()
        self.std_floored = np.flatnonzero(floored).tolist()
        self.fallback = np.flatnonzero(scale <= 0).tolist()
        self.center = center
        self.scale = np.where(scale > 0, scale, 1.0)
        self.n_fit = int(X.shape[0])
        self.fit_sha256 = array_digest(X)
        return self

    def transform(self, X):
        return ((np.asarray(X, np.float64) - self.center) / self.scale).astype(np.float32)

    def describe(self):
        out = {"method": self.method, "n_fit": self.n_fit, "fit_sha256": self.fit_sha256,
               "n_features": int(self.center.size), "unit_scale_fallback": self.fallback}
        if self.method == "robust":
            out.update({"scale_rule": "max(IQR / 1.349, std_floor * std)", "std_floor": self.std_floor,
                        "std_floored": self.std_floored, "iqr_zero": self.iqr_zero})
        return out


def topk_mean(values, k):
    values = np.asarray(values)
    f = values.shape[1]
    k = f if k == "all" else int(k)
    if k < 1:
        raise ValueError("k must be >= 1 or 'all'")
    if k >= f:
        return values.mean(axis=1, dtype=np.float64)
    return np.partition(values, f - k, axis=1)[:, f - k:].mean(axis=1, dtype=np.float64)


def segment_positions(segment):
    starts, stops = run_bounds(segment)
    return np.arange(len(segment)) - np.repeat(starts, stops - starts)


def apply_warmup(scores, segment, warmup):
    out = np.array(scores, dtype=np.float64)
    out[segment_positions(segment) < warmup] = NAN
    return out


def _scored_bounds(scores, segment):
    out = []
    for a, b in zip(*run_bounds(segment)):
        scored = ~np.isnan(scores[a:b])
        first = int(np.argmax(scored)) if scored.any() else b - a
        if not scored[first:].all():
            raise ValueError("unscored rows (NaN) are only allowed at the start of a segment")
        if first < b - a:
            out.append((a + first, b))
    return out


def smooth(scores, segment, window):
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    out = np.array(scores, dtype=np.float64)
    if window == 1:
        return out
    for a, b in _scored_bounds(out, segment):
        c = np.r_[0.0, np.cumsum(out[a:b])]
        i = np.arange(1, b - a + 1)
        lo = np.maximum(i - window, 0)
        out[a:b] = (c[i] - c[lo]) / (i - lo)
    return out


def segment_series(scores, segment):
    return [scores[a:b] for a, b in _scored_bounds(scores, segment)]


def alarm_time_threshold(parts, max_fraction):
    s = np.sort(np.concatenate(parts))[::-1]
    k = int(np.floor(max_fraction * s.size + 1e-9))
    if k >= s.size:
        return float(s[-1])
    return float(np.nextafter(s[k], np.inf))


def normal_thresholds(parts, targets, period_s=1.0, merge_gap=0, max_false_alarm_s=None):
    if not parts or sum(p.size for p in parts) == 0:
        raise ValueError("no normal scores")
    out = {}
    for target in targets:
        thr = threshold_for_false_alarm_rate(parts, target, period_s, merge_gap)
        if max_false_alarm_s is not None:
            thr = max(thr, alarm_time_threshold(parts, target * max_false_alarm_s / 3600.0))
        out[float(target)] = thr
    return out


def fit_thresholds(val_scores, val_segment, targets, period_s=1.0, merge_gap=0, max_false_alarm_s=None):
    parts = segment_series(np.asarray(val_scores, np.float64), val_segment)
    return normal_thresholds(parts, targets, period_s, merge_gap, max_false_alarm_s)


def covered_by_events(labels, grace_n):
    y = np.asarray(labels).astype(bool)
    d = np.diff(np.r_[0, y.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    ends = np.minimum(ends + grace_n, np.r_[starts[1:], y.size])
    delta = np.zeros(y.size + 1, np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return np.cumsum(delta[:-1]) > 0


def normal_series(scores, labels, segment, grace_s=0.0, period_s=1.0):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels).astype(bool)
    grace_n = int(np.ceil(grace_s / period_s - 1e-9))
    out = []
    for a, b in _scored_bounds(scores, segment):
        normal = ~covered_by_events(labels[a:b], grace_n)
        starts, stops = run_bounds(normal)
        out += [scores[a + i:a + j] for i, j in zip(starts, stops) if normal[i]]
    return out


def _joined(parts, dtype):
    return np.concatenate(parts) if parts else np.zeros(0, dtype)


def _delay_summary(delays, detected):
    found = delays[detected]
    return (float(found.mean()) if found.size else NAN), (float(np.median(found)) if found.size else NAN)


def alarm_report(scores, labels, segment, threshold, period_s=1.0, merge_gap=0, grace_s=0.0,
                 pre_tolerance_s=60.0, fa_unit_s=60.0):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels).astype(bool)
    grace_n = int(np.ceil(grace_s / period_s - 1e-9))
    lenient, delays, strict, strict_delays, pre_alarmed, offsets = [], [], [], [], [], []
    n_false, n_false_dur, false_alarm_s, normal_alarm_s = 0, 0, 0.0, 0.0
    normal_rows, n_alarm, n_rows = 0, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for a, b in _scored_bounds(scores, segment):
            alarms, y = scores[a:b] >= threshold, labels[a:b]
            ev = event_metrics(alarms, y, period_s, grace_s)
            lenient.append(ev["detected"])
            delays.append(ev["delays_s"])
            st = strict_event_metrics(alarms, y, period_s, grace_s, pre_tolerance_s, merge_gap)
            strict.append(st["detected"])
            strict_delays.append(st["delays_s"])
            pre_alarmed.append(st["pre_alarmed"])
            offsets.append(st["onset_offset_s"])
            dur = duration_aware_false_alarms(alarms, y, period_s, grace_s, fa_unit_s, pre_tolerance_s, merge_gap)
            n_false_dur += dur["n_false_alarms"]
            false_alarm_s += dur["false_alarm_s"]
            normal_alarm_s += normal_alarm_time(alarms, y, period_s, grace_s)["normal_alarm_s"]
            covered = np.r_[0, np.cumsum(covered_by_events(y, grace_n))]
            runs = alarm_segments(alarms, merge_gap)
            n_false += int(np.sum(covered[runs[:, 1]] == covered[runs[:, 0]]))
            normal_rows += int(b - a - covered[-1])
            n_alarm += int(alarms.sum())
            n_rows += b - a
    detected, delays = _joined(lenient, bool), _joined(delays, float)
    strict, strict_delays = _joined(strict, bool), _joined(strict_delays, float)
    normal_hours = normal_rows * period_s / 3600.0
    mean_delay, median_delay = _delay_summary(delays, detected)
    strict_mean, strict_median = _delay_summary(strict_delays, strict)
    return {
        "threshold": float(threshold),
        "n_events": int(detected.size),
        "n_detected": int(detected.sum()),
        "event_recall": float(detected.mean()) if detected.size else NAN,
        "mean_delay_s": mean_delay,
        "median_delay_s": median_delay,
        "n_false_alarms": n_false,
        "normal_hours": normal_hours,
        "fa_per_hour": n_false / normal_hours if normal_hours else NAN,
        "alarm_fraction": n_alarm / n_rows if n_rows else NAN,
        "n_strict_detected": int(strict.sum()),
        "strict_event_recall": float(strict.mean()) if strict.size else NAN,
        "n_pre_alarmed": int(_joined(pre_alarmed, bool).sum()),
        "strict_mean_delay_s": strict_mean,
        "strict_median_delay_s": strict_median,
        "n_false_alarms_duration": n_false_dur,
        "fa_per_hour_duration": n_false_dur / normal_hours if normal_hours else NAN,
        "false_alarm_s": false_alarm_s,
        "normal_alarm_s": normal_alarm_s,
        "normal_alarm_fraction": normal_alarm_s / (normal_rows * period_s) if normal_rows else NAN,
        "pre_tolerance_s": float(pre_tolerance_s),
        "fa_unit_s": float(fa_unit_s),
        "detected": detected,
        "delays_s": delays,
        "strict_detected": strict,
        "strict_delays_s": strict_delays,
        "pre_alarmed": _joined(pre_alarmed, bool),
        "onset_offset_s": _joined(offsets, float),
    }


def time_blocks(segment, block_rows):
    starts, stops = run_bounds(segment)
    seg_index = np.repeat(np.arange(starts.size), stops - starts)
    block = segment_positions(segment) // int(block_rows)
    return seg_index * (int(block.max()) + 1 if block.size else 1) + block


def _point_recall(y, alarms):
    return float(alarms[y].mean()) if y.any() else NAN


def _mean(x):
    return float(x.mean()) if x.size else NAN


def _nanmean(x):
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else NAN


def _seed_mean(metric, labelled):
    if labelled:
        return lambda y, *xs: float(np.mean([metric(y, x) for x in xs]))
    return lambda *xs: float(np.mean([metric(x) for x in xs]))


def _interval(stat, arrays, clusters, boot, min_clusters):
    k = int(np.unique(clusters).size)
    if k < min_clusters:
        return {"point": float(stat(*arrays)), "lo": NAN, "hi": NAN, "n_clusters": k, "n_boot": 0, "n_valid": 0}
    return cluster_bootstrap_ci(stat, arrays, clusters, **boot)


def _intervals(scores, labels, segment, thresholds, alarm_args, boot, block_s, min_clusters):
    scores = [np.asarray(s, np.float64) for s in scores]
    labels = np.asarray(labels).astype(bool)
    valid = ~np.isnan(scores[0])
    if any(not np.array_equal(~np.isnan(s), valid) for s in scores[1:]):
        raise ValueError("every seed must score the same rows")
    y, seg = labels[valid], np.asarray(segment)[valid]
    s = [x[valid] for x in scores]
    blocks = time_blocks(seg, max(1, int(round(block_s / alarm_args["period_s"]))))
    cis = {"n_blocks": int(np.unique(blocks).size),
           "average_precision": _interval(_seed_mean(average_precision, True), [y, *s], blocks, boot, min_clusters),
           "roc_auc": _interval(_seed_mean(roc_auc, True), [y, *s], blocks, boot, min_clusters),
           "targets": {}}
    reports = [{} for _ in scores]
    for target in thresholds[0]:
        thr = [t[target] for t in thresholds]
        reps = [alarm_report(x, labels, segment, t, **alarm_args) for x, t in zip(scores, thr)]
        out = {"point_recall": _interval(_seed_mean(_point_recall, True), [y, *(x >= t for x, t in zip(s, thr))],
                                         blocks, boot, min_clusters)}
        for name, det_key, delay_key in (("", "detected", "delays_s"),
                                         ("strict_", "strict_detected", "strict_delays_s")):
            det = [r[det_key] for r in reps]
            out[f"{name}event_recall"] = _interval(_seed_mean(_mean, False), [d.astype(float) for d in det],
                                                   np.arange(det[0].size), boot, min_clusters)
            found = np.any(det, axis=0)
            out[f"{name}mean_delay_s"] = _interval(_seed_mean(_nanmean, False), [r[delay_key][found] for r in reps],
                                                   np.arange(int(found.sum())), boot, min_clusters)
        cis["targets"][str(target)] = out
        for report, rep in zip(reports, reps):
            report[str(target)] = rep
    return cis, reports


def _alarm_args(period_s, merge_gap, grace_s, pre_tolerance_s, fa_unit_s):
    return {"period_s": period_s, "merge_gap": merge_gap, "grace_s": grace_s, "pre_tolerance_s": pre_tolerance_s,
            "fa_unit_s": fa_unit_s}


def evaluate(scores, labels, segment, thresholds, period_s=1.0, merge_gap=0, grace_s=0.0, pre_tolerance_s=60.0,
             fa_unit_s=60.0, block_s=3600, n_boot=1000, alpha=0.05, seed=0, min_clusters=MIN_CI_CLUSTERS):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels).astype(bool)
    valid = ~np.isnan(scores)
    y, s = labels[valid], scores[valid]
    boot = {"n_boot": n_boot, "alpha": alpha, "seed": seed}
    args = _alarm_args(period_s, merge_gap, grace_s, pre_tolerance_s, fa_unit_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        cis, (reports,) = _intervals([scores], labels, segment, [thresholds], args, boot, block_s, min_clusters)
        out = {
            "rows_scored": int(valid.sum()),
            "rows_unscored": int((~valid).sum()),
            "labelled_rows_unscored": int(labels[~valid].sum()),
            "label_fraction": float(y.mean()) if y.size else NAN,
            "n_blocks": cis["n_blocks"],
            "average_precision": cis["average_precision"],
            "roc_auc": cis["roc_auc"],
            "targets": {},
        }
        for target, thr in thresholds.items():
            rep = dict(reports[str(target)])
            arrays = {key: rep.pop(key) for key in ("detected", "delays_s", "strict_detected", "strict_delays_s",
                                                    "pre_alarmed", "onset_offset_s")}
            rep["point_precision"] = precision_recall_f1(y, s, thr)["precision"]
            rep.update(cis["targets"][str(target)])
            rep["event_detected"] = arrays["detected"].tolist()
            rep["event_delays_s"] = arrays["delays_s"].tolist()
            rep["event_strict_detected"] = arrays["strict_detected"].tolist()
            rep["event_strict_delays_s"] = arrays["strict_delays_s"].tolist()
            rep["event_pre_alarmed"] = arrays["pre_alarmed"].tolist()
            rep["event_onset_offset_s"] = arrays["onset_offset_s"].tolist()
            out["targets"][str(target)] = rep
    return out


def seed_mean_intervals(scores_per_seed, labels, segment, thresholds_per_seed, period_s=1.0, merge_gap=0,
                        grace_s=0.0, pre_tolerance_s=60.0, fa_unit_s=60.0, block_s=3600, n_boot=1000, alpha=0.05,
                        seed=0, min_clusters=MIN_CI_CLUSTERS):
    boot = {"n_boot": n_boot, "alpha": alpha, "seed": seed}
    args = _alarm_args(period_s, merge_gap, grace_s, pre_tolerance_s, fa_unit_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        cis, _ = _intervals(scores_per_seed, labels, segment, thresholds_per_seed, args, boot, block_s, min_clusters)
    return {"n_seeds": len(scores_per_seed), **cis}
