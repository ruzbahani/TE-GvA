import warnings

import numpy as np

NAN = float("nan")


class UndefinedMetricWarning(UserWarning):
    pass


def _warn(message):
    warnings.warn(message, UndefinedMetricWarning, stacklevel=3)


def _labels(labels, n=None, name="labels"):
    y = np.asarray(labels).ravel()
    if y.dtype != bool:
        if not ((y == 0) | (y == 1)).all():
            raise ValueError(f"{name} must be binary (0/1 or bool)")
        y = y.astype(bool)
    if n is not None and y.size != n:
        raise ValueError(f"{name} has length {y.size}, expected {n}")
    return y


def _values(values, n=None, name="scores"):
    x = np.asarray(values, dtype=float).ravel()
    if n is not None and x.size != n:
        raise ValueError(f"{name} has length {x.size}, expected {n}")
    if np.isnan(x).any():
        raise ValueError(f"{name} contain NaN")
    return x


def _probs(probs, n):
    p = _values(probs, n, "probs")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probs must lie in [0, 1]")
    return p


def _both_classes(y, what):
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        _warn(f"{what} is undefined when labels contain a single class")
        return False
    return True


def _nonempty(x, what):
    if x.size == 0:
        _warn(f"{what} is undefined for empty input")
        return False
    return True


def _pr_points(y, s):
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), s.size - 1]
    tps = np.cumsum(y)[last]
    fps = last + 1 - tps
    return s[last], tps, fps


def average_precision(labels, scores):
    y = _labels(labels)
    s = _values(scores, y.size)
    if not _both_classes(y, "average precision"):
        return NAN
    _, tps, fps = _pr_points(y, s)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _average_ranks(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    bounds = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1, x.size]
    ranks = np.empty(x.size)
    ranks[order] = np.repeat((bounds[:-1] + 1 + bounds[1:]) / 2.0, np.diff(bounds))
    return ranks


def roc_auc(labels, scores):
    y = _labels(labels)
    s = _values(scores, y.size)
    if not _both_classes(y, "ROC-AUC"):
        return NAN
    n_pos = float(y.sum())
    n_neg = y.size - n_pos
    rank_sum = _average_ranks(s)[y].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _prf(y, pred):
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    if tp + fn == 0:
        _warn("recall and F1 are undefined without positive labels")
        recall = f1 = NAN
    else:
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    precision = tp / (tp + fp) if tp + fp else NAN
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def precision_recall_f1(labels, scores, threshold):
    y = _labels(labels)
    return _prf(y, _values(scores, y.size) >= threshold)


def best_f1(labels, scores):
    y = _labels(labels)
    s = _values(scores, y.size)
    if not _both_classes(y, "best F1"):
        return {"f1": NAN, "threshold": NAN, "precision": NAN, "recall": NAN}
    thresholds, tps, fps = _pr_points(y, s)
    n_pos = tps[-1]
    f1 = 2 * tps / (tps + fps + n_pos)
    k = int(np.argmax(f1))
    return {
        "f1": float(f1[k]),
        "threshold": float(thresholds[k]),
        "precision": float(tps[k] / (tps[k] + fps[k])),
        "recall": float(tps[k] / n_pos),
    }


def point_adjust_f1(labels, scores, threshold):
    y = _labels(labels)
    pred = _values(scores, y.size) >= threshold
    adjusted = pred.copy()
    for start, end in _runs(y):
        if pred[start:end].any():
            adjusted[start:end] = True
    return _prf(y, adjusted)


def _runs(binary):
    d = np.diff(np.r_[0, binary.astype(np.int8), 0])
    return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)])


def _gap(merge_gap):
    if merge_gap < 0 or merge_gap != int(merge_gap):
        raise ValueError("merge_gap must be a non-negative whole number of samples")
    return int(merge_gap)


def alarm_segments(binary, merge_gap=0):
    merge_gap = _gap(merge_gap)
    seg = _runs(_labels(binary, name="alarms"))
    if len(seg) > 1 and merge_gap > 0:
        keep = seg[1:, 0] - seg[:-1, 1] > merge_gap
        seg = np.column_stack([seg[np.r_[True, keep], 0], seg[np.r_[keep, True], 1]])
    return seg


def _check_period(period_s):
    if not 0 < period_s < np.inf:
        raise ValueError(f"period_s must be positive and finite, got {period_s}")


def _grace_samples(grace_s, period_s):
    _check_period(period_s)
    if not 0 <= grace_s < np.inf:
        raise ValueError(f"grace_s must be finite and >= 0, got {grace_s}")
    return int(np.ceil(grace_s / period_s - 1e-9))


def _pre_samples(pre_tolerance_s, period_s):
    _check_period(period_s)
    if not 0 <= pre_tolerance_s < np.inf:
        raise ValueError(f"pre_tolerance_s must be finite and >= 0, got {pre_tolerance_s}")
    return int(np.floor(pre_tolerance_s / period_s + 1e-9))


def _event_windows(y, grace_n):
    events = _runs(y)
    ends = np.minimum(events[:, 1] + grace_n, np.r_[events[1:, 0], y.size])
    return events, ends


def _cover(n, starts, ends):
    delta = np.zeros(n + 1, np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return np.cumsum(delta[:-1]) > 0


def _first_at_or_after(idx, starts, n):
    if idx.size == 0:
        return np.full(len(starts), n)
    pos = np.searchsorted(idx, starts)
    return np.where(pos < idx.size, idx[np.minimum(pos, idx.size - 1)], n)


def _delay_stats(delays, detected):
    found = delays[detected]
    return {
        "mean_delay_s": float(found.mean()) if found.size else NAN,
        "median_delay_s": float(np.median(found)) if found.size else NAN,
    }


def _false_alarms(a, y, period_s, merge_gap, grace_n):
    events, ends = _event_windows(y, grace_n)
    covered = _cover(a.size, events[:, 0], ends)
    c = np.r_[0, np.cumsum(covered)]
    seg = alarm_segments(a, merge_gap)
    count = int(np.sum(c[seg[:, 1]] == c[seg[:, 0]]))
    normal_hours = (a.size - int(covered.sum())) * period_s / 3600.0
    return count, normal_hours


def false_alarms_per_hour(alarms, labels=None, period_s=1.0, merge_gap=0, grace_s=0.0):
    a = _labels(alarms, name="alarms")
    y = np.zeros(a.size, bool) if labels is None else _labels(labels, a.size)
    count, normal_hours = _false_alarms(a, y, period_s, merge_gap, _grace_samples(grace_s, period_s))
    if normal_hours == 0:
        _warn("false-alarm rate is undefined without normal time")
        return NAN
    return count / normal_hours


def event_metrics(alarms, labels, period_s=1.0, grace_s=0.0):
    a = _labels(alarms, name="alarms")
    y = _labels(labels, a.size)
    events, ends = _event_windows(y, _grace_samples(grace_s, period_s))
    first = _first_at_or_after(np.flatnonzero(a), events[:, 0], a.size)
    detected = first < ends
    delays = np.where(detected, (first - events[:, 0]) * period_s, np.nan)
    if len(events):
        recall = float(detected.mean())
    else:
        _warn("event recall is undefined without labelled events")
        recall = NAN
    return {
        "n_events": int(len(events)),
        "n_detected": int(detected.sum()),
        "event_recall": recall,
        **_delay_stats(delays, detected),
        "detected": detected,
        "delays_s": delays,
        "events": events,
    }


def _running_max(x, w):
    if w == 1:
        return x.copy()
    n = x.size
    n_blocks = -(-(n + w - 1) // w)
    z = np.full(n_blocks * w, -np.inf)
    z[w - 1 : w - 1 + n] = x
    blocks = z.reshape(n_blocks, w)
    prefix = np.maximum.accumulate(blocks, axis=1).ravel()
    suffix = np.maximum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    i = np.arange(n)
    return np.maximum(suffix[i], prefix[i + w - 1])


def _segment_counts(parts, levels, merge_gap):
    merge_gap = _gap(merge_gap)
    active, continuing = [], []
    for s in parts:
        if s.size:
            prev = np.r_[-np.inf, _running_max(s, merge_gap + 1)[:-1]]
            active.append(s)
            continuing.append(np.minimum(s, prev))
    active = np.sort(np.concatenate(active))
    continuing = np.sort(np.concatenate(continuing))
    n = active.size
    return (n - np.searchsorted(active, levels)) - (n - np.searchsorted(continuing, levels))


def _series_list(x):
    if isinstance(x, (list, tuple)) and len(x) and np.ndim(x[0]) > 0:
        return [_values(part, name="normal_scores") for part in x]
    return [_values(x, name="normal_scores")]


def threshold_for_false_alarm_rate(normal_scores, target_per_hour, period_s=1.0, merge_gap=0):
    if not target_per_hour >= 0:
        raise ValueError(f"target_per_hour must be >= 0, got {target_per_hour}")
    _check_period(period_s)
    parts = _series_list(normal_scores)
    s = np.concatenate(parts)
    if s.size == 0:
        raise ValueError("normal_scores is empty")
    if not np.isfinite(s).all():
        raise ValueError("normal_scores must be finite")
    levels = np.unique(s)[::-1]
    rate = _segment_counts(parts, levels, merge_gap) / (s.size * period_s / 3600.0)
    over = np.flatnonzero(rate > target_per_hour)
    if over.size == 0:
        return float(levels[-1])
    if over[0] == 0:
        return float(np.nextafter(levels[0], np.inf))
    return float(levels[over[0] - 1])


def recall_at_fa_rate(scores_test, labels_test, normal_scores_val, target_per_hour,
                      period_s=1.0, merge_gap=0, grace_s=0.0):
    thr = threshold_for_false_alarm_rate(normal_scores_val, target_per_hour, period_s, merge_gap)
    y = _labels(labels_test)
    alarms = _values(scores_test, y.size) >= thr
    events = event_metrics(alarms, y, period_s, grace_s)
    point = _prf(y, alarms)
    n_false, normal_hours = _false_alarms(alarms, y, period_s, merge_gap, _grace_samples(grace_s, period_s))
    return {
        "threshold": thr,
        "target_fa_per_hour": float(target_per_hour),
        "test_fa_per_hour": n_false / normal_hours if normal_hours else NAN,
        "n_false_alarms": n_false,
        "event_recall": events["event_recall"],
        "n_events": events["n_events"],
        "n_detected": events["n_detected"],
        "mean_delay_s": events["mean_delay_s"],
        "median_delay_s": events["median_delay_s"],
        "point_precision": point["precision"],
        "point_recall": point["recall"],
        "point_f1": point["f1"],
        "detected": events["detected"],
        "delays_s": events["delays_s"],
    }


def normal_alarm_time(alarms, labels=None, period_s=1.0, grace_s=0.0):
    a = _labels(alarms, name="alarms")
    y = np.zeros(a.size, bool) if labels is None else _labels(labels, a.size)
    events, ends = _event_windows(y, _grace_samples(grace_s, period_s))
    normal = ~_cover(a.size, events[:, 0], ends)
    normal_s = int(normal.sum()) * period_s
    alarm_s = int(np.sum(a & normal)) * period_s
    if normal_s == 0:
        _warn("normal-time alarm fraction is undefined without normal time")
        fraction = NAN
    else:
        fraction = alarm_s / normal_s
    return {"normal_alarm_s": float(alarm_s), "normal_s": float(normal_s), "normal_alarm_fraction": fraction}


def duration_aware_false_alarms(alarms, labels=None, period_s=1.0, grace_s=0.0, unit_s=60.0,
                                pre_tolerance_s=60.0, merge_gap=0):
    a = _labels(alarms, name="alarms")
    y = np.zeros(a.size, bool) if labels is None else _labels(labels, a.size)
    grace_n = _grace_samples(grace_s, period_s)
    pre_n = _pre_samples(pre_tolerance_s, period_s)
    if not 0 < unit_s < np.inf:
        raise ValueError(f"unit_s must be positive and finite, got {unit_s}")
    events, ends = _event_windows(y, grace_n)
    covered = _cover(a.size, events[:, 0], ends)
    excused = _cover(a.size, np.maximum(events[:, 0] - pre_n, 0), ends)
    seg = alarm_segments(a, merge_gap)
    parts = _runs(_cover(a.size, seg[:, 0], seg[:, 1]) & ~excused)
    n_alarms = np.r_[0, np.cumsum(a)]
    parts = parts[n_alarms[parts[:, 1]] > n_alarms[parts[:, 0]]]
    lengths = parts[:, 1] - parts[:, 0]
    count = int(np.maximum(1, np.ceil(lengths * period_s / unit_s - 1e-9)).sum())
    normal_hours = (a.size - int(covered.sum())) * period_s / 3600.0
    if normal_hours == 0:
        _warn("false-alarm rate is undefined without normal time")
        rate = NAN
    else:
        rate = count / normal_hours
    return {
        "n_false_alarms": count,
        "fa_per_hour": rate,
        "false_alarm_s": float(int(lengths.sum()) * period_s),
        "normal_hours": normal_hours,
    }


def strict_event_metrics(alarms, labels, period_s=1.0, grace_s=0.0, pre_tolerance_s=60.0, merge_gap=0):
    a = _labels(alarms, name="alarms")
    y = _labels(labels, a.size)
    events, ends = _event_windows(y, _grace_samples(grace_s, period_s))
    pre_n = _pre_samples(pre_tolerance_s, period_s)
    starts = events[:, 0]
    window_starts = np.maximum(starts - pre_n, 0)
    onsets = alarm_segments(a, merge_gap)[:, 0]
    onset = _first_at_or_after(onsets, window_starts, a.size)
    detected = onset < ends
    lenient = _first_at_or_after(np.flatnonzero(a), starts, a.size) < ends
    pre_alarmed = lenient & ~detected
    covering = np.r_[0, onsets][np.searchsorted(onsets, window_starts)]
    offset = np.where(detected, onset - starts, np.where(pre_alarmed, covering - starts, 0)) * period_s
    delays = np.where(detected, np.maximum(offset, 0.0), np.nan)
    if len(events):
        strict_recall, lenient_recall = float(detected.mean()), float(lenient.mean())
    else:
        _warn("event recall is undefined without labelled events")
        strict_recall = lenient_recall = NAN
    return {
        "n_events": int(len(events)),
        "n_detected": int(detected.sum()),
        "strict_recall": strict_recall,
        "n_lenient_detected": int(lenient.sum()),
        "lenient_recall": lenient_recall,
        "n_pre_alarmed": int(pre_alarmed.sum()),
        **_delay_stats(delays, detected),
        "detected": detected,
        "lenient_detected": lenient,
        "pre_alarmed": pre_alarmed,
        "delays_s": delays,
        "onset_offset_s": np.where(detected | pre_alarmed, offset, np.nan),
        "events": events,
    }


def brier_score(labels, probs):
    y = _labels(labels)
    p = _probs(probs, y.size)
    if not _nonempty(y, "Brier score"):
        return NAN
    return float(np.mean((p - y) ** 2))


def binary_nll(labels, probs, eps=1e-12):
    y = _labels(labels)
    p = np.clip(_probs(probs, y.size), eps, 1 - eps)
    if not _nonempty(y, "NLL"):
        return NAN
    return float(-np.mean(np.where(y, np.log(p), np.log1p(-p))))


def reliability_curve(labels, probs, n_bins=15, strategy="uniform"):
    y = _labels(labels)
    p = _probs(probs, y.size)
    if not _nonempty(y, "reliability curve"):
        empty = np.zeros(0)
        return {"bin": empty.astype(int), "count": empty.astype(int), "mean_prob": empty, "frac_pos": empty}
    if strategy == "uniform":
        b = np.searchsorted(np.arange(1, n_bins) / n_bins, p, side="left")
    elif strategy == "quantile":
        b = np.searchsorted(np.sort(p)[(np.arange(1, n_bins) * p.size) // n_bins], p, side="right")
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    count = np.bincount(b, minlength=n_bins)
    keep = count > 0
    sum_p = np.bincount(b, weights=p, minlength=n_bins)[keep]
    sum_y = np.bincount(b, weights=y.astype(float), minlength=n_bins)[keep]
    return {
        "bin": np.flatnonzero(keep),
        "count": count[keep],
        "mean_prob": sum_p / count[keep],
        "frac_pos": sum_y / count[keep],
    }


def ece(labels, probs, n_bins=15, strategy="uniform"):
    curve = reliability_curve(labels, probs, n_bins, strategy)
    n = curve["count"].sum()
    if n == 0:
        return NAN
    return float(np.sum(curve["count"] / n * np.abs(curve["frac_pos"] - curve["mean_prob"])))


def selective_inputs(labels, probs):
    y = _labels(labels)
    p = _probs(probs, y.size)
    return ((p >= 0.5) != y).astype(float), np.maximum(p, 1.0 - p)


def risk_coverage(errors, confidence):
    e = _values(errors, name="errors")
    c = _values(confidence, e.size, "confidence")
    if not _nonempty(e, "risk-coverage curve"):
        empty = np.zeros(0)
        return {"coverage": empty, "risk": empty, "threshold": empty}
    order = np.argsort(-c, kind="mergesort")
    cs, es = c[order], e[order]
    last = np.r_[np.flatnonzero(cs[1:] != cs[:-1]), cs.size - 1]
    kept = last + 1
    return {"coverage": kept / e.size, "risk": np.cumsum(es)[last] / kept, "threshold": cs[last]}


def aurc(errors, confidence):
    curve = risk_coverage(errors, confidence)
    if curve["coverage"].size == 0:
        return NAN
    return float(np.sum(np.diff(np.r_[0.0, curve["coverage"]]) * curve["risk"]))


def latency_percentiles(latencies, percentiles=(50, 95, 99), method="inverted_cdf"):
    x = _values(latencies, name="latencies")
    keys = [f"p{q:g}" for q in percentiles]
    if not _nonempty(x, "latency percentiles"):
        return {"n": 0, "mean": NAN, "max": NAN, **{k: NAN for k in keys}}
    out = {"n": int(x.size), "mean": float(np.mean(x)), "max": float(np.max(x))}
    for key, value in zip(keys, np.percentile(x, percentiles, method=method)):
        out[key] = float(value)
    return out


def deadline_miss_rate(latencies, deadlines):
    x = _values(latencies, name="latencies")
    d = _values(deadlines, name="deadlines")
    if d.size not in (1, x.size):
        raise ValueError("deadlines must be a scalar or match latencies")
    if not _nonempty(x, "deadline miss rate"):
        return NAN
    return float(np.mean(x > d))


def contiguous_blocks(n, block_len):
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    return np.arange(n) // int(block_len)


def _aligned(arrays, n=None):
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0] if n is None else n
    for a in arrays:
        if a.shape[0] != n:
            raise ValueError("all arrays must have the same length along axis 0")
    return arrays


def _cluster_index(clusters, n):
    codes = np.unique(np.asarray(clusters).ravel(), return_inverse=True)[1].ravel()
    if codes.size != n:
        raise ValueError(f"clusters has length {codes.size}, expected {n}")
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return order, starts, sizes


def _bootstrap(stat, n, clusters, n_boot, seed):
    order, starts, sizes = _cluster_index(clusters, n)
    k = sizes.size
    if k == 0:
        _warn("bootstrap interval is undefined for empty input")
        return np.full(n_boot, NAN), 0
    if k < 10:
        warnings.warn(f"only {k} clusters: percentile bootstrap intervals are unreliable", stacklevel=3)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for b in range(n_boot):
            draw = rng.integers(0, k, size=k)
            sz = sizes[draw]
            ends = np.cumsum(sz)
            idx = order[np.arange(ends[-1]) + np.repeat(starts[draw] - (ends - sz), sz)]
            out[b] = stat(idx)
    return out, k


def _interval(point, boot, alpha, k):
    valid = boot[~np.isnan(boot)]
    if valid.size < boot.size and k:
        _warn(f"{boot.size - valid.size} of {boot.size} bootstrap replicates were undefined and dropped")
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2]) if valid.size else (NAN, NAN)
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_clusters": int(k),
        "n_boot": int(boot.size),
        "n_valid": int(valid.size),
    }


def cluster_bootstrap_ci(metric_fn, arrays, clusters, n_boot=2000, alpha=0.05, seed=0):
    arrays = _aligned(arrays)
    n = arrays[0].shape[0]
    point = metric_fn(*arrays)
    boot, k = _bootstrap(lambda idx: metric_fn(*(a[idx] for a in arrays)), n, clusters, n_boot, seed)
    return _interval(point, boot, alpha, k)


def paired_cluster_bootstrap_diff(metric_fn, arrays_a, arrays_b, clusters, n_boot=2000, alpha=0.05, seed=0):
    arrays_a = _aligned(arrays_a)
    n = arrays_a[0].shape[0]
    arrays_b = _aligned(arrays_b, n)
    point = metric_fn(*arrays_a) - metric_fn(*arrays_b)

    def diff(idx):
        return metric_fn(*(a[idx] for a in arrays_a)) - metric_fn(*(b[idx] for b in arrays_b))

    boot, k = _bootstrap(diff, n, clusters, n_boot, seed)
    out = _interval(point, boot, alpha, k)
    valid = boot[~np.isnan(boot)]
    if valid.size:
        out["p_two_sided"] = float(min(1.0, 2 * min(np.mean(valid <= 0), np.mean(valid >= 0))))
    else:
        out["p_two_sided"] = NAN
    return out
