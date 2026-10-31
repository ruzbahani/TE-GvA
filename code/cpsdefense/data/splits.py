import numpy as np


def _as_1d(values, dtype=None):
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {arr.shape}")
    return arr


def _check_sorted(t):
    if np.isnan(t).any():
        raise ValueError("timestamps contain NaN")
    if t.size > 1 and np.any(np.diff(t) < 0):
        raise ValueError("timestamps must be non-decreasing")


def run_bounds(values):
    values = _as_1d(values)
    if values.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    stops = np.r_[starts[1:], values.size]
    return starts.astype(np.int64), stops.astype(np.int64)


def segment_by_gaps(timestamps_seconds, max_gap):
    t = _as_1d(timestamps_seconds, np.float64)
    _check_sorted(t)
    if t.size == 0:
        return np.zeros(0, np.int64)
    return np.r_[0, np.cumsum(np.diff(t) > max_gap)].astype(np.int64)


def events_from_labels(labels, segments=None):
    positive = _as_1d(labels) != 0
    key = positive.astype(np.int64)
    if segments is not None:
        segments = _as_1d(segments)
        if segments.shape != positive.shape:
            raise ValueError("labels and segments differ in length")
        change = np.r_[True, (positive[1:] != positive[:-1]) | (segments[1:] != segments[:-1])]
        key = np.cumsum(change)
    starts, stops = run_bounds(key)
    return [(int(a), int(b)) for a, b in zip(starts, stops) if positive[a]]


def time_split_with_purge(timestamps, fractions, purge=0.0):
    t = _as_1d(timestamps, np.float64)
    _check_sorted(t)
    fractions = _as_1d(fractions, np.float64)
    if fractions.size == 0 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions.tolist()}")
    if purge < 0:
        raise ValueError("purge must be non-negative")
    n = t.size
    stops = np.rint(np.cumsum(fractions) * n).astype(np.int64)
    stops[-1] = n
    starts = np.r_[0, stops[:-1]]
    out = np.empty(n, np.int64)
    for k, (a, b) in enumerate(zip(starts, stops)):
        if b <= a:
            raise ValueError(f"split {k} is empty for {n} rows and fractions {fractions.tolist()}")
        out[a:b] = k
    if purge > 0:
        for k in range(1, fractions.size):
            prev = np.arange(starts[k - 1], starts[k])
            out[prev[t[prev] > t[starts[k]] - purge]] = -1
    for k in range(fractions.size):
        if not np.any(out == k):
            raise ValueError(f"purge of {purge} removed split {k} entirely")
    return out


def window_starts(segment_ids, length, stride=1):
    if length < 1 or stride < 1:
        raise ValueError("length and stride must be >= 1")
    starts, stops = run_bounds(segment_ids)
    parts = [np.arange(a, b - length + 1, stride) for a, b in zip(starts, stops) if b - a >= length]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)


def grouped_folds(groups, n_folds, seed=None):
    groups = _as_1d(groups)
    uniq, first, inverse, counts = np.unique(groups, return_index=True, return_inverse=True, return_counts=True)
    if n_folds < 2 or n_folds > uniq.size:
        raise ValueError(f"need 2 <= n_folds <= number of groups ({uniq.size}), got {n_folds}")
    order = np.argsort(first, kind="stable")
    if seed is not None:
        order = np.random.default_rng(seed).permutation(order)
    cum = np.r_[0, np.cumsum(counts[order])]
    total = cum[-1]
    cuts = [0]
    for k in range(1, n_folds):
        lo, hi = cuts[-1] + 1, uniq.size - (n_folds - k)
        candidates = np.arange(lo, hi + 1)
        cuts.append(int(candidates[np.argmin(np.abs(cum[candidates] - k * total / n_folds))]))
    cuts.append(uniq.size)
    fold_of_group = np.empty(uniq.size, np.int64)
    for k in range(n_folds):
        fold_of_group[order[cuts[k]:cuts[k + 1]]] = k
    fold = fold_of_group[inverse]
    all_idx = np.arange(groups.size)
    return [(all_idx[fold != k], all_idx[fold == k]) for k in range(n_folds)]
