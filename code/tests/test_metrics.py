import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve

from cpsdefense.eval import metrics as M
from cpsdefense.eval.metrics import UndefinedMetricWarning


def _random_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 400))
    y = rng.random(n) < rng.uniform(0.02, 0.9)
    y[0], y[-1] = True, False
    kind = seed % 3
    s = rng.normal(size=n) + y * rng.uniform(-0.5, 2.0)
    if kind == 1:
        s = np.round(s, 1)
    elif kind == 2:
        s = rng.integers(0, 4, size=n) + y * rng.integers(0, 2, size=n)
    return y, s


@pytest.mark.parametrize("seed", range(300))
def test_ap_and_auc_match_sklearn(seed):
    y, s = _random_case(seed)
    assert M.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)
    assert M.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)


def test_ranking_metrics_ignore_order_of_ties():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.3
    s = rng.integers(0, 5, size=500).astype(float)
    perm = rng.permutation(500)
    assert M.average_precision(y, s) == M.average_precision(y[perm], s[perm])
    assert M.roc_auc(y, s) == pytest.approx(M.roc_auc(y[perm], s[perm]), abs=1e-15)


def test_constant_scores():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    s = np.full(8, 0.7)
    assert M.average_precision(y, s) == pytest.approx(0.25)
    assert M.roc_auc(y, s) == pytest.approx(0.5)
    best = M.best_f1(y, s)
    assert best["threshold"] == 0.7
    assert best["recall"] == 1.0
    assert best["f1"] == pytest.approx(2 * 0.25 / 1.25)


@pytest.mark.parametrize("y", [np.zeros(10, int), np.ones(10, int)])
def test_single_class_gives_nan_and_warning(y):
    s = np.arange(10.0)
    for fn in (M.average_precision, M.roc_auc):
        with pytest.warns(UndefinedMetricWarning):
            assert np.isnan(fn(y, s))
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(M.best_f1(y, s)["f1"])


def test_input_validation():
    with pytest.raises(ValueError):
        M.average_precision([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        M.roc_auc([0, 1, 1], [0.1, np.nan, 0.3])
    with pytest.raises(ValueError):
        M.roc_auc([0, 1, 1], [0.1, 0.3])


def test_precision_recall_f1_by_hand():
    y = np.array([0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.6, 0.7, 0.2, 0.9, 0.5])
    out = M.precision_recall_f1(y, s, 0.6)
    assert (out["tp"], out["fp"], out["fn"]) == (2, 1, 1)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(skm.f1_score(y, s >= 0.6))
    none = M.precision_recall_f1(y, s, 5.0)
    assert np.isnan(none["precision"]) and none["recall"] == 0.0 and none["f1"] == 0.0
    with pytest.warns(UndefinedMetricWarning):
        out = M.precision_recall_f1(np.zeros(6), s, 0.5)
    assert np.isnan(out["recall"]) and np.isnan(out["f1"]) and out["precision"] == 0.0


@pytest.mark.parametrize("seed", range(40))
def test_best_f1_matches_brute_force_and_sklearn(seed):
    y, s = _random_case(seed)
    brute = max(M.precision_recall_f1(y, s, t)["f1"] for t in np.unique(s))
    best = M.best_f1(y, s)
    assert best["f1"] == pytest.approx(brute, abs=1e-12)
    assert M.precision_recall_f1(y, s, best["threshold"])["f1"] == pytest.approx(best["f1"], abs=1e-12)
    p, r, _ = skm.precision_recall_curve(y, s)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = np.nanmax(2 * p * r / (p + r))
    assert best["f1"] == pytest.approx(sk, abs=1e-12)


def test_point_adjust_by_hand():
    y = np.zeros(40, int)
    y[10:20] = 1
    y[30:35] = 1
    s = np.zeros(40)
    s[15] = 1.0
    s[2] = 1.0
    pa = M.point_adjust_f1(y, s, 0.5)
    assert (pa["tp"], pa["fp"], pa["fn"]) == (10, 1, 5)
    assert pa["recall"] == pytest.approx(10 / 15)
    raw = M.precision_recall_f1(y, s, 0.5)
    assert raw["recall"] == pytest.approx(1 / 15)


@pytest.mark.parametrize("seed", range(20))
def test_point_adjust_never_below_point_f1_and_inflates_random_scores(seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(5000, bool)
    for start in rng.choice(4800, size=8, replace=False):
        y[start : start + 150] = True
    s = rng.random(5000)
    thr = np.quantile(s, 0.99)
    pa = M.point_adjust_f1(y, s, thr)["f1"]
    raw = M.precision_recall_f1(y, s, thr)["f1"]
    assert pa >= raw
    assert pa > raw + 0.3


def test_alarm_segments_and_merge_gap():
    a = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    assert M.alarm_segments(a).tolist() == [[1, 3], [5, 6], [9, 10]]
    assert M.alarm_segments(a, merge_gap=1).tolist() == [[1, 3], [5, 6], [9, 10]]
    assert M.alarm_segments(a, merge_gap=2).tolist() == [[1, 6], [9, 10]]
    assert M.alarm_segments(a, merge_gap=3).tolist() == [[1, 10]]
    assert M.alarm_segments(np.zeros(5)).shape == (0, 2)
    assert M.alarm_segments(np.ones(4)).tolist() == [[0, 4]]
    assert M.alarm_segments(np.zeros(0)).shape == (0, 2)


def _hand_series(period_s):
    n = 7200
    y = np.zeros(n, bool)
    y[1000:1100] = True
    y[5000:5300] = True
    a = np.zeros(n, bool)
    for start, end in [(200, 210), (215, 220), (1050, 1200), (3000, 3001), (5310, 5320), (6000, 6005)]:
        a[start:end] = True
    return a, y


@pytest.mark.parametrize("period_s", [1.0, 10.0])
def test_false_alarms_per_hour_by_hand(period_s):
    a, y = _hand_series(period_s)
    hours = 7200 * period_s / 3600
    fa = M.false_alarms_per_hour(a, y, period_s, merge_gap=0, grace_s=0)
    assert fa == pytest.approx(5 / (6800 * period_s / 3600))
    fa = M.false_alarms_per_hour(a, y, period_s, merge_gap=10, grace_s=30 * period_s)
    assert fa == pytest.approx(3 / (6740 * period_s / 3600))
    assert M.false_alarms_per_hour(a, None, period_s) == pytest.approx(6 / hours)
    assert M.false_alarms_per_hour(np.zeros(7200), y, period_s) == 0.0


@pytest.mark.parametrize("period_s", [1.0, 10.0])
def test_event_metrics_by_hand(period_s):
    a, y = _hand_series(period_s)
    ev = M.event_metrics(a, y, period_s, grace_s=0)
    assert ev["n_events"] == 2 and ev["n_detected"] == 1
    assert ev["event_recall"] == 0.5
    assert ev["detected"].tolist() == [True, False]
    assert ev["delays_s"][0] == pytest.approx(50 * period_s) and np.isnan(ev["delays_s"][1])
    assert ev["events"].tolist() == [[1000, 1100], [5000, 5300]]

    ev = M.event_metrics(a, y, period_s, grace_s=30 * period_s)
    assert ev["event_recall"] == 1.0
    assert ev["delays_s"].tolist() == pytest.approx([50 * period_s, 310 * period_s])
    assert ev["mean_delay_s"] == pytest.approx(180 * period_s)
    assert ev["median_delay_s"] == pytest.approx(180 * period_s)

    ev = M.event_metrics(a, y, period_s, grace_s=10 * period_s)
    assert ev["detected"].tolist() == [True, False]


def test_event_delay_zero_when_alarm_already_on_and_grace_cut_at_next_event():
    y = np.zeros(60, bool)
    y[10:20] = True
    y[25:30] = True
    a = np.zeros(60, bool)
    a[5:12] = True
    ev = M.event_metrics(a, y, 1.0, grace_s=10)
    assert ev["detected"].tolist() == [True, False]
    assert ev["delays_s"][0] == 0.0

    a = np.zeros(60, bool)
    a[26] = True
    ev = M.event_metrics(a, y, 1.0, grace_s=10)
    assert ev["detected"].tolist() == [False, True]
    assert ev["delays_s"][1] == 1.0

    a = np.zeros(60, bool)
    a[34] = True
    assert M.event_metrics(a, y, 1.0, grace_s=5)["detected"].tolist() == [False, True]
    assert M.event_metrics(a, y, 1.0, grace_s=4)["detected"].tolist() == [False, False]


def test_event_metrics_edge_cases():
    y = np.zeros(20, bool)
    y[5:8] = True
    ev = M.event_metrics(np.zeros(20), y)
    assert ev["event_recall"] == 0.0 and np.isnan(ev["mean_delay_s"]) and np.isnan(ev["median_delay_s"])
    with pytest.warns(UndefinedMetricWarning):
        ev = M.event_metrics(np.ones(20), np.zeros(20))
    assert np.isnan(ev["event_recall"]) and ev["n_events"] == 0
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(M.false_alarms_per_hour(np.ones(20), np.ones(20)))
    with pytest.raises(ValueError):
        M.event_metrics(np.zeros(20), np.zeros(19))


@pytest.mark.parametrize("seed", range(30))
def test_segment_counts_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    parts = [np.round(rng.normal(size=int(rng.integers(1, 120))), 1) for _ in range(int(rng.integers(1, 4)))]
    levels = np.unique(np.concatenate(parts))[::-1]
    for gap in (0, 1, 2, 5):
        fast = M._segment_counts(parts, levels, gap)
        brute = [sum(len(M.alarm_segments(p >= u, gap)) for p in parts) for u in levels]
        assert fast.tolist() == brute


def _ar1(n, seed, phi=0.95):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


@pytest.mark.parametrize("merge_gap", [0, 5, 30])
@pytest.mark.parametrize("target", [0.5, 2.0, 10.0])
def test_threshold_meets_target_and_is_tight(merge_gap, target):
    period_s = 1.0
    s = np.round(_ar1(36_000, seed=merge_gap), 3)
    thr = M.threshold_for_false_alarm_rate(s, target, period_s, merge_gap)
    assert M.false_alarms_per_hour(s >= thr, None, period_s, merge_gap) <= target
    levels = np.unique(s)[::-1]
    above = levels[levels >= thr]
    rates = [M.false_alarms_per_hour(s >= u, None, period_s, merge_gap) for u in above[:: max(1, len(above) // 50)]]
    assert max(rates) <= target
    lower = levels[levels < thr]
    if lower.size:
        assert M.false_alarms_per_hour(s >= lower[0], None, period_s, merge_gap) > target


def test_threshold_not_fooled_by_merged_segments():
    s = np.random.default_rng(0).normal(scale=0.1, size=36_000)
    s[::600] = 5.0
    assert M.false_alarms_per_hour(s >= s.min()) == pytest.approx(0.1)
    thr = M.threshold_for_false_alarm_rate(s, target_per_hour=1.0, period_s=1.0)
    assert thr > 5.0
    assert M.false_alarms_per_hour(s >= thr) == 0.0
    assert M.threshold_for_false_alarm_rate(s, target_per_hour=6.0, period_s=1.0) == 5.0


def test_threshold_edge_cases():
    s = np.random.default_rng(1).normal(size=3600)
    thr = M.threshold_for_false_alarm_rate(s, 0.0)
    assert thr > s.max() and not (s >= thr).any()
    assert M.threshold_for_false_alarm_rate(np.full(3600, 2.0), 5.0) == 2.0
    assert M.threshold_for_false_alarm_rate(np.full(3600, 2.0), 0.5) > 2.0
    with pytest.raises(ValueError):
        M.threshold_for_false_alarm_rate(np.array([0.0, np.inf]), 1.0)
    with pytest.raises(ValueError):
        M.threshold_for_false_alarm_rate(np.zeros(10), -1.0)
    for bad_gap in (-1, 2.5):
        with pytest.raises(ValueError):
            M.threshold_for_false_alarm_rate(np.zeros(10), 1.0, merge_gap=bad_gap)
        with pytest.raises(ValueError):
            M.alarm_segments(np.zeros(10), merge_gap=bad_gap)


def test_nan_and_infinite_rates_periods_and_windows_raise():
    s = np.random.default_rng(1).normal(size=3600)
    a, y = np.zeros(100, bool), np.zeros(100, bool)
    y[40:50] = True
    for bad in (np.nan, -1.0):
        with pytest.raises(ValueError, match="target_per_hour"):
            M.threshold_for_false_alarm_rate(s, bad)
        with pytest.raises(ValueError, match="target_per_hour"):
            M.recall_at_fa_rate(s, np.zeros(3600, bool), s, bad)
    for bad in (np.nan, np.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match="period_s"):
            M.threshold_for_false_alarm_rate(s, 1.0, period_s=bad)
        with pytest.raises(ValueError, match="period_s"):
            M.false_alarms_per_hour(a, y, period_s=bad)
        with pytest.raises(ValueError, match="period_s"):
            M.event_metrics(a, y, period_s=bad)
    for bad in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="grace_s"):
            M.false_alarms_per_hour(a, y, grace_s=bad)
        with pytest.raises(ValueError, match="grace_s"):
            M.event_metrics(a, y, grace_s=bad)


def test_threshold_on_separate_series():
    a = np.zeros(3600)
    a[1000] = 0.5
    a[-1] = 1.0
    b = np.zeros(3600)
    b[0] = 1.0
    assert M._segment_counts([np.r_[a, b]], np.array([1.0]), 0).tolist() == [1]
    assert M._segment_counts([a, b], np.array([1.0]), 0).tolist() == [2]
    assert M.threshold_for_false_alarm_rate(np.r_[a, b], 0.5) == 1.0
    assert M.threshold_for_false_alarm_rate([a, b], 0.5) > 1.0


def test_threshold_speed_on_a_long_series():
    import time

    s = np.random.default_rng(2).normal(size=1_000_000)
    t0 = time.perf_counter()
    M.threshold_for_false_alarm_rate(s, 1.0, 1.0, merge_gap=60)
    assert time.perf_counter() - t0 < 10.0


def test_recall_at_fa_rate_threshold_comes_only_from_validation():
    rng = np.random.default_rng(3)
    val = _ar1(36_000, seed=4)
    test = _ar1(36_000, seed=5)
    y = np.zeros(36_000, bool)
    for start in range(2000, 34_000, 4000):
        y[start : start + 300] = True
    shifted = test + 20.0 * y
    out = M.recall_at_fa_rate(shifted, y, val, 1.0, period_s=1.0, merge_gap=10, grace_s=60)
    assert out["threshold"] == M.threshold_for_false_alarm_rate(val, 1.0, 1.0, 10)
    assert out["n_events"] == 8 and out["event_recall"] == 1.0
    assert out["test_fa_per_hour"] < 5.0
    assert 0.0 <= out["mean_delay_s"] < 60

    other = M.recall_at_fa_rate(rng.normal(size=36_000), np.roll(y, 777), val, 1.0, merge_gap=10, grace_s=60)
    assert other["threshold"] == out["threshold"]
    assert other["event_recall"] < out["event_recall"]


def _naive_runs(binary):
    runs, i = [], 0
    while i < len(binary):
        if binary[i]:
            j = i
            while j < len(binary) and binary[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _naive_merged(alarms, merge_gap):
    runs = []
    for start, end in _naive_runs(alarms):
        if runs and start - runs[-1][1] <= merge_gap:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
    return runs


def _naive_windows(labels, period_s, grace_s, pre_s=0.0):
    p, g, pre = Fraction(period_s), Fraction(grace_s), Fraction(pre_s)
    events = _naive_runs(labels)
    out = []
    for k, (start, end) in enumerate(events):
        nxt = events[k + 1][0] if k + 1 < len(events) else len(labels)
        lo = min(i for i in range(start + 1) if i * p >= start * p - pre)
        hi = max(i for i in range(start, nxt) if i * p < end * p + g) + 1
        out.append((start, lo, hi))
    return out


def _naive_normal(alarms, labels, period_s, grace_s):
    win = _naive_windows(labels, period_s, grace_s)
    normal = [not any(s <= i < hi for s, _, hi in win) for i in range(len(labels))]
    p = Fraction(period_s)
    return sum(a and n for a, n in zip(alarms, normal)) * p, sum(normal) * p


def _naive_duration_fa(alarms, labels, period_s, grace_s, unit_s, pre_s, merge_gap):
    win = _naive_windows(labels, period_s, grace_s, pre_s)
    excused = [any(lo <= i < hi for _, lo, hi in win) for i in range(len(labels))]
    p, unit = Fraction(period_s), Fraction(unit_s)
    count, outside = 0, 0
    for start, end in _naive_merged(alarms, merge_gap):
        i = start
        while i < end:
            if excused[i]:
                i += 1
                continue
            j = i
            while j < end and not excused[j]:
                j += 1
            if any(alarms[i:j]):
                count += max(1, math.ceil((j - i) * p / unit))
                outside += j - i
            i = j
    return count, outside * p


def _naive_strict(alarms, labels, period_s, grace_s, pre_s, merge_gap):
    p = Fraction(period_s)
    runs = _naive_merged(alarms, merge_gap)
    rows = []
    for start, lo, hi in _naive_windows(labels, period_s, grace_s, pre_s):
        onsets = [r0 for r0, _ in runs if lo <= r0 < hi]
        hits = [i for i in range(start, hi) if alarms[i]]
        strict, lenient = bool(onsets), bool(hits)
        if strict:
            offset = (onsets[0] - start) * p
        elif lenient:
            offset = (next(r0 for r0, r1 in runs if r0 <= hits[0] < r1) - start) * p
        else:
            offset = None
        delay = max(0, onsets[0] - start) * p if strict else None
        rows.append((strict, lenient, lenient and not strict, delay, offset))
    return rows


def _duration_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 250))

    def blocky(p_on, p_off):
        x, on = np.zeros(n, bool), rng.random() < 0.3
        for i in range(n):
            x[i] = on
            on = rng.random() >= p_off if on else rng.random() < p_on
        return x

    y = blocky(rng.uniform(0.005, 0.08), rng.uniform(0.05, 0.4))
    a = blocky(rng.uniform(0.01, 0.3), rng.uniform(0.05, 0.9))
    period_s = float(rng.choice([0.5, 1.0, 2.0, 10.0]))
    grace_s = float(rng.choice([0.0, 0.5, 1.0, 2.5, 7.3, 30.0]))
    pre_s = float(rng.choice([0.0, 0.5, 3.0, 7.3, 60.0]))
    unit_s = float(rng.choice([0.5, 1.0, 7.0, 30.5, 60.0]))
    return a, y, period_s, grace_s, pre_s, unit_s, int(rng.integers(0, 7))


@pytest.mark.parametrize("seed", range(400))
def test_duration_aware_metrics_match_naive(seed):
    a, y, period_s, grace_s, pre_s, unit_s, gap = _duration_case(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        nt = M.normal_alarm_time(a, y, period_s, grace_s)
        fa = M.duration_aware_false_alarms(a, y, period_s, grace_s, unit_s=unit_s, pre_tolerance_s=pre_s,
                                           merge_gap=gap)
        st = M.strict_event_metrics(a, y, period_s, grace_s, pre_tolerance_s=pre_s, merge_gap=gap)
        old = M.event_metrics(a, y, period_s, grace_s)
        old_fa = M._false_alarms(a, y, period_s, gap, M._grace_samples(grace_s, period_s))[0]

    alarm_s, normal_s = _naive_normal(a, y, period_s, grace_s)
    assert nt["normal_alarm_s"] == pytest.approx(float(alarm_s))
    assert nt["normal_s"] == pytest.approx(float(normal_s))
    if normal_s:
        assert nt["normal_alarm_fraction"] == pytest.approx(float(alarm_s / normal_s))
    else:
        assert np.isnan(nt["normal_alarm_fraction"])

    count, outside_s = _naive_duration_fa(a, y, period_s, grace_s, unit_s, pre_s, gap)
    assert fa["n_false_alarms"] == count
    assert fa["false_alarm_s"] == pytest.approx(float(outside_s))
    assert fa["normal_hours"] == pytest.approx(float(normal_s) / 3600)
    if normal_s:
        assert fa["fa_per_hour"] == pytest.approx(count / (float(normal_s) / 3600))
    else:
        assert np.isnan(fa["fa_per_hour"])
    if pre_s == 0:
        assert count >= old_fa

    rows = _naive_strict(a, y, period_s, grace_s, pre_s, gap)
    assert st["n_events"] == len(rows) == old["n_events"]
    assert st["detected"].tolist() == [r[0] for r in rows]
    assert st["lenient_detected"].tolist() == [r[1] for r in rows] == old["detected"].tolist()
    assert st["pre_alarmed"].tolist() == [r[2] for r in rows]
    assert st["n_detected"] == sum(r[0] for r in rows)
    assert st["n_pre_alarmed"] == sum(r[2] for r in rows)
    assert st["events"].tolist() == old["events"].tolist()
    for got, want in ((st["delays_s"], [r[3] for r in rows]), (st["onset_offset_s"], [r[4] for r in rows])):
        assert np.isnan(got).tolist() == [w is None for w in want]
        assert got[~np.isnan(got)].tolist() == pytest.approx([float(w) for w in want if w is not None])
    if rows:
        assert st["strict_recall"] == pytest.approx(np.mean([r[0] for r in rows]))
        assert st["lenient_recall"] == old["event_recall"]
        found = [float(r[3]) for r in rows if r[0]]
        if found:
            assert st["mean_delay_s"] == pytest.approx(np.mean(found))
            assert st["median_delay_s"] == pytest.approx(np.median(found))
        else:
            assert np.isnan(st["mean_delay_s"]) and np.isnan(st["median_delay_s"])
    else:
        assert np.isnan(st["strict_recall"]) and np.isnan(st["lenient_recall"])


def _review_scenario():
    n = 8 * 3600
    y = np.zeros(n, bool)
    starts = np.arange(1, 6) * 5000
    for s in starts:
        y[s : s + 300] = True
    return y, starts


def test_always_on_detector_is_no_longer_free():
    y, starts = _review_scenario()
    a = np.ones(y.size, bool)
    kw = {"period_s": 1.0, "grace_s": 60}
    assert M.false_alarms_per_hour(a, y, merge_gap=60, **kw) == 0.0
    old = M.event_metrics(a, y, **kw)
    assert old["event_recall"] == 1.0 and old["mean_delay_s"] == 0.0
    nt = M.normal_alarm_time(a, y, **kw)
    assert nt["normal_alarm_fraction"] == 1.0
    assert nt["normal_s"] == y.size - 5 * 360
    fa = M.duration_aware_false_alarms(a, y, **kw, unit_s=60, pre_tolerance_s=60, merge_gap=60)
    lengths = [4940] + [4580] * 4 + [y.size - (starts[-1] + 360)]
    assert fa["n_false_alarms"] == sum(math.ceil(d / 60) for d in lengths)
    assert fa["false_alarm_s"] == sum(lengths)
    assert fa["fa_per_hour"] > 50
    st = M.strict_event_metrics(a, y, **kw, pre_tolerance_s=60, merge_gap=60)
    assert st["strict_recall"] == 0.0 and st["n_detected"] == 0
    assert st["lenient_recall"] == 1.0 and st["n_pre_alarmed"] == 5
    assert np.isnan(st["mean_delay_s"]) and np.isnan(st["delays_s"]).all()
    assert st["onset_offset_s"].tolist() == (-starts).astype(float).tolist()


def test_perfect_but_late_detector_keeps_recall_and_its_delay():
    y, starts = _review_scenario()
    a = np.zeros(y.size, bool)
    for s in starts:
        a[s + 30 : s + 330] = True
    kw = {"period_s": 1.0, "grace_s": 60}
    st = M.strict_event_metrics(a, y, **kw, pre_tolerance_s=60, merge_gap=60)
    assert st["strict_recall"] == 1.0 and st["lenient_recall"] == 1.0 and st["n_pre_alarmed"] == 0
    assert st["delays_s"].tolist() == [30.0] * 5 and st["mean_delay_s"] == 30.0
    fa = M.duration_aware_false_alarms(a, y, **kw, unit_s=60, pre_tolerance_s=60, merge_gap=60)
    assert fa["n_false_alarms"] == 0 and fa["fa_per_hour"] == 0.0 and fa["false_alarm_s"] == 0.0
    assert M.normal_alarm_time(a, y, **kw)["normal_alarm_s"] == 0.0


def test_events_inside_a_long_preexisting_alarm_are_pre_alarmed():
    n = 12 * 3600
    y = np.zeros(n, bool)
    starts = [2 * 3600] + [int(h * 3600) for h in (4, 5.5, 7, 8.5, 10)]
    for s in starts:
        y[s : s + 300] = True
    a = np.zeros(n, bool)
    a[starts[0] + 10 : starts[0] + 200] = True
    a[3 * 3600 :] = True
    kw = {"period_s": 1.0, "grace_s": 60}
    old = M.event_metrics(a, y, **kw)
    assert old["event_recall"] == 1.0 and old["delays_s"].tolist() == [10.0] + [0.0] * 5
    assert M.false_alarms_per_hour(a, y, merge_gap=60, **kw) == 0.0

    st = M.strict_event_metrics(a, y, **kw, pre_tolerance_s=60, merge_gap=60)
    assert st["detected"].tolist() == [True] + [False] * 5
    assert st["pre_alarmed"].tolist() == [False] + [True] * 5
    assert st["strict_recall"] == pytest.approx(1 / 6) and st["lenient_recall"] == 1.0
    assert st["delays_s"][0] == 10.0 and st["mean_delay_s"] == 10.0
    assert st["onset_offset_s"].tolist() == [10.0] + [float(3 * 3600 - s) for s in starts[1:]]

    fa = M.duration_aware_false_alarms(a, y, **kw, unit_s=60, pre_tolerance_s=60, merge_gap=60)
    inside = 5 * (60 + 300 + 60)
    assert fa["false_alarm_s"] == 9 * 3600 - inside
    assert fa["n_false_alarms"] >= (9 * 3600 - inside) / 60
    nt = M.normal_alarm_time(a, y, **kw)
    assert nt["normal_alarm_s"] == 9 * 3600 - 5 * 360


def test_strict_events_by_hand():
    y = np.zeros(400, bool)
    y[100:150] = True
    y[300:320] = True
    kw = {"period_s": 1.0, "grace_s": 10, "pre_tolerance_s": 20}
    a = np.zeros(400, bool)
    a[80:120] = True
    st = M.strict_event_metrics(a, y, **kw)
    assert st["detected"].tolist() == [True, False] and st["delays_s"][0] == 0.0
    assert st["onset_offset_s"][0] == -20.0
    a[79] = True
    st = M.strict_event_metrics(a, y, **kw)
    assert st["detected"].tolist() == [False, False] and st["pre_alarmed"].tolist() == [True, False]
    assert st["onset_offset_s"][0] == -21.0 and np.isnan(st["delays_s"][0])
    a[130:135] = True
    st = M.strict_event_metrics(a, y, **kw)
    assert st["detected"].tolist() == [True, False] and st["delays_s"][0] == 30.0
    assert not st["pre_alarmed"][0]
    st = M.strict_event_metrics(a, y, **kw, merge_gap=10)
    assert st["pre_alarmed"].tolist() == [True, False]
    a = np.zeros(400, bool)
    a[329] = True
    assert M.strict_event_metrics(a, y, **kw)["delays_s"][1] == 29.0
    a = np.zeros(400, bool)
    a[330] = True
    assert not M.strict_event_metrics(a, y, **kw)["detected"][1]
    a = np.zeros(400, bool)
    a[285:290] = True
    st = M.strict_event_metrics(a, y, **kw)
    assert st["detected"].tolist() == [False, True] and st["lenient_detected"].tolist() == [False, False]
    assert M.duration_aware_false_alarms(a, y, 1.0, 10, pre_tolerance_s=20)["n_false_alarms"] == 0
    assert M.duration_aware_false_alarms(a, y, 1.0, 10, pre_tolerance_s=0)["n_false_alarms"] == 1


def test_duration_aware_false_alarms_by_hand():
    y = np.zeros(1000, bool)
    y[500:600] = True
    a = np.zeros(1000, bool)
    a[100:160] = True
    a[200:261] = True
    a[400:700] = True
    kw = {"period_s": 1.0, "grace_s": 20, "unit_s": 60, "pre_tolerance_s": 30}
    fa = M.duration_aware_false_alarms(a, y, **kw)
    assert fa["n_false_alarms"] == 1 + 2 + 2 + 2
    assert fa["false_alarm_s"] == 60 + 61 + 70 + 80
    assert fa["normal_hours"] == pytest.approx((1000 - 120) / 3600)
    assert fa["fa_per_hour"] == pytest.approx(7 / ((1000 - 120) / 3600))
    assert M.false_alarms_per_hour(a, y, 1.0, grace_s=20) == pytest.approx(2 / ((1000 - 120) / 3600))
    fa = M.duration_aware_false_alarms(a, y, **{**kw, "unit_s": 1000}, merge_gap=40)
    assert fa["false_alarm_s"] == 161 + 70 + 80 and fa["n_false_alarms"] == 3
    y = np.zeros(300, bool)
    y[100:110] = True
    y[140:150] = True
    a = np.zeros(300, bool)
    a[120:130] = True
    a[136:140] = True
    bridge = {"period_s": 1.0, "grace_s": 20, "unit_s": 60, "pre_tolerance_s": 5, "merge_gap": 10}
    fa = M.duration_aware_false_alarms(a, y, **bridge)
    assert fa["n_false_alarms"] == 0 and fa["false_alarm_s"] == 0.0
    a[132] = True
    fa = M.duration_aware_false_alarms(a, y, **bridge)
    assert fa["n_false_alarms"] == 1 and fa["false_alarm_s"] == 5.0
    assert M.duration_aware_false_alarms(a, y, **{**bridge, "merge_gap": 0})["false_alarm_s"] == 1.0
    b = np.zeros(100, bool)
    b[10:17] = True
    assert M.duration_aware_false_alarms(b, None, 10.0, 0, unit_s=60)["n_false_alarms"] == 2
    assert M.duration_aware_false_alarms(b, None, 10.0, 0, unit_s=70)["n_false_alarms"] == 1
    nt = M.normal_alarm_time(b, None, 10.0)
    assert (nt["normal_alarm_s"], nt["normal_s"]) == (70.0, 1000.0)
    assert nt["normal_alarm_fraction"] == pytest.approx(0.07)


def test_duration_aware_metrics_edge_cases():
    y = np.zeros(20, bool)
    y[5:8] = True
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(M.normal_alarm_time(np.ones(20), np.ones(20))["normal_alarm_fraction"])
    with pytest.warns(UndefinedMetricWarning):
        fa = M.duration_aware_false_alarms(np.zeros(0), np.zeros(0), 1.0, 0.0)
    assert fa["n_false_alarms"] == 0 and np.isnan(fa["fa_per_hour"])
    with pytest.warns(UndefinedMetricWarning):
        st = M.strict_event_metrics(np.ones(20), np.zeros(20), 1.0, 0.0)
    assert np.isnan(st["strict_recall"]) and np.isnan(st["lenient_recall"]) and st["n_events"] == 0
    st = M.strict_event_metrics(np.zeros(20), y)
    assert st["strict_recall"] == 0.0 and st["n_pre_alarmed"] == 0 and np.isnan(st["mean_delay_s"])
    assert M.strict_event_metrics(np.ones(20), y, pre_tolerance_s=5)["detected"].tolist() == [True]
    assert M.strict_event_metrics(np.ones(20), y, pre_tolerance_s=4)["pre_alarmed"].tolist() == [True]
    for bad in (np.nan, np.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match="unit_s"):
            M.duration_aware_false_alarms(np.zeros(20), y, 1.0, 0.0, unit_s=bad)
    for bad in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="pre_tolerance_s"):
            M.duration_aware_false_alarms(np.zeros(20), y, 1.0, 0.0, pre_tolerance_s=bad)
        with pytest.raises(ValueError, match="pre_tolerance_s"):
            M.strict_event_metrics(np.zeros(20), y, 1.0, 0.0, pre_tolerance_s=bad)
    for bad in (np.nan, 0.0):
        with pytest.raises(ValueError, match="period_s"):
            M.normal_alarm_time(np.zeros(20), y, period_s=bad)
    with pytest.raises(ValueError):
        M.strict_event_metrics(np.zeros(20), y, merge_gap=-1)
    with pytest.raises(ValueError):
        M.normal_alarm_time(np.zeros(20), np.zeros(19))


@pytest.mark.parametrize("seed", range(10))
def test_brier_and_nll_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, size=500)
    y = rng.random(500) < p
    assert M.brier_score(y, p) == pytest.approx(skm.brier_score_loss(y, p), abs=1e-12)
    assert M.binary_nll(y, p) == pytest.approx(skm.log_loss(y, p), abs=1e-12)


def test_nll_is_clipped():
    assert np.isfinite(M.binary_nll([1, 0], [0.0, 1.0]))
    assert M.binary_nll([1, 0], [0.0, 1.0], eps=1e-6) == pytest.approx(-np.log(1e-6))
    with pytest.raises(ValueError):
        M.brier_score([1, 0], [0.5, 1.2])


def test_ece_by_hand():
    y = np.r_[np.zeros(9), 1, np.zeros(5), np.ones(5)]
    p = np.r_[np.full(10, 0.1), np.full(10, 0.9)]
    assert M.ece(y, p, n_bins=10) == pytest.approx(0.5 * 0.4)
    assert M.ece(y, p, n_bins=2, strategy="quantile") == pytest.approx(0.5 * 0.4)
    curve = M.reliability_curve(y, p, n_bins=10)
    assert curve["bin"].tolist() == [0, 8]
    assert curve["count"].tolist() == [10, 10]
    assert curve["frac_pos"].tolist() == pytest.approx([0.1, 0.5])


def test_ece_calibrated_vs_miscalibrated():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = rng.random(200_000) < p
    for strategy in ("uniform", "quantile"):
        assert M.ece(y, p, strategy=strategy) < 0.01
        assert M.ece(y, p**3, strategy=strategy) > 0.15


def test_reliability_curve_matches_sklearn_and_quantile_bins():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    y = rng.random(1000) < p
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="uniform")
    curve = M.reliability_curve(y, p, n_bins=10)
    assert curve["frac_pos"] == pytest.approx(frac)
    assert curve["mean_prob"] == pytest.approx(mean)
    assert M.reliability_curve(y, p, n_bins=10, strategy="quantile")["count"].tolist() == [100] * 10
    tied = np.repeat([0.2, 0.4, 0.6, 0.8], [100, 300, 50, 550])
    q = M.reliability_curve(np.zeros(1000), tied, n_bins=10, strategy="quantile")
    assert sorted(q["count"].tolist()) == [50, 100, 300, 550]


def test_uniform_bins_are_right_closed_on_the_edges():
    y, p = np.array([0, 1, 1]), np.array([0.25, 0.5, 0.75])
    frac, mean = calibration_curve(y, p, n_bins=2)
    curve = M.reliability_curve(y, p, n_bins=2)
    assert curve["frac_pos"].tolist() == pytest.approx([0.5, 1.0]) == frac.tolist()
    assert curve["mean_prob"] == pytest.approx(mean)
    assert M.ece(y, p, n_bins=2) == pytest.approx(1 / 6)
    for n_bins in (2, 3, 5, 6, 7, 10, 12, 15, 20):
        k = np.arange(n_bins + 1)
        curve = M.reliability_curve(np.zeros(k.size), k / n_bins, n_bins)
        assert curve["bin"].tolist() == list(range(n_bins))
        assert curve["count"].tolist() == [2] + [1] * (n_bins - 1)
    rng = np.random.default_rng(0)
    for n_bins in (2, 5, 10):
        p = np.r_[rng.integers(0, 6, 200) / 5, rng.integers(0, 11, 300) / 10, rng.integers(0, 21, 300) / 20]
        y = rng.random(p.size) < p
        frac, mean = calibration_curve(y, p, n_bins=n_bins)
        curve = M.reliability_curve(y, p, n_bins=n_bins)
        assert curve["frac_pos"] == pytest.approx(frac, abs=1e-12)
        assert curve["mean_prob"] == pytest.approx(mean, abs=1e-12)


def test_risk_coverage_and_aurc_by_hand():
    curve = M.risk_coverage([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6])
    assert curve["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert curve["risk"].tolist() == pytest.approx([0, 1 / 2, 1 / 3, 1 / 2])
    assert M.aurc([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6]) == pytest.approx((0 + 1 / 2 + 1 / 3 + 1 / 2) / 4)
    assert M.aurc([1, 0, 0], [0.9, 0.9, 0.5]) == pytest.approx(2 / 3 * 1 / 2 + 1 / 3 * 1 / 3)
    assert M.aurc([0, 1, 0], [0.9, 0.9, 0.5]) == pytest.approx(4 / 9)


def test_selective_inputs_and_perfect_ranking():
    errors, conf = M.selective_inputs([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6])
    assert errors.tolist() == [0, 0, 1, 1]
    assert conf.tolist() == pytest.approx([0.9, 0.8, 0.6, 0.6])
    rng = np.random.default_rng(0)
    e = (rng.random(1000) < 0.2).astype(float)
    perfect = M.aurc(e, -e)
    assert perfect < M.aurc(e, rng.random(1000)) < M.aurc(e, e)


def test_latency_percentiles():
    x = np.arange(1, 101, dtype=float)
    out = M.latency_percentiles(x)
    assert (out["p50"], out["p95"], out["p99"]) == (50.0, 95.0, 99.0)
    assert out["n"] == 100 and out["mean"] == 50.5 and out["max"] == 100.0
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=777)
    out = M.latency_percentiles(x, (50, 90, 99.9), method="linear")
    assert out["p99.9"] == pytest.approx(np.percentile(x, 99.9))
    out = M.latency_percentiles([1.0, 2.0, np.inf, np.inf])
    assert out["p50"] == 2.0 and out["p99"] == np.inf
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(M.latency_percentiles([])["p95"])


def test_deadline_miss_rate():
    assert M.deadline_miss_rate([1.0, 2.0, 3.0, np.inf], 2.0) == 0.5
    assert M.deadline_miss_rate([1.0, 2.0, 3.0], [0.5, 2.0, 5.0]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        M.deadline_miss_rate([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(M.deadline_miss_rate([], 1.0))


def test_contiguous_blocks():
    assert M.contiguous_blocks(10, 4).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        M.contiguous_blocks(10, 0)


def _clustered(seed, k=30, m=20):
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(k), m)
    x = rng.normal(size=k)[clusters] + rng.normal(size=k * m)
    return x, clusters


def test_bootstrap_is_reproducible_and_point_is_full_metric():
    x, clusters = _clustered(0)
    a = M.cluster_bootstrap_ci(np.mean, [x], clusters, n_boot=300, seed=7)
    b = M.cluster_bootstrap_ci(np.mean, [x], clusters, n_boot=300, seed=7)
    c = M.cluster_bootstrap_ci(np.mean, [x], clusters, n_boot=300, seed=8)
    assert a == b
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])
    assert a["point"] == pytest.approx(x.mean())
    assert a["lo"] < a["point"] < a["hi"]
    assert a["n_clusters"] == 30 and a["n_valid"] == 300


def test_bootstrap_resamples_whole_clusters():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 9, size=25)
    clusters = rng.permutation(np.repeat(np.arange(25), sizes))
    within = np.zeros(clusters.size, int)
    for c in range(25):
        within[clusters == c] = np.arange(sizes[c])
    seen = []

    def check(cl, pos):
        ids, counts = np.unique(cl, return_counts=True)
        assert all(n % sizes[i] == 0 for i, n in zip(ids, counts))
        for i in ids:
            order = pos[cl == i]
            assert order.tolist() == list(range(sizes[i])) * (order.size // sizes[i])
        seen.append(len(ids))
        return float(cl.size)

    M.cluster_bootstrap_ci(check, [clusters, within], clusters, n_boot=50, seed=0)
    assert len(seen) == 51 and min(seen) < 25


def test_cluster_bootstrap_coverage_and_iid_undercoverage():
    hits_cluster = hits_iid = 0
    reps = 150
    for rep in range(reps):
        x, clusters = _clustered(1000 + rep)
        ci = M.cluster_bootstrap_ci(np.mean, [x], clusters, n_boot=400, alpha=0.1, seed=rep)
        hits_cluster += ci["lo"] <= 0.0 <= ci["hi"]
        ci = M.cluster_bootstrap_ci(np.mean, [x], np.arange(x.size), n_boot=400, alpha=0.1, seed=rep)
        hits_iid += ci["lo"] <= 0.0 <= ci["hi"]
    assert 0.80 <= hits_cluster / reps <= 0.97
    assert hits_iid / reps < 0.6


def test_bootstrap_drops_undefined_replicates():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(12), 50)
    y = np.zeros(600, bool)
    y[:50] = True
    s = rng.normal(size=600) + y
    with pytest.warns(UndefinedMetricWarning, match="dropped"):
        ci = M.cluster_bootstrap_ci(M.roc_auc, [y, s], clusters, n_boot=200, seed=0)
    assert 0 < ci["n_valid"] < 200
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_few_clusters_warns():
    with pytest.warns(UserWarning, match="clusters"):
        M.cluster_bootstrap_ci(np.mean, [np.arange(20.0)], M.contiguous_blocks(20, 5), n_boot=20)


def test_paired_bootstrap_identical_methods():
    x, clusters = _clustered(0)
    y = x > 0.5
    out = M.paired_cluster_bootstrap_diff(M.roc_auc, [y, x], [y, x.copy()], clusters, n_boot=200)
    assert out["point"] == 0.0 and out["lo"] == 0.0 and out["hi"] == 0.0
    assert out["p_two_sided"] == 1.0


def test_paired_bootstrap_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 100)
    difficulty = rng.normal(scale=2.0, size=40)[clusters]
    y = rng.random(4000) < 0.3
    base = y * 1.0 + difficulty + rng.normal(size=4000)
    better = base + 0.5 * y
    paired = M.paired_cluster_bootstrap_diff(M.roc_auc, [y, better], [y, base], clusters, n_boot=300, seed=1)
    assert paired["point"] > 0 and paired["lo"] > 0 and paired["p_two_sided"] < 0.01
    a = M.cluster_bootstrap_ci(M.roc_auc, [y, better], clusters, n_boot=300, seed=1)
    b = M.cluster_bootstrap_ci(M.roc_auc, [y, base], clusters, n_boot=300, seed=1)
    assert a["lo"] < b["hi"]
    assert paired["hi"] - paired["lo"] < 0.5 * min(a["hi"] - a["lo"], b["hi"] - b["lo"])


def test_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        M.cluster_bootstrap_ci(np.mean, [np.zeros(10)], np.zeros(9), n_boot=5)
    with pytest.raises(ValueError):
        M.paired_cluster_bootstrap_diff(np.mean, [np.zeros(10)], [np.zeros(9)], np.zeros(10), n_boot=5)


def test_bootstrap_on_empty_input_gives_nan_with_warning():
    y, s = np.zeros(0, bool), np.zeros(0)
    with pytest.warns(UndefinedMetricWarning, match="empty input"):
        warnings.filterwarnings("ignore", message="average precision", category=UndefinedMetricWarning)
        ci = M.cluster_bootstrap_ci(M.average_precision, [y, s], np.zeros(0, int), n_boot=50)
    assert np.isnan([ci["point"], ci["lo"], ci["hi"]]).all()
    assert (ci["n_clusters"], ci["n_boot"], ci["n_valid"]) == (0, 50, 0)
    with pytest.warns(UndefinedMetricWarning, match="empty input"):
        warnings.filterwarnings("ignore", message="average precision", category=UndefinedMetricWarning)
        d = M.paired_cluster_bootstrap_diff(M.average_precision, [y, s], [y, s], [], n_boot=50)
    assert np.isnan([d["point"], d["lo"], d["hi"], d["p_two_sided"]]).all()
    assert (d["n_clusters"], d["n_valid"]) == (0, 0)


def test_no_warnings_on_regular_inputs():
    y, s = _random_case(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        M.average_precision(y, s)
        M.roc_auc(y, s)
        M.best_f1(y, s)
        M.event_metrics(s > 0, y)
        M.strict_event_metrics(s > 0, y)
        M.duration_aware_false_alarms(s > 0, y)
        M.normal_alarm_time(s > 0, y)
