import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cpsdefense.data.splits import run_bounds
from cpsdefense.detect.baselines import PCAReconstruction, RobustZScore
from cpsdefense.detect.forecast import GRUForecaster
from cpsdefense.detect.scoring import (
    FeatureScaler,
    alarm_report,
    apply_warmup,
    array_digest,
    evaluate,
    alarm_time_threshold,
    fit_thresholds,
    normal_series,
    normal_thresholds,
    seed_mean_intervals,
    segment_positions,
    smooth,
    time_blocks,
    topk_mean,
)
from cpsdefense.detect.train import (
    dev_criterion,
    format_results,
    format_table,
    run_detector,
    select_setting,
    summarize,
    warmup_rows,
)
from cpsdefense.eval.metrics import (
    UndefinedMetricWarning,
    average_precision,
    cluster_bootstrap_ci,
    duration_aware_false_alarms,
    event_metrics,
    false_alarms_per_hour,
    normal_alarm_time,
    roc_auc,
    strict_event_metrics,
    threshold_for_false_alarm_rate,
)
from cpsdefense.utils.config import load_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FEATURES = 8
MIXING = np.random.default_rng(123).normal(size=(3, N_FEATURES))


def make_split(n, seed, segments=1, n_anomalies=0, name="s", anomaly_len=40):
    rng = np.random.default_rng(seed)
    per = n // segments
    parts, times = [], []
    for k in range(segments):
        t = np.arange(per)
        phase = rng.uniform(0, 2 * np.pi)
        ar = np.zeros(per)
        eps = rng.normal(0, 0.05, per)
        for i in range(1, per):
            ar[i] = 0.99 * ar[i - 1] + eps[i]
        latent = np.column_stack([np.sin(2 * np.pi * t / 200 + phase), np.cos(2 * np.pi * t / 77 + 2 * phase), ar])
        parts.append(latent @ MIXING + 0.05 * rng.normal(size=(per, N_FEATURES)))
        times.append(k * (per + 3600) + t)
    X = np.concatenate(parts)
    segment = np.repeat(np.arange(segments), per)
    y = np.zeros(X.shape[0], np.int8)
    if n_anomalies:
        starts = np.linspace(150, X.shape[0] - anomaly_len - 50, n_anomalies).astype(int)
        for a in starts:
            feats = rng.choice(N_FEATURES, 3, replace=False)
            size = rng.choice([-1, 1], (anomaly_len, 3)) * rng.uniform(3, 6, (anomaly_len, 3))
            X[a:a + anomaly_len, feats] += size
            y[a:a + anomaly_len] = 1
    return {
        "X": X.astype(np.float32),
        "y": y,
        "timestamp": np.datetime64("2024-01-01T00:00:00", "ns") + np.concatenate(times).astype("timedelta64[s]"),
        "segment": segment.astype(np.int64),
        "file": np.array([f"{name}{k}" for k in segment]),
        "columns": [f"f{i}" for i in range(N_FEATURES)],
    }


@pytest.fixture(scope="module")
def data():
    return {
        "train": make_split(20000, 1, segments=2, name="train"),
        "val": make_split(6000, 2, segments=2, name="train"),
        "dev_test": make_split(6000, 3, n_anomalies=10, name="dev"),
        "test": make_split(6000, 4, n_anomalies=10, name="test"),
    }


def small_config(detector_seeds=(0,)):
    return {
        "period_s": 1,
        "eval": {"fa_targets_per_hour": [20.0, 5.0], "merge_gap": 5, "grace_s": 5, "max_false_alarm_s": 30,
                 "pre_tolerance_s": 10, "fa_unit_s": 30, "smoothing": [1, 5], "block_s": 600, "n_boot": 50,
                 "alpha": 0.1, "boot_seed": 0, "min_ci_clusters": 10},
        "zscore": {"topk": [1, 3]},
        "pca": {"variance": [0.9, 0.99]},
        "gru": {"seeds": list(detector_seeds), "window": 16, "hidden": 32, "layers": 1, "batch_size": 128,
                "steps_per_epoch": 60, "max_epochs": 6, "patience": 3, "max_minutes": 2, "topk": [1, "all"],
                "device": DEVICE},
    }


def fit_gru(data, seed=0):
    cfg = dict(small_config()["gru"])
    cfg.pop("seeds")
    return GRUForecaster(**cfg, seed=seed).fit(data["train"], data["val"])


@pytest.fixture(scope="module")
def gru(data):
    return fit_gru(data)


def test_topk_mean():
    v = np.array([[1.0, 5.0, 3.0, 2.0], [4.0, 4.0, 0.0, 1.0], [np.nan] * 4])
    assert np.allclose(topk_mean(v, 1)[:2], [5, 4])
    assert np.allclose(topk_mean(v, 2)[:2], [4, 4])
    assert np.allclose(topk_mean(v, "all")[:2], [2.75, 2.25])
    assert np.allclose(topk_mean(v, 10)[:2], [2.75, 2.25])
    assert np.isnan(topk_mean(v, 2)[2]) and np.isnan(topk_mean(v, "all")[2])
    with pytest.raises(ValueError):
        topk_mean(v, 0)


def test_feature_scaler_robust_standard_and_fallbacks():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(5, 2, 4000), np.r_[np.zeros(3990), np.ones(10)], np.full(4000, 7.0)])
    X = X.astype(np.float32)
    robust = FeatureScaler("robust").fit(X)
    q25, med, q75 = np.percentile(X[:, 0], [25, 50, 75])
    assert robust.center[0] == pytest.approx(med)
    assert robust.scale[0] == pytest.approx((q75 - q25) / 1.3489795, rel=1e-6)
    assert robust.scale[1] == pytest.approx(0.1 * X[:, 1].std(dtype=np.float64))
    assert robust.scale[2] == 1.0
    assert robust.iqr_zero == [1, 2] and robust.std_floored == [1] and robust.fallback == [2]
    assert robust.describe()["std_floored"] == [1] and robust.describe()["std_floor"] == 0.1
    z = robust.transform(X)
    assert z.dtype == np.float32 and np.all(z[:, 2] == 0)
    standard = FeatureScaler("standard").fit(X)
    assert np.allclose(standard.center, X.mean(axis=0, dtype=np.float64))
    assert standard.n_fit == 4000 and standard.fit_sha256 == array_digest(X)
    with pytest.raises(ValueError):
        FeatureScaler("minmax")
    with pytest.raises(ValueError):
        FeatureScaler().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))


def _point_mass(fraction, n=20000, seed=0):
    k = int(round(fraction * n))
    return np.r_[np.zeros(k), np.random.default_rng(seed).uniform(0.1, 2.0, n - k)]


def test_robust_scale_is_floored_relative_to_spread():
    rng = np.random.default_rng(8)
    n = 20000
    tight = np.where(rng.random(n) < 0.8, 0.364 + 0.004 * rng.normal(size=n), rng.uniform(0.4, 1.5, n))
    X = np.column_stack([_point_mass(0.749), _point_mass(0.751), tight, rng.normal(5, 2, n)]).astype(np.float32)
    sc = FeatureScaler("robust").fit(X)
    std = X.std(axis=0, dtype=np.float64)
    q25, q75 = np.percentile(X, [25, 75], axis=0)
    iqr_scale = (q75 - q25) / 1.3489795003921634
    assert np.allclose(sc.scale, np.maximum(iqr_scale, 0.1 * std), rtol=1e-12)
    assert iqr_scale[2] < 0.05 * std[2] and sc.scale[2] == pytest.approx(0.1 * std[2])
    assert sc.std_floored == [1, 2] and sc.iqr_zero == [1] and sc.fallback == []
    assert sc.scale[3] == pytest.approx(iqr_scale[3])
    z = np.abs(sc.transform(np.array([[1.0, 1.0, 1.0, 5.0]], np.float32)))[0]
    assert z[0] / z[1] < 1.5
    assert z[2] < 30
    loose = FeatureScaler("robust", std_floor=0.01).fit(X)
    assert loose.std_floored == [1] and loose.scale[2] == pytest.approx(iqr_scale[2])
    for bad in (0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="std_floor"):
            FeatureScaler("robust", std_floor=bad)


def test_zscore_records_floored_columns_in_the_run(data, tmp_path):
    train = dict(data["train"], X=data["train"]["X"].copy())
    rng = np.random.default_rng(9)
    n = len(train["y"])
    train["X"][:, 2] = np.where(rng.random(n) < 0.8, 1.0 + 0.001 * rng.normal(size=n), rng.uniform(2, 6, n))
    det = RobustZScore((1, 3)).fit(train)
    assert det.floored_columns == ["f2"] and det.describe()["std_floored_columns"] == ["f2"]
    config = small_config()
    config["zscore"]["std_floor"] = 0.2
    lines = []
    m = run_detector("zscore", "synthetic", RecordingLoader(dict(data, train=train)), config, skip_test=True,
                     run_root=tmp_path, log=lines.append)[0]
    assert m["model"]["std_floor"] == 0.2 and m["model"]["scaler"]["std_floor"] == 0.2
    assert "f2" in m["model"]["std_floored_columns"]
    record = json.loads((Path(m["run_dir"]) / "run.json").read_text())
    assert "f2" in record["metrics"]["model"]["std_floored_columns"]
    assert any("robust scale floored" in line and "f2" in line for line in lines)


def test_array_digest_depends_on_values_shape_and_dtype():
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert array_digest(X) == array_digest(X.copy())
    assert array_digest(X) != array_digest(X.reshape(4, 3))
    assert array_digest(X) != array_digest(X.astype(np.float64))
    Y = X.copy()
    Y[2, 3] += 1
    assert array_digest(X) != array_digest(Y)


def test_warmup_masks_the_start_of_every_segment():
    seg = np.array([0] * 5 + [1] * 3 + [0] * 4)
    assert segment_positions(seg).tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3]
    out = apply_warmup(np.arange(12.0), seg, 2)
    assert np.isnan(out).tolist() == [True, True, False, False, False, True, True, False, True, True, False, False]
    assert np.array_equal(apply_warmup(np.arange(12.0), seg, 0), np.arange(12.0))


def test_smooth_is_causal_and_restarts_at_segments():
    rng = np.random.default_rng(1)
    s = rng.normal(size=40)
    seg = np.repeat([0, 1], 20)
    s[[0, 1, 20, 21, 22]] = np.nan
    out = smooth(s, seg, 4)
    assert np.isnan(out[[0, 1, 20, 21, 22]]).all()
    assert out[2] == pytest.approx(s[2])
    assert out[3] == pytest.approx(s[2:4].mean())
    assert out[10] == pytest.approx(s[7:11].mean())
    assert out[23] == pytest.approx(s[23])
    changed = s.copy()
    changed[15:] += 100
    assert np.allclose(smooth(changed, seg, 4)[2:15], out[2:15])
    changed = s.copy()
    changed[2:20] += 100
    assert np.allclose(smooth(changed, seg, 4)[23:], out[23:])
    assert np.array_equal(smooth(s, seg, 1), s, equal_nan=True)
    bad = s.copy()
    bad[30] = np.nan
    with pytest.raises(ValueError):
        smooth(bad, seg, 3)


def test_time_blocks_never_cross_segments():
    seg = np.repeat([0, 1, 2], [25, 7, 12])
    blocks = time_blocks(seg, 10)
    for b in np.unique(blocks):
        assert np.unique(seg[blocks == b]).size == 1
    assert np.unique(blocks).size == 3 + 1 + 2
    assert np.bincount(np.unique(blocks, return_inverse=True)[1]).tolist() == [10, 10, 5, 7, 10, 2]


def test_alarm_report_matches_metrics_on_one_series():
    rng = np.random.default_rng(2)
    for trial in range(20):
        y = np.zeros(3000, bool)
        for a in rng.choice(2900, 6, replace=False):
            y[a:a + rng.integers(5, 60)] = True
        s = rng.normal(size=3000) + 2 * y
        thr, gap, grace = 1.5, int(rng.integers(0, 6)), float(rng.integers(0, 30))
        rep = alarm_report(s, y, np.zeros(3000, int), thr, 1.0, gap, grace)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            rate = false_alarms_per_hour(s >= thr, y, 1.0, gap, grace)
            ev = event_metrics(s >= thr, y, 1.0, grace)
        assert rep["n_false_alarms"] / rep["normal_hours"] == pytest.approx(rate)
        assert rep["n_events"] == ev["n_events"] and rep["n_detected"] == ev["n_detected"]
        assert np.array_equal(rep["delays_s"], ev["delays_s"], equal_nan=True)


def test_alarm_report_duration_metrics_match_metrics_on_one_series():
    rng = np.random.default_rng(12)
    for trial in range(30):
        y = np.zeros(3000, bool)
        for a in rng.choice(2900, 6, replace=False):
            y[a:a + rng.integers(5, 60)] = True
        s = rng.normal(size=3000) + 2 * y
        thr, gap, grace = 1.2, int(rng.integers(0, 6)), float(rng.integers(0, 30))
        pre, unit = float(rng.integers(0, 40)), float(rng.choice([1, 10, 60]))
        rep = alarm_report(s, y, np.zeros(3000, int), thr, 1.0, gap, grace, pre, unit)
        a = s >= thr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            st = strict_event_metrics(a, y, 1.0, grace, pre, gap)
            dur = duration_aware_false_alarms(a, y, 1.0, grace, unit, pre, gap)
            nat = normal_alarm_time(a, y, 1.0, grace)
        assert rep["n_strict_detected"] == st["n_detected"] and rep["n_pre_alarmed"] == st["n_pre_alarmed"]
        assert rep["strict_event_recall"] == pytest.approx(st["strict_recall"])
        assert np.array_equal(rep["strict_detected"], st["detected"])
        assert np.array_equal(rep["strict_delays_s"], st["delays_s"], equal_nan=True)
        assert np.array_equal(rep["pre_alarmed"], st["pre_alarmed"])
        assert np.array_equal(rep["detected"], st["lenient_detected"])
        assert rep["strict_mean_delay_s"] == pytest.approx(st["mean_delay_s"], nan_ok=True)
        assert rep["n_false_alarms_duration"] == dur["n_false_alarms"]
        assert rep["fa_per_hour_duration"] == pytest.approx(dur["fa_per_hour"])
        assert rep["false_alarm_s"] == dur["false_alarm_s"]
        assert rep["normal_hours"] == pytest.approx(dur["normal_hours"])
        assert rep["normal_alarm_s"] == nat["normal_alarm_s"]
        assert rep["normal_alarm_fraction"] == pytest.approx(nat["normal_alarm_fraction"])


def test_alarm_report_charges_alarm_duration():
    y = np.zeros(8 * 3600, bool)
    for a in (3000, 8000, 13000, 18000, 23000):
        y[a:a + 300] = True
    seg = np.zeros(y.size, int)
    args = {"period_s": 1.0, "merge_gap": 60, "grace_s": 60, "pre_tolerance_s": 60, "fa_unit_s": 60}
    always = alarm_report(np.ones(y.size), y, seg, 0.5, **args)
    late = alarm_report(np.roll(y, 30).astype(float), y, seg, 0.5, **args)
    for rep in (always, late):
        assert rep["fa_per_hour"] == 0 and rep["event_recall"] == 1 and rep["n_detected"] == 5
    assert always["mean_delay_s"] == 0 and late["mean_delay_s"] == 30
    assert always["strict_event_recall"] == 0 and always["n_pre_alarmed"] == 5
    assert always["normal_alarm_fraction"] == 1 and always["normal_hours"] == 7.5
    assert always["n_false_alarms_duration"] == 448 and always["fa_per_hour_duration"] == pytest.approx(448 / 7.5)
    assert late["strict_event_recall"] == 1 and late["n_pre_alarmed"] == 0
    assert late["strict_mean_delay_s"] == 30 and late["strict_median_delay_s"] == 30
    assert late["n_false_alarms_duration"] == 0 and late["normal_alarm_fraction"] == 0
    assert np.array_equal(always["pre_alarmed"], np.ones(5, bool))
    assert np.allclose(always["onset_offset_s"], -np.array([3000, 8000, 13000, 18000, 23000]))


def test_alarm_report_keeps_segments_apart():
    y = np.zeros(200, int)
    y[90:100] = 1
    seg = np.repeat([0, 1], 100)
    s = np.zeros(200)
    s[100:103] = 5.0
    rep = alarm_report(s, y, seg, 1.0, merge_gap=10, grace_s=20)
    assert rep["n_events"] == 1 and rep["n_detected"] == 0
    assert rep["n_false_alarms"] == 1
    single = alarm_report(s, y, np.zeros(200, int), 1.0, merge_gap=10, grace_s=20)
    assert single["n_detected"] == 1 and single["n_false_alarms"] == 0


def test_duration_metrics_keep_segments_apart():
    y = np.zeros(500, int)
    y[90:100] = 1
    seg = np.repeat([0, 1], [100, 400])
    s = np.zeros(500)
    s[95:400] = 5.0
    args = {"merge_gap": 10, "grace_s": 60, "pre_tolerance_s": 60, "fa_unit_s": 60}
    rep = alarm_report(s, y, seg, 1.0, **args)
    assert rep["n_strict_detected"] == 1 and rep["n_false_alarms_duration"] == 5 and rep["false_alarm_s"] == 300
    assert rep["normal_alarm_s"] == 300
    single = alarm_report(s, y, np.zeros(500, int), 1.0, **args)
    assert single["n_false_alarms_duration"] == 4 and single["false_alarm_s"] == 240


def test_fit_thresholds_uses_per_segment_normal_scores():
    rng = np.random.default_rng(3)
    seg = np.repeat([0, 1, 2], 4000)
    s = np.empty(seg.size)
    for k in range(3):
        e = rng.normal(size=4000)
        s[seg == k] = np.convolve(e, np.ones(5) / 5, mode="same")
    s = apply_warmup(s, seg, 7)
    thresholds = fit_thresholds(s, seg, [10.0, 2.0], period_s=1.0, merge_gap=3)
    parts = [s[seg == k][7:] for k in range(3)]
    for target, thr in thresholds.items():
        assert thr == threshold_for_false_alarm_rate(parts, target, 1.0, 3)
        rep = alarm_report(s, np.zeros(seg.size), seg, thr, 1.0, 3, 0.0)
        assert rep["fa_per_hour"] <= target
    assert thresholds[2.0] > thresholds[10.0]


def test_alarm_time_threshold_is_the_lowest_within_budget():
    rng = np.random.default_rng(5)
    parts = [np.round(rng.normal(size=n), 1) for n in (700, 300)]
    s = np.concatenate(parts)
    for frac in (0.0, 0.01, 0.05, 0.3):
        thr = alarm_time_threshold(parts, frac)
        assert np.mean(s >= thr) <= frac + 1e-12
        lower = s[s < thr].max()
        assert np.mean(s >= lower) > frac
    assert alarm_time_threshold(parts, 1.0) == s.min()


def test_normal_thresholds_cap_long_alarms():
    t = np.arange(36000)
    bump = np.where((t >= 5000) & (t < 12000), 3.0 * np.sin(np.pi * (t - 5000) / 7000), 0.0)
    slow = np.where(bump > 0, bump, 0.3 * np.random.default_rng(6).normal(size=t.size))
    parts = [slow]
    plain = normal_thresholds(parts, [1.0], merge_gap=60)
    capped = normal_thresholds(parts, [1.0], merge_gap=60, max_false_alarm_s=60)
    assert np.mean(slow >= plain[1.0]) > 0.1
    assert np.mean(slow >= capped[1.0]) <= 60 / 3600
    assert capped[1.0] >= plain[1.0]
    quiet = [np.random.default_rng(7).normal(size=36000)]
    assert normal_thresholds(quiet, [1.0], merge_gap=60, max_false_alarm_s=3600)[1.0] == \
        threshold_for_false_alarm_rate(quiet, 1.0, 1.0, 60)
    with pytest.raises(ValueError):
        normal_thresholds([], [1.0])


def test_evaluate_reports_cis_and_counts():
    rng = np.random.default_rng(4)
    y = np.zeros(8000, int)
    for a in range(300, 7800, 700):
        y[a:a + 50] = 1
    seg = np.repeat([0, 1], 4000)
    s = apply_warmup(rng.normal(size=8000) + 1.5 * y, seg, 10)
    out = evaluate(s, y, seg, {1.0: 2.0}, block_s=500, n_boot=100, alpha=0.1, seed=0)
    assert out["rows_scored"] == 8000 - 20 and out["labelled_rows_unscored"] == 0
    assert out["n_blocks"] == 16
    ap = out["average_precision"]
    assert ap["lo"] <= ap["point"] <= ap["hi"] and ap["n_clusters"] == 16
    t = out["targets"]["1.0"]
    assert t["n_events"] == 11 and len(t["event_detected"]) == 11
    assert t["event_recall"]["n_clusters"] == 11
    assert t["point_recall"]["point"] == pytest.approx(np.mean(s[y == 1] >= 2.0))


def _three_events():
    rng = np.random.default_rng(10)
    y = np.zeros(8000, int)
    for a in (1000, 3000, 6000):
        y[a:a + 50] = 1
    seg = np.repeat([0, 1], 4000)
    return apply_warmup(rng.normal(size=8000) + 3 * y, seg, 10), y, seg


def test_evaluate_reports_both_alarm_families_and_skips_cis_on_few_clusters():
    s, y, seg = _three_events()
    kw = {"merge_gap": 5, "grace_s": 30, "pre_tolerance_s": 20, "fa_unit_s": 30, "block_s": 500, "n_boot": 100,
          "alpha": 0.1, "seed": 0}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = evaluate(s, y, seg, {1.0: 2.5}, **kw)
    assert not [w for w in caught if "clusters" in str(w.message)]
    t = out["targets"]["1.0"]
    rep = alarm_report(s, y, seg, 2.5, 1.0, 5, 30.0, 20.0, 30.0)
    for key in ("n_strict_detected", "n_pre_alarmed", "n_false_alarms_duration", "fa_per_hour_duration",
                "false_alarm_s", "normal_alarm_fraction", "fa_per_hour", "strict_median_delay_s"):
        assert t[key] == pytest.approx(rep[key], nan_ok=True), key
    assert t["point_precision"] == pytest.approx(np.mean(y[~np.isnan(s)][s[~np.isnan(s)] >= 2.5]))
    assert t["event_strict_detected"] == rep["strict_detected"].tolist() and len(t["event_pre_alarmed"]) == 3
    for key in ("event_recall", "strict_event_recall", "mean_delay_s", "strict_mean_delay_s"):
        assert t[key]["n_clusters"] <= 3 and t[key]["n_boot"] == 0, key
        assert np.isnan(t[key]["lo"]) and np.isnan(t[key]["hi"]), key
    assert t["strict_event_recall"]["point"] == pytest.approx(rep["strict_event_recall"])
    assert t["strict_mean_delay_s"]["point"] == pytest.approx(rep["strict_mean_delay_s"], nan_ok=True)
    ap = out["average_precision"]
    assert ap["n_clusters"] == 16 and ap["lo"] <= ap["point"] <= ap["hi"]
    with pytest.warns(UserWarning, match="only 3 clusters"):
        low = evaluate(s, y, seg, {1.0: 2.5}, min_clusters=1, **kw)
    assert low["targets"]["1.0"]["event_recall"]["n_boot"] == 100


def test_seed_mean_intervals_resample_all_seeds_jointly():
    rng = np.random.default_rng(11)
    y = np.zeros(12000, int)
    for a in range(400, 11600, 900):
        y[a:a + 60] = 1
    seg = np.repeat([0, 1], 6000)
    s1 = apply_warmup(rng.normal(size=12000) + 2 * y, seg, 10)
    s2 = s1 + 0.7 * rng.normal(size=12000)
    thr1, thr2 = {1.0: 2.0, 0.5: 2.5}, {1.0: 2.2, 0.5: 2.7}
    kw = {"merge_gap": 5, "grace_s": 30, "pre_tolerance_s": 20, "fa_unit_s": 30, "block_s": 500, "n_boot": 200,
          "alpha": 0.1, "seed": 0}
    e1, e2 = evaluate(s1, y, seg, thr1, **kw), evaluate(s2, y, seg, thr2, **kw)
    one = seed_mean_intervals([s1], y, seg, [thr1], **kw)
    joint = seed_mean_intervals([s1, s2], y, seg, [thr1, thr2], **kw)
    twin = seed_mean_intervals([s1, s1], y, seg, [thr1, thr1], **kw)
    assert joint["n_seeds"] == 2
    interval_keys = ("point_recall", "event_recall", "strict_event_recall", "mean_delay_s", "strict_mean_delay_s")
    for key in ("average_precision", "roc_auc"):
        assert one[key] == e1[key] and twin[key] == e1[key]
        assert joint[key]["point"] == pytest.approx((e1[key]["point"] + e2[key]["point"]) / 2)
    for t in ("1.0", "0.5"):
        for key in interval_keys:
            assert one["targets"][t][key] == e1["targets"][t][key], key
            assert joint["targets"][t][key]["point"] == pytest.approx(
                (e1["targets"][t][key]["point"] + e2["targets"][t][key]["point"]) / 2), key
    ok = ~np.isnan(s1)
    blocks = time_blocks(seg[ok], 500)
    ref = cluster_bootstrap_ci(lambda yy, a, b: (average_precision(yy, a) + average_precision(yy, b)) / 2,
                               [y[ok].astype(bool), s1[ok], s2[ok]], blocks, n_boot=200, alpha=0.1, seed=0)
    assert joint["average_precision"] == ref
    bad = s2.copy()
    bad[5000] = np.nan
    with pytest.raises(ValueError, match="same rows"):
        seed_mean_intervals([s1, bad], y, seg, [thr1, thr2], **kw)


@pytest.mark.parametrize("name", ["zscore", "pca", "gru"])
def test_detectors_rank_anomalies_above_normal(data, gru, name):
    det = {"zscore": lambda: RobustZScore((1, 3)).fit(data["train"]),
           "pca": lambda: PCAReconstruction((0.9, 0.99)).fit(data["train"]),
           "gru": lambda: gru}[name]()
    test = data["test"]
    for setting, s in det.scores(test["X"], test["segment"]).items():
        s = apply_warmup(s, test["segment"], 15)
        ok = ~np.isnan(s)
        auc = roc_auc(test["y"][ok], s[ok])
        assert auc > 0.95, (name, setting, auc)
        assert np.median(s[ok][test["y"][ok] == 1]) > np.quantile(s[ok][test["y"][ok] == 0], 0.99)


def test_pca_matches_sklearn_reconstruction(data):
    from sklearn.decomposition import PCA

    det = PCAReconstruction((0.6, 0.99)).fit(data["train"])
    Z = det.scaler.transform(data["train"]["X"]).astype(np.float64)
    Zt = det.scaler.transform(data["test"]["X"]).astype(np.float64)
    scores = det.scores(data["test"]["X"])
    full = PCA().fit(Z)
    for v, k in det.n_components.items():
        assert k == int(np.argmax(np.cumsum(full.explained_variance_ratio_) >= v - 1e-12)) + 1
        ref = PCA(n_components=k).fit(Z)
        err = ((Zt - ref.inverse_transform(ref.transform(Zt))) ** 2).sum(axis=1)
        assert np.allclose(scores[f"var{v:g}"], err, rtol=1e-6, atol=1e-8)
    assert det.n_components[0.6] < det.n_components[0.99] == 3


def test_scaling_and_statistics_fit_on_train_only(data, gru):
    train = data["train"]
    X = train["X"]
    shifted = {k: dict(v) for k, v in data.items()}
    for split in ("val", "dev_test", "test"):
        shifted[split]["X"] = data[split]["X"] * 3 + 100
    q25, med, q75 = np.percentile(X, [25, 50, 75], axis=0)

    z = RobustZScore().fit(train, data["val"])
    z2 = RobustZScore().fit(shifted["train"], shifted["val"])
    assert np.allclose(z.scaler.center, med)
    assert np.allclose(z.scaler.scale, (q75 - q25) / 1.3489795, rtol=1e-6)
    assert np.array_equal(z.scaler.scale, z2.scaler.scale) and np.array_equal(z.scaler.center, z2.scaler.center)

    p = PCAReconstruction().fit(train, data["val"])
    p2 = PCAReconstruction().fit(shifted["train"], shifted["val"])
    assert np.allclose(p.scaler.center, X.mean(axis=0, dtype=np.float64))
    assert np.allclose(p.scaler.scale, X.std(axis=0, dtype=np.float64))
    assert np.array_equal(p.components, p2.components)
    test_x = data["test"]["X"]
    assert all(np.array_equal(p.scores(test_x)[k], p2.scores(test_x)[k]) for k in p.scores(test_x))

    for det in (z, p, gru):
        assert det.scaler.n_fit == len(X) and det.scaler.fit_sha256 == array_digest(X)
    assert np.allclose(gru.scaler.center, X.mean(axis=0, dtype=np.float64))
    assert np.allclose(gru.scaler.scale, X.std(axis=0, dtype=np.float64))


def test_forecaster_windows_stay_inside_segments(data, gru):
    split = data["dev_test"]
    X, seg = split["X"].copy(), np.repeat([0, 1], 3000)
    r = gru.residuals(X, seg)
    missing = np.isnan(r).any(axis=1)
    assert np.array_equal(missing, segment_positions(seg) < 15)
    assert np.isnan(r).all(axis=1)[missing].all()
    alone = gru.residuals(X[3000:], seg[3000:])
    assert np.allclose(r[3000:], alone, equal_nan=True, atol=1e-5)
    X[:3000] += 50
    assert np.allclose(gru.residuals(X, seg)[3000:], alone, equal_nan=True, atol=1e-5)
    s = gru.scores(split["X"], seg)
    assert set(s) == {"top1", "topall"} and np.isnan(s["top1"][:15]).all() and np.isnan(s["top1"][3000:3015]).all()
    assert gru.train_info["epochs"] >= 1 and gru.train_info["stop_reason"] in ("patience", "max_epochs", "time_limit")
    assert gru.residual_scale.shape == (N_FEATURES,) and (gru.residual_scale >= gru.residual_floor).all()


def test_forecaster_is_reproducible_on_cpu(data):
    cfg = dict(small_config()["gru"], device="cpu", max_epochs=2, steps_per_epoch=20)
    cfg.pop("seeds")
    a = GRUForecaster(**cfg, seed=5).fit(data["train"], data["val"])
    b = GRUForecaster(**cfg, seed=5).fit(data["train"], data["val"])
    c = GRUForecaster(**cfg, seed=6).fit(data["train"], data["val"])
    x, seg = data["test"]["X"][:500], data["test"]["segment"][:500]
    assert np.allclose(a.residuals(x, seg), b.residuals(x, seg), equal_nan=True)
    assert not np.allclose(a.residuals(x, seg), c.residuals(x, seg), equal_nan=True)


class RecordingLoader:
    def __init__(self, splits):
        self.splits = splits
        self.calls = []

    def __call__(self, split):
        self.calls.append(split)
        return self.splits[split]


def test_pipeline_selects_on_dev_and_loads_test_last(data, tmp_path):
    config = small_config()
    loader = RecordingLoader(data)
    results = run_detector("zscore", "synthetic", loader, config, run_root=tmp_path, log=lambda *_: None)
    assert loader.calls == ["train", "val", "dev_test", "test"]
    m = results[0]
    assert len(m["selection"]["table"]) == 2 * 2
    assert m["setting"] in ("top1", "top3") and m["smoothing"] in (1, 5)
    run_dir = tmp_path / "detect"
    (run,) = list(run_dir.iterdir())
    record = json.loads((run / "run.json").read_text())
    assert record["kind"] == "detect" and record["seed"] == 0 and "finished_utc" in record
    assert json.loads((run / "metrics.json").read_text())["thresholds"] == m["thresholds"]
    for split in ("dev_test", "test"):
        frame = pd.read_parquet(run / f"scores_{split}.parquet")
        assert len(frame) == len(data[split]["y"])
        assert {"timestamp", "file", "segment", "label", "scored", "score_raw", "score",
                "alarm_fa20", "alarm_fa5"} <= set(frame.columns)
        assert (~frame["scored"]).sum() == 15
        assert not frame.loc[~frame["scored"], ["alarm_fa20", "alarm_fa5"]].any().any()
        assert m[split]["rows_scored"] == len(frame) - 15
        assert m[split]["average_precision"]["point"] > 0.5
    for target, rep in m["val"].items():
        assert rep["fa_per_hour"] <= float(target)
        assert rep["alarm_fraction"] <= float(target) * 30 / 3600 + 1e-12


def test_pipeline_thresholds_do_not_depend_on_test(data, tmp_path):
    config = small_config()
    other = dict(data, test=make_split(6000, 99, n_anomalies=25, name="test"))
    quiet = {"log": lambda *_: None}
    a = run_detector("pca", "synthetic", RecordingLoader(data), config, run_root=tmp_path / "a", **quiet)[0]
    b = run_detector("pca", "synthetic", RecordingLoader(other), config, run_root=tmp_path / "b", **quiet)[0]
    assert a["thresholds"] == b["thresholds"]
    assert a["selection"] == b["selection"]
    assert a["dev_test"] == b["dev_test"]
    assert a["test"] != b["test"]
    loader = RecordingLoader(data)
    c = run_detector("pca", "synthetic", loader, config, skip_test=True, run_root=tmp_path / "c", **quiet)[0]
    assert "test" not in loader.calls and "test" not in c
    assert c["thresholds"] == a["thresholds"]
    assert not (tmp_path / "c" / "detect" / next((tmp_path / "c" / "detect").iterdir()).name
                / "scores_test.parquet").exists()


def test_pipeline_rejects_labelled_validation(data, tmp_path):
    bad_val = dict(data["val"], y=data["val"]["y"].copy())
    bad_val["y"][100] = 1
    with pytest.raises(ValueError, match="normal-only"):
        run_detector("zscore", "synthetic", RecordingLoader(dict(data, val=bad_val)), small_config(),
                     run_root=tmp_path, log=lambda *_: None)


def test_selection_averages_over_seeds(data):
    dev = data["dev_test"]
    rng = np.random.default_rng(0)
    good = dev["y"] * 3.0 + rng.normal(size=dev["y"].size)
    weak = dev["y"] * 1.0 + rng.normal(size=dev["y"].size)
    per_seed = [{"a": good, "b": weak}, {"a": weak, "b": good * 1.1 + 0.2 * dev["y"]}]
    out = select_setting(per_seed, dev, small_config())
    assert len(out["table"]) == 4
    for row in out["table"]:
        assert row["criterion"] == pytest.approx(np.mean(row["per_seed"]))
        assert len(row["per_seed"]) == 2 and len(row["recall_per_target"]) == 2
    assert out["criterion"] == max(r["criterion"] for r in out["table"])
    alone = select_setting([{"a": good, "b": weak}], dev, small_config())
    assert alone["setting"] == "a"


def test_dev_criterion_is_immune_to_a_uniform_score_shift(data):
    dev = data["dev_test"]
    rng = np.random.default_rng(1)
    s = dev["y"] * 2.0 + rng.normal(size=dev["y"].size)
    config = small_config()
    base = dev_criterion(s, dev, [20.0, 5.0], config)
    shifted = dev_criterion(s + 10.0, dev, [20.0, 5.0], config)
    assert base[0] == pytest.approx(shifted[0]) and base[2] == pytest.approx(shifted[2])
    parts = normal_series(apply_warmup(s, dev["segment"], 0), dev["y"], dev["segment"], grace_s=5)
    assert sum(p.size for p in parts) == (dev["y"] == 0).sum() - 5 * 10
    assert len(parts) == 11


def test_gru_pipeline_with_two_seeds_and_summary(data, tmp_path):
    config = small_config(detector_seeds=(0, 1))
    config["gru"]["max_epochs"] = 2
    results = run_detector("gru", "synthetic", RecordingLoader(data), config, run_root=tmp_path,
                           log=lambda *_: None)
    assert [m["seed"] for m in results] == [0, 1]
    assert results[0]["setting"] == results[1]["setting"] and results[0]["smoothing"] == results[1]["smoothing"]
    assert results[0]["selection"]["table"][0]["per_seed"].__len__() == 2
    rows, agg = summarize(results)
    assert len(rows) == 4 and len(agg) == 2
    assert (agg["n_seeds"] == 2).all()
    dev = rows[rows["split"] == "dev_test"]["ap"]
    assert agg.loc[agg["split"] == "dev_test", "ap"].iloc[0] == pytest.approx(dev.mean())
    assert agg.loc[agg["split"] == "dev_test", "ap_std"].iloc[0] == pytest.approx(dev.std(ddof=1))
    table = format_table(agg, [20.0, 5.0])
    assert table.count("| synthetic | gru (n=2) |") == 2 + 2 * 2 and "±" not in table
    for split in ("dev_test", "test"):
        joint = results[0]["seed_mean"][split]
        assert results[1]["seed_mean"] == results[0]["seed_mean"] and results[0]["seed_mean"]["seeds"] == [0, 1]
        row = agg[agg["split"] == split].iloc[0]
        assert row["ap"] == pytest.approx(joint["average_precision"]["point"])
        assert row["ap_lo"] == joint["average_precision"]["lo"] and row["ap_hi"] == joint["average_precision"]["hi"]
        assert row["ap_n_clusters"] == joint["average_precision"]["n_clusters"]
        t = joint["targets"]["20.0"]
        assert row["strict_recall@20"] == pytest.approx(t["strict_event_recall"]["point"])
        assert row["strict_recall@20_n_clusters"] == t["strict_event_recall"]["n_clusters"]
        assert row["recall@20_lo"] == t["point_recall"]["lo"]
        per_seed = rows[rows["split"] == split]
        assert per_seed["ap_lo"].tolist() == [m[split]["average_precision"]["lo"] for m in results]
        cells = next(line for line in table.splitlines() if f"| gru (n=2) | {split} | top" in line).split(" | ")
        assert cells[4] == f"{row['ap']:.3f} [{row['ap_lo']:.3f}, {row['ap_hi']:.3f}]"
        assert cells[6].rstrip(" |") == f"{row['ap_std']:.3f} / {row['auc_std']:.3f}"


def test_results_markdown_has_note_legend_and_both_alarm_families(data, tmp_path):
    config = small_config()
    quiet = {"log": lambda *_: None}
    results = (run_detector("zscore", "synthetic", RecordingLoader(data), config, run_root=tmp_path, **quiet)
               + run_detector("pca", "synthetic", RecordingLoader(data), config, run_root=tmp_path, **quiet))
    rows, agg = summarize(results)
    assert (agg["n_seeds"] == 1).all() and len(agg) == 4
    for (_, r), m in zip(agg.iterrows(), [results[0], results[0], results[1], results[1]]):
        ci = m[r["split"]]["average_precision"]
        assert (r["ap"], r["ap_lo"], r["ap_hi"], r["ap_n_clusters"]) == (ci["point"], ci["lo"], ci["hi"],
                                                                          ci["n_clusters"])
        assert np.isnan(r["ap_std"])
        t = m[r["split"]]["targets"]["20.0"]
        assert r["fa_per_hour_duration@20"] == t["fa_per_hour_duration"]
        assert r["n_pre_alarmed@20"] == t["n_pre_alarmed"] and r["n_strict_detected@20"] == t["n_strict_detected"]
        assert r["normal_alarm_pct@20"] == pytest.approx(100 * t["normal_alarm_fraction"])
    for m in results:
        assert set(m["val"]["20.0"]) >= {"fa_per_hour", "fa_per_hour_duration", "normal_alarm_fraction"}
    md = format_results(agg, config, note="v2 provenance note")
    head, rest = md.split("## Legend")
    assert "v2 provenance note" in head
    legend_text, tables = rest.split("## Results")
    for phrase in ("95% percentile cluster-bootstrap CI", "seed sd", "CI of the seed mean", "strict event recall",
                   "pre-alarmed", "FA/h duration", "normal alarm %", "lenient event recall", "GRU use of val",
                   ">= 10 clusters", "fixed a priori"):
        assert phrase in legend_text, phrase
    for column in ("AP [95% CI]", "point precision", "strict event recall [95% CI] (k/n)", "pre-alarmed",
                   "lenient event recall [95% CI] (k/n)", "FA/h segment", "FA/h duration", "normal alarm %",
                   "strict delay s: mean [95% CI] / median (n)"):
        assert column in tables, column
    assert tables.count("| synthetic | zscore |") == 2 + 2 * 2 and tables.count("| synthetic | pca |") == 6


@pytest.mark.data
@pytest.mark.parametrize("version", ["23.05", "22.04"])
def test_heldout_guard_refuses_the_test_split(processed_dir, heldout_names, version):
    from cpsdefense.data import hai

    meta = hai.load_meta(version, processed_dir)
    assert set(meta["splits"]["test"]["files"]) <= heldout_names
    with pytest.raises(AssertionError, match="held-out"):
        hai.load_split(version, "test", processed_dir)
    for stem in meta["splits"]["test"]["files"]:
        with pytest.raises(AssertionError, match="held-out"):
            pd.read_parquet(hai.processed_release_dir(version, processed_dir) / f"{stem}.parquet")


@pytest.mark.data
@pytest.mark.parametrize("version", ["23.05", "22.04"])
def test_hai_features_and_warmup(processed_dir, version):
    from cpsdefense.data import hai

    warmup = warmup_rows(load_config("detect/hai.yaml"))
    meta = hai.load_meta(version, processed_dir)
    splits = {s: hai.load_split(version, s, processed_dir) for s in ("train", "val", "dev_test")}
    train = splits["train"]
    assert "P2_Emerg" not in train["columns"] and "P2_OnOff" not in train["columns"]
    assert train["columns"] == meta["feature_columns"]
    for name, d in splits.items():
        starts, stops = run_bounds(d["segment"])
        assert (stops - starts).min() > warmup, name
        assert d["y"][segment_positions(d["segment"]) < warmup].sum() == 0, name
        assert len(d["y"]) == meta["splits"][name]["rows"] and starts.size == meta["splits"][name]["n_segments"]
        assert d["y"].sum() == meta["splits"][name]["attack_rows"]
    test, lengths = meta["splits"]["test"], []
    for stem in test["files"]:
        info = meta["files"][stem]
        for r in (r for r in info["split_ranges"] if r["split"] == "test"):
            gaps = [g["row"] for g in info["cadence"]["gaps"] if r["start_row"] < g["row"] < r["stop_row"]]
            lengths += np.diff([r["start_row"], *gaps, r["stop_row"]]).tolist()
    assert sum(lengths) == test["rows"] and len(lengths) == test["n_segments"]
    assert min(lengths) > warmup
    det = RobustZScore().fit(train)
    assert det.scaler.fit_sha256 == array_digest(train["X"])
