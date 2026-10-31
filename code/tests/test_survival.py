import warnings

import numpy as np
import pytest
from scipy import stats

from cpsdefense.eval import survival as sv
from cpsdefense.eval.metrics import UndefinedMetricWarning
from cpsdefense.model import timecalc as tc


def _censored_sample(seed, n=300, ties=False):
    rng = np.random.default_rng(seed)
    t_true = rng.gamma(2.0, 2.0, n)
    c = rng.exponential(6.0, n)
    time = np.minimum(t_true, c)
    if ties:
        time = np.round(time)
    return time, t_true <= c, rng


def _exp_pmf(rate, edges):
    c = 1 - np.exp(-np.asarray(rate)[:, None] * edges[None, :])
    return np.concatenate([np.diff(c, axis=1), 1 - c[:, -1:]], axis=1)


def _heterogeneous(seed, n=20000):
    rng = np.random.default_rng(seed)
    rate = 0.2 * np.exp(1.5 * rng.uniform(size=n))
    t_true = rng.exponential(1 / rate)
    c = rng.exponential(8.0, n)
    return rate, t_true, np.minimum(t_true, c), t_true <= c


def _km_loop(time, event):
    out, s = {}, 1.0
    for u in np.unique(time):
        at_risk = np.sum(time >= u)
        deaths = np.sum((time == u) & event)
        s *= 1 - deaths / at_risk
        out[u] = s
    return out


def _step(curve, t, left=False):
    keys = [k for k in curve if (k < t if left else k <= t)]
    return curve[max(keys)] if keys else 1.0


def test_kaplan_meier_hand_example():
    time = np.array([1, 2, 2, 3, 4, 5.0])
    event = np.array([1, 1, 0, 1, 0, 1])
    km = sv.kaplan_meier(time, event)
    np.testing.assert_allclose(km.survival, [5 / 6, 4 / 6, 4 / 9, 4 / 9, 0.0])
    np.testing.assert_array_equal(km.at_risk, [6, 5, 3, 2, 1])
    np.testing.assert_allclose(km([0.5, 2.0, 2.5, 4.9, 10.0]), [1, 4 / 6, 4 / 6, 4 / 9, 0])
    np.testing.assert_allclose(km([2.0, 1.0], left=True), [5 / 6, 1.0])
    g = sv.censoring_km(time, event)
    np.testing.assert_allclose(g([1.9, 2.0, 4.0]), [1.0, 4 / 5, 4 / 5 * 1 / 2])


@pytest.mark.parametrize("seed", range(10))
def test_kaplan_meier_matches_loop_with_ties(seed):
    time, event, _ = _censored_sample(seed, n=120, ties=True)
    km = sv.kaplan_meier(time, event)
    ref = _km_loop(time, event)
    probe = np.r_[np.unique(time), np.unique(time) + 0.5, -1.0]
    np.testing.assert_allclose(km(probe), [_step(ref, t) for t in probe], atol=1e-12)
    np.testing.assert_allclose(km(probe, left=True), [_step(ref, t, left=True) for t in probe], atol=1e-12)


def test_kaplan_meier_without_censoring_is_one_minus_ecdf():
    rng = np.random.default_rng(0)
    time = rng.exponential(size=200)
    km = sv.kaplan_meier(time, np.ones(200, bool))
    probe = np.linspace(0, time.max() + 1, 50)
    np.testing.assert_allclose(km(probe), [np.mean(time > t) for t in probe], atol=1e-12)


def test_km_pmf_and_baseline():
    time = np.array([0.5, 1.0, 1.5, 2.5, 2.5, 4.0, 6.0])
    event = np.ones(7, bool)
    edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(sv.km_pmf(sv.kaplan_meier(time, event), edges),
                               np.array([2, 1, 2, 1, 1]) / 7)
    groups = np.array(["a", "a", "a", "b", "b", "b", "c"])
    base = sv.KMBaseline(edges, min_events=2).fit(time, event, groups)
    rows = base.pmf(groups=["a", "b", "c", "unseen"])
    np.testing.assert_allclose(rows[0], [2 / 3, 1 / 3, 0, 0, 0])
    np.testing.assert_allclose(rows[1], [0, 0, 2 / 3, 1 / 3, 0])
    np.testing.assert_allclose(rows[2], base.marginal)
    np.testing.assert_allclose(rows[3], base.marginal)
    assert base.pmf(n=3).shape == (3, 5)
    km = sv.kaplan_meier([1.0, 2.0, 3.0], [1, 0, 0])
    np.testing.assert_allclose(sv.km_pmf(km, [0.0, 1.0, 10.0]), [1 / 3, 0, 2 / 3])


def test_km_baseline_is_d_calibrated_on_its_training_data():
    rng = np.random.default_rng(1)
    n = 20000
    t_true = rng.lognormal(1.0, 0.6, n)
    c = rng.uniform(0, 12, n)
    time, event = np.minimum(t_true, c), t_true <= c
    base = sv.KMBaseline(np.linspace(0, time.max(), 101)).fit(time, event)
    dcal = sv.d_calibration(base.pmf(n=n), base.edges, time, event, n_bins=10, rule="linear")
    assert np.abs(dcal["proportions"] - 0.1).max() < 0.005
    assert dcal["p_value"] > 0.2
    assert sv.c_index_td(base.pmf(n=n)[:2000], base.edges, time[:2000], event[:2000]) == 0.5


def test_nll_hand_values():
    edges = np.array([0.0, 1.0, 3.0])
    pmf = np.tile([0.5, 0.3, 0.2], (5, 1))
    time = np.array([2.0, 2.0, 5.0, 0.0, 5.0])
    event = np.array([1, 0, 1, 0, 0])
    want = -np.log([0.3, 0.35, 0.2, 1.0, 0.2])
    np.testing.assert_allclose(sv.nll_discrete(pmf, time, event, edges, reduction="none"), want)
    assert sv.nll_discrete(pmf, time, event, edges) == pytest.approx(want.mean())
    assert sv.nll_discrete(pmf, time, event, edges, reduction="sum") == pytest.approx(want.sum())
    np.testing.assert_allclose(sv.nll_discrete(tc.pmf_to_hazards(pmf), time, event, edges, kind="hazard",
                                               reduction="none"), want)
    exp_tail = sv.nll_discrete(pmf, time, event, edges, tail="exp", reduction="none")
    assert exp_tail[4] == pytest.approx(-np.log(0.2 * 0.4))


def test_perfect_predictor():
    rng = np.random.default_rng(2)
    n = 60
    t_true = rng.permutation(np.arange(1, n + 1)).astype(float)
    c = rng.uniform(0, n + 5, n)
    time, event = np.minimum(t_true, c), t_true <= c
    edges = np.arange(0, n + 1, dtype=float)
    pmf = np.zeros((n, n + 1))
    pmf[np.arange(n), t_true.astype(int) - 1] = 1.0
    assert sv.nll_discrete(pmf, time, event, edges, rule="upper") == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(sv.brier_ipcw(pmf, edges, time, event, [5.5, 20.0, 40.0], rule="upper"), 0.0)
    assert sv.integrated_brier(pmf, edges, time, event, np.linspace(0, 50, 101), rule="upper") == 0.0
    assert sv.c_index(-t_true, time, event) == 1.0
    assert sv.c_index_td(pmf, edges, time, event, rule="upper") == 1.0
    assert sv.c_index(t_true, time, event) == 0.0


def test_uniform_predictor():
    rng = np.random.default_rng(3)
    n = 20000
    t_true = rng.uniform(0, 10, n)
    c = rng.uniform(0, 20, n)
    time, event = np.minimum(t_true, c), t_true <= c
    edges = np.linspace(0, 10, 11)
    pmf = np.tile(np.r_[np.full(10, 0.1), 0.0], (n, 1))
    assert sv.c_index(np.zeros(n), time, event) == 0.5
    assert sv.c_index_td(pmf[:3000], edges, time[:3000], event[:3000]) == 0.5
    assert sv.d_calibration(pmf, edges, time, event)["p_value"] > 0.01
    for t in (2.0, 5.0, 8.0):
        assert sv.brier_ipcw(pmf, edges, time, event, t) == pytest.approx(t / 10 * (1 - t / 10), abs=0.01)
    assert sv.calibration_at(pmf, edges, time, event, 5.0)["ece"] < 0.01


def _brier_loop(s_t, time, event, t):
    g = _km_loop(time, ~event)
    total = 0.0
    for s, ti, di in zip(s_t, time, event):
        if ti <= t and di:
            total += s**2 / _step(g, ti, left=True)
        elif ti > t:
            total += (1 - s) ** 2 / _step(g, t)
    return total / time.size


@pytest.mark.parametrize("seed", range(8))
def test_brier_matches_loop_reference(seed):
    time, event, rng = _censored_sample(seed, n=150, ties=seed % 2 == 0)
    edges = np.linspace(0, 12, 9)
    pmf = tc.hazards_to_pmf(rng.uniform(0.05, 0.5, size=(time.size, 8)))
    horizons = np.array([0.5, 2.0, 3.0, 5.5, 9.0])
    got = sv.brier_ipcw(pmf, edges, time, event, horizons, rule="pch", tail="exp")
    s = tc.survival_at(pmf[:, None, :], edges, horizons, "pch", "exp")
    want = [_brier_loop(s[:, j], time, event, h) for j, h in enumerate(horizons)]
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert sv.brier_ipcw(pmf, edges, time, event, 3.0, rule="pch", tail="exp") == pytest.approx(want[2])


def test_brier_without_censoring_is_plain_mean_squared_error():
    rng = np.random.default_rng(4)
    time = rng.exponential(3.0, 500)
    edges = np.linspace(0, 10, 21)
    pmf = tc.hazards_to_pmf(rng.uniform(0.02, 0.3, size=(500, 20)))
    for t in (0.7, 3.0, 9.0):
        s = tc.survival_at(pmf, edges, t)
        assert sv.brier_ipcw(pmf, edges, time, np.ones(500, bool), t) == pytest.approx(np.mean(((time > t) - s) ** 2))


def test_ipcw_brier_recovers_the_uncensored_score():
    rate, t_true, time, event = _heterogeneous(5)
    edges = np.linspace(0, 30, 61)
    for name, pmf in (("true", _exp_pmf(rate, edges)), ("wrong", _exp_pmf(np.full(rate.size, 0.4), edges))):
        for t in (1.0, 3.0, 8.0):
            s = tc.survival_at(pmf, edges, t, "pch", "exp")
            oracle = np.mean(((t_true > t) - s) ** 2)
            assert sv.brier_ipcw(pmf, edges, time, event, t, "pch", "exp") == pytest.approx(oracle, abs=0.005), name


def test_integrated_brier_is_the_normalised_trapezoid_and_ranks_models():
    rate, _, time, event = _heterogeneous(6, n=5000)
    edges = np.linspace(0, 30, 61)
    grid = np.linspace(0, 12, 49)
    true_pmf = _exp_pmf(rate, edges)
    bs = sv.brier_ipcw(true_pmf, edges, time, event, grid, "pch", "exp")
    ibs = sv.integrated_brier(true_pmf, edges, time, event, grid, "pch", "exp")
    assert ibs == pytest.approx(np.trapezoid(bs, grid) / 12)
    marginal = sv.KMBaseline(edges).fit(time, event).pmf(n=time.size)
    assert ibs < sv.integrated_brier(marginal, edges, time, event, grid, "linear")
    with pytest.raises(ValueError):
        sv.integrated_brier(true_pmf, edges, time, event, [1.0])


def _comparable(ti, tj, di, dj):
    return di and (tj > ti or (tj == ti and not dj))


def _harrell_loop(risk, time, event):
    num = den = 0.0
    for i in range(time.size):
        for j in range(time.size):
            if i != j and _comparable(time[i], time[j], event[i], event[j]):
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den


def _antolini_loop(pmf, edges, time, event):
    num = den = 0.0
    for i in range(time.size):
        for j in range(time.size):
            if i != j and _comparable(time[i], time[j], event[i], event[j]):
                si = tc.survival_at(pmf[i], edges, time[i])
                sj = tc.survival_at(pmf[j], edges, time[i])
                den += 1
                num += 1.0 if si < sj else 0.5 if si == sj else 0.0
    return num / den


@pytest.mark.parametrize("seed", range(10))
def test_c_index_matches_brute_force_with_ties(seed):
    time, event, rng = _censored_sample(seed, n=80, ties=True)
    risk = np.round(rng.normal(size=80) - 0.2 * time, 1 if seed % 2 else 3)
    assert sv.c_index(risk, time, event) == pytest.approx(_harrell_loop(risk, time, event), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_time_dependent_c_index_matches_brute_force(seed):
    time, event, rng = _censored_sample(seed, n=50, ties=True)
    edges = np.linspace(0, 10, 11)
    pmf = tc.hazards_to_pmf(rng.uniform(0.05, 0.6, size=(50, 10)))
    pmf[::7] = pmf[1]
    assert sv.c_index_td(pmf, edges, time, event) == pytest.approx(_antolini_loop(pmf, edges, time, event),
                                                                   abs=1e-12)


def test_time_dependent_c_index_equals_harrell_for_non_crossing_curves():
    rate, _, time, event = _heterogeneous(7, n=3000)
    edges = np.linspace(0, 30, 61)
    pmf = _exp_pmf(rate, edges)
    td = sv.c_index_td(pmf, edges, time, event, "pch", "exp")
    assert td == pytest.approx(sv.c_index(rate, time, event), abs=1e-12)
    assert td == pytest.approx(sv.c_index(1 - tc.survival_at(pmf, edges, 4.0, "pch", "exp"), time, event),
                               abs=1e-12)


def test_c_index_undefined_without_comparable_pairs():
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(sv.c_index([0.1, 0.2], [1.0, 2.0], [0, 0]))


def test_d_calibration_hand_example():
    edges = np.array([0.0, 1.0])
    pmf = np.tile([1.0, 0.0], (4, 1))
    time = np.array([0.1, 0.6, 0.5, 0.2])
    event = np.array([1, 1, 0, 0])
    out = sv.d_calibration(pmf, edges, time, event, n_bins=4)
    want = np.array([0.5 + 0.3125, 1 + 0.5 + 0.3125, 0.3125, 1 + 0.0625])
    np.testing.assert_allclose(out["counts"], want)
    stat = np.sum((want - 1) ** 2)
    assert out["statistic"] == pytest.approx(stat)
    assert out["p_value"] == pytest.approx(stats.chi2.sf(stat, 3))


def test_d_calibration_separates_calibrated_and_miscalibrated_models():
    rate, _, time, event = _heterogeneous(8)
    edges = np.linspace(0, 30, 61)
    good = sv.d_calibration(_exp_pmf(rate, edges), edges, time, event, rule="pch", tail="exp")
    bad = sv.d_calibration(_exp_pmf(2 * rate, edges), edges, time, event, rule="pch", tail="exp")
    assert good["counts"].sum() == pytest.approx(time.size)
    assert good["p_value"] > 0.01 and np.abs(good["proportions"] - 0.1).max() < 0.01
    assert bad["p_value"] < 1e-6


def test_calibration_at_matches_hand_weights():
    edges = np.array([0.0, 10.0])
    pmf = np.array([[0.2, 0.8], [0.2, 0.8], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    time = np.array([1.0, 6.0, 2.0, 3.0, 7.0])
    event = np.array([1, 0, 0, 1, 1])
    out = sv.calibration_at(pmf, edges, time, event, 5.0, n_bins=2)
    np.testing.assert_array_equal(out["bin"], [0])
    out = sv.calibration_at(pmf, edges, time, event, 5.0, n_bins=10)
    np.testing.assert_array_equal(out["bin"], [0, 2])
    np.testing.assert_array_equal(out["count"], [2, 3])
    np.testing.assert_array_equal(out["n_known"], [2, 2])
    np.testing.assert_allclose(out["mean_pred"], [0.1, 0.25])
    np.testing.assert_allclose(out["obs_freq"], [1 / (1 + 4 / 3), 0.5])
    assert out["ece"] == pytest.approx(2 / 5 * abs(3 / 7 - 0.1) + 3 / 5 * 0.25)


def test_calibration_at_corrects_censoring_bias():
    rate, t_true, time, event = _heterogeneous(9)
    edges = np.linspace(0, 30, 61)
    pmf = _exp_pmf(rate, edges)
    for t in (1.0, 3.0, 8.0):
        out = sv.calibration_at(pmf, edges, time, event, t, rule="pch", tail="exp")
        overall = np.sum(out["obs_freq"] * out["count"]) / time.size
        truth = np.mean(t_true <= t)
        known = (event & (time <= t)) | (time > t)
        naive = np.mean((time <= t)[known])
        assert overall == pytest.approx(truth, abs=0.01)
        assert naive - truth > 0.015
        assert out["ece"] < 0.02
    wrong = sv.calibration_at(_exp_pmf(2 * rate, edges), edges, time, event, 3.0, rule="pch", tail="exp")
    assert wrong["ece"] > 0.1


def test_empty_input_warns_and_returns_nan():
    edges = np.array([0.0, 1.0])
    empty = np.zeros((0, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        with pytest.raises(UndefinedMetricWarning):
            sv.nll_discrete(empty, [], [], edges)
        with pytest.raises(UndefinedMetricWarning):
            sv.brier_ipcw(empty, edges, [], [], 0.5)
        with pytest.raises(UndefinedMetricWarning):
            sv.d_calibration(empty, edges, [], [])
    with pytest.raises(ValueError):
        sv.kaplan_meier([1.0, -1.0], [1, 1])
    with pytest.raises(ValueError):
        sv.brier_ipcw(np.tile([1.0, 0.0], (3, 1)), edges, [0.1, 0.2], [1, 1], 0.5)
