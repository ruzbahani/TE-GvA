import itertools

import numpy as np
import pytest
import torch
from scipy import stats
from scipy.optimize import brentq

from cpsdefense.eval import survival as sv
from cpsdefense.model import timecalc as tc

E2 = np.array([0.0, 1.0, 3.0])
P2 = np.array([0.5, 0.3, 0.2])


def _random_pmf(rng, shape, k, tail=True):
    h = rng.uniform(0.0, 0.6, size=shape + (k,))
    if not tail:
        h[..., -1] = 1.0
    return tc.hazards_to_pmf(h)


def _random_edges(rng, k):
    return np.r_[0.0, np.cumsum(rng.uniform(0.2, 2.0, k))]


@pytest.mark.parametrize("seed", range(20))
def test_hazards_give_monotone_normalised_distribution(seed):
    rng = np.random.default_rng(seed)
    h = rng.uniform(size=(4, 3, 12))
    h[0, 0, 3] = 0.0
    h[1, 2, 5] = 1.0
    pmf = tc.hazards_to_pmf(h)
    s = tc.hazards_to_survival(h)
    assert (pmf >= 0).all()
    np.testing.assert_allclose(pmf.sum(-1), 1.0, atol=1e-12)
    assert (np.diff(s, axis=-1) <= 0).all()
    np.testing.assert_allclose(s, 1.0 - np.cumsum(pmf[..., :-1], axis=-1), atol=1e-12)
    np.testing.assert_allclose(pmf[..., -1], s[..., -1])
    back = tc.pmf_to_hazards(pmf)
    reached = np.concatenate([np.ones_like(s[..., :1]), s[..., :-1]], axis=-1) > 1e-12
    np.testing.assert_allclose(back[reached], h[reached], atol=1e-9)


def test_hazards_reject_values_outside_unit_interval():
    for bad in ([0.2, 1.1], [-0.1, 0.5], [np.nan, 0.5]):
        with pytest.raises(ValueError):
            tc.hazards_to_pmf(bad)


def test_torch_hazards_match_numpy_and_are_differentiable():
    rng = np.random.default_rng(0)
    h = rng.uniform(0.05, 0.95, size=(3, 7))
    ht = torch.tensor(h, dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(tc.hazards_to_pmf_torch(ht).detach().numpy(), tc.hazards_to_pmf(h), atol=1e-14)
    np.testing.assert_allclose(tc.hazards_to_survival_torch(ht).detach().numpy(), tc.hazards_to_survival(h),
                               atol=1e-14)
    assert torch.autograd.gradcheck(tc.hazards_to_pmf_torch, (ht,))
    z = torch.tensor(rng.normal(size=(3, 7)) * 3, dtype=torch.float64, requires_grad=True)
    log_pmf = tc.log_pmf_from_logits(z)
    np.testing.assert_allclose(log_pmf.exp().detach().numpy(), tc.hazards_to_pmf(torch.sigmoid(z).detach().numpy()),
                               atol=1e-13)
    assert torch.autograd.gradcheck(tc.log_pmf_from_logits, (z,))


def test_log_pmf_from_logits_is_stable_for_extreme_logits():
    z = torch.tensor([[40.0, -40.0, 40.0], [-60.0, -60.0, -60.0]], dtype=torch.float64)
    log_pmf = tc.log_pmf_from_logits(z)
    assert torch.isfinite(log_pmf).all()
    np.testing.assert_allclose(torch.logsumexp(log_pmf, -1).numpy(), 0.0, atol=1e-12)


@pytest.mark.parametrize("rule", tc.RULES)
@pytest.mark.parametrize("tail", ["inf", "exp"])
def test_torch_nll_matches_numpy_nll(rule, tail):
    rng = np.random.default_rng(1)
    edges = _random_edges(rng, 6)
    n = 200
    z = torch.tensor(rng.normal(size=(n, 6)), dtype=torch.float64, requires_grad=True)
    time = np.r_[0.0, edges[1:], rng.uniform(0, edges[-1] * 1.4, n - edges.size)]
    event = rng.random(n) < 0.5
    got = tc.nll_from_logits(z, torch.tensor(time), torch.tensor(event), edges, rule, tail, reduction="none")
    pmf = tc.hazards_to_pmf(torch.sigmoid(z).detach().numpy())
    want = sv.nll_discrete(pmf, time, event, edges, rule=rule, tail=tail, reduction="none")
    np.testing.assert_allclose(got.detach().numpy(), want, rtol=1e-10, atol=1e-10)
    got.mean().backward()
    assert torch.isfinite(z.grad).all()


def test_torch_nll_gradcheck_and_refuses_edge_tail():
    rng = np.random.default_rng(2)
    edges = _random_edges(rng, 4)
    z = torch.tensor(rng.normal(size=(9, 4)), dtype=torch.float64, requires_grad=True)
    time = torch.tensor(rng.uniform(0, edges[-1] * 1.3, 9))
    event = torch.tensor(rng.random(9) < 0.5)
    for rule in ("linear", "pch"):
        assert torch.autograd.gradcheck(lambda x: tc.nll_from_logits(x, time, event, edges, rule, "exp"), (z,))
    with pytest.raises(ValueError):
        tc.nll_from_logits(z, time, event, edges, "linear", "edge")


def test_survival_hand_values():
    t = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    want = {
        "linear": [1, 1, 0.75, 0.5, 0.35],
        "pch": [1, 1, np.sqrt(0.5), 0.5, np.sqrt(0.1)],
        "upper": [1, 1, 1, 0.5, 0.5],
        "lower": [1, 0.5, 0.5, 0.2, 0.2],
    }
    tail_at_5 = {"inf": 0.2, "edge": 0.0, "exp": 0.2 * 0.4}
    for rule, values in want.items():
        for tail, beyond in tail_at_5.items():
            s = tc.survival_at(P2, E2, t, rule, tail)
            at_3 = 0.0 if tail == "edge" else 0.2
            np.testing.assert_allclose(s, values + [at_3, beyond], atol=1e-12, err_msg=f"{rule}/{tail}")
            np.testing.assert_allclose(tc.cdf_at(P2, E2, t, rule, tail), 1 - s, atol=1e-12)
    ll = tc.nll_from_logits(torch.logit(torch.tensor([[0.5, 0.6]], dtype=torch.float64)), torch.tensor([1.0]),
                            torch.tensor([False]), E2, "lower")
    assert ll.item() == pytest.approx(-np.log(0.2))


def test_mean_hand_values():
    tail_mean = 3 + 2 / -np.log(0.4)
    assert tc.mean(P2, E2, "linear", "edge") == pytest.approx(0.25 + 0.6 + 0.6)
    assert tc.mean(P2, E2, "linear", "exp") == pytest.approx(0.85 + 0.2 * tail_mean)
    assert tc.mean(P2, E2, "linear", "inf") == np.inf
    assert tc.mean(P2, E2, "upper", "edge") == pytest.approx(0.5 + 0.9 + 0.6)
    assert tc.mean(P2, E2, "lower", "edge") == pytest.approx(0.3 + 0.6)
    closed = np.array([0.5, 0.5, 0.0])
    assert tc.mean(closed, E2, "linear", "inf") == pytest.approx(0.25 + 1.0)


def test_quantile_hand_values():
    q = np.array([0.0, 0.25, 0.65, 0.8, 0.9, 1.0])
    v = np.log(0.5) / np.log(0.4)
    np.testing.assert_allclose(tc.quantile(P2, E2, q, "linear", "exp"), [0, 0.5, 2, 3, 3 + 2 * v, np.inf])
    np.testing.assert_allclose(tc.quantile(P2, E2, q, "linear", "inf"), [0, 0.5, 2, 3, np.inf, np.inf])
    np.testing.assert_allclose(tc.quantile(P2, E2, q, "linear", "edge"), [0, 0.5, 2, 3, 3, 3])
    u = np.log(0.5 / 0.35) / np.log(0.5 / 0.2)
    np.testing.assert_allclose(tc.quantile(P2, E2, q[:4], "pch"), [0, -np.log(0.75) / np.log(2), 1 + 2 * u, 3])
    np.testing.assert_allclose(tc.quantile(P2, E2, q[:4], "upper"), [0, 1, 3, 3])
    np.testing.assert_allclose(tc.quantile(P2, E2, q[:4], "lower"), [0, 0, 1, 1])


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("rule", ["linear", "pch"])
def test_quantile_inverts_continuous_cdf(seed, rule):
    rng = np.random.default_rng(seed)
    edges = _random_edges(rng, 9)
    pmf = _random_pmf(rng, (5,), 9)
    q = rng.uniform(0.001, 0.999, size=(5, 40)) * (1 - pmf[:, -1:])
    t = tc.quantile(pmf[:, None, :], edges, q, rule)
    np.testing.assert_allclose(tc.cdf_at(pmf[:, None, :], edges, t, rule), q, atol=1e-12)
    t = rng.uniform(0, edges[-1], size=(5, 40))
    np.testing.assert_allclose(tc.quantile(pmf[:, None, :], edges, tc.cdf_at(pmf[:, None, :], edges, t, rule), rule),
                               t, atol=1e-9)


@pytest.mark.parametrize("rule", ["upper", "lower"])
def test_quantile_is_generalised_inverse_for_point_rules(rule):
    rng = np.random.default_rng(3)
    edges = _random_edges(rng, 8)
    pmf = _random_pmf(rng, (6,), 8)
    q = rng.uniform(0.001, 0.999, size=(6, 30)) * (1 - pmf[:, -1:])
    t = tc.quantile(pmf[:, None, :], edges, q, rule)
    assert (tc.cdf_at(pmf[:, None, :], edges, t, rule) >= q - 1e-12).all()
    assert (tc.cdf_at(pmf[:, None, :], edges, t - 1e-9, rule) < q).all()


@pytest.mark.parametrize("rule", tc.RULES)
@pytest.mark.parametrize("tail", ["exp", "edge"])
def test_mean_is_integral_of_survival(rule, tail):
    rng = np.random.default_rng(4)
    edges = _random_edges(rng, 7)
    pmf = _random_pmf(rng, (3,), 7)
    end = edges[-1] * (10 if tail == "exp" else 1.01)
    grid = np.linspace(0, end, 1_200_001)
    s = tc.survival_at(pmf[:, None, :], edges, grid, rule, tail)
    integral = np.sum((s[:, 1:] + s[:, :-1]) / 2 * np.diff(grid), axis=-1)
    if tail == "exp":
        r = tc._reverse_cumsum(pmf)
        integral += s[:, -1] * (edges[-1] - edges[-2]) / -np.log(r[:, -1] / r[:, -2])
    np.testing.assert_allclose(tc.mean(pmf, edges, rule, tail), integral, rtol=2e-4)


def test_pch_reproduces_an_exponential_law_exactly():
    law = stats.expon(scale=1 / 0.7)
    edges = np.array([0.0, 0.3, 1.0, 1.1, 2.5, 5.0])
    c = law.cdf(edges)
    pmf = np.r_[np.diff(c), 1 - c[-1]]
    t = np.array([0.05, 0.3, 0.77, 2.0, 4.99, 6.3, 12.0])
    np.testing.assert_allclose(tc.cdf_at(pmf, edges, t, "pch", "exp"), law.cdf(t), atol=1e-14)
    q = np.array([0.01, 0.3, 0.9, 0.99, 0.999])
    np.testing.assert_allclose(tc.quantile(pmf, edges, q, "pch", "exp"), law.ppf(q), rtol=1e-12)
    assert tc.mean(pmf, edges, "pch", "exp") == pytest.approx(law.mean(), rel=1e-12)


def test_linear_reproduces_a_uniform_law_exactly():
    edges = np.array([0.0, 0.5, 3.0, 3.2, 7.0, 10.0])
    pmf = np.r_[np.diff(edges) / 10.0, 0.0]
    t = np.linspace(0, 10, 57)
    np.testing.assert_allclose(tc.cdf_at(pmf, edges, t, "linear"), t / 10, atol=1e-14)
    assert tc.mean(pmf, edges, "linear") == pytest.approx(5.0)


@pytest.mark.parametrize("rule", tc.RULES)
@pytest.mark.parametrize("tail", tc.TAILS)
def test_regrid_is_exact_at_the_new_edges(rule, tail):
    rng = np.random.default_rng(5)
    edges = _random_edges(rng, 6)
    pmf = _random_pmf(rng, (4,), 6)
    new_edges = np.r_[0.0, np.sort(rng.uniform(0, edges[-1] * 1.5, 9))]
    new = tc.regrid(pmf, edges, new_edges, rule, tail)
    np.testing.assert_allclose(new.sum(-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(tc.survival_at(new[:, None, :], new_edges, new_edges[1:], "upper"),
                               tc.survival_at(pmf[:, None, :], edges, new_edges[1:], rule, tail), atol=1e-12)


@pytest.mark.parametrize("rule", tc.RULES)
@pytest.mark.parametrize("tail", ["inf", "exp"])
def test_regrid_onto_a_refinement_keeps_the_law(rule, tail):
    rng = np.random.default_rng(5)
    edges = _random_edges(rng, 6)
    pmf = _random_pmf(rng, (4,), 6)
    fine = np.unique(np.r_[edges, rng.uniform(0, edges[-1] * 1.5, 25)])
    new = tc.regrid(pmf, edges, fine, rule, tail)
    t = np.linspace(0, fine[-1], 301)
    old_s = tc.survival_at(pmf[:, None, :], edges, t, rule, tail)
    new_s = tc.survival_at(new[:, None, :], fine, t, rule, "inf")
    if rule == "lower":
        assert (new_s <= old_s + 1e-12).all()
    else:
        inside = t <= edges[-1] if tail == "exp" else slice(None)
        np.testing.assert_allclose(new_s[:, inside], old_s[:, inside], atol=1e-12)


def test_elementwise_broadcasting_matches_row_loop():
    rng = np.random.default_rng(6)
    edges = _random_edges(rng, 5)
    pmf = _random_pmf(rng, (7,), 5)
    t = rng.uniform(0, edges[-1] * 1.2, 11)
    grid = tc.survival_at(pmf[:, None, :], edges, t, "pch", "exp")
    assert grid.shape == (7, 11)
    for i in range(7):
        np.testing.assert_allclose(grid[i], tc.survival_at(pmf[i], edges, t, "pch", "exp"))
    paired = tc.survival_at(pmf, edges, t[:7], "linear")
    np.testing.assert_allclose(paired, [tc.survival_at(pmf[i], edges, t[i], "linear") for i in range(7)])


def test_input_validation():
    with pytest.raises(ValueError):
        tc.check_edges([0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        tc.check_edges([0.5, 1.0])
    with pytest.raises(ValueError):
        tc.check_edges([0.0, np.inf])
    with pytest.raises(ValueError):
        tc.survival_at([0.5, 0.4, 0.2], E2, 1.0)
    with pytest.raises(ValueError):
        tc.survival_at([0.5, 0.5], E2, 1.0)
    with pytest.raises(ValueError):
        tc.survival_at(P2, E2, 1.0, rule="spline")
    with pytest.raises(ValueError):
        tc.quantile(P2, E2, 1.5)
    with pytest.raises(ValueError):
        tc.bin_index(E2, [-1.0])
    np.testing.assert_array_equal(tc.bin_index(E2, [0.0, 0.5, 1.0, 1.01, 3.0, 3.5]), [0, 0, 0, 1, 1, 2])


def _binned(law, n_bins=60, lo_q=0.001, hi_q=0.999):
    edges = np.r_[0.0, np.geomspace(law.ppf(lo_q), law.ppf(hi_q), n_bins)]
    c = law.cdf(edges)
    c[0] = 0.0
    return np.r_[np.diff(c), 1 - c[-1]], edges


class _Mixture:
    def __init__(self, w, a, b):
        self.w, self.a, self.b = w, a, b

    def cdf(self, x):
        return self.w * self.a.cdf(x) + (1 - self.w) * self.b.cdf(x)

    def ppf(self, q):
        return brentq(lambda x: self.cdf(x) - q, 0.0, 1e7)

    def rvs(self, n, random_state):
        pick = random_state.random(n) < self.w
        return np.where(pick, self.a.rvs(n, random_state=random_state), self.b.rvs(n, random_state=random_state))

PIPELINE = [
    stats.gamma(2, scale=0.5),
    stats.lognorm(0.25, scale=0.3),
    _Mixture(0.9, stats.lognorm(0.3, scale=2.0), stats.lognorm(0.5, scale=20.0)),
    stats.lognorm(1.2, scale=5.0),
    stats.gamma(3, scale=2.0),
]


def _mc_sum(laws, n, seed):
    rng = np.random.default_rng(seed)
    return sum(law.rvs(n, random_state=rng) for law in laws)


def _sample_binned(pmf, edges, n, rng, rule):
    k = rng.choice(pmf.size, size=n, p=pmf / pmf.sum())
    lo = edges[np.minimum(k, edges.size - 2)]
    w = np.diff(edges)[np.minimum(k, edges.size - 2)]
    r = tc._reverse_cumsum(pmf)
    if rule == "linear":
        x = lo + w * rng.random(n)
    else:
        ratio = np.where(r[:-1] > 0, r[1:] / np.where(r[:-1] > 0, r[:-1], 1), 1)[np.minimum(k, edges.size - 2)]
        u = rng.random(n)
        x = lo + w * np.log1p(-u * (1 - ratio)) / np.log(np.where(ratio > 0, ratio, 1e-300))
    rate = -np.log(r[-1] / r[-2]) / (edges[-1] - edges[-2])
    return np.where(k == edges.size - 1, edges[-1] + rng.exponential(1 / rate, n), x)


def test_exact_lattice_case_matches_brute_force_enumeration():
    rng = np.random.default_rng(7)
    comps = [(_random_pmf(rng, (), 4), np.array([0.0, 1, 2, 4, 7])),
             (_random_pmf(rng, (), 3), np.array([0.0, 2, 3, 5])),
             (_random_pmf(rng, (), 2, tail=False), np.array([0.0, 1, 3]))]
    points = [list(zip(e[1:], p[:-1])) for p, e in comps]
    for d in range(0, 16):
        exact = sum(np.prod([m for _, m in combo]) for combo in itertools.product(*points)
                    if sum(x for x, _ in combo) <= d)
        got = tc.prob_total_within(comps, float(d), rule="upper", tail="inf", bound="lower", n_grid=max(d, 1))
        assert got == pytest.approx(exact, abs=1e-13), d


def test_lower_bound_never_rounds_a_deadline_up_to_a_point_mass():
    comps = [(np.array([1.0, 0.0]), np.array([0.0, 10.0]))]
    for gap in (1e-12, 1e-10, 1e-8):
        d = np.array([2048.0, 10.0 - gap])
        lower = tc.prob_total_within(comps, d, rule="upper", tail="inf", bound="lower")
        assert lower[1] == 0.0, gap
        assert not tc.feasible(comps, d, delta=0.05, rule="upper", tail="inf")[1]
    on_point = tc.prob_total_within(comps, np.array([2048.0, 10.0]), rule="upper", tail="inf", bound="lower")
    assert on_point[1] == 1.0


def test_convolve_places_the_same_index_law_at_both_ends():
    rng = np.random.default_rng(8)
    comps = [(_random_pmf(rng, (), 5), _random_edges(rng, 5)) for _ in range(3)]
    d = np.array([1.01, 4.02, 7.53, 11.0])
    dt = d.max() / 220
    up, grid = tc.convolve(comps, dt, 12.0, "linear", "exp", place="upper")
    low, grid2 = tc.convolve(comps, dt, 12.0, "linear", "exp", place="lower")
    np.testing.assert_array_equal(grid, grid2)
    np.testing.assert_allclose(up[2:-1], low[: up.size - 3], atol=1e-15)
    assert np.allclose(up.sum(), 1) and np.allclose(low.sum(), 1)
    lower = tc.prob_total_within(comps, d, "linear", "exp", "lower", n_grid=220)
    upper = tc.prob_total_within(comps, d, "linear", "exp", "upper", n_grid=220)
    np.testing.assert_allclose(tc.cdf_at(up, grid, d[:3], "upper"), lower[:3], atol=1e-12)
    np.testing.assert_allclose(tc.cdf_at(low, grid, d[:3], "lower"), upper[:3], atol=1e-12)


@pytest.mark.parametrize("rule", ["linear", "pch"])
def test_bounds_bracket_monte_carlo_of_the_binned_law(rule):
    rng = np.random.default_rng(9)
    comps = [_binned(law, n_bins=25) for law in PIPELINE]
    n = 1_000_000
    total = sum(_sample_binned(p, e, n, rng, rule) for p, e in comps)
    d = np.quantile(total, [0.1, 0.5, 0.9, 0.95, 0.99])
    p_mc = np.array([np.mean(total <= x) for x in d])
    sigma = np.sqrt(p_mc * (1 - p_mc) / n)
    gaps = []
    for n_grid in (256, 1024):
        lower = np.array([tc.prob_total_within(comps, x, rule, "exp", "lower", n_grid) for x in d])
        upper = np.array([tc.prob_total_within(comps, x, rule, "exp", "upper", n_grid) for x in d])
        assert (lower <= p_mc + 4 * sigma).all() and (upper >= p_mc - 4 * sigma).all()
        gaps.append(upper - lower)
    assert (gaps[1] < 0.35 * gaps[0]).all()
    mid = np.array([tc.prob_total_within(comps, x, rule, "exp", "mid", 2048) for x in d])
    np.testing.assert_allclose(mid, p_mc, atol=4 * sigma.max() + 1e-4)


def test_convolution_matches_monte_carlo_of_continuous_components():
    n = 1_000_000
    total = _mc_sum(PIPELINE, n, seed=10)
    comps = [_binned(law) for law in PIPELINE]
    for q in (0.05, 0.5, 0.9, 0.95, 0.99, 0.999):
        d = np.quantile(total, q)
        p_mc = np.mean(total <= d)
        sigma = np.sqrt(p_mc * (1 - p_mc) / n)
        lower = tc.prob_total_within(comps, d, rule="upper", tail="inf", bound="lower")
        upper = tc.prob_total_within(comps, d, rule="lower", tail="edge", bound="upper")
        assert lower <= p_mc + 4 * sigma and upper >= p_mc - 4 * sigma, q
        for rule in ("linear", "pch"):
            est = tc.prob_total_within(comps, d, rule=rule, tail="exp", bound="mid")
            assert abs(est - p_mc) < 4 * sigma + 2e-3, (q, rule)
            assert abs((1 - est) - (1 - p_mc)) < 4 * sigma + 0.1 * (1 - p_mc), (q, rule)
    law, grid = tc.convolve(comps, 0.05, 400.0, "pch", "exp", place="upper")
    for q in (0.5, 0.95, 0.99):
        assert tc.quantile(law, grid, q, "upper") == pytest.approx(np.quantile(total, q), rel=0.02)


def test_heavy_tailed_sum_tail_probability():
    laws = [stats.lognorm(1.5, scale=1.0)] * 3 + [stats.pareto(2.5, scale=0.5)]
    n = 2_000_000
    total = _mc_sum(laws, n, seed=11)
    comps = [_binned(law, n_bins=80, hi_q=0.9999) for law in laws]
    for q in (0.99, 0.999):
        d = np.quantile(total, q)
        miss_mc = np.mean(total > d)
        est = 1 - tc.prob_total_within(comps, d, rule="pch", tail="exp", bound="mid", n_grid=4096)
        assert abs(est - miss_mc) / miss_mc < 0.1, q
        conservative = 1 - tc.prob_total_within(comps, d, rule="upper", tail="inf", bound="lower", n_grid=4096)
        assert conservative >= miss_mc * (1 - 4 / np.sqrt(miss_mc * n))


def test_quantile_sum_is_anti_conservative_with_rare_large_delays():
    law = _Mixture(0.97, stats.gamma(2, scale=0.5), stats.lognorm(0.3, scale=30.0))
    comps = [_binned(law, n_bins=80)] * 5
    total = _mc_sum([law] * 5, 1_000_000, seed=12)
    d = float(tc.quantile_sum(comps, 0.95, "pch", "exp"))
    assert d < 20
    p_mc = np.mean(total <= d)
    assert p_mc < 0.9
    assert tc.prob_total_within(comps, d, "pch", "exp", "mid") == pytest.approx(p_mc, abs=5e-3)
    assert not tc.feasible(comps, d, delta=0.05, rule="pch", tail="exp")


def test_quantile_sum_is_over_conservative_for_light_tailed_components():
    law = stats.gamma(2, scale=0.5)
    comps = [_binned(law, n_bins=60)] * 5
    total = _mc_sum([law] * 5, 1_000_000, seed=13)
    qsum = float(tc.quantile_sum(comps, 0.95, "pch", "exp"))
    true_q95 = stats.gamma(10, scale=0.5).ppf(0.95)
    assert qsum == pytest.approx(5 * law.ppf(0.95), rel=1e-3) and qsum > 1.4 * true_q95
    d = (qsum + true_q95) / 2
    assert np.mean(total <= d) > 0.99
    assert tc.feasible(comps, d, delta=0.05, rule="pch", tail="exp")
    assert qsum > d


def test_feasible_answers_are_conservative_on_random_cases():
    rng = np.random.default_rng(14)
    for _ in range(20):
        comps = [(_random_pmf(rng, (), 6), _random_edges(rng, 6)) for _ in range(int(rng.integers(1, 5)))]
        d = rng.uniform(1, 20)
        delta = rng.choice([0.01, 0.05, 0.1])
        truth = tc.prob_total_within(comps, d, "linear", "exp", "lower", n_grid=16384)
        if tc.feasible(comps, d, delta, "linear", "exp", n_grid=256):
            assert truth >= 1 - delta
        lo = tc.prob_total_within(comps, d, "linear", "exp", "lower", n_grid=256)
        hi = tc.prob_total_within(comps, d, "linear", "exp", "upper", n_grid=256)
        fine_hi = tc.prob_total_within(comps, d, "linear", "exp", "upper", n_grid=16384)
        assert lo <= truth + 1e-12 and truth <= fine_hi + 1e-12 and fine_hi <= hi + 1e-12


def test_batched_candidates_with_shared_pipeline_components():
    rng = np.random.default_rng(15)
    edges = _random_edges(rng, 8)
    actions = _random_pmf(rng, (4, 3), 8)
    shared = [(_random_pmf(rng, (), 5), _random_edges(rng, 5)) for _ in range(2)]
    deadline = rng.uniform(3, 15, size=(4, 1))
    got = tc.prob_total_within([(actions, edges)] + shared, deadline, "pch", "exp", "lower", n_grid=512)
    assert got.shape == (4, 3)
    for i, j in itertools.product(range(4), range(3)):
        same_grid = np.array([deadline[i, 0], deadline.max()])
        row = tc.prob_total_within([(actions[i, j], edges)] + shared, same_grid, "pch", "exp", "lower", n_grid=512)
        assert got[i, j] == pytest.approx(row[0], abs=1e-12)
    ok = tc.feasible([(actions, edges)] + shared, deadline, 0.1, "pch", "exp", n_grid=512)
    np.testing.assert_array_equal(ok, got >= 0.9)


def test_non_positive_deadline_is_never_met():
    comps = [(P2, E2)]
    assert tc.prob_total_within(comps, 0.0) == 0.0
    np.testing.assert_array_equal(tc.prob_total_within(comps, np.array([-1.0, 0.0])), [0.0, 0.0])
    with pytest.raises(ValueError):
        tc.feasible(comps, 1.0, delta=1.0)
