import warnings

import numpy as np
from scipy import stats

from cpsdefense.eval.metrics import UndefinedMetricWarning
from cpsdefense.model import timecalc

NAN = float("nan")


def _warn(message):
    warnings.warn(message, UndefinedMetricWarning, stacklevel=3)


def _observed(time, event):
    t = np.asarray(time, dtype=float).ravel()
    d = np.asarray(event).ravel()
    if d.dtype != bool:
        if not ((d == 0) | (d == 1)).all():
            raise ValueError("event must be binary (0/1 or bool)")
        d = d.astype(bool)
    if t.size != d.size:
        raise ValueError("time and event must have the same length")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("time must be finite and >= 0")
    return t, d


def _predictions(pmf, edges, n):
    e = timecalc.check_edges(edges)
    p = timecalc.check_pmf(pmf, e)
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError(f"pmf must have shape ({n}, {e.size})")
    return p, e


class KaplanMeier:
    def __init__(self, time, event):
        t, d = _observed(time, event)
        if t.size == 0:
            raise ValueError("Kaplan-Meier needs at least one observation")
        self.time, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
        self.events = np.bincount(inverse, weights=d, minlength=self.time.size)
        self.at_risk = t.size - np.r_[0, np.cumsum(counts)[:-1]]
        self.censored = counts - self.events
        self.survival = np.cumprod(1.0 - self.events / self.at_risk)

    def __call__(self, t, left=False):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.time, t, side="left" if left else "right") - 1
        return np.where(idx >= 0, self.survival[np.maximum(idx, 0)], 1.0)


def kaplan_meier(time, event):
    return KaplanMeier(time, event)


def censoring_km(time, event):
    t, d = _observed(time, event)
    return KaplanMeier(t, ~d)


def km_pmf(km, edges):
    e = timecalc.check_edges(edges)
    s = km(e[1:])
    above = np.r_[1.0, s]
    return np.clip(np.r_[above[:-1] - above[1:], s[-1]], 0.0, None)


class KMBaseline:
    def __init__(self, edges, min_events=1):
        self.edges = timecalc.check_edges(edges)
        self.min_events = int(min_events)
        self.marginal = None
        self.by_group = {}

    def fit(self, time, event, groups=None):
        t, d = _observed(time, event)
        self.marginal = km_pmf(KaplanMeier(t, d), self.edges)
        self.by_group = {}
        if groups is not None:
            g = np.asarray(groups).ravel()
            if g.size != t.size:
                raise ValueError("groups must match time")
            for label in np.unique(g):
                sel = g == label
                if d[sel].sum() >= self.min_events:
                    self.by_group[label] = km_pmf(KaplanMeier(t[sel], d[sel]), self.edges)
        return self

    def pmf(self, groups=None, n=None):
        if self.marginal is None:
            raise RuntimeError("fit() first")
        if groups is None:
            if n is None:
                raise ValueError("give groups or n")
            return np.tile(self.marginal, (int(n), 1))
        rows = [self.by_group.get(label, self.marginal) for label in np.asarray(groups).ravel()]
        return np.stack(rows) if rows else np.zeros((0, self.edges.size))


def nll_discrete(pred, time, event, edges, kind="pmf", rule="linear", tail="inf", eps=1e-12, reduction="mean"):
    if kind not in ("pmf", "hazard"):
        raise ValueError("kind must be 'pmf' or 'hazard'")
    t, d = _observed(time, event)
    p = timecalc.hazards_to_pmf(pred) if kind == "hazard" else pred
    p, e = _predictions(p, edges, t.size)
    k = timecalc.bin_index(e, t)
    ll_event = np.log(np.maximum(p[np.arange(t.size), k], eps))
    ll_cens = np.log(np.maximum(timecalc.survival_at(p, e, t, rule, tail), eps))
    nll = -np.where(d, ll_event, ll_cens)
    if reduction == "none":
        return nll
    if not t.size:
        _warn("NLL is undefined for empty input")
        return NAN
    return float(nll.mean() if reduction == "mean" else nll.sum())


def _ipcw_status(t, d, horizons, censor):
    died = d[:, None] & (t[:, None] <= horizons[None, :])
    alive = t[:, None] > horizons[None, :]
    g_event = censor(t, left=True)[:, None]
    g_horizon = censor(horizons)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(died, 1.0 / g_event, np.where(alive, 1.0 / g_horizon, 0.0))
    return died, alive, weight


def brier_ipcw(pmf, edges, time, event, t, rule="linear", tail="inf", censor=None):
    tt, d = _observed(time, event)
    p, e = _predictions(pmf, edges, tt.size)
    horizons = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    if np.isnan(horizons).any():
        raise ValueError("t contains NaN")
    if not tt.size:
        _warn("Brier score is undefined for empty input")
        out = np.full(horizons.shape, NAN)
    else:
        censor = censoring_km(tt, d) if censor is None else censor
        s = timecalc.survival_at(p[:, None, :], e, horizons[None, :], rule, tail)
        died, alive, weight = _ipcw_status(tt, d, horizons, censor)
        loss = np.where(died, s**2, np.where(alive, (1.0 - s) ** 2, 0.0))
        out = np.mean(np.where(weight > 0, loss * weight, 0.0), axis=0)
    return float(out[0]) if np.ndim(t) == 0 else out


def integrated_brier(pmf, edges, time, event, t_grid, rule="linear", tail="inf", censor=None):
    grid = np.asarray(t_grid, dtype=float).ravel()
    if grid.size < 2 or not (np.diff(grid) > 0).all():
        raise ValueError("t_grid must have at least two strictly increasing times")
    bs = brier_ipcw(pmf, edges, time, event, grid, rule, tail, censor)
    return float(np.trapezoid(bs, grid) / (grid[-1] - grid[0]))


def _comparable_chunks(t, d, chunk=512):
    events = np.flatnonzero(d)
    for start in range(0, events.size, chunk):
        i = events[start : start + chunk]
        ti = t[i][:, None]
        yield i, (t[None, :] > ti) | ((t[None, :] == ti) & ~d[None, :])


def c_index(risk, time, event):
    t, d = _observed(time, event)
    r = np.asarray(risk, dtype=float).ravel()
    if r.size != t.size or np.isnan(r).any():
        raise ValueError("risk must be one non-NaN value per subject")
    num = den = 0.0
    for i, mask in _comparable_chunks(t, d):
        ri = r[i][:, None]
        num += np.sum(mask * ((ri > r[None, :]) + 0.5 * (ri == r[None, :])))
        den += mask.sum()
    if den == 0:
        _warn("concordance is undefined without comparable pairs")
        return NAN
    return float(num / den)


def c_index_td(pmf, edges, time, event, rule="linear", tail="inf"):
    t, d = _observed(time, event)
    p, e = _predictions(pmf, edges, t.size)
    num = den = 0.0
    for i, mask in _comparable_chunks(t, d):
        s_own = timecalc.survival_at(p[i], e, t[i], rule, tail)[:, None]
        s_all = timecalc.survival_at(p[None, :, :], e, t[i][:, None], rule, tail)
        num += np.sum(mask * ((s_own < s_all) + 0.5 * (s_own == s_all)))
        den += mask.sum()
    if den == 0:
        _warn("concordance is undefined without comparable pairs")
        return NAN
    return float(num / den)


def d_calibration(pmf, edges, time, event, n_bins=10, rule="linear", tail="inf"):
    t, d = _observed(time, event)
    p, e = _predictions(pmf, edges, t.size)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    counts = np.zeros(n_bins)
    if not t.size:
        _warn("D-calibration is undefined for empty input")
        return {"counts": counts, "proportions": counts, "statistic": NAN, "p_value": NAN}
    s = np.clip(timecalc.survival_at(p, e, t, rule, tail), 0.0, 1.0)
    b = np.minimum((s * n_bins).astype(np.int64), n_bins - 1)
    counts += np.bincount(b[d], minlength=n_bins)
    sc, bc = s[~d], b[~d]
    lower = np.arange(n_bins)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(lower < bc[:, None], 1.0 / (n_bins * sc[:, None]),
                         np.where(lower == bc[:, None], np.clip(sc[:, None] - bc[:, None] / n_bins, 0.0, None)
                                  / sc[:, None], 0.0))
    share = np.where(sc[:, None] > 0, share, lower == 0)
    counts += share.sum(axis=0)
    expected = t.size / n_bins
    statistic = float(np.sum((counts - expected) ** 2) / expected)
    return {
        "counts": counts,
        "proportions": counts / t.size,
        "statistic": statistic,
        "p_value": float(stats.chi2.sf(statistic, n_bins - 1)),
    }


def _bins(prob, n_bins, strategy):
    if strategy == "uniform":
        return np.searchsorted(np.arange(1, n_bins) / n_bins, prob, side="left")
    if strategy == "quantile":
        return np.searchsorted(np.sort(prob)[(np.arange(1, n_bins) * prob.size) // n_bins], prob, side="right")
    raise ValueError(f"unknown strategy {strategy!r}")


def calibration_at(pmf, edges, time, event, t, n_bins=10, strategy="uniform", rule="linear", tail="inf",
                   censor=None):
    tt, d = _observed(time, event)
    p, e = _predictions(pmf, edges, tt.size)
    horizon = float(t)
    prob = np.clip(timecalc.cdf_at(p, e, horizon, rule, tail), 0.0, 1.0)
    empty = np.zeros(0)
    if not tt.size:
        _warn("calibration is undefined for empty input")
        return {"bin": empty.astype(int), "count": empty.astype(int), "n_known": empty.astype(int),
                "mean_pred": empty, "obs_freq": empty, "ece": NAN}
    censor = censoring_km(tt, d) if censor is None else censor
    died, alive, weight = (a[:, 0] for a in _ipcw_status(tt, d, np.array([horizon]), censor))
    known = died | alive
    weight = np.where(known, weight, 0.0)
    b = _bins(prob, n_bins, strategy)
    count = np.bincount(b, minlength=n_bins)
    keep = count > 0
    n_known = np.bincount(b, weights=known, minlength=n_bins)[keep].astype(int)
    w_sum = np.bincount(b, weights=weight, minlength=n_bins)[keep]
    wy_sum = np.bincount(b, weights=weight * died, minlength=n_bins)[keep]
    mean_pred = np.bincount(b, weights=prob, minlength=n_bins)[keep] / count[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.where(w_sum > 0, wy_sum / w_sum, NAN)
    ok = w_sum > 0
    if ok.any():
        share = count[keep][ok] / count[keep][ok].sum()
        ece = float(np.sum(share * np.abs(obs[ok] - mean_pred[ok])))
    else:
        _warn("calibration is undefined when no subject's status at t is known")
        ece = NAN
    return {"bin": np.flatnonzero(keep), "count": count[keep], "n_known": n_known,
            "mean_pred": mean_pred, "obs_freq": obs, "ece": ece}
