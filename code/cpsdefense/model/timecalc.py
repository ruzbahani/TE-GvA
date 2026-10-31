import numpy as np
import scipy.fft

RULES = ("linear", "pch", "upper", "lower")
TAILS = ("inf", "exp", "edge")


def check_edges(edges):
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges must be 1-D with at least two values")
    if not np.isfinite(e).all() or e[0] != 0 or not (np.diff(e) > 0).all():
        raise ValueError("edges must be finite, start at 0 and increase strictly")
    return e


def check_pmf(pmf, edges, atol=1e-6):
    p = np.asarray(pmf, dtype=float)
    if p.ndim == 0 or p.shape[-1] != len(edges):
        raise ValueError(f"pmf must have a last axis of length {len(edges)} (K finite bins + open bin)")
    if np.isnan(p).any() or (p < 0).any():
        raise ValueError("pmf must be non-negative and free of NaN")
    if p.size and not np.allclose(p.sum(-1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("pmf rows must sum to 1 (the open last bin holds P(T > e_K))")
    return p


def _check_rule(rule, tail):
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}, got {tail!r}")


def _times(t, name="t"):
    t = np.asarray(t, dtype=float)
    if np.isnan(t).any():
        raise ValueError(f"{name} contains NaN")
    return t


def _check_hazards(hazards):
    h = np.asarray(hazards, dtype=float)
    if h.ndim == 0 or h.shape[-1] == 0:
        raise ValueError("hazards must have a non-empty last axis")
    if np.isnan(h).any() or ((h < 0) | (h > 1)).any():
        raise ValueError("hazards must lie in [0, 1]")
    return h


def hazards_to_survival(hazards):
    return np.cumprod(1.0 - _check_hazards(hazards), axis=-1)


def hazards_to_pmf(hazards):
    h = _check_hazards(hazards)
    s = np.cumprod(1.0 - h, axis=-1)
    prev = np.concatenate([np.ones_like(s[..., :1]), s[..., :-1]], axis=-1)
    return np.concatenate([prev * h, s[..., -1:]], axis=-1)


def pmf_to_hazards(pmf):
    p = np.asarray(pmf, dtype=float)
    r = _reverse_cumsum(p)[..., :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, np.clip(p[..., :-1] / np.where(r > 0, r, 1.0), 0.0, 1.0), 1.0)


def hazards_to_survival_torch(hazards):
    import torch

    return torch.cumprod(1 - hazards, dim=-1)


def hazards_to_pmf_torch(hazards):
    import torch

    s = torch.cumprod(1 - hazards, dim=-1)
    prev = torch.cat([torch.ones_like(s[..., :1]), s[..., :-1]], dim=-1)
    return torch.cat([prev * hazards, s[..., -1:]], dim=-1)


def log_pmf_from_logits(logits):
    import torch
    import torch.nn.functional as F

    log_s = torch.cumsum(F.logsigmoid(-logits), dim=-1)
    log_prev = torch.cat([torch.zeros_like(log_s[..., :1]), log_s[..., :-1]], dim=-1)
    return torch.cat([log_prev + F.logsigmoid(logits), log_s[..., -1:]], dim=-1)


def nll_from_logits(logits, time, event, edges, rule="linear", tail="inf", reduction="mean"):
    import torch
    import torch.nn.functional as F

    _check_rule(rule, tail)
    if tail == "edge":
        raise ValueError("tail='edge' has no likelihood for censoring beyond e_K")
    e = torch.as_tensor(check_edges(edges), dtype=logits.dtype, device=logits.device)
    n_bins = e.numel() - 1
    if logits.shape[-1] != n_bins:
        raise ValueError(f"logits must have {n_bins} bins")
    t = torch.as_tensor(time, dtype=logits.dtype, device=logits.device)
    ev = torch.as_tensor(event, device=logits.device).bool()
    log_pmf = log_pmf_from_logits(logits)
    log_keep = F.logsigmoid(-logits)
    log_r = torch.cat([torch.zeros_like(log_keep[..., :1]), torch.cumsum(log_keep, dim=-1)], dim=-1)

    k_event = (torch.searchsorted(e, t, right=False) - 1).clamp(0, n_bins)
    ll_event = log_pmf.gather(-1, k_event[..., None])[..., 0]

    k = k_event.clamp(max=n_bins - 1)
    u = ((t - e[k]) / (e[k + 1] - e[k])).clamp(0, 1)
    lo = log_r.gather(-1, k[..., None])[..., 0]
    hi = log_r.gather(-1, (k + 1)[..., None])[..., 0]
    if rule == "linear":
        ll_cens = torch.logaddexp(hi, log_pmf.gather(-1, k[..., None])[..., 0] + torch.log1p(-u))
    elif rule == "pch":
        ll_cens = lo + u * log_keep.gather(-1, k[..., None])[..., 0]
    elif rule == "upper":
        ll_cens = torch.where(u < 1, lo, hi)
    else:
        j = (torch.searchsorted(e, t, right=True) - 1).clamp(0, n_bins - 1)
        ll_cens = log_r.gather(-1, (j + 1)[..., None])[..., 0]
    if tail == "exp":
        beyond = ((t - e[-1]) / (e[-1] - e[-2])).clamp(min=0)
        ll_cens = ll_cens + beyond * log_keep[..., -1]
    nll = -torch.where(ev, ll_event, ll_cens)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"unknown reduction {reduction!r}")


def _reverse_cumsum(p):
    return np.flip(np.cumsum(np.flip(p, -1), -1), -1)


def _gather(a, idx):
    shape = np.broadcast_shapes(a.shape[:-1], np.shape(idx))
    a = a.reshape((1,) * (len(shape) - a.ndim + 1) + a.shape)
    idx = np.broadcast_to(idx, shape)
    return np.take_along_axis(a, idx[..., None], axis=-1)[..., 0]


def _last_bin_ratio(r):
    prev, last = r[..., -2], r[..., -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(prev > 0, last / np.where(prev > 0, prev, 1.0), 0.0)


def bin_index(edges, t):
    e = check_edges(edges)
    t = _times(t)
    if (t < 0).any():
        raise ValueError("times must be >= 0")
    return np.clip(np.searchsorted(e, t, side="left") - 1, 0, e.size - 1)


def survival_at(pmf, edges, t, rule="linear", tail="inf"):
    e = check_edges(edges)
    p = check_pmf(pmf, e)
    _check_rule(rule, tail)
    t = _times(t)
    n_bins = e.size - 1
    r = _reverse_cumsum(p)
    k = np.clip(np.searchsorted(e, t, side="left") - 1, 0, n_bins - 1)
    u = np.clip((t - e[k]) / (e[k + 1] - e[k]), 0.0, 1.0)
    lo, hi = _gather(r, k), _gather(r, k + 1)
    if rule == "linear":
        s = hi + _gather(p, k) * (1.0 - u)
    elif rule == "pch":
        s = lo ** (1.0 - u) * hi**u
    elif rule == "upper":
        s = np.where(u < 1, lo, hi)
    else:
        s = _gather(r, np.clip(np.searchsorted(e, t, side="right") - 1, 0, n_bins - 1) + 1)
    if tail == "edge":
        s = np.where(t >= e[-1], 0.0, s)
    elif tail == "exp":
        beyond = np.maximum(t - e[-1], 0.0) / (e[-1] - e[-2])
        with np.errstate(invalid="ignore"):
            decay = np.where(beyond > 0, _last_bin_ratio(r) ** beyond, 1.0)
        s = s * decay
    return np.where(t < 0, r[..., 0], s)


def cdf_at(pmf, edges, t, rule="linear", tail="inf"):
    return 1.0 - survival_at(pmf, edges, t, rule, tail)


def quantile(pmf, edges, q, rule="linear", tail="inf"):
    e = check_edges(edges)
    p = check_pmf(pmf, e)
    _check_rule(rule, tail)
    q = _times(q, "q")
    if ((q < 0) | (q > 1)).any():
        raise ValueError("q must lie in [0, 1]")
    n_bins = e.size - 1
    r = _reverse_cumsum(p)
    s = 1.0 - q
    k = np.sum(1.0 - r[..., 1:] < q[..., None], axis=-1)
    kk = np.minimum(k, n_bins - 1)
    lo, hi, mass = _gather(r, kk), _gather(r, kk + 1), _gather(p, kk)
    with np.errstate(divide="ignore", invalid="ignore"):
        if rule == "linear":
            u = np.where(mass > 0, 1.0 - (s - hi) / np.where(mass > 0, mass, 1.0), 0.0)
        elif rule == "pch":
            u = np.where(hi > 0, np.log(lo / s) / np.log(lo / hi), 0.0)
        elif rule == "upper":
            u = np.ones_like(s * lo)
        else:
            u = np.zeros_like(s * lo)
        t = e[kk] + np.clip(np.nan_to_num(u, nan=0.0), 0.0, 1.0) * (e[kk + 1] - e[kk])
        if tail == "inf":
            t_tail = np.full_like(t, np.inf)
        elif tail == "edge":
            t_tail = np.full_like(t, e[-1])
        else:
            ratio = _last_bin_ratio(r)
            r_last = r[..., -1]
            beyond = np.maximum(np.log(s / r_last) / np.log(ratio), 0.0)
            t_tail = np.where(ratio < 1, e[-1] + (e[-1] - e[-2]) * beyond, np.inf)
    t = np.where(k >= n_bins, t_tail, t)
    return np.where(q == 0, 0.0, t)


def _trunc_exp_mean_frac(a):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exact = 1.0 / a - 1.0 / np.expm1(a)
    return np.where(a < 1e-4, 0.5 - a / 12.0, exact)


def mean(pmf, edges, rule="linear", tail="inf"):
    e = check_edges(edges)
    p = check_pmf(pmf, e)
    _check_rule(rule, tail)
    w = np.diff(e)
    r = _reverse_cumsum(p)
    if rule == "linear":
        at = e[:-1] + w / 2
    elif rule == "upper":
        at = e[1:]
    elif rule == "lower":
        at = e[:-1]
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(r[..., :-1] > 0, r[..., 1:] / np.where(r[..., :-1] > 0, r[..., :-1], 1.0), 1.0)
            at = e[:-1] + w * _trunc_exp_mean_frac(-np.log(ratio))
    finite = np.sum(p[..., :-1] * at, axis=-1)
    r_last = r[..., -1]
    if tail == "edge":
        return finite + r_last * e[-1]
    if tail == "inf":
        return np.where(r_last > 0, np.inf, finite)
    with np.errstate(divide="ignore"):
        rate_w = -np.log(_last_bin_ratio(r))
        tail_mean = np.where(rate_w > 0, e[-1] + (e[-1] - e[-2]) / rate_w, np.inf)
    return finite + np.where(r_last > 0, r_last * tail_mean, 0.0)


def regrid(pmf, edges, new_edges, rule="linear", tail="inf"):
    e = check_edges(edges)
    g = check_edges(new_edges)
    p = check_pmf(pmf, e)
    s = survival_at(p[..., None, :], e, g[1:], rule, tail)
    above = np.concatenate([p.sum(-1, keepdims=True), s], axis=-1)
    return np.clip(np.concatenate([above[..., :-1] - above[..., 1:], s[..., -1:]], axis=-1), 0.0, None)


def _components(dists):
    dists = list(dists)
    if not dists:
        raise ValueError("need at least one component (pmf, edges)")
    return dists


def _conv_truncated(a, b, n):
    size = scipy.fft.next_fast_len(a.shape[-1] + b.shape[-1] - 1, real=True)
    out = scipy.fft.irfft(scipy.fft.rfft(a, size, axis=-1) * scipy.fft.rfft(b, size, axis=-1), size, axis=-1)
    return np.clip(out[..., :n], 0.0, None)


def _index_pmf(dists, dt, n, rule, tail):
    grid = dt * np.arange(n + 1)
    out = None
    for pmf, edges in dists:
        q = regrid(pmf, edges, grid, rule, tail)[..., :n]
        out = q if out is None else _conv_truncated(out, q, n)
    return out


def convolve(dists, dt, horizon, rule="linear", tail="inf", place="upper"):
    dists = _components(dists)
    if place not in ("upper", "lower"):
        raise ValueError("place must be 'upper' or 'lower'")
    if not (dt > 0 and horizon > 0 and np.isfinite(horizon)):
        raise ValueError("dt and horizon must be positive and finite")
    n = int(np.ceil(horizon / dt - 1e-9))
    c = _index_pmf(dists, dt, n, rule, tail)
    if place == "upper":
        body = np.zeros_like(c)
        shift = len(dists) - 1
        body[..., shift:] = c[..., : max(n - shift, 0)]
    else:
        body = c
    rest = np.clip(1.0 - body.sum(-1, keepdims=True), 0.0, None)
    return np.concatenate([body, rest], axis=-1), dt * np.arange(n + 1)


def prob_total_within(dists, deadline, rule="linear", tail="inf", bound="lower", n_grid=2048):
    dists = _components(dists)
    if bound not in ("lower", "upper", "mid"):
        raise ValueError("bound must be 'lower', 'upper' or 'mid'")
    d = _times(deadline, "deadline")
    if not np.isfinite(d).all():
        raise ValueError("deadline must be finite")
    if n_grid < 1:
        raise ValueError("n_grid must be >= 1")
    d_max = float(d.max()) if d.size else 0.0
    if d_max <= 0:
        batch = np.broadcast_shapes(d.shape, *(np.shape(p)[:-1] for p, _ in dists))
        return np.zeros(batch)
    dt = d_max / n_grid
    cdf = np.cumsum(_index_pmf(dists, dt, n_grid + 1, rule, tail), axis=-1)
    ratio = np.maximum(d, 0.0) / dt
    m = np.floor(ratio + 1e-9).astype(np.int64)
    upper = np.where(d > 0, _gather(cdf, m), 0.0)
    m_low = np.floor(ratio * (1 + 4 * np.finfo(np.float64).eps)).astype(np.int64) - len(dists)
    lower = np.where((d > 0) & (m_low >= 0), _gather(cdf, np.maximum(m_low, 0)), 0.0)
    if bound == "lower":
        return lower
    if bound == "upper":
        return upper
    return (lower + upper) / 2


def feasible(dists, deadline, delta, rule="linear", tail="inf", n_grid=2048):
    if not 0 <= delta < 1:
        raise ValueError("delta must lie in [0, 1)")
    return prob_total_within(dists, deadline, rule, tail, "lower", n_grid) >= 1.0 - delta


def quantile_sum(dists, q, rule="linear", tail="inf"):
    return sum(quantile(pmf, edges, q, rule, tail) for pmf, edges in _components(dists))
