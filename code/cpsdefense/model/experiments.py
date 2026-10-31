import time
import warnings

import numpy as np
import torch
from scipy import stats

from cpsdefense.eval import survival
from cpsdefense.eval.metrics import (UndefinedMetricWarning, cluster_bootstrap_ci, latency_percentiles,
                                     paired_cluster_bootstrap_diff)
from cpsdefense.model import timecalc
from cpsdefense.model.train import BASELINES, VARIANTS, predict

NAN = float("nan")
ORDER = (*BASELINES, *VARIANTS)


def decision_blocks(segment, decision_time, block_s=3600.0):
    segment = np.asarray(segment)
    tau = np.asarray(decision_time, np.float64)
    first = {s: tau[segment == s].min() for s in np.unique(segment)}
    block = np.floor((tau - np.array([first[s] for s in segment])) / float(block_s)).astype(np.int64)
    return np.unique(np.stack([segment, block], 1), axis=0, return_inverse=True)[1].ravel()


def _ci(metric_fn, arrays, clusters, boot):
    boot = dict(boot)
    min_clusters = int(boot.pop("min_clusters", 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = cluster_bootstrap_ci(metric_fn, arrays, clusters, **boot)
    if out["n_clusters"] < min_clusters:
        out["lo"] = out["hi"] = None
    return out


def mean_ci(rows, clusters, boot):
    rows = np.asarray(rows, np.float64)
    rows = rows[:, None] if rows.ndim == 1 else rows
    return _ci(lambda a: float(np.mean(a)), [rows], clusters, boot)


def paired_mean_diff(rows_a, rows_b, clusters, boot):
    a = np.asarray(rows_a, np.float64)
    b = np.asarray(rows_b, np.float64)
    a, b = (a[:, None] if a.ndim == 1 else a), (b[:, None] if b.ndim == 1 else b)
    boot = dict(boot)
    min_clusters = int(boot.pop("min_clusters", 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = paired_cluster_bootstrap_diff(lambda x: float(np.mean(x)), [a], [b], clusters, **boot)
    if out["n_clusters"] < min_clusters:
        out["lo"] = out["hi"] = out["p_two_sided"] = None
    return out


def cdf_at_edge(pmf, edges, t):
    k = int(np.searchsorted(edges, t))
    if k >= len(edges) or edges[k] != t:
        raise ValueError(f"{t} is not a bin edge")
    return pmf[:, :k].sum(-1)


def nll_rows(pmf, edges, time, event, rule="linear", tail="inf"):
    return survival.nll_discrete(pmf, time, event, edges, rule=rule, tail=tail, reduction="none")


def ipcw_status(time, event, horizons, censor):
    t = np.asarray(time, np.float64)
    d = np.asarray(event, bool)
    h = np.atleast_1d(np.asarray(horizons, np.float64))
    died = d[:, None] & (t[:, None] <= h[None, :])
    alive = t[:, None] > h[None, :]
    g_event = censor(t, left=True)[:, None]
    g_horizon = censor(h)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(died, 1.0 / g_event, np.where(alive, 1.0 / g_horizon, 0.0))
    return died, alive, weight


def brier_rows(pmf, edges, time, event, horizons, censor, rule="linear", tail="inf"):
    h = np.atleast_1d(np.asarray(horizons, np.float64))
    s = timecalc.survival_at(pmf[:, None, :], edges, h[None, :], rule, tail)
    died, alive, weight = ipcw_status(time, event, h, censor)
    loss = np.where(died, s**2, np.where(alive, (1.0 - s) ** 2, 0.0))
    return np.where(weight > 0, loss * weight, 0.0)


def ibs_rows(pmf, edges, time, event, grid, censor, rule="linear", tail="inf"):
    grid = np.asarray(grid, np.float64)
    b = brier_rows(pmf, edges, time, event, grid, censor, rule, tail)
    return np.trapezoid(b, grid, axis=1) / (grid[-1] - grid[0])


def ece_ipcw(prob, died, weight, n_bins=10):
    b = np.searchsorted(np.arange(1, n_bins) / n_bins, prob, side="left")
    count = np.bincount(b, minlength=n_bins)
    w_sum = np.bincount(b, weights=weight, minlength=n_bins)
    wy_sum = np.bincount(b, weights=weight * died, minlength=n_bins)
    p_sum = np.bincount(b, weights=prob, minlength=n_bins)
    ok = (count > 0) & (w_sum > 0)
    if not ok.any():
        return NAN
    share = count[ok] / count[ok].sum()
    return float(np.sum(share * np.abs(wy_sum[ok] / w_sum[ok] - p_sum[ok] / count[ok])))


def c_index_td_fast(pmf, edges, time, event, rule="linear", tail="inf"):
    t = np.asarray(time, np.float64)
    d = np.asarray(event, bool)
    num = den = 0.0
    for u in np.unique(t[d]):
        s = timecalc.survival_at(pmf, edges, np.full(t.size, u), rule, tail)
        own = s[d & (t == u)]
        partners = np.sort(s[(t > u) | ((t == u) & ~d)])
        if partners.size == 0:
            continue
        right = np.searchsorted(partners, own, side="right")
        left = np.searchsorted(partners, own, side="left")
        num += float(np.sum(partners.size - right) + 0.5 * np.sum(right - left))
        den += float(own.size * partners.size)
    return num / den if den else NAN


def d_calibration_summary(pmf, edges, time, event, n_bins=10, rule="linear", tail="inf"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        dc = survival.d_calibration(pmf, edges, time, event, n_bins, rule, tail)
    return {"statistic": dc["statistic"], "p_value": dc["p_value"],
            "max_dev": float(np.max(np.abs(dc["proportions"] - 1.0 / n_bins)))}


def _stack(values):
    return np.stack(values, 1)


def _seed_summary(values):
    v = np.asarray(values, np.float64)
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if v.size > 1 else None,
            "per_seed": v.tolist()}


def tte_rows(preds, split, q_index, edges, ev):
    name = split.names[q_index]
    pmf = preds["pmf"][name]
    t, d = split.time[:, q_index], split.event[:, q_index]
    rule, tail = ev["rule"], ev["tail"]
    censor = survival.censoring_km(t, d)
    grid = np.arange(*ev["ibs_grid_s"], dtype=np.float64)
    horizons = [float(h) for h in ev["calib_horizons_s"]]
    died, _, weight = ipcw_status(t, d, horizons, censor)
    out = {"nll": nll_rows(pmf, edges, t, d, rule, tail), "ibs": ibs_rows(pmf, edges, t, d, grid, censor, rule, tail)}
    b = brier_rows(pmf, edges, t, d, horizons, censor, rule, tail)
    for j, h in enumerate(horizons):
        out[f"brier{h:g}"] = b[:, j]
        out[f"prob{h:g}"] = np.clip(cdf_at_edge(pmf, edges, h), 0.0, 1.0)
        out[f"died{h:g}"] = died[:, j]
        out[f"weight{h:g}"] = weight[:, j]
    return out


def tte_metrics(pred_list, split, q_index, edges, ev, clusters, boot):
    name = split.names[q_index]
    t, d = split.time[:, q_index], split.event[:, q_index]
    rows = [tte_rows(p, split, q_index, edges, ev) for p in pred_list]
    horizons = [float(h) for h in ev["calib_horizons_s"]]
    n_bins = int(ev["calib_bins"])
    out = {"n": int(split.n), "events": int(d.sum())}
    for key in ["nll", "ibs"] + [f"brier{h:g}" for h in horizons]:
        stacked = _stack([r[key] for r in rows])
        out[key] = {**mean_ci(stacked, clusters, boot), **_seed_summary(stacked.mean(0))}
    for h in horizons:
        prob = _stack([r[f"prob{h:g}"] for r in rows])
        died, weight = rows[0][f"died{h:g}"].astype(np.float64), rows[0][f"weight{h:g}"]

        def ece(p, y, w, n_bins=n_bins):
            return float(np.mean([ece_ipcw(p[:, s], y, w, n_bins) for s in range(p.shape[1])]))

        ci = _ci(ece, [prob, died, weight], clusters, boot)
        out[f"ece{h:g}"] = {**ci, **_seed_summary([ece_ipcw(prob[:, s], died, weight, n_bins)
                                                   for s in range(prob.shape[1])])}
        known = weight > 0
        out[f"obs{h:g}"] = float(np.sum(weight * died) / np.sum(weight[known])) if known.any() else NAN
        out[f"pred{h:g}"] = float(prob.mean())
    c = [c_index_td_fast(p["pmf"][name], edges, t, d, ev["rule"], ev["tail"]) for p in pred_list]
    out["c_td"] = {"point": float(np.mean(c)), **_seed_summary(c)}
    dcal = [d_calibration_summary(p["pmf"][name], edges, t, d, int(ev["dcal_bins"]), ev["rule"], ev["tail"])
            for p in pred_list]
    out["dcal"] = {k: _seed_summary([x[k] for x in dcal]) for k in ("statistic", "max_dev", "p_value")}
    out["_nll_rows"] = _stack([r["nll"] for r in rows])
    out["_ibs_rows"] = _stack([r["ibs"] for r in rows])
    return out


def score_metrics(pred_list, split, clusters, boot):
    ok = split.score >= 0
    y = split.score[ok]
    nll = _stack([-np.log(np.clip(p["score"][ok][np.arange(y.size), y], 1e-12, None)) for p in pred_list])
    acc = _stack([(p["score"][ok].argmax(1) == y).astype(np.float64) for p in pred_list])
    c = clusters[ok]
    return {"n": int(ok.sum()), "nll": {**mean_ci(nll, c, boot), **_seed_summary(nll.mean(0))},
            "accuracy": {**mean_ci(acc, c, boot), **_seed_summary(acc.mean(0))}, "_nll_rows": nll, "_ok": ok}


def coherence(pred_list, split, q_index, edges, within_s, clusters, boot):
    name = split.names[q_index]
    t, d = split.time[:, q_index], split.event[:, q_index]
    y = np.where(d & (t <= within_s), 1.0, np.where(t >= within_s, 0.0, np.nan))
    known = ~np.isnan(y)
    gap = _stack([p["noul"][name] - cdf_at_edge(p["pmf"][name], edges, within_s) for p in pred_list])
    absgap = np.abs(gap)
    brier_noul = _stack([(p["noul"][name][known] - y[known]) ** 2 for p in pred_list])
    brier_hazard = _stack([(cdf_at_edge(p["pmf"][name], edges, within_s)[known] - y[known]) ** 2 for p in pred_list])
    return {
        "mean_abs_gap": {**mean_ci(absgap, clusters, boot), **_seed_summary(absgap.mean(0))},
        "mean_gap": _seed_summary(gap.mean(0)),
        "p95_abs_gap": _seed_summary(np.quantile(absgap, 0.95, axis=0)),
        "max_abs_gap": _seed_summary(absgap.max(0)),
        "brier_noul": {**mean_ci(brier_noul, clusters[known], boot), **_seed_summary(brier_noul.mean(0))},
        "brier_hazard": {**mean_ci(brier_hazard, clusters[known], boot), **_seed_summary(brier_hazard.mean(0))},
        "brier_diff_noul_minus_hazard": paired_mean_diff(brier_noul, brier_hazard, clusters[known], boot),
        "n_known": int(known.sum()),
    }


def hour_partner(segment, decision_time, hour_s=3600.0):
    segment = np.asarray(segment)
    tau = np.asarray(decision_time, np.float64)
    key = {(s, t): i for i, (s, t) in enumerate(zip(segment.tolist(), tau.tolist()))}
    out = np.full(tau.size, -1, np.int64)
    for i, (s, t) in enumerate(zip(segment.tolist(), tau.tolist())):
        out[i] = key.get((s, t - hour_s), key.get((s, t + hour_s), -1))
    return out


def fill_transform(fill, gpu_batch, partner=None, random_std=3.0, huge=1e30, seed=0):
    gen = torch.Generator(device=gpu_batch.device).manual_seed(int(seed))
    partner_t = None if partner is None else torch.as_tensor(partner, device=gpu_batch.device)

    def transform(sub, rows):
        invalid = ~sub.validity()
        payload = sub.payload
        shape = payload.shape
        if fill == "random":
            new = torch.randn(shape, generator=gen, device=payload.device, dtype=payload.dtype) * random_std
        elif fill in ("huge", "inf"):
            sign = torch.where(torch.rand(shape, generator=gen, device=payload.device) < 0.5, -1.0, 1.0)
            new = (sign * (huge if fill == "huge" else float("inf"))).to(payload.dtype)
        elif fill == "nan":
            new = torch.full_like(payload, float("nan"))
        elif fill == "hour_earlier":
            p = partner_t[rows]
            has = p >= 0
            src = gpu_batch.take(p.clamp(min=0))
            same = has[:, None] & src.present & (src.source == sub.source)
            invalid = invalid & same
            new = src.payload
        else:
            raise ValueError(f"unknown fill {fill!r}")
        return sub.replace(payload=torch.where(invalid[..., None], new, payload))

    return transform


def _survival_from_pmf(pmf):
    return 1.0 - np.cumsum(pmf.astype(np.float64), -1)[..., :-1]


def compare_outputs(base, other, spec, split, ev):
    names = list(base)
    same = np.ones(split.n, bool)
    finite = np.ones(split.n, bool)
    for k in names:
        a, b = base[k].reshape(split.n, -1), other[k].reshape(split.n, -1)
        same &= np.all(a.view(np.uint32) == b.view(np.uint32), axis=1)
        finite &= np.all(np.isfinite(b), axis=1)
    surv, noul, score = [], [], []
    dnll, dnll_abs = {}, {}
    for i, name in enumerate(spec.names):
        sa, sb = _survival_from_pmf(base[name]), _survival_from_pmf(other[name])
        surv.append(np.max(np.abs(sa - sb), axis=1))
        nk = f"{name}_within{spec.within_s:g}"
        noul.append(np.abs(base[nk].astype(np.float64) - other[nk].astype(np.float64)))
        rows = finite
        if rows.any():
            pa = np.clip(base[name][rows].astype(np.float64), 0, None)
            pb = np.clip(other[name][rows].astype(np.float64), 0, None)
            pa, pb = pa / pa.sum(-1, keepdims=True), pb / pb.sum(-1, keepdims=True)
            t, d = split.time[rows, i], split.event[rows, i]
            diff = nll_rows(pb, spec.edges, t, d, ev["rule"], ev["tail"]) - nll_rows(pa, spec.edges, t, d,
                                                                                    ev["rule"], ev["tail"])
            dnll[name], dnll_abs[name] = float(diff.mean()), float(np.abs(diff).mean())
        else:
            dnll[name] = dnll_abs[name] = NAN
    if "level_bin" in base:
        score.append(np.max(np.abs(base["level_bin"].astype(np.float64) - other["level_bin"].astype(np.float64)), 1))
    per_row = np.max(np.stack(surv + noul + score, 1), axis=1)
    fin = np.isfinite(per_row)

    def fmax(x):
        x = np.asarray(x)[fin]
        return float(x.max()) if x.size else NAN

    return {
        "bitwise_equal_rate": float(same.mean()),
        "nonfinite_rate": float(1.0 - finite.mean()),
        "max_abs_change": fmax(per_row),
        "max_abs_change_survival": fmax(np.max(np.stack(surv, 1), 1)),
        "max_abs_change_noul": fmax(np.max(np.stack(noul, 1), 1)),
        "mean_abs_change": float(per_row[fin].mean()) if fin.any() else NAN,
        "frac_change_gt_0.01": float(np.mean(per_row[fin] > 0.01)) if fin.any() else NAN,
        "delta_nll_mean": dnll,
        "delta_nll_abs_mean": dnll_abs,
    }


def invariance_audit(model, gpu_batch, split, spec, config, base=None):
    inv = config["invariance"]
    ev = config["eval"]
    questions = spec.questions()
    chunk = int(config["train"]["eval_chunk"])
    base = predict(model, gpu_batch, questions, chunk) if base is None else base
    valid = split.batch.validity()
    known = split.batch.known()
    out = {"n_decisions": int(split.n), "invalid_items": int((split.batch.present & ~valid).sum()),
           "arrived_invalid_items": int((known & ~valid).sum()), "fills": {}}
    partner = hour_partner(split.segment, split.decision_time, float(inv["hour_s"]))
    out["hour_partner_earlier"] = int(np.sum((partner >= 0)
                                             & (split.decision_time[np.maximum(partner, 0)] < split.decision_time)))
    out["hour_partner_missing"] = int(np.sum(partner < 0))
    for k, fill in enumerate(inv["fills"]):
        transform = fill_transform(fill, gpu_batch, partner, float(inv["random_std"]), float(inv["huge"]), seed=k)
        other = predict(model, gpu_batch, questions, chunk, transform=transform)
        out["fills"][fill] = compare_outputs(base, other, spec, split, ev)
    used = []
    with torch.no_grad():
        for start in range(0, gpu_batch.n, chunk):
            rows = torch.arange(start, min(start + chunk, gpu_batch.n), device=gpu_batch.device)
            used.append(model.encoder(gpu_batch.take(rows)).used.cpu().numpy())
    used = np.concatenate(used)
    expected = known if model.mode == "unmasked" else valid
    out["used_equals_expected_mask"] = bool(np.array_equal(used, expected))
    out["used_items_per_decision"] = float(used.sum(1).mean())
    return out


def revalidation_stats(batch, deltas):
    tau = batch.decision_time
    valid = batch.validity()
    known = batch.known()
    out = {"n_decisions": int(batch.n), "dependency_size_mean": float(valid.sum(1).mean()),
           "unmasked_dependency_size_mean": float(known.sum(1).mean()),
           "unmasked_depends_on_invalid_rate": float(np.mean((known & ~valid).any(1))), "deltas": {}}
    for delta in deltas:
        t = tau + float(delta)
        later = batch.validity(t)
        lost = valid & ~later
        expired = lost & ~(t[:, None] < batch.valid_until)
        retracted = lost & ~(t[:, None] < batch.retracted_at)
        gained = later & ~valid
        out["deltas"][f"{delta:g}"] = {
            "frac_lose_any": float(lost.any(1).mean()),
            "lost_per_decision": float(lost.sum(1).mean()),
            "frac_lose_by_expiry": float(expired.any(1).mean()),
            "frac_lose_by_retraction": float(retracted.any(1).mean()),
            "frac_gain_any": float(gained.any(1).mean()),
            "gained_per_decision": float(gained.sum(1).mean()),
        }
    return out


def _timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _ms(latencies):
    p = latency_percentiles(np.asarray(latencies))
    return {k: (v * 1e3 if k != "n" else v) for k, v in p.items()}


def latency_benchmark(models, batch, spec, config, log=print):
    lc = config["latency"]
    questions = spec.questions()
    delta = float(lc["delta_s"])
    rng = np.random.default_rng(int(lc.get("seed", 0)))
    out = {"delta_s": delta, "cpu_threads": int(lc["cpu_threads"]), "devices": {}}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(lc["cpu_threads"]))
    try:
        for device in lc["devices"]:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            dev_out = {}
            for bs in [int(b) for b in lc["batch_sizes"]]:
                repeats = int(lc["repeats"][str(bs)] if str(bs) in lc["repeats"] else lc["repeats"][bs])
                rows_list = [rng.choice(batch.n, size=bs, replace=False) for _ in range(repeats + int(lc["warmup"]))]
                res = {}
                for variant, model in models.items():
                    m = model.to(device).eval()
                    times = {"fresh": [], "rescore": [], "decide_all_questions": [], "decide_per_question": []}
                    encoded = {"fresh": [], "rescore": []}
                    for r, rows in enumerate(rows_list):
                        sub = batch.take(rows).to(device)
                        t_exec = sub.decision_time + delta
                        with torch.no_grad():
                            dt_all, _ = _timed(lambda: m(sub, questions=questions, keep_cache=False), device)
                            dt_per, _ = _timed(lambda: [m(sub, questions=[q], keep_cache=False) for q in questions],
                                               device)
                            cache = m(sub, questions=questions)["cache"]
                            dt_fresh, fresh = _timed(lambda: m(sub, t_exec, questions, keep_cache=False), device)
                            dt_re, re = _timed(lambda: m.score_at(sub, t_exec, questions, cache=cache), device)
                        if r < int(lc["warmup"]):
                            continue
                        times["decide_all_questions"].append(dt_all)
                        times["decide_per_question"].append(dt_per)
                        times["fresh"].append(dt_fresh)
                        times["rescore"].append(dt_re)
                        encoded["fresh"].append(fresh["stats"]["n_encoded"])
                        encoded["rescore"].append(re["stats"]["n_encoded"])
                    res[variant] = {k: _ms(v) for k, v in times.items()}
                    res[variant]["items_encoded_mean"] = {k: float(np.mean(v)) for k, v in encoded.items()}
                    log(f"  latency {device} bs {bs} {variant}: " + ", ".join(
                        f"{k} p50 {res[variant][k]['p50']:.2f} ms" for k in times))
                dev_out[str(bs)] = res
            out["devices"][device] = dev_out
    finally:
        torch.set_num_threads(previous_threads)
        for model in models.values():
            model.to("cuda" if torch.cuda.is_available() else "cpu")
    return out


def _component_cdf(comp, x):
    kind = comp["type"]
    if kind == "lognormal":
        return stats.lognorm.cdf(x, s=comp["sigma"], scale=comp["median"])
    if kind == "gamma":
        return stats.gamma.cdf(x, a=comp["shape"], scale=comp["scale"])
    if kind == "uniform":
        return stats.uniform.cdf(x, loc=comp["low"], scale=comp["high"] - comp["low"])
    if kind == "mixture":
        return sum(w * _component_cdf(c, x) for w, c in zip(comp["weights"], comp["parts"]))
    raise ValueError(f"unknown component type {kind!r}")


def _component_sample(comp, rng, n):
    kind = comp["type"]
    if kind == "lognormal":
        return rng.lognormal(np.log(comp["median"]), comp["sigma"], n)
    if kind == "gamma":
        return rng.gamma(comp["shape"], comp["scale"], n)
    if kind == "uniform":
        return rng.uniform(comp["low"], comp["high"], n)
    if kind == "mixture":
        which = rng.choice(len(comp["weights"]), size=n, p=np.asarray(comp["weights"]) / np.sum(comp["weights"]))
        draws = np.stack([_component_sample(c, rng, n) for c in comp["parts"]], 1)
        return draws[np.arange(n), which]
    raise ValueError(f"unknown component type {kind!r}")


def pipeline_components(dc):
    edges = np.arange(0.0, float(dc["grid_max_s"]) + 1e-9, float(dc["grid_step_s"]))
    out = []
    for comp in dc["pipeline"]:
        cdf = _component_cdf(comp, edges)
        cdf[0] = 0.0
        pmf = np.r_[np.diff(cdf), 1.0 - cdf[-1]]
        out.append((np.clip(pmf, 0.0, None) / np.clip(pmf, 0.0, None).sum(), edges))
    return out


def pipeline_draws(dc, n, seed):
    rng = np.random.default_rng(seed)
    return sum(_component_sample(comp, rng, n) for comp in dc["pipeline"])


def deadline_outcomes(time, event, latency, deadline):
    on_time = event & (time + latency <= deadline)
    known = event | (time + latency >= deadline)
    return on_time, known


def deadline_gates(pmf, edges, components, deadlines, deltas, rule="linear", tail="inf", n_grid=2048, chunk=2048):
    deadlines = np.asarray(deadlines, np.float64)
    n = pmf.shape[0]
    lower = np.zeros((n, deadlines.size))
    mid = np.zeros((n, deadlines.size))
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        dists = [(pmf[sl][:, None, :], edges)] + list(components)
        lower[sl] = timecalc.prob_total_within(dists, deadlines[None, :], rule, tail, "lower", n_grid)
        mid[sl] = timecalc.prob_total_within(dists, deadlines[None, :], rule, tail, "mid", n_grid)
    gates = {}
    for delta in deltas:
        q = 1.0 - float(delta)
        qsum = timecalc.quantile(pmf, edges, np.full(n, q), rule, tail)
        qsum = qsum + sum(float(timecalc.quantile(p, e, q, rule, tail)) for p, e in components)
        gates[f"{delta:g}"] = {"conv": lower >= q, "qsum": qsum[:, None] <= deadlines[None, :], "qsum_value": qsum}
    return {"lower": lower, "mid": mid, "gates": gates}


def sample_pmf(pmf, edges, rng):
    cdf = np.cumsum(pmf, -1)
    u = rng.random(pmf.shape[0]) * cdf[:, -1]
    k = np.minimum((cdf < u[:, None]).sum(1), pmf.shape[1] - 1)
    inside = k < len(edges) - 1
    kk = np.minimum(k, len(edges) - 2)
    t = edges[kk] + rng.random(pmf.shape[0]) * (edges[kk + 1] - edges[kk])
    return np.where(inside, t, np.inf)


def deadline_demo(pmf_list, split, q_index, edges, config, clusters, boot):
    dc = config["deadline"]
    components = pipeline_components(dc)
    t, d = split.time[:, q_index], split.event[:, q_index]
    latency = pipeline_draws(dc, split.n, int(dc["seed"]) + q_index)
    deadlines = [float(x) for x in dc["deadlines_s"]]
    deltas = [float(x) for x in dc["deltas"]]
    results = [deadline_gates(p, edges, components, deadlines, deltas, dc["rule"], dc["tail"], int(dc["n_grid"]))
               for p in pmf_list]
    sim_t = [sample_pmf(p, edges, np.random.default_rng([int(dc["seed"]), q_index, s]))
             for s, p in enumerate(pmf_list)]
    out = {"deadlines": {}, "n": int(split.n)}
    for j, D in enumerate(deadlines):
        on_time, known = deadline_outcomes(t, d, latency, D)
        sim_on_time = _stack([ts + latency <= D for ts in sim_t])
        row = {"realised_rate": float(on_time[known].mean()), "n_unknown": int((~known).sum()), "gates": {}}
        mids = _stack([r["mid"][:, j] for r in results])
        row["pred_mid_mean"] = float(mids[known].mean())
        row["ece_mid"] = _seed_summary([_plain_ece(mids[known, s], on_time[known]) for s in range(mids.shape[1])])
        row["sim_ece_mid"] = _seed_summary([_plain_ece(mids[:, s], sim_on_time[:, s]) for s in range(mids.shape[1])])
        for delta in deltas:
            for rule_name in ("conv", "qsum"):
                feas = _stack([r["gates"][f"{delta:g}"][rule_name][:, j] for r in results])
                sel = feas & known[:, None]
                n_sel = sel.sum(0)
                rates = [float(on_time[sel[:, s]].mean()) if n_sel[s] else NAN for s in range(sel.shape[1])]
                sim = [float(sim_on_time[feas[:, s], s].mean()) if feas[:, s].any() else NAN
                       for s in range(feas.shape[1])]
                cell = {"n_feasible": _seed_summary(n_sel), "frac_feasible": _seed_summary(n_sel / known.sum()),
                        "on_time_rate": _seed_summary(rates), "sim_on_time_rate": _seed_summary(sim),
                        "nominal": 1.0 - delta,
                        "on_time_pooled": float((sel & on_time[:, None]).sum() / n_sel.sum()) if n_sel.sum() else NAN,
                        "sim_on_time_pooled": float((feas & sim_on_time).sum() / feas.sum()) if feas.any() else NAN,
                        "n_seeds_feasible": int((n_sel > 0).sum())}
                if sel.shape[1] == 1 and n_sel[0]:
                    s0 = sel[:, 0]
                    cell["on_time_ci"] = mean_ci(on_time[s0].astype(np.float64), clusters[s0], boot)
                row["gates"][f"{rule_name}@{delta:g}"] = cell
        out["deadlines"][f"{D:g}"] = row
    return out


def _plain_ece(prob, y, n_bins=10):
    return ece_ipcw(prob, y.astype(np.float64), np.ones_like(prob), n_bins)


def boot_args(config):
    ev = config["eval"]
    return {"n_boot": int(ev["n_boot"]), "alpha": float(ev["alpha"]), "seed": int(ev["boot_seed"]),
            "min_clusters": int(ev["min_ci_clusters"])}


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def evaluate_split(split, preds, spec, config, log=print):
    ev, boot = config["eval"], boot_args(config)
    clusters = decision_blocks(split.segment, split.decision_time, float(ev["block_s"]))
    variants = [v for v in ORDER if v in preds]
    out = {"n": int(split.n), "n_blocks": int(clusters.max() + 1), "hours": float(split.n * 10 / 3600),
           "attack_in_horizon_rate": float(np.mean(split.attack_in_horizon)), "questions": {}, "paired": {},
           "coherence": {}, "deadline": {}, "score": {}, "score_paired": {}}
    for i, name in enumerate(split.names):
        log(f"  [{split.version} {split.split}] {name}: time-to-event metrics")
        m = {v: tte_metrics(preds[v], split, i, spec.edges, ev, clusters, boot) for v in variants}
        out["paired"][name] = {
            f"{a} - {b}": {"nll": paired_mean_diff(m[a]["_nll_rows"], m[b]["_nll_rows"], clusters, boot),
                           "ibs": paired_mean_diff(m[a]["_ibs_rows"], m[b]["_ibs_rows"], clusters, boot)}
            for a, b in config["compare"] if a in m and b in m}
        out["questions"][name] = strip_private(m)
        out["coherence"][name] = {v: coherence(preds[v], split, i, spec.edges, spec.within_s, clusters, boot)
                                  for v in variants if v in VARIANTS}
        dl = {}
        for v in config["deadline"]["predictors"]:
            if v in preds:
                dl[v] = deadline_demo([p["pmf"][name] for p in preds[v]], split, i, spec.edges, config, clusters, boot)
        out["deadline"][name] = dl
    if split.score is not None and spec.n_levels:
        sm = {v: score_metrics(preds[v], split, clusters, boot) for v in variants if preds[v][0]["score"] is not None}
        for a, b in config["compare"]:
            if a in sm and b in sm:
                ok = sm[a]["_ok"]
                out["score_paired"][f"{a} - {b}"] = paired_mean_diff(sm[a]["_nll_rows"], sm[b]["_nll_rows"],
                                                                     clusters[ok], boot)
        out["score"] = strip_private(sm)
    return out


def aggregate_audits(audits):
    out = {"n_seeds": len(audits), "used_equals_expected_mask": all(a["used_equals_expected_mask"] for a in audits),
           "used_items_per_decision": float(np.mean([a["used_items_per_decision"] for a in audits])),
           "invalid_items": audits[0]["invalid_items"], "arrived_invalid_items": audits[0]["arrived_invalid_items"],
           "hour_partner_missing": audits[0].get("hour_partner_missing"), "fills": {}}
    def num(values):
        return np.array([NAN if v is None else v for v in values], np.float64)

    def nanstat(fn, values):
        v = num(values)
        return float(fn(v[np.isfinite(v)])) if np.isfinite(v).any() else NAN

    for fill in audits[0]["fills"]:
        rows = [a["fills"][fill] for a in audits]
        dnll = [nanstat(np.mean, r["delta_nll_mean"].values()) for r in rows]
        dnll_abs = [nanstat(np.mean, r["delta_nll_abs_mean"].values()) for r in rows]
        out["fills"][fill] = {
            "bitwise_equal_rate_min": float(min(r["bitwise_equal_rate"] for r in rows)),
            "bitwise_equal_rate_mean": float(np.mean([r["bitwise_equal_rate"] for r in rows])),
            "nonfinite_rate_mean": float(np.mean([r["nonfinite_rate"] for r in rows])),
            "max_abs_change": nanstat(np.max, [r["max_abs_change"] for r in rows]),
            "max_abs_change_survival": nanstat(np.max, [r["max_abs_change_survival"] for r in rows]),
            "max_abs_change_noul": nanstat(np.max, [r["max_abs_change_noul"] for r in rows]),
            "mean_abs_change": nanstat(np.mean, [r["mean_abs_change"] for r in rows]),
            "frac_change_gt_0.01": nanstat(np.mean, [r["frac_change_gt_0.01"] for r in rows]),
            "delta_nll_mean": nanstat(np.mean, dnll),
            "delta_nll_abs_mean": nanstat(np.mean, dnll_abs),
        }
    return out


def _f(x, d=3):
    if x is None:
        return "-"
    try:
        if np.isnan(x):
            return "-"
    except TypeError:
        return str(x)
    return f"{x:.{d}f}"


def _g(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2g}"


def _p(x, n_boot):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    floor = 2.0 / n_boot
    return f"< {floor:.2g}" if x < floor else f"{x:.2g}"


def _pooled(c, key):
    if f"{key}_pooled" in c:
        return c[f"{key}_pooled"]
    rates = np.asarray(c[f"{key}_rate"]["per_seed"], np.float64)
    n = np.asarray(c["n_feasible"]["per_seed"], np.float64)
    ok = (n > 0) & np.isfinite(rates)
    return float((rates[ok] * n[ok]).sum() / n[ok].sum()) if ok.any() else NAN


def _cell(c, d=3):
    point = c.get("point", c.get("mean"))
    if c.get("lo") is None:
        return _f(point, d)
    return f"{_f(point, d)} [{_f(c['lo'], d)}, {_f(c['hi'], d)}]"


def _sd(c, d=3):
    return _f(c.get("sd"), d)


def _row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(head, rows):
    return [_row(head), "|" + "---|" * len(head), *(_row(r) for r in rows)]


def _label(v, n_seeds):
    return f"{v} (n={n_seeds})" if v in VARIANTS else v


def tte_table(res_split, horizons, n_seeds):
    head = ["question", "variant", "NLL [95% CI]", "IBS [95% CI]", "C-td", "D-cal max dev (chi2)"]
    head += [f"ECE@{h:g} [95% CI]" for h in horizons] + [f"Brier@{h:g}" for h in horizons]
    head += [f"pred / obs P(T<={h:g})" for h in horizons] + ["seed sd NLL / IBS / C-td"]
    rows = []
    for name, m in res_split["questions"].items():
        for v, r in m.items():
            cells = [name, _label(v, n_seeds.get(v, 1)), _cell(r["nll"]), _cell(r["ibs"], 4), _f(r["c_td"]["point"]),
                     f"{_f(r['dcal']['max_dev']['mean'])} ({_f(r['dcal']['statistic']['mean'], 0)})"]
            cells += [_cell(r[f"ece{h:g}"]) for h in horizons] + [_cell(r[f"brier{h:g}"], 4) for h in horizons]
            cells += [f"{_f(r[f'pred{h:g}'])} / {_f(r[f'obs{h:g}'])}" for h in horizons]
            cells.append(f"{_sd(r['nll'])} / {_sd(r['ibs'], 4)} / {_sd(r['c_td'])}" if v in VARIANTS else "-")
            rows.append(cells)
    return _table(head, rows)


def paired_table(res, split_names, n_boot):
    head = ["split", "question", "comparison (a - b)", "dNLL [95% CI]", "p", "dIBS [95% CI]", "p"]
    rows = []
    for split in split_names:
        for name, comps in res[split]["paired"].items():
            for comp, d in comps.items():
                rows.append([split, name, comp, _cell(d["nll"], 4), _p(d["nll"].get("p_two_sided"), n_boot),
                             _cell(d["ibs"], 5), _p(d["ibs"].get("p_two_sided"), n_boot)])
    return _table(head, rows)


def score_table(res, split_names, n_seeds, n_boot):
    head = ["split", "variant", "NLL [95% CI]", "accuracy [95% CI]", "seed sd NLL / acc"]
    rows = []
    for split in split_names:
        for v, r in res[split]["score"].items():
            rows.append([split, _label(v, n_seeds.get(v, 1)), _cell(r["nll"]), _cell(r["accuracy"]),
                         f"{_sd(r['nll'])} / {_sd(r['accuracy'])}" if v in VARIANTS else "-"])
        for comp, d in res[split]["score_paired"].items():
            rows.append([split, f"diff {comp}", _cell(d, 4) + f" (p {_p(d.get('p_two_sided'), n_boot)})", "", ""])
    return _table(head, rows)


def coherence_table(res, split_names, within_s):
    head = ["split", "question", "variant", "mean |gap| [95% CI]", "mean gap", "p95 |gap|", "max |gap|",
            "Brier noul", f"Brier 1-S({within_s:g})", "Brier diff noul - hazard [95% CI]"]
    rows = []
    for split in split_names:
        for name, by_v in res[split]["coherence"].items():
            for v, c in by_v.items():
                rows.append([split, name, v, _cell(c["mean_abs_gap"], 4), _f(c["mean_gap"]["mean"], 4),
                             _f(c["p95_abs_gap"]["mean"], 4), _f(c["max_abs_gap"]["mean"], 3),
                             _f(c["brier_noul"]["point"], 4), _f(c["brier_hazard"]["point"], 4),
                             _cell(c["brier_diff_noul_minus_hazard"], 5)])
    return _table(head, rows)


def invariance_table(inv):
    head = ["split", "variant", "fill", "bitwise-equal rate (min over seeds)", "max |change| (surv / noul)",
            "share of decisions changed > 0.01", "non-finite outputs", "mean dNLL (mean |dNLL|)",
            "used slots = mask"]
    rows = []
    for split, by_v in inv.items():
        for v, a in by_v.items():
            for fill, r in a["fills"].items():
                rows.append([split, v, fill, _f(r["bitwise_equal_rate_min"], 4),
                             f"{_g(r['max_abs_change_survival'])} / {_g(r['max_abs_change_noul'])}",
                             _f(r["frac_change_gt_0.01"], 3), _f(r["nonfinite_rate_mean"], 3),
                             f"{_g(r['delta_nll_mean'])} ({_g(r['delta_nll_abs_mean'])})",
                             "yes" if a["used_equals_expected_mask"] else "NO"])
    return _table(head, rows)


def revalidation_table(reval):
    head = ["split", "Delta s", "decisions losing >= 1 item", "lost items / decision", "by expiry", "by retraction",
            "decisions gaining >= 1 new valid item", "valid items / decision at tau"]
    rows = []
    for split, r in reval.items():
        for delta, d in r["deltas"].items():
            rows.append([split, delta, _f(d["frac_lose_any"]), _f(d["lost_per_decision"], 2),
                         _f(d["frac_lose_by_expiry"]), _f(d["frac_lose_by_retraction"], 4),
                         _f(d["frac_gain_any"]), _f(r["dependency_size_mean"], 1)])
    return _table(head, rows)


def latency_table(lat):
    head = ["device", "batch", "variant", "decide, all questions in 1 pass", "decide, 1 pass per question",
            "fresh pass at tau+Delta", "re-score (score_at)", "items encoded fresh / re-score"]
    rows = []
    for device, by_bs in lat["devices"].items():
        for bs, by_v in by_bs.items():
            for v, r in by_v.items():
                cell = lambda k: f"{r[k]['p50']:.2f} / {r[k]['p95']:.2f} / {r[k]['p99']:.2f}"
                rows.append([device, bs, v, cell("decide_all_questions"), cell("decide_per_question"), cell("fresh"),
                             cell("rescore"), f"{r['items_encoded_mean']['fresh']:.1f} / "
                                              f"{r['items_encoded_mean']['rescore']:.1f}"])
    return _table(head, rows)


def deadline_table(res, split_names, deltas):
    head = ["split", "question", "predictor", "D s", "realised P(on time)", "mean pred P (mid)",
            "ECE mid: real / sim"]
    for delta in deltas:
        head += [f"conv@{delta:g}: n feasible / on-time real / sim", f"qsum@{delta:g}: n feasible / on-time real / sim"]
    rows = []
    for split in split_names:
        for name, by_p in res[split]["deadline"].items():
            for p, r in by_p.items():
                for D, d in r["deadlines"].items():
                    cells = [split, name, p, D, _f(d["realised_rate"]), _f(d["pred_mid_mean"]),
                             f"{_f(d['ece_mid']['mean'])} / {_f(d['sim_ece_mid']['mean'])}"]
                    for delta in deltas:
                        for g in ("conv", "qsum"):
                            c = d["gates"][f"{g}@{delta:g}"]
                            n_seeds = len(c["n_feasible"]["per_seed"])
                            k = c.get("n_seeds_feasible", int(sum(n > 0 for n in c["n_feasible"]["per_seed"])))
                            note = f" ({k}/{n_seeds} seeds)" if 0 < k < n_seeds else ""
                            cells.append(f"{_f(c['n_feasible']['mean'], 0)} / {_f(_pooled(c, 'on_time'))} / "
                                         f"{_f(_pooled(c, 'sim_on_time'))}{note}")
                    rows.append(cells)
    return _table(head, rows)


def legend(config):
    ev, dc, inv = config["eval"], config["deadline"], config["invariance"]
    q = config["questions"]
    h = [float(x) for x in ev["calib_horizons_s"]]
    return [
        "- Variants. km_marginal: Kaplan-Meier curve per question fitted on train. km_state: KM per state at tau "
        "(the per-type histogram). km_state_age: KM per (state, time-since-last-event bin), shrunk toward km_state "
        f"with {config['baselines']['shrink_events']} pseudo-events. The KM covariates are undelayed raw data "
        "(privileged: the network sees only delayed evidence). unmasked: every arrived item incl. expired and "
        "retracted ones, with age, time to expiry and expired / retracted flags. unmasked_novalidity: the same "
        "items without time to expiry and flags; validity stays largely inferable from the source and the item's "
        "age (lifetimes are per source), so this is not a validity-blind model. masked: items valid at tau gathered before any computation, "
        "full attention among them. block_local: masked, no cross-item attention in the encoder.",
        "- Neural variants share one architecture and budget (configs/model/hai_backbone.yaml) and are trained "
        "on train with early stopping and learning-rate decay on the val loss (the only choices). (n=3) = seeds "
        "0-2: values are seed means; [lo, hi] is the joint 95% cluster-bootstrap CI of the seed mean (every seed "
        "scored on the same resampled blocks); seed sd = std over seeds (ddof 1).",
        f"- CIs: percentile bootstrap over contiguous {ev['block_s']:g} s blocks of decisions inside segments, "
        f"{ev['n_boot']} resamples, shown only with >= {ev['min_ci_clusters']} blocks. Decisions are 10 s apart and "
        "strongly autocorrelated; blocks are the independent units. Differences between methods are paired "
        "bootstraps (p = two-sided percentile approximation, resolution 2 / n_boot: values below it are shown as "
        "'< 2 / n_boot', never as 0); a difference is claimed only when its CI excludes 0.",
        f"- Hazard bins (s): {q['hazard_edges']} plus the open bin (> {q['hazard_edges'][-1]} s); right-closed "
        f"bins. Horizon 600 s: decisions without an event by then are censored (also at segment ends). Within-bin "
        f"rule '{ev['rule']}', open-bin tail '{ev['tail']}' where a metric needs S(t) between edges.",
        "- NLL: mean censoring-aware negative log-likelihood per decision (event: -log mass of its bin; censored "
        "at c: -log S(c)); comparable only between models on the same bins. IBS: IPCW integrated Brier score over "
        f"t in [{ev['ibs_grid_s'][0]}, {ev['ibs_grid_s'][1]}) s step {ev['ibs_grid_s'][2]} s. C-td: Antolini "
        "time-dependent concordance (point, seed mean). D-cal: max |share - 0.1| over 10 D-calibration bins "
        "(chi-square statistic in brackets; its p-value assumes independent decisions and is not used).",
        f"- ECE@t / Brier@t: IPCW calibration error (10 uniform bins, Hajek observed frequency) and IPCW Brier of "
        f"P(T <= t), t in {h} s. IPCW weights use the censoring KM of the evaluated split, fixed across bootstrap "
        "replicates. pred / obs: mean predicted P(T <= t) vs IPCW observed frequency.",
        f"- Coherence: gap = P_noul(event within {q['noul_within_s']} s) - (1 - S({q['noul_within_s']})) of the "
        "hazard head of the same question and model; Brier of each against the known labels (decisions censored "
        "before the horizon excluded). KM baselines have no separate head.",
        f"- Invariance audit: for every decision the payload of every item invalid at tau (expired, retracted, "
        f"not yet arrived, padding) is overwritten: random = N(0, {inv['random_std']}^2), huge = +-{float(inv['huge']):g}, "
        "nan, inf = +-inf, hour_earlier = the payload of the same slot (same source and lag) in the decision "
        f"{float(inv['hour_s']):g} s earlier in the segment (later where no earlier decision exists; unchanged where that "
        "slot is empty). Outputs compared: hazard pmf, noul and score probabilities; bitwise = identical float32 "
        "bit patterns for every output of the decision; both passes run under eval_determinism with the same "
        "chunking. dNLL: change of the hazard NLL (mean over questions) on decisions with finite outputs.",
        "- Revalidation: for each execution delay Delta, share of decisions whose dependency set (items valid at "
        "tau) loses an item by tau + Delta (expiry or retraction; the executor's revalidation catches these) and "
        "share that gain a newly valid item (not in the set; only a state-version check catches that).",
        "- Latency: ms p50 / p95 / p99 per call (nearest rank), random decisions per call, CUDA synchronised. "
        "re-score = score_at at tau + Delta: block_local re-uses the item states cached at tau and encodes only "
        "newly valid items; masked recomputes everything (identical to a fresh pass).",
        f"- Deadline demo (MACHINERY CHECK, not a decision-quality result): T_total = T (the predicted "
        f"time-to-event law of the question, treated as a completion time) + independent synthetic pipeline "
        f"latencies {[c['name'] for c in dc['pipeline']]} with known laws. conv: feasible iff the timecalc lower "
        f"bound of P(T_total <= D) >= 1 - delta (n_grid {dc['n_grid']}); qsum: feasible iff the sum of per-"
        "component (1 - delta)-quantiles <= D. on-time = realised T + L <= D with L drawn from the same laws "
        "(seeded); coverage holds when the on-time rate among feasible decisions is >= 1 - delta. On-time rates "
        "pool the feasible decisions of all seeds; '(k/n seeds)' marks cells where some seeds had none. mid = average "
        "of the lower and upper bounds (calibration check of the composed probability). real = realised process "
        "outcomes (tests the predicted law and the machinery together; a gap below 1 - delta is model "
        "miscalibration). sim = T drawn from each decision's own predicted law (uniform inside bins, the open bin "
        "never arrives): tests the machinery alone; conv must reach 1 - delta up to Monte Carlo noise, qsum need "
        "not (two components stall with probability 0.04 < 0.05, which per-component quantiles ignore).",
        "- Score question: km_marginal row = train class frequencies; km_state and km_state_age rows = train "
        "distribution of the bin 60 s ahead given the current bin (undelayed, privileged); the two rows are "
        "identical by construction.",
    ]


def format_results(results, config, note=None):
    parts = ["# Decision backbone on HAI tasks", ""]
    parts += ["## Provenance", "",
              "Data: SEMI-SYNTHETIC evidence timing (delivery delay, validity lifetime, outages, retractions with "
              "replayed payloads drawn from seeded generators) over real HAI content and real process targets "
              "(cpsdefense.data.hai_tasks). Splits: train (fit), val (normal, in-distribution; early stopping) and "
              "dev_test (shifted, contains attacks; reported separately). The held-out HAI test split was never "
              "loaded.", ""]
    if note:
        parts += [note, ""]
    parts += ["## Legend", "", *legend(config), ""]
    h = [float(x) for x in config["eval"]["calib_horizons_s"]]
    deltas = [float(x) for x in config["deadline"]["deltas"]]
    for version, res in results["versions"].items():
        n_seeds = res.get("n_seeds", {})
        splits = [s for s in ("val", "dev_test") if s in res["splits"]]
        parts += [f"## HAI {version}", ""]
        tr = res.get("training")
        if tr:
            parts += ["### Training", "", *_table(
                ["variant", "seeds", "best step (per seed)", "stop", "train min (mean)", "val loss at best (mean)",
                 "parameters"],
                [[v, t["n_seeds"], t["best_steps"], t["stops"], _f(t["train_minutes_mean"], 1),
                  _f(t["best_val_total_mean"], 4), t["n_parameters"]] for v, t in tr.items()]), ""]
        for split in splits:
            r = res["splits"][split]
            parts += [f"### 1. Time-to-event quality: {split} ({r['n']} decisions, {r['hours']:.1f} h, "
                      f"{r['n_blocks']} blocks, attack in horizon {r['attack_in_horizon_rate']:.3f})", "",
                      *tte_table(r, h, n_seeds), ""]
        n_boot = int(config["eval"]["n_boot"])
        parts += ["### 1b. Paired differences (seed means)", "", *paired_table(res["splits"], splits, n_boot), ""]
        parts += ["### 1c. Score question (level bin 60 s ahead)", "",
                  *score_table(res["splits"], splits, n_seeds, n_boot), ""]
        parts += ["### 2. Coherence of the separate noul head", "",
                  *coherence_table(res["splits"], splits, float(config["questions"]["noul_within_s"])), ""]
        if res.get("invariance"):
            parts += ["### 3. Invariance audit (E3a)", "", *invariance_table(res["invariance"]), ""]
        if res.get("revalidation"):
            parts += ["### 4a. Revalidation (E3b)", "", *revalidation_table(res["revalidation"]), ""]
        parts += ["### 5. Deadline feasibility demo (machinery check)", "",
                  *deadline_table(res["splits"], splits, deltas), ""]
    if results.get("latency"):
        lat = results["latency"]
        parts += [f"## 4b. Latency (HAI {lat['version']} {lat['split']}, Delta = {lat['delta_s']:g} s, CPU threads "
                  f"{lat['cpu_threads']}; ms p50 / p95 / p99)", "", *latency_table(lat), ""]
    return "\n".join(parts) + "\n"
