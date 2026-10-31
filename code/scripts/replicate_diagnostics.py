import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpsdefense.experience.dataset import belief_labels, split_of
from cpsdefense.paths import PROCESSED_DIR, ROOT
from cpsdefense.utils.runs import start_run
from dataset_stats import (
    DEV_SPLITS, TOL, best_stats, decision_row, episode_files, jsonable, load_episodes, result_matrix, summary)

SETS = {"natural": PROCESSED_DIR / "tep_experience" / "20260927-003712-v2-natural",
        "stress": PROCESSED_DIR / "tep_experience" / "20260927-085004-v2-stress"}
FRACTIONS = (0.6, 0.2, 0.2)
N_BOOT = 400
SEED = 20261030
SUBSETS = ("all", "ambiguous_family", "twin_weight_pos", "high_stakes")

DEFINITIONS = {
    "development_pool": "train and val episodes of the set (split_of(seed, (0.6, 0.2, 0.2)) from the file name, "
                        "before unpickling); the test split is never read",
    "decision_subsets": {
        "all": "every labelled development decision",
        "ambiguous_family": "decisions of episodes whose incident is a link fault on sensor:<g> or an I/O-module "
                            "fault on <g> (dataset.in_family = arrays dec_family; the paper's 'ambiguous "
                            "decisions', 42.3% / 58.2% of development decisions)",
        "twin_weight_pos": "ambiguous_family decisions whose twin was labelled and has posterior P(c' | E) > 0 "
                           "(deterministic evidence did not exclude the twin); the verifier's 'ambiguous' and "
                           "the investigator's 'ambiguous (twin weight > 0)' in the triage reports",
        "high_stakes": "the hindsight trip probability differs across candidates: max p_trip > min p_trip + TOL "
                       "(dataset_stats stakes; the paper's high-stakes decisions)",
        "family_with_twin_record": "ambiguous_family decisions with a twin record and a defined "
                                   "pi_after_memory_differs (the denominator of dataset_stats.twin_stats)",
    },
    "TOL": f"{TOL}: exact equality of costs (dataset_stats.TOL)",
    "J": "hindsight label: mean over the R replicates of the branch total cost under the true cause "
         "(dataset.candidate_labels)",
    "J_belief": "P(c | E) J(a | c) + P(c' | E) J(a | c', h) for mixed decisions (dataset.belief_labels); J "
                "elsewhere",
    "twin_weight": "P(c' | E), the posterior of the twin cause (belief.posterior of the twin's belief record)",
    "argmin_set": "candidates with J <= min J + TOL (ties included)",
    "exact_tie": "the argmin set has more than one candidate",
    "gathering_optimal": "decision with both observe and non-observe candidates whose best observe J is below "
                         "the best non-observe J by more than TOL (dataset_stats); undefined without both",
    "pi_after_memory_differs": "stored per decision in the twin record: the twin world's own playbook (history "
                               "h') remembers other trace results or actuator link-loss reports at tau than the "
                               "defender's playbook (history h) that the twin labels use",
    "remembered_trace": "the latest-arriving (ties: largest evidence id) trace_link item on sensor:<g> among the "
                        "decision's valid_evidence; its world health and reported health come from the twin's "
                        "belief record (matched by event time)",
    "trace_rule": "a remembered trace exists and the two worlds' link health differed when it was taken",
    "remembered_trace_wrong": "the remembered trace's reported health differs from the true world's health",
    "flagged_share": "share of twin_weight_pos decisions with pi_after_memory_differs True",
    "abs_J_belief_minus_J_per_candidate": "|J_belief - J| pooled over every candidate of the group's decisions "
                                          "(mean and median per candidate, not per decision)",
    "citation_changed": "the twin branches' first-action citation differs from the primary's (list of evidence "
                        "ids): primary ids not citable in the twin were dropped, or replaced by shared world "
                        "evidence when none was left",
    "citation_replaced": "the twin's first-action citation holds none of the primary's cited ids",
    "replicate_sd": "per candidate, the SD (ddof = 1) of the branch total over its R replicates; primary: every "
                    "candidate of every decision; twin: every candidate of every decision with a labelled twin "
                    "(twin weight 0 included)",
    "p_trip_strictly_between": "share of candidates (every decision, primary world) whose trip probability over "
                               "the R replicates is strictly between 0 and 1",
    "runner_up_unresolved_t95": "decisions with a runner-up (a candidate above min J + TOL): gap to the first "
                                "runner-up <= t_{0.975, R-1} times the SE of the paired replicate difference "
                                "(dataset_stats.best_stats unresolved_t95_share); _2se with 2 instead of t",
    "tie_identical": "every pair of tied candidates has costs within TOL in every replicate (for belief ties: "
                     "in both worlds where the twin enters J_belief)",
    "tie_monitor_only": "every tied candidate is a monitor action",
    "bootstrap": "per decision, n_boot draws of R replicate ids with replacement, one draw shared by all "
                 "candidates and both worlds; J and J_belief recomputed from the drawn replicates (posterior "
                 "fixed); p_disjoint = mean share of draws whose argmin set shares no candidate with the "
                 "original one; p_set_differs = mean share whose argmin set is not identical; p_flip = mean "
                 "share whose gathering-optimal answer differs from the original (decisions with both "
                 "families). Means over the decisions of the subset; *_mc_se = Monte Carlo SE of that mean "
                 "from the finite number of draws, sqrt(sum p (1 - p) / (n_boot - 1)) / n. *_gather_optimal "
                 "subsets: decisions whose original answer (under that label set) is gathering optimal",
    "split_halves": "the first R/2 replicates against the last R/2 (R = 4: replicate ids {0, 1} against {2, 3}), "
                    "in both worlds: argmin sets overlap (share a candidate), and the gathering-optimal answer "
                    "agrees (decisions with both families)",
}
NOTES = {
    "ties": "Ties at the minimum are ties of identical trajectories under common random numbers, not ties of "
            "noisy means: in all but a handful of tie decisions (tied_identical_every_replicate) the tied "
            "candidates, mostly the two monitor durations (300 s and 900 s), have identical costs in every "
            "stored replicate. Whether they stay identical in further replicates depends on whether pi_after "
            "acts during the longer wait, a stochastic event.",
    "bootstrap": "A bootstrap over R = 4 values understates the replicate variance (a resampled mean has "
                 "(R - 1) / R of it), so these probabilities probably understate the label instability; they are "
                 "not proven bounds.",
    "citation": "The triage reports' 'twin's first action cites substitute evidence' (11.6% / 9.1%) is "
                "citation_changed, in most of which primary ids were only dropped; substitution proper (none of "
                "the primary's ids left) is citation_replaced.",
    "triage_subset_names": "The triage reports' C5 'family' = ambiguous_family; the verifier's 'ambiguous' and "
                           "its corrected LaTeX ('13.7% / 14.0% of resamples of ambiguous decisions', 'flips "
                           "... in 6.7%') = twin_weight_pos; 'gather flips among belief-gather-optimal "
                           "decisions, n = 205 / 640' = ambiguous_family_gather_optimal, n = 218 / 654 = "
                           "all_gather_optimal.",
}


def dev_splits(splits):
    splits = tuple(splits)
    if not splits or not set(splits) <= set(DEV_SPLITS):
        raise AssertionError(f"only the development splits {DEV_SPLITS} may be read, got {splits}")
    return splits


def count(mask):
    mask = np.asarray(mask, bool)
    return {"k": int(mask.sum()), "n": int(mask.size), "share": float(mask.mean()) if mask.size else None}


def resample_mean(p, n_boot):
    p = np.asarray(p, np.float64)
    if not p.size:
        return None, None
    return float(p.mean()), float(np.sqrt(np.sum(p * (1 - p)) / (n_boot - 1)) / p.size)


def rounded(obj, digits=4):
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, digits) for v in obj]
    return round(obj, digits) if isinstance(obj, float) else obj


def argmin_mask(J):
    J = np.asarray(J, np.float64)
    return J <= J.min(-1, keepdims=True) + TOL


def gather_optimal(J, observe):
    J = np.asarray(J, np.float64)
    return np.where(observe, J, np.inf).min(-1) < np.where(~observe, J, np.inf).min(-1) - TOL


def replicate_labels(P, T, w, idx):
    Jh = np.moveaxis(P[:, idx].mean(-1), 0, -1)
    if w[1] == 0:
        return Jh, Jh
    return Jh, w[0] * Jh + w[1] * np.moveaxis(T[:, idx].mean(-1), 0, -1)


def resample(P, T, w, observe, has_both, idx):
    J0 = replicate_labels(P, T, w, np.arange(P.shape[1]))
    J1 = replicate_labels(P, T, w, idx)
    out = {}
    for name, a, b in zip(("hindsight", "belief"), J0, J1):
        m0, m1 = argmin_mask(a), argmin_mask(b)
        out[f"{name}_disjoint"] = float(np.mean(~(m1 & m0).any(-1)))
        out[f"{name}_differs"] = float(np.mean((m1 != m0).any(-1)))
        out[f"{name}_flip"] = (float(np.mean(gather_optimal(b, observe) != gather_optimal(a, observe)))
                               if has_both else np.nan)
    return out


def split_halves(P, T, w, observe, has_both):
    R = P.shape[1]
    A = replicate_labels(P, T, w, np.arange(R // 2))
    B = replicate_labels(P, T, w, np.arange(R // 2, R))
    out = {}
    for name, a, b in zip(("hindsight", "belief"), A, B):
        out[f"{name}_overlap"] = bool((argmin_mask(a) & argmin_mask(b)).any())
        out[f"{name}_gather_agree"] = (bool(gather_optimal(a, observe) == gather_optimal(b, observe))
                                       if has_both else None)
    return out


def identical_rows(M, rows):
    return bool(np.all(np.abs(M[rows] - M[rows[0]]) <= TOL))


def remembered_trace(items, decision, group):
    link = f"sensor:{group}"
    traces = [items[e] for e in decision["valid_evidence"]
              if e in items and items[e]["kind"] == "trace_link" and items[e]["subject_id"] == link]
    if not traces:
        return None
    last = max(traces, key=lambda i: (i["arrival_time"], i["evidence_id"]))
    match = [t for t in decision["twin"]["belief"]["traces"] if abs(t["time"] - float(last["event_time"])) < 1e-6]
    if len(match) != 1:
        raise ValueError(f"trace {last['evidence_id']}: {len(match)} matches in the twin's belief record")
    t = match[0]
    return bool(t["healthy"][0] != t["healthy"][1]), bool(t["reported_healthy"] != t["healthy"][0])


def decision_record(episode, d, items):
    row = decision_row(episode, d)
    bl = belief_labels(d, row["family"])
    P = result_matrix(d, "total")
    twin = d.get("twin") or {}
    T = None
    if twin.get("results") is not None:
        T = result_matrix({"candidates": d["candidates"], "results": twin["results"]}, "total")
    mixed = not bl["belief_fallback"] and bl["twin_weight"] > 0
    w = (bl["posterior"], bl["twin_weight"]) if mixed else (1.0, 0.0)
    if mixed and sorted({r["replicate"] for r in d["results"]}) != sorted({r["replicate"] for r in twin["results"]}):
        raise ValueError(f"seed {episode['seed']}, tau {d['tau']}: primary and twin replicates differ")
    if np.isnan(P).any() or (mixed and np.isnan(T).any()):
        raise ValueError(f"seed {episode['seed']}, tau {d['tau']}: missing replicate results")
    if not np.array_equal(replicate_labels(P, T, w, np.arange(P.shape[1]))[1], bl["J_belief"]):
        raise AssertionError(f"seed {episode['seed']}, tau {d['tau']}: J_belief not reproduced from replicates")
    observe = np.array([c["family"] == "observe" for c in d["candidates"]])
    flag = twin.get("pi_after_memory_differs") if twin else None
    cited, cited_twin = list(d["cited_evidence"]), list(twin.get("cited_evidence") or [])
    return {
        "row": row, "P": P, "T": T, "w": w, "observe": observe, "has_both": bool(observe.any() and (~observe).any()),
        "J": row["cand"]["J"], "J_belief": bl["J_belief"], "twin_weight": bl["twin_weight"],
        "twin_labelled": T is not None, "flag": flag,
        "trace": remembered_trace(items, d, row["family_group"]) if flag is not None else None,
        "citation_changed": cited_twin != cited, "citation_replaced": bool(cited) and not set(cited) & set(cited_twin),
    }


def records_of(episodes):
    out = []
    for e in episodes:
        items = {i["evidence_id"]: i for i in e["items"]}
        out += [decision_record(e, d, items) for d in e["decisions"]]
    return out


def subset_masks(records):
    row = [r["row"] for r in records]
    fam = np.array([x["family"] for x in row])
    return {"all": np.ones(len(records), bool), "ambiguous_family": fam,
            "twin_weight_pos": fam & np.array([r["twin_labelled"] and r["twin_weight"] > 0 for r in records]),
            "high_stakes": np.array([x["p_max"] > x["p_min"] + TOL for x in row])}


def twin_memory(records, masks):
    with_flag = np.array([r["row"]["family"] and r["row"]["twin_record"] and r["flag"] is not None for r in records])
    rec = [r for r, m in zip(records, with_flag) if m]
    flag = np.array([bool(r["flag"]) for r in rec])
    rule = np.array([r["trace"] is not None and r["trace"][0] for r in rec])
    crosstab = {f"flag={a},trace_rule={b}": int(((flag == a) & (rule == b)).sum()) for a in (True, False)
                for b in (True, False)}
    pos = [r for r, m in zip(records, masks["twin_weight_pos"]) if m]
    groups = {}
    for name, value in (("flagged", True), ("unflagged", False)):
        g = [r for r in pos if bool(r["flag"]) == value]
        absd = np.concatenate([np.abs(r["J_belief"] - r["J"]) for r in g]) if g else np.zeros(0)
        groups[name] = {
            "n": len(g), "twin_weight": summary([r["twin_weight"] for r in g]),
            "abs_J_belief_minus_J_per_candidate": {"n": int(absd.size), "mean": float(absd.mean()) if g else None,
                                                   "median": float(np.median(absd)) if g else None},
            "remembered_trace_wrong": count([r["trace"][1] for r in g if r["trace"] is not None])}
    return {
        "flag_share_family": count(flag),
        "flag_equals_trace_rule": {**count(flag == rule), "crosstab": crosstab},
        "twin_weight_pos": {"n": len(pos), "flagged_share": count([bool(r["flag"]) for r in pos]), "by_flag": groups,
                            "citation_changed": count([r["citation_changed"] for r in pos]),
                            "citation_replaced": count([r["citation_replaced"] for r in pos])},
    }


def ties(records, masks):
    out = {}
    for labels, key in (("hindsight", "J"), ("belief", "J_belief")):
        best = [np.flatnonzero(argmin_mask(r[key])) for r in records]
        tie = np.array([b.size > 1 for b in best])
        tied = [(r, b) for r, b, t in zip(records, best, tie) if t]
        same = [identical_rows(r["P"], b) and (labels == "hindsight" or r["w"][1] == 0 or identical_rows(r["T"], b))
                for r, b in tied]
        out[labels] = {**{s: count(tie[m]) for s, m in masks.items()},
                       "tied_identical_every_replicate": count(same),
                       "tied_monitor_only": count([all(r["row"]["cand"]["tool"][i] == "monitor" for i in b)
                                                   for r, b in tied])}
    return out


def replicate_noise(records, masks, n_boot, seed):
    rows = [r["row"] for r in records]
    col = {k: np.array([x[k] for x in rows]) for k in ("gap", "gap_se", "n_replicates", "incident_class", "n_best")}
    best = best_stats(rows, col)
    sd_twin = np.concatenate([r["T"].std(1, ddof=1) for r in records if r["T"] is not None])
    p_trip = np.concatenate([x["cand"]["p_trip"] for x in rows])
    rng = np.random.default_rng(seed)
    boot, halves = [], []
    for r in records:
        R = r["P"].shape[1]
        args = (r["P"], r["T"], r["w"], r["observe"], r["has_both"])
        boot.append(resample(*args, rng.integers(0, R, (n_boot, R))))
        halves.append(split_halves(*args))
    both = np.array([r["has_both"] for r in records])
    out = {
        "replicates_per_decision": {str(k): int(v) for k, v in zip(*np.unique(col["n_replicates"],
                                                                                return_counts=True))},
        "replicate_sd_primary": summary(np.concatenate([x["cand"]["J_sd"] for x in rows])),
        "replicate_sd_twin": summary(sd_twin),
        "p_trip_strictly_between": count((p_trip > 0) & (p_trip < 1)),
        "runner_up": {"defined": int(np.isfinite(col["gap"]).sum()), "t95_critical": best["t95_critical"],
                      "unresolved_t95_share": best["unresolved_t95_share"],
                      "unresolved_2se_share": best["unresolved_2se_share"]},
        "ties": ties(records, masks),
        "bootstrap": {"n_boot": n_boot, "seed": seed},
        "split_halves": {},
    }
    for labels, key in (("hindsight", "J"), ("belief", "J_belief")):
        dis = np.array([b[f"{labels}_disjoint"] for b in boot])
        dif = np.array([b[f"{labels}_differs"] for b in boot])
        flip = np.array([b[f"{labels}_flip"] for b in boot])
        g0 = both & np.array([bool(gather_optimal(r[key], r["observe"])) for r in records])
        sel = {**{s: m & both for s, m in masks.items()},
               "all_gather_optimal": g0, "ambiguous_family_gather_optimal": masks["ambiguous_family"] & g0}
        argmin, gather = {}, {}
        for s, m in masks.items():
            p, se = resample_mean(dis[m], n_boot)
            argmin[s] = {"n": int(m.sum()), "p_disjoint": p, "p_disjoint_mc_se": se,
                         "p_set_differs": resample_mean(dif[m], n_boot)[0]}
        for s, m in sel.items():
            p, se = resample_mean(flip[m], n_boot)
            gather[s] = {"n": int(m.sum()), "p_flip": p, "p_flip_mc_se": se}
        out["bootstrap"][labels] = {"argmin": argmin, "gather_flip": gather}
        overlap = np.array([h[f"{labels}_overlap"] for h in halves])
        agree = np.array([bool(h[f"{labels}_gather_agree"]) for h in halves])
        out["split_halves"][labels] = {"argmin_overlap": {s: count(overlap[m]) for s, m in masks.items()},
                                       "gather_agree": {s: count(agree[m & both]) for s, m in masks.items()}}
    return out


def analyse(episodes, n_boot, seed):
    records = records_of(episodes)
    masks = subset_masks(records)
    return {"decisions": len(records), "subset_sizes": {s: int(m.sum()) for s, m in masks.items()},
            "twin_memory": twin_memory(records, masks),
            "replicate_noise": replicate_noise(records, masks, n_boot, seed)}


def pct(c):
    return f"{c['k']:,}/{c['n']:,} = {100 * c['k'] / c['n']:.1f}%" if c["n"] else "n/a"


def joined(*values, digits=1, scale=100.0, unit="%"):
    return " / ".join("n/a" if v is None else f"{scale * v:.{digits}f}{unit}" for v in values)


def table_rows(s):
    tm, rn = s["twin_memory"], s["replicate_noise"]
    tp, fl = tm["twin_weight_pos"], tm["twin_weight_pos"]["by_flag"]
    ties_h, ties_b = rn["ties"]["hindsight"], rn["ties"]["belief"]
    bh, bb = rn["bootstrap"]["hindsight"], rn["bootstrap"]["belief"]
    sh, sb = rn["split_halves"]["hindsight"], rn["split_halves"]["belief"]
    return [
        ("decisions (all / ambiguous_family / twin_weight_pos / high_stakes)",
         " / ".join(f"{v:,}" for v in s["subset_sizes"].values())),
        ("C1 memory flag, family decisions with a twin record", pct(tm["flag_share_family"])),
        ("C1 flag equals the trace rule", pct(tm["flag_equals_trace_rule"])),
        ("C1 flagged, twin_weight_pos", pct(tp["flagged_share"])),
        ("C1 median twin weight, flagged / unflagged",
         joined(fl["flagged"]["twin_weight"]["median"], fl["unflagged"]["twin_weight"]["median"], digits=4,
                scale=1.0, unit="")),
        ("C1 mean abs(J_belief - J) per candidate, flagged / unflagged",
         joined(fl["flagged"]["abs_J_belief_minus_J_per_candidate"]["mean"],
                fl["unflagged"]["abs_J_belief_minus_J_per_candidate"]["mean"], digits=3, scale=1.0, unit="")),
        ("C1 remembered trace wrong, flagged", pct(fl["flagged"]["remembered_trace_wrong"])),
        ("C1 twin citation changed / replaced, twin_weight_pos",
         f"{pct(tp['citation_changed'])}; {pct(tp['citation_replaced'])}"),
        ("C5 replicates per decision", ", ".join(f"R={k}: {v:,}" for k, v in rn["replicates_per_decision"].items())),
        ("C5 replicate SD of J, median / p95, primary",
         joined(rn["replicate_sd_primary"]["median"], rn["replicate_sd_primary"]["q95"], digits=2, scale=1.0,
                unit="")),
        ("C5 replicate SD of J, median / p95, twin",
         joined(rn["replicate_sd_twin"]["median"], rn["replicate_sd_twin"]["q95"], digits=2, scale=1.0, unit="")),
        ("C5 candidates with 0 < p_trip < 1", pct(rn["p_trip_strictly_between"])),
        ("C5 runner-up unresolved, t95 / 2SE",
         joined(rn["runner_up"]["unresolved_t95_share"], rn["runner_up"]["unresolved_2se_share"])),
        ("C5 exact ties, hindsight / belief / hindsight high-stakes",
         joined(ties_h["all"]["share"], ties_b["all"]["share"], ties_h["high_stakes"]["share"])),
        ("C5 hindsight ties identical in every replicate", pct(ties_h["tied_identical_every_replicate"])),
        ("C5 hindsight ties among monitor candidates only", pct(ties_h["tied_monitor_only"])),
        ("C5 P(argmin set disjoint), hindsight: all / high_stakes",
         joined(bh["argmin"]["all"]["p_disjoint"], bh["argmin"]["high_stakes"]["p_disjoint"], digits=2)),
        ("C5 P(argmin set disjoint), belief: all / ambiguous_family / twin_weight_pos / high_stakes",
         joined(*(bb["argmin"][k]["p_disjoint"] for k in SUBSETS), digits=2)),
        ("C5 P(gather flip), belief: all / ambiguous_family / twin_weight_pos",
         joined(*(bb["gather_flip"][k]["p_flip"] for k in SUBSETS[:3]), digits=2)),
        ("C5 Monte Carlo SE of the bootstrap means above, max",
         joined(max([bh["argmin"][k]["p_disjoint_mc_se"] for k in ("all", "high_stakes")]
                    + [bb["argmin"][k]["p_disjoint_mc_se"] for k in SUBSETS]
                    + [bb["gather_flip"][k]["p_flip_mc_se"] for k in SUBSETS[:3]]), digits=2)),
        ("C5 split halves, argmin overlap: hindsight all / belief all / belief family / belief high_stakes",
         joined(sh["argmin_overlap"]["all"]["share"], sb["argmin_overlap"]["all"]["share"],
                sb["argmin_overlap"]["ambiguous_family"]["share"], sb["argmin_overlap"]["high_stakes"]["share"])),
        ("C5 split halves, gather answer agrees: hindsight / belief",
         joined(sh["gather_agree"]["all"]["share"], sb["gather_agree"]["all"]["share"])),
    ]


def markdown(results):
    names = list(results["sets"])
    rows = {n: table_rows(s) for n, s in results["sets"].items()}
    cfg = results["config"]
    lines = ["# Replicate noise and twin memory (C1, C5)", "",
             f"Development pool ({' + '.join(cfg['splits'])}) of each set, raw episodes; replicate bootstrap B = "
             f"{cfg['n_boot']}, seed {cfg['seed']}. Subsets and every quantity are defined under `definitions` "
             f"in results.json.", "",
             "| quantity | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for i, (label, _) in enumerate(rows[names[0]]):
        lines.append(f"| {label} | " + " | ".join(rows[n][i][1] for n in names) + " |")
    lines += ["", NOTES["ties"], "", NOTES["bootstrap"], ""]
    return "\n".join(lines)


def headline(results):
    out = {}
    for name, s in results["sets"].items():
        tm, rn = s["twin_memory"], s["replicate_noise"]
        out[name] = {
            "memory_flag_share_family": tm["flag_share_family"]["share"],
            "flag_equals_trace_rule_share": tm["flag_equals_trace_rule"]["share"],
            "runner_up_unresolved_t95_share": rn["runner_up"]["unresolved_t95_share"],
            "tie_share_hindsight": rn["ties"]["hindsight"]["all"]["share"],
            "belief_p_disjoint_ambiguous_family": rn["bootstrap"]["belief"]["argmin"]["ambiguous_family"]["p_disjoint"],
            "belief_p_gather_flip_ambiguous_family":
                rn["bootstrap"]["belief"]["gather_flip"]["ambiguous_family"]["p_flip"]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=[f"{k}={v}" for k, v in SETS.items()],
                        help="name=raw episode directory (default: the v2 natural and stress sets)")
    parser.add_argument("--splits", nargs="+", default=list(DEV_SPLITS), help="train and/or val; nothing else")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--name", default="dev")
    args = parser.parse_args()
    splits = dev_splits(args.splits)
    sets = {k: ROOT / v for k, v in (s.split("=", 1) for s in args.sets)}
    t0 = time.perf_counter()
    config = {"sets": {k: v.relative_to(ROOT).as_posix() if v.is_relative_to(ROOT) else str(v)
                       for k, v in sets.items()},
              "splits": list(splits), "fractions": list(FRACTIONS), "n_boot": args.n_boot, "seed": args.seed,
              "tol": TOL}
    run = start_run("replicate_diagnostics", config, seed=args.seed, name=args.name)
    results = {"config": config, "definitions": DEFINITIONS, "notes": NOTES, "sets": {}}
    for name, root in sets.items():
        by_split = load_episodes(root, splits, FRACTIONS)
        episodes = [e for s in splits for e in by_split[s]]
        if any(split_of(e["seed"], FRACTIONS) not in splits for e in episodes):
            raise AssertionError(f"{name}: an episode outside {splits} was loaded")
        out = analyse(episodes, args.n_boot, args.seed)
        out["episodes"] = {s: len(by_split[s]) for s in splits}
        out["test_episode_files_not_opened"] = sum(split_of(k, FRACTIONS) == "test" for k in episode_files(root))
        results["sets"][name] = out
        print(f"{name}: {len(episodes)} episodes, {out['decisions']} decisions ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    results["wall_s"] = round(time.perf_counter() - t0, 1)
    results = jsonable(results)
    text = markdown(results)
    run.save_json("results.json", {**results, "sets": rounded(results["sets"])})
    (run.dir / "summary.md").write_text(text, encoding="utf-8")
    run.finish(rounded(headline(results)))
    print(text)
    print(f"-> {run.dir}")

if __name__ == "__main__":
    main()
