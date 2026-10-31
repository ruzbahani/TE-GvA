import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from belief_sensitivity import twin_labels
from cpsdefense.eval.metrics import paired_cluster_bootstrap_diff
from cpsdefense.paths import ROOT
from cpsdefense.utils.runs import start_run
from dataset_stats import TOL, jsonable, share
from train_direct import (
    FLAT_GRID, context, flat_predictions, label_view, load_split, m_mean, m_mean_where, rows_for_choice, subsets)

DEV_SPLITS = ("train", "val")
SETS = ("data/processed/tep_decisions/v2-natural", "data/processed/tep_decisions/v2-stress")
COMPARE_RUNS = ("runs/model_direct/20260927-224653-v2n-compare-flatref",
                "runs/model_direct/20260927-224653-v2s-compare-flatref")
LABEL_SETS = ("belief", "hindsight")
BINS = ("zero", "T1", "T2", "T3")
CONTRAST_SEED = 1
DEFINITIONS = {
    "E1": "per decision, regret of the flat GBM trained on the train split's belief labels (J_belief) minus "
          "regret of the flat GBM trained on its hindsight labels (J); each GBM's FLAT_GRID point is chosen on val "
          "regret under its own label set (train_direct.flat_predictions); negative favours belief labels",
    "regret": "J of the chosen candidate minus the lowest J among the decision's candidates, with J from the "
              "scoring label set (belief_scoring: J_belief; hindsight_scoring: J)",
    "gbm_belief / gbm_hindsight": "the flat GBM trained on belief / hindsight labels",
    "ambiguous family": "dec_family: the episode's incident is a link or remote I/O module fault of one sensor "
                        "group (the paper's ambiguous family); view 'family'",
    "backup": "dec_backup_available (views backup_available / backup_unavailable)",
    "n / episodes": "decisions / distinct episodes among them",
    "pi": "dec_posterior, the posterior of the true cause given the defender's evidence (1 outside the family "
          "and where the twin was excluded)",
    "v": "between-cause term per decision: mean over the decision's candidates of pi (1 - pi) (J - J_twin)^2, "
         "J_twin = (J_belief - pi J) / (1 - pi) (belief_sensitivity.twin_labels); 0 where 1 - pi < 1e-12; uses "
         "the stored label estimates",
    "a": "pi (1 - pi) per decision (0 where pi is not finite)",
    "bins": "zero: score <= TOL; T1-T3: terciles of the score over the decisions with score > TOL "
            "(np.quantile at 1/3 and 2/3; a value equal to a cut point goes to the upper bin). v_bins: the set's "
            "own cut points; v_bins_reference_cuts: the cut points of the reference set (--cuts-from)",
    "TOL": f"{TOL:g}: values closer than this count as equal (scripts/dataset_stats.py TOL)",
    "gather_optimal_belief_share": "share of decisions with both observe and non-observe candidates in which the "
                                   "best observe candidate's J_belief is below the best non-observe candidate's "
                                   "by more than TOL (scripts/dataset_stats.py, gather_belief)",
    "high_stakes_share": "share of decisions whose candidates' hindsight trip probabilities differ by more than "
                         "TOL (scripts/dataset_stats.py, the paper's high-stakes decisions); "
                         "high_stakes_belief_share: the same on p_trip_belief",
    "E1 intervals": "point, 95% percentile interval and two-sided bootstrap p "
                    "(cpsdefense.eval.metrics.paired_cluster_bootstrap_diff): per subset, a paired bootstrap over "
                    "the subset's episodes, seed 0",
    "contrasts": "E1 on one subset minus E1 on another (no_backup_minus_backup, top_minus_zero), "
                 "paired_cluster_bootstrap_diff of the subset means on joint resamples of all val episodes, seed 1",
    "v_above_median": "v above the median of v over the reference set's val decisions with v > TOL",
    "differ_share": "share of decisions where the two GBMs choose different candidates",
    "E1_given_differ": "mean E1 over the decisions where the two GBMs choose different candidates (E1 is 0 where "
                       "they choose the same one)",
    "switch_belief_optimal_share": "the candidate with the lowest J_belief (the first on exact ties) is a "
                                   "switch_to_backup action; switch_belief_co_optimal_share: some "
                                   "switch_to_backup candidate is within TOL of the lowest J_belief",
    "gbm_switch_share": "share of decisions where the GBM chooses a switch_to_backup action",
}
CAVEATS = (
    "Validation only and exploratory: not pre-specified, and the GBM grid points were chosen on val.",
    "v-bins select on label realizations (v is computed from the stored label estimates J and J_belief), so the "
    "argument that hindsight-scored differences are unbiased does not carry over to them; it does carry over to "
    "the a-bins, which use the posterior only.",
    "v uses the label estimates in place of the expected costs, so it overstates the between-cause term by "
    "pi (1 - pi) Var(J - J_twin) on average.",
    "In the zero bin of v the two label sets are (up to decisions with 1 - pi < 1e-12) identical, so E1 there "
    "comes from training only (spillover).",
    "first_actions describes the first action only; what the follow-up policy does with the backup channel "
    "afterwards is not measured here.",
)


def load_dev(root, splits=DEV_SPLITS):
    if not all(s in DEV_SPLITS for s in splits):
        raise AssertionError(f"only the {DEV_SPLITS} splits are read, got {tuple(splits)}")
    return {s: load_split(root, s) for s in splits}


def between_cause(split):
    dec = split["cand"]["decision"]
    p = split["dec"]["posterior"].astype(np.float64)
    J_twin = twin_labels(split)[0]
    pc = p[dec]
    term = np.where(np.isfinite(J_twin), pc * (1 - pc) * (split["cand"]["J"].astype(np.float64) - J_twin) ** 2, 0.0)
    v = np.bincount(dec, term, minlength=split["n"]) / np.bincount(dec, minlength=split["n"])
    return v, np.where(np.isfinite(p), p * (1 - p), 0.0)


def tercile_cuts(score):
    return np.quantile(score[score > TOL], [1 / 3, 2 / 3])


def bin_of(score, cuts):
    out = np.zeros(score.size, np.int64)
    pos = score > TOL
    out[pos] = 1 + np.searchsorted(np.asarray(cuts), score[pos], side="right")
    return out


def gather_optimal(ctx):
    return ctx["has_both"] & (ctx["obs_min"] < ctx["act_min"] - TOL)


def high_stakes(ctx):
    return np.where(ctx["mask"], ctx["p_trip"], -np.inf).max(1) > ctx["pmin"] + TOL


def decision_arrays(root, log):
    vocab = json.loads((root / "meta.json").read_text(encoding="utf-8"))["vocab"]
    data = load_dev(root)
    pred, flat = {}, {}
    for labels in LABEL_SETS:
        out, table, params = flat_predictions({s: label_view(d, labels) for s, d in data.items()}, vocab, log)
        pred[labels] = out["flat_gbm"]["val"][0][0]
        flat[labels] = {"grid": table, "chosen": params}
    val = data["val"]
    ctx = {sc: context(val, vocab, labels=sc) for sc in LABEL_SETS}
    choice = {m: np.where(val["mask"], pred[m], np.inf).argmin(1) for m in LABEL_SETS}
    regret = {sc: {m: rows_for_choice(ctx[sc], choice[m])["regret"] for m in LABEL_SETS} for sc in LABEL_SETS}
    v, a = between_cause(val)
    lo, hi = vocab["action_blocks"]["tool"]
    d = {"n": val["n"], "cluster": val["dec"]["episode_seed"], "family": val["dec"]["family"].astype(bool),
         "backup": val["dec"]["backup_available"].astype(bool), "subsets": subsets(val), "v": v, "a": a,
         "choice": choice, "regret": regret, "e1": {sc: regret[sc]["belief"] - regret[sc]["hindsight"]
                                                    for sc in LABEL_SETS},
         "has_both": ctx["belief"]["has_both"], "gather_optimal": gather_optimal(ctx["belief"]),
         "high_stakes": high_stakes(ctx["hindsight"]), "high_stakes_belief": high_stakes(ctx["belief"]),
         "mask": val["mask"], "J_belief": ctx["belief"]["J"], "tool": val["pad"]["features"][..., lo:hi].argmax(-1),
         "switch": vocab["tools"].index("switch_to_backup")}
    return d, flat


def cell(d, sel, n_boot):
    sel = np.asarray(sel, bool)
    out = {"n": int(sel.sum()), "episodes": int(np.unique(d["cluster"][sel]).size)}
    if not sel.any():
        return out
    out.update(
        mean_v=float(d["v"][sel].mean()), family_share=share(d["family"][sel]),
        gather_optimal_belief_share=share(d["gather_optimal"][sel & d["has_both"]]),
        high_stakes_share=share(d["high_stakes"][sel]), high_stakes_belief_share=share(d["high_stakes_belief"][sel]),
        regret={f"{sc}_scoring": {f"gbm_{m}": m_mean(d["regret"][sc][m][sel]) for m in LABEL_SETS}
                for sc in LABEL_SETS},
        E1={f"{sc}_scoring": paired_cluster_bootstrap_diff(m_mean, [d["regret"][sc]["belief"][sel]],
                                                          [d["regret"][sc]["hindsight"][sel]], d["cluster"][sel],
                                                          n_boot=n_boot, seed=0) for sc in LABEL_SETS})
    return out


def contrast(d, sel_a, sel_b, n_boot):
    return {f"{sc}_scoring": paired_cluster_bootstrap_diff(m_mean_where, [d["e1"][sc], sel_a], [d["e1"][sc], sel_b],
                                                          d["cluster"], n_boot=n_boot, seed=CONTRAST_SEED)
            for sc in LABEL_SETS}


def binned(d, score, cuts, n_boot):
    b = bin_of(score, cuts)
    out = {"cuts": [float(c) for c in cuts], "bins": {}}
    for k, name in enumerate(BINS):
        sel = b == k
        out["bins"][name] = {**cell(d, sel, n_boot), "mean_score": float(score[sel].mean()) if sel.any() else None,
                             "range": [float(score[sel].min()), float(score[sel].max())] if sel.any() else None}
    out["top_minus_zero"] = contrast(d, b == len(BINS) - 1, b == 0, n_boot)
    return out


def first_actions(d, sel):
    sel = np.asarray(sel, bool)
    r = np.arange(sel.size)
    differ = d["choice"]["belief"] != d["choice"]["hindsight"]
    switch = d["mask"] & (d["tool"] == d["switch"])
    best = switch[r, d["J_belief"].argmin(1)]
    co_best = (switch & (d["J_belief"] <= d["J_belief"].min(1, keepdims=True) + TOL)).any(1)
    return {"n": int(sel.sum()), "episodes": int(np.unique(d["cluster"][sel]).size),
            "differ_share": share(differ[sel]),
            "E1_mean": {f"{sc}_scoring": m_mean(d["e1"][sc][sel]) for sc in LABEL_SETS},
            "E1_given_differ": {f"{sc}_scoring": m_mean(d["e1"][sc][sel & differ]) for sc in LABEL_SETS},
            "E1_zero_where_same": bool(all(np.all(d["e1"][sc][sel & ~differ] == 0) for sc in LABEL_SETS)),
            "switch_candidate_share": share(switch.any(1)[sel]),
            "switch_belief_optimal_share": share(best[sel]), "switch_belief_optimal_n": int(best[sel].sum()),
            "switch_belief_co_optimal_share": share(co_best[sel]),
            "gbm_switch_share": {f"gbm_{m}": share(switch[r, d["choice"][m]][sel]) for m in LABEL_SETS}}


def analyse(d, ref_cuts, threshold, n_boot):
    res = {"decisions": d["n"], "episodes": int(np.unique(d["cluster"]).size),
           "views": {name: cell(d, sel, n_boot) for name, sel in d["subsets"]},
           "v_bins": binned(d, d["v"], tercile_cuts(d["v"]), n_boot)}
    if ref_cuts is not None:
        res["v_bins_reference_cuts"] = binned(d, d["v"], ref_cuts, n_boot)
    res["a_bins"] = binned(d, d["a"], tercile_cuts(d["a"]), n_boot)
    if 0 < d["backup"].sum() < d["n"]:
        res["no_backup_minus_backup"] = contrast(d, ~d["backup"], d["backup"], n_boot)
        scopes = (("all", np.ones(d["n"], bool)), ("v_above_median", d["v"] > threshold))
        strata = (("backup_available", d["backup"]), ("backup_unavailable", ~d["backup"]))
        res["first_actions"] = {f"{scope}/{stratum}": first_actions(d, s & b)
                                for scope, s in scopes for stratum, b in strata}
    return res


def recorded_check(run_dir, res, flat):
    rec = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    diffs = []
    for view, c in res["views"].items():
        for sc in LABEL_SETS:
            r = rec["results"]["val"][f"{sc}/{view}"]
            if r["reference"] != "flat_gbm_belief":
                raise ValueError(f"{run_dir}: reference {r['reference']!r}, expected 'flat_gbm_belief'")
            p = r["paired"]["flat_gbm_hindsight"]["regret"]
            diffs += [abs(p[k] - c["E1"][f"{sc}_scoring"][k]) for k in ("point", "lo", "hi", "p_two_sided")]
            diffs += [abs(r["scores"][f"flat_gbm_{m}"]["regret"]["point"] - c["regret"][f"{sc}_scoring"][f"gbm_{m}"])
                      for m in LABEL_SETS]
    return {"run": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
            "chosen_equal": all(rec["flat"][m]["chosen"] == flat[m]["chosen"] for m in LABEL_SETS),
            "max_abs_diff": float(max(diffs))}


def rounded(obj, digits=4):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {str(k): rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, digits) for v in obj]
    obj = jsonable(obj)
    return round(obj, digits) if isinstance(obj, float) else obj


def num(x, spec="+.3f"):
    return "-" if x is None else format(x, spec)


def fmt(ci):
    if ci is None or ci.get("point") is None:
        return "-"
    if ci.get("lo") is None or ci.get("hi") is None:
        return num(ci["point"])
    p = ci.get("p_two_sided")
    return f"{ci['point']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]" + ("" if p is None else f", p={p:.3f}")


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def summary_markdown(res):
    lines = ["# E1 by backup availability and by the between-cause term (validation, exploratory)", "",
             "E1 = regret of the GBM trained on belief labels minus regret of the GBM trained on hindsight labels "
             "(negative favours belief labels), scored with belief labels (b) and hindsight labels (h). Per-subset "
             "values: paired 95% cluster-bootstrap interval over the subset's episodes and two-sided bootstrap p; "
             "contrasts: joint resamples of all val episodes. Definitions in results.json.", "",
             "## Reproduction (val)", "",
             "| set | chosen, belief labels | chosen, hindsight labels | regret b (GBM b / GBM h) | "
             "regret h (GBM b / GBM h) | recorded run: same choice, max abs diff |", "|---" * 6 + "|"]
    for name, s in res["sets"].items():
        reg = {sc: " / ".join(f"{x:.4f}" for x in r.values()) for sc, r in s["views"]["all"]["regret"].items()}
        rep = s.get("reproduction")
        lines.append(f"| {name} | {s['flat']['belief']['chosen']} | {s['flat']['hindsight']['chosen']} | "
                     f"{reg['belief_scoring']} | {reg['hindsight_scoring']} | "
                     + ("-" if rep is None else f"{rep['chosen_equal']}, {rep['max_abs_diff']:.1e}") + " |")
    lines += ["", "## E1 views", "", "| set | view | n (episodes) | E1 b | E1 h |", "|---" * 5 + "|"]
    for name, s in res["sets"].items():
        for view, c in s["views"].items():
            lines.append(f"| {name} | {view} | {c['n']} ({c['episodes']}) | {fmt(c['E1']['belief_scoring'])} | "
                         f"{fmt(c['E1']['hindsight_scoring'])} |")
        if "no_backup_minus_backup" in s:
            nb = s["no_backup_minus_backup"]
            lines.append(f"| {name} | no backup minus backup (joint) | {s['decisions']} ({s['episodes']}) | "
                         f"{fmt(nb['belief_scoring'])} | {fmt(nb['hindsight_scoring'])} |")
    for key, title in (("v_bins", "v, own cut points"), ("v_bins_reference_cuts", "v, reference cut points"),
                       ("a_bins", "a = pi (1 - pi), own cut points")):
        for name, s in res["sets"].items():
            if key not in s:
                continue
            t = s[key]
            lines += ["", f"## {name}: bins of {title} (cut points {t['cuts'][0]:.4g}, {t['cuts'][1]:.4g})", "",
                      "| bin | range | n (episodes) | mean | gather optimal (b) | high-stakes | E1 b | E1 h |",
                      "|---" * 8 + "|"]
            for b, c in t["bins"].items():
                rng = "-" if c.get("range") is None else f"{c['range'][0]:.4g} - {c['range'][1]:.4g}"
                mean = num(c.get("mean_score"), ".4g")
                lines.append(f"| {b} | {rng} | {c['n']} ({c['episodes']}) | {mean} | "
                             f"{pct(c.get('gather_optimal_belief_share'))} | {pct(c.get('high_stakes_share'))} | "
                             f"{fmt(c.get('E1', {}).get('belief_scoring'))} | "
                             f"{fmt(c.get('E1', {}).get('hindsight_scoring'))} |")
            lines.append(f"| top minus zero | | | | | | {fmt(t['top_minus_zero']['belief_scoring'])} | "
                         f"{fmt(t['top_minus_zero']['hindsight_scoring'])} |")
    rows = [(name, k, c) for name, s in res["sets"].items() for k, c in s.get("first_actions", {}).items()]
    if rows:
        lines += ["", f"## First actions (v_above_median: v > {res['v_median_threshold']:.4g})", "",
                  "| set | decisions | n (episodes) | GBMs differ | E1 b | E1 b given differ | E1 h given differ | "
                  "switch_to_backup belief-optimal (co-optimal) | GBM b / h choose switch |", "|---" * 9 + "|"]
        for name, k, c in rows:
            picks = " / ".join(pct(x) for x in c["gbm_switch_share"].values())
            lines.append(f"| {name} | {k} | {c['n']} ({c['episodes']}) | {pct(c['differ_share'])} | "
                         f"{num(c['E1_mean']['belief_scoring'])} | {num(c['E1_given_differ']['belief_scoring'])} | "
                         f"{num(c['E1_given_differ']['hindsight_scoring'])} | "
                         f"{pct(c['switch_belief_optimal_share'])} ({pct(c['switch_belief_co_optimal_share'])}) | "
                         f"{picks} |")
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in res["caveats"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=list(SETS), help="decision sets (directories with meta.json)")
    parser.add_argument("--cuts-from", default="v2-stress",
                        help="reference set (directory name): its v tercile cut points are also applied to the other "
                             "sets and its median of v > 0 is the first-action threshold")
    parser.add_argument("--compare-runs", nargs="*", default=list(COMPARE_RUNS),
                        help="frozen comparison run per --data entry whose val views are checked (none: no check)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--name", default="val")
    args = parser.parse_args()
    if args.compare_runs and len(args.compare_runs) != len(args.data):
        raise ValueError("--compare-runs needs one run per --data entry")
    t0 = time.perf_counter()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    roots = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.data]
    if args.cuts_from not in [r.name for r in roots]:
        raise ValueError(f"--cuts-from {args.cuts_from!r} is not one of the --data sets")
    run = start_run("backup_strata", {"data": args.data, "splits": list(DEV_SPLITS), "cuts_from": args.cuts_from,
                                      "compare_runs": args.compare_runs, "n_boot": args.n_boot, "flat_grid": FLAT_GRID,
                                      "contrast_seed": CONTRAST_SEED}, seed=None, name=args.name)
    arrays, flat = {}, {}
    for root in roots:
        log(f"{root.name}: fitting the flat GBM on train belief and hindsight labels (grid chosen on val)")
        arrays[root.name], flat[root.name] = decision_arrays(root, log)
    ref = arrays[args.cuts_from]
    ref_cuts = tercile_cuts(ref["v"])
    threshold = float(np.median(ref["v"][ref["v"] > TOL]))
    results = {"definitions": DEFINITIONS, "caveats": list(CAVEATS), "reference_set": args.cuts_from,
               "v_median_threshold": threshold, "sets": {}}
    for i, (name, d) in enumerate(arrays.items()):
        res = analyse(d, None if name == args.cuts_from else ref_cuts, threshold, args.n_boot)
        res["flat"] = flat[name]
        if args.compare_runs:
            run_dir = Path(args.compare_runs[i])
            res["reproduction"] = recorded_check(run_dir if run_dir.is_absolute() else ROOT / run_dir, res, flat[name])
            log(f"{name}: recorded val views reproduced: {res['reproduction']}")
        all_ = res["views"]["all"]
        log(f"{name}: val regret (belief scoring) GBM belief {all_['regret']['belief_scoring']['gbm_belief']:.4f}, "
            f"GBM hindsight {all_['regret']['belief_scoring']['gbm_hindsight']:.4f}; E1 "
            f"{fmt(all_['E1']['belief_scoring'])}; v > 0 in {int((d['v'] > TOL).sum())} of {d['n']} decisions")
        if "no_backup_minus_backup" in res:
            log(f"{name}: no backup minus backup {fmt(res['no_backup_minus_backup']['belief_scoring'])} (belief), "
                f"{fmt(res['no_backup_minus_backup']['hindsight_scoring'])} (hindsight)")
        results["sets"][name] = res
    full = jsonable({**results, "log": lines, "wall_s": time.perf_counter() - t0})
    out = rounded(full)
    run.save_json("results.json", out)
    (run.dir / "summary.md").write_text(summary_markdown(full), encoding="utf-8")
    print(f"wrote {run.dir}", flush=True)
    metrics = {}
    for name, s in out["sets"].items():
        metrics[name] = {"E1_all": s["views"]["all"]["E1"]["belief_scoring"]["point"],
                         "v_top_minus_zero": s["v_bins"]["top_minus_zero"]["belief_scoring"]["point"]}
        if "no_backup_minus_backup" in s:
            metrics[name]["no_backup_minus_backup"] = s["no_backup_minus_backup"]["belief_scoring"]["point"]
    run.finish(metrics)

if __name__ == "__main__":
    main()
