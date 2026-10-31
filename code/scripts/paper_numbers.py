import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpsdefense.paths import ROOT
from train_direct import context, label_view, load_split, paired, rows_for_predictions, subsets, take

RUNS = ROOT / "runs" / "model_direct"
SETS = {"natural": "data/processed/tep_decisions/v2-natural", "stress": "data/processed/tep_decisions/v2-stress"}
METRICS = ("regret", "regret_family", "regret_within", "cf_labelled_family_regret", "cf_always_act_regret",
           "forced_act_regret", "regret_high_stakes", "trip_rate_high_stakes", "gather_rate", "gva_accuracy",
           "gva_accuracy_material")
DIAG = ("pred_gather_share", "true_gather_share", "gap_spearman", "gap_auc_true_gather", "tail_decisions",
        "tail_family_regret", "tail_gather_share", "act_corr_high_stakes", "centred_slope", "centred_corr",
        "bias_gap", "bias_gap_zero_predictor", "regret_family", "regret_within_family")
EXTRA_PAIRS = (("summary", "base"), ("summary_ensemble", "base"), ("summary_ensemble", "base_ensemble"),
               ("summary_items", "summary"), ("summary_items", "base"), ("summary_aux", "base"),
               ("summary_ensemble", "summary"))


def one(pattern):
    hits = sorted(RUNS.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {len(hits)} matches")
    return hits[0]


def ci(d, digits=4):
    return {k: (round(float(d[k]), digits) if d.get(k) is not None and np.isfinite(d[k]) else None)
            for k in ("point", "lo", "hi", "p_two_sided") if k in d}


def main_runs(name, kind):
    return one(f"*-test-{name}-{kind}")


def extract_compare(run):
    res = json.loads((run / "results.json").read_text(encoding="utf-8"))
    out = {}
    for split, views in res["results"].items():
        out[split] = {}
        for view, rep in views.items():
            out[split][view] = {
                "decisions": rep["decisions"], "episodes": rep["episodes"], "high_stakes": rep["high_stakes"],
                "has_both": rep["has_both"], "material": rep["material"], "reference": rep["reference"],
                "scores": {m: {k: ci(v[k]) for k in METRICS if k in v} for m, v in rep["scores"].items()},
                "seed_std": {m: {k: v[k].get("seed_std_ddof1") for k in ("regret",) if "seed_std_ddof1" in v.get(k, {})}
                             for m, v in rep["scores"].items()},
                "paired": {m: {k: ci(v[k]) for k in METRICS if k in v} for m, v in rep["paired"].items()},
                "diagnostics": {m: {k: round(float(v[k]), 4) for k in DIAG if k in v}
                                for m, v in rep["gva_diagnostics"].items()}}
    flat = {k: v["chosen"] for k, v in res.get("flat", {}).items()}
    return out, flat


def extra_pairs(set_name, compare_run):
    cfg = json.loads((compare_run / "run.json").read_text(encoding="utf-8"))["config"]
    runs = cfg["compare"]
    meta = json.loads((ROOT / SETS[set_name] / "meta.json").read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    out = {}
    for split in ("val", "test"):
        data = load_split(ROOT / SETS[set_name], split)
        preds = {}
        for name, r in runs.items():
            p = np.load(ROOT / r / f"predictions_{split}.npz")
            if not np.array_equal(p["episode_seed"], data["dec"]["episode_seed"]):
                raise ValueError(f"{r}: predictions do not match the {split} decisions")
            preds[name] = p["J_pred"]
            preds[f"{name}_ensemble"] = p["J_pred"].mean(0, keepdims=True)
        out[split] = {}
        for labels in ("belief", "hindsight"):
            for sub, sel in subsets(data):
                if sub not in ("all", "family", "family_backup_unavailable"):
                    continue
                ctx = context(take(label_view(data, "hindsight"), sel), vocab, cfg["eval"]["material_gap"], labels)
                rows = {m: rows_for_predictions(ctx, J[:, sel]) for m, J in preds.items()}
                view = {}
                for a, b in EXTRA_PAIRS:
                    d = paired(ctx, rows[a], rows[b], cfg["eval"]["n_boot"])
                    view[f"{a} - {b}"] = {k: ci(d[k]) for k in ("regret", "regret_family", "gva_accuracy") if k in d}
                out[split][f"{labels}/{sub}"] = view
    return out


def sensitivity(set_name):
    run = sorted((ROOT / "runs" / "belief_sensitivity").glob(f"*-sens-{set_name}-test"))
    if len(run) != 1:
        return None
    res = json.loads((run[0] / "results.json").read_text(encoding="utf-8"))
    out = {"prior_link": res["prior_link"], "settled_share": res["settled_share"], "configs": []}
    for c in res["configs"]:
        entry = {"T": c["T"], "delta": c["delta"], "splits": {}}
        for split, s in c["splits"].items():
            entry["splits"][split] = {
                "mean_abs_posterior_shift_family": round(s["mean_abs_posterior_shift_family"], 4),
                "mean_abs_label_shift_family": round(s["mean_abs_label_shift_family"], 4),
                "views": {v: {k: ci(x) for k, x in d.items()} for v, d in s["views"].items()}}
        out["configs"].append(entry)
    out["references"] = {k: {s: {v: ci(x) for v, x in views.items()} for s, views in per.items()}
                         for k, per in res["references"].items()}
    return out


def learning_curves():
    out = {}
    for set_name, tag in (("natural", "v2n"), ("stress", "v2s")):
        rows = {}
        for frac in ("0.25", "0.5"):
            base = json.loads((one(f"*-{tag}-lc-base-f{frac}") / "results.json").read_text(encoding="utf-8"))
            noit = json.loads((one(f"*-{tag}-lc-noitems-f{frac}") / "results.json").read_text(encoding="utf-8"))
            sb = base["results"]["val"]["belief/all"]["scores"]
            sn = noit["results"]["val"]["belief/all"]["scores"]
            rows[frac] = {"base": sb[f"{tag}-lc-base-f{frac}"]["regret"]["point"],
                          "summary": sn[f"{tag}-lc-noitems-f{frac}"]["regret"]["point"],
                          "summary_ensemble": sn[f"{tag}-lc-noitems-f{frac}_ensemble"]["regret"]["point"],
                          "flat_gbm": sn["flat_gbm"]["regret"]["point"],
                          "train_decisions": None}
            for line in noit["log"]:
                if line.startswith("train_fraction"):
                    rows[frac]["train_episodes_decisions"] = line
        main = json.loads((main_runs(set_name, "main") / "results.json").read_text(encoding="utf-8"))
        sv = main["results"]["val"]["belief/all"]["scores"]
        rows["1.0"] = {"base": sv["base"]["regret"]["point"], "summary": sv["summary"]["regret"]["point"],
                       "summary_ensemble": sv["summary_ensemble"]["regret"]["point"],
                       "flat_gbm": sv["flat_gbm_belief"]["regret"]["point"]}
        out[set_name] = {k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in r.items()}
                         for k, r in rows.items()}
    return out


def counts():
    out = {}
    for set_name, root in SETS.items():
        meta = json.loads((ROOT / root / "meta.json").read_text(encoding="utf-8"))
        c = {"episodes_with_decisions": 0, "decisions": 0, "candidates": 0, "family_decisions": 0,
             "family_candidates": 0, "splits": {}}
        for split in ("train", "val", "test"):
            with np.load(ROOT / root / f"{split}.npz", allow_pickle=False) as a:
                n = len(a["dec_tau"])
                fam = a["dec_family"].astype(bool)
                k = len(a["cand_decision"])
                fam_c = int(fam[a["cand_decision"]].sum())
                ep = len(np.unique(a["dec_episode_seed"]))
            c["splits"][split] = {"episodes": ep, "decisions": n, "candidates": k, "family_decisions": int(fam.sum())}
            c["episodes_with_decisions"] += ep
            c["decisions"] += n
            c["candidates"] += k
            c["family_decisions"] += int(fam.sum())
            c["family_candidates"] += fam_c
        c["hindsight_branches"] = 4 * c["candidates"]
        c["twin_branches"] = 4 * c["family_candidates"]
        c["vocab"] = {"sources": len(meta["vocab"]["sources"]), "kinds": len(meta["vocab"]["kinds"]),
                      "action_columns": len(meta["vocab"]["action_columns"]), "tools": len(meta["vocab"]["tools"])}
        out[set_name] = c
    return out


def main():
    numbers = {"counts": counts(), "learning_curves": learning_curves()}
    for set_name in SETS:
        run = main_runs(set_name, "main")
        numbers[f"{set_name}_main"], numbers[f"{set_name}_flat_params"] = extract_compare(run)
        numbers[f"{set_name}_main_run"] = run.name
        lab = main_runs(set_name, "labels")
        numbers[f"{set_name}_labels"], _ = extract_compare(lab)
        numbers[f"{set_name}_labels_run"] = lab.name
        numbers[f"{set_name}_extra_pairs"] = extra_pairs(set_name, run)
        numbers[f"{set_name}_sensitivity"] = sensitivity(set_name)
    out = ROOT / "paper" / "numbers.json"
    out.write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    for set_name in SETS:
        m = numbers[f"{set_name}_main"]
        for split in ("val", "test"):
            for view in m[split]:
                d = m[split][view]["paired"]["flat_gbm_hindsight"]["regret"]
                print(f"E1 {set_name:7s} {split:4s} {view:36s} GBM(belief) - GBM(hindsight) {d['point']:+.3f} "
                      f"[{d['lo']:+.3f}, {d['hi']:+.3f}] p={d.get('p_two_sided')}")
        for split in ("val", "test"):
            for view, pairs in numbers[f"{set_name}_extra_pairs"][split].items():
                for k, v in pairs.items():
                    r = v["regret"]
                    print(f"E2 {set_name:7s} {split:4s} {view:30s} {k:34s} {r['point']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
        lab = numbers[f"{set_name}_labels"]
        for split in ("val", "test"):
            for view in ("belief/all", "hindsight/all", "belief/family"):
                d = lab[split][view]["paired"]["direct_hindsight"]["regret"]
                print(f"E3 {set_name:7s} {split:4s} {view:16s} direct(belief) - direct(hindsight) {d['point']:+.3f} "
                      f"[{d['lo']:+.3f}, {d['hi']:+.3f}]")

if __name__ == "__main__":
    main()
