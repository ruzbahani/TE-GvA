import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpsdefense.experience.dataset import belief_labels, candidate_labels, in_family
from cpsdefense.model.direct import derived_voi
from cpsdefense.paths import ROOT
from cpsdefense.utils.config import load_config
from cpsdefense.utils.runs import start_run
from dataset_stats import DEV_SPLITS, TOL, by_group, jsonable, load_episodes, share, summary
from train_direct import (
    FLAT_GRID, context, flat_predictions, label_view, load_split, paired, rows_for_predictions, score, subsets, take)

SETS = {"natural": "data/processed/tep_decisions/v2-natural", "stress": "data/processed/tep_decisions/v2-stress"}
LUMP = 100.0
LUMPS = (25, 50, 100, 200, 400)
E1_LUMPS = (100, 50, 200, 400)
SHORT_S = 3600.0
SCORING = ("belief", "hindsight")
DEFINITIONS = {
    "development": "train and val pooled; the test split is never read",
    "labels_under_L": "J_L = J + (L - 100) p_trip and J_belief_L = J_belief + (L - 100) p_trip_belief per candidate "
                      "(exact: the lump is charged once per tripped replicate and nothing accrues after a trip); "
                      "VOI recomputed from J_L",
    "ambiguous": "decisions of the link-vs-module family (dec_family); data.ambiguous_with_belief_fallback counts "
                 "those whose belief labels fell back to hindsight labels",
    "both": "decisions with at least one observe and one non-observe candidate",
    "gathering_optimal": "among decisions with both: min J over the observe candidates < min J over the "
                         "non-observe candidates - TOL, under the named label set (scripts/dataset_stats.py, "
                         "TOL = 1e-9)",
    "hindsight_best": "candidates with hindsight J <= min hindsight J + TOL",
    "hindsight_best_belief_regret": "min J_belief over the hindsight-best candidates minus min J_belief "
                                    "(dataset_stats hindsight_choice_belief_regret); positive: > TOL",
    "trip_part": "regret: L * (p_trip_belief of the hindsight-best candidate with the lowest J_belief - "
                 "p_trip_belief of the first argmin of J_belief), the rest is non-trip cost; mean J: L * mean "
                 "p_trip over all candidates",
    "belief_best_changed": "ambiguous decisions whose set {J_belief <= min J_belief + TOL} differs from that at "
                           "L = 100",
    "gather_flipped": "ambiguous decisions with both whose belief gathering-optimal answer differs from L = 100",
    "hindsight_best_changed": "all decisions whose hindsight-best set differs from that at L = 100",
    "high_stakes": "max - min p_trip over the candidates > TOL (dataset_stats; train_direct's max > min gives "
                   "the same decisions, p_trip being a share of replicates)",
    "high_stakes_1h": "the same with p_trip counting only the primary replicates that tripped within 1 h of tau "
                      "(y_trip_event, y_trip_time)",
    "trips_within_1h_share": "primary replicates (train and val) that tripped within 1 h of tau among those that "
                             "tripped within their label horizon (3 h, shorter where the episode ended first)",
    "e1": "regret (validation) of the flat GBM fitted on belief labels minus that fitted on hindsight labels, "
          "both refitted on the L labels of their label set; scored with the L labels of the scoring label set "
          "(view <scoring>/<subset>, subsets as train_direct.subsets); 95% paired cluster bootstrap over "
          "episodes, 2,000 resamples, seed 0",
    "e1_reselected": "each GBM's FLAT_GRID point chosen on val regret under its training labels at that L",
    "e1_fixed": "each GBM's grid point fixed to its L = 100 choice",
    "relative": "diff regret point / regret point of the hindsight-trained GBM",
}


def dev_split(root, split):
    assert split in DEV_SPLITS, f"only the development splits {DEV_SPLITS} are read, not {split!r}"
    return load_split(root, split)


def lump_view(split, L):
    k = float(L) - LUMP
    pad, cand = dict(split["pad"]), dict(split["cand"])
    for J, p in (("J", "p_trip"), ("J_belief", "p_trip_belief")):
        pad[J] = split["pad"][J] + k * split["pad"][p]
        cand[J] = split["cand"][J] + k * split["cand"][p]
    mask, observe = torch.as_tensor(split["mask"]), torch.as_tensor(split["pad"]["observe"].astype(bool))
    for voi, J in (("voi", "J"), ("voi_belief", "J_belief")):
        pad[voi] = derived_voi(torch.as_tensor(pad[J]), observe, mask).numpy()
    return {**split, "pad": pad, "cand": cand}


def count(mask, sel):
    mask, sel = np.asarray(mask, bool), np.asarray(sel, bool)
    return {"k": int(mask[sel].sum()), "n": int(sel.sum()), "share": share(mask[sel])}


def high_stakes(p, mask):
    return np.where(mask, p, -np.inf).max(1) > np.where(mask, p, np.inf).min(1) + TOL


def decision_table(view, L):
    mask, pad = view["mask"], view["pad"]
    observe = mask & pad["observe"].astype(bool)
    act = mask & ~observe
    both = observe.any(1) & act.any(1)
    out = {"both": both}
    for labels, key in (("hindsight", "J"), ("belief", "J_belief")):
        J = np.where(mask, pad[key], np.inf)
        out[f"gather_{labels}"] = both & (np.where(observe, J, np.inf).min(1) < np.where(act, J, np.inf).min(1) - TOL)
        out[f"best_{labels}"] = J <= J.min(1, keepdims=True) + TOL
    Jb, pb, r = np.where(mask, pad["J_belief"], np.inf), pad["p_trip_belief"], np.arange(view["n"])
    pick = np.where(out["best_hindsight"], Jb, np.inf).argmin(1)
    top = Jb.argmin(1)
    out["regret"] = Jb[r, pick] - Jb[r, top]
    out["regret_trip"] = L * (pb[r, pick] - pb[r, top])
    return out


def label_stats(splits, L):
    parts, J, p = [], [], []
    for s in splits:
        view = lump_view(s, L)
        t, ref = decision_table(view, L), decision_table(lump_view(s, LUMP), LUMP)
        parts.append({**{k: v for k, v in t.items() if not k.startswith("best_")},
                      "belief_best_changed": (t["best_belief"] != ref["best_belief"]).any(1),
                      "hindsight_best_changed": (t["best_hindsight"] != ref["best_hindsight"]).any(1),
                      "gather_flipped": t["gather_belief"] != ref["gather_belief"],
                      "ambiguous": s["dec"]["family"].astype(bool),
                      "backup": np.where(s["dec"]["backup_available"].astype(bool), "available", "unavailable")})
        J.append(view["cand"]["J"])
        p.append(view["cand"]["p_trip"])
    d = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    J, p = np.concatenate(J), np.concatenate(p)
    amb, both = d["ambiguous"], d["both"]
    gb = both & amb
    regret = d["regret"][amb]
    return {
        "gather_belief_ambiguous": count(d["gather_belief"], gb),
        "gather_belief_ambiguous_by_backup": by_group(d["backup"][gb], d["gather_belief"][gb]),
        "gather_hindsight_all": count(d["gather_hindsight"], both),
        "gather_hindsight_ambiguous": count(d["gather_hindsight"], gb),
        "hindsight_best_belief_regret_ambiguous": {
            **summary(regret), "positive": count(regret > TOL, np.ones(regret.size, bool)),
            "trip_part_mean": float(d["regret_trip"][amb].mean()),
            "other_part_mean": float((regret - d["regret_trip"][amb]).mean()),
            "by_backup": by_group(d["backup"][amb], np.ones(regret.size, bool), regret)},
        "belief_best_changed_ambiguous": count(d["belief_best_changed"], amb),
        "gather_flipped_ambiguous": count(d["gather_flipped"], gb),
        "hindsight_best_changed_all": count(d["hindsight_best_changed"], np.ones(both.size, bool)),
        "mean_J": float(J.mean()), "mean_J_trip_part": float(L * p.mean()),
    }


def horizon_stats(splits):
    replicates = trips = early = 0
    times, hs, hs_short = [], [], []
    for s in splits:
        mask, pad = s["mask"], s["pad"]
        valid = mask[..., None] & np.isfinite(pad["trip_event"])
        trip = valid & (np.nan_to_num(pad["trip_event"]) > 0.5)
        within = trip & (np.nan_to_num(pad["trip_time"], nan=np.inf) <= SHORT_S)
        n = np.maximum(valid.sum(-1), 1)
        if np.abs(trip.sum(-1) / n - pad["p_trip"])[mask].max() > 1e-12:
            raise ValueError(f"{s.get('name')}: the replicate trip events do not reproduce p_trip")
        hs.append(high_stakes(pad["p_trip"], mask))
        hs_short.append(high_stakes(within.sum(-1) / n, mask))
        times.append(pad["trip_time"][trip])
        replicates, trips, early = replicates + int(valid.sum()), trips + int(trip.sum()), early + int(within.sum())
    hs, hs_short = np.concatenate(hs), np.concatenate(hs_short)
    every = np.ones(hs.size, bool)
    return {"decisions": int(hs.size), "replicates": replicates, "trips": trips, "trips_within_1h": early,
            "trips_within_1h_share": early / trips if trips else None,
            "trip_time_min": summary(np.concatenate(times), 1 / 60),
            "high_stakes": count(hs, every), "high_stakes_1h": count(hs_short, every),
            "high_stakes_still_at_1h": count(hs_short, hs), "high_stakes_only_at_1h": int((hs_short & ~hs).sum())}


def interval(d):
    return {k: d[k] for k in ("point", "lo", "hi", "p_two_sided") if k in d}


def fit_val(dl, vocab, labels, grid, log):
    flat, table, chosen = flat_predictions({s: label_view(d, labels) for s, d in dl.items()}, vocab, log, grid)
    return flat["flat_gbm"]["val"][0], table, chosen


def e1_views(val, vocab, preds, n_boot, material_gap):
    out = {}
    for scoring in SCORING:
        for sub, sel in subsets(val):
            ctx = context(take(val, sel), vocab, material_gap, scoring)
            rows = {k: {"regret": rows_for_predictions(ctx, J[:, sel])["regret"]} for k, J in preds.items()}
            sc = {k: score(ctx, r, n_boot) for k, r in rows.items()}
            diff = paired(ctx, rows["belief"], rows["hindsight"], n_boot)
            ref = sc["hindsight"]["regret"]["point"]
            out[f"{scoring}/{sub}"] = {
                "decisions": int(ctx["n"]), "episodes": int(np.unique(ctx["cluster"]).size),
                "high_stakes": int(ctx["stakes"].sum()),
                **{f"{k}_trained": {m: interval(v) for m, v in sc[k].items()} for k in preds},
                "diff": {m: interval(v) for m, v in diff.items()},
                "relative": diff["regret"]["point"] / ref if ref else None}
    return out


def e1(data, vocab, n_boot, material_gap, log):
    assert E1_LUMPS[0] == LUMP, "the fixed variant needs the L = 100 choices first"
    out, fits, fixed = {"reselected": {}, "fixed": {}}, {}, None
    for L in E1_LUMPS:
        dl = {s: lump_view(d, L) for s, d in data.items()}
        for variant in ("reselected", "fixed"):
            entry, preds = {"chosen": {}}, {}
            for labels in SCORING:
                if variant == "reselected":
                    J, table, chosen = fit_val(dl, vocab, labels, FLAT_GRID, log)
                    entry.setdefault("grid", {})[labels] = table
                    fits[(L, labels, tuple(sorted(chosen.items())))] = J
                else:
                    chosen = fixed[labels]
                    J = fits.get((L, labels, tuple(sorted(chosen.items()))))
                    if J is None:
                        J = fit_val(dl, vocab, labels, [chosen], log)[0]
                preds[labels], entry["chosen"][labels] = J, chosen
            if fixed is None:
                fixed = dict(entry["chosen"])
            entry["views"] = e1_views(dl["val"], vocab, preds, n_boot, material_gap)
            out[variant][L] = entry
            v = entry["views"]["belief/all"]
            d = v["diff"]["regret"]
            log(f"  e1 L={L} {variant}: belief/all belief-trained {v['belief_trained']['regret']['point']:.3f}, "
                f"hindsight-trained {v['hindsight_trained']['regret']['point']:.3f}, diff {d['point']:+.3f} "
                f"[{d['lo']:+.3f}, {d['hi']:+.3f}] p={d['p_two_sided']:.3f}")
    return out


def shifted_decision(decision, L):
    def shift(results):
        return [{**r, "total": r["total"] + (L - LUMP) * float(bool(r["shutdown"]))} for r in results]
    out = {**decision, "results": shift(decision["results"])}
    twin = decision.get("twin")
    if twin is not None and twin.get("results") is not None:
        out["twin"] = {**twin, "results": shift(twin["results"])}
    return out


def new_counts():
    return {"replicates": 0, "trips": 0, "trips_within_1h": 0, "total_not_sum_of_components": 0,
            "lump_without_trip_or_trip_without_lump": 0, "near_limit_cost_exceeds_time": 0,
            "service_cost_exceeds_time": 0, "trip_time_inconsistent": 0, "physical_tripped": [np.inf, -np.inf],
            "physical_untripped_max": -np.inf}


def replicate_checks(results, costs, acc):
    near_h = costs["physical"]["near_limit_per_h"]
    service_h = costs["service"]["production_shortfall_per_h"] + costs["service"]["degraded_per_h"]
    for r in results:
        c, trip, tts = r["cost"], bool(r["shutdown"]), r["time_to_shutdown"]
        hours = (tts if trip and tts is not None else r["horizon_s"]) / 3600.0
        rest = c["physical"] - LUMP * trip
        acc["replicates"] += 1
        acc["trips"] += int(trip)
        acc["trips_within_1h"] += int(trip and tts is not None and tts <= SHORT_S)
        acc["total_not_sum_of_components"] += int(abs(r["total"] - sum(c.values())) > TOL)
        acc["lump_without_trip_or_trip_without_lump"] += int((c["physical"] >= LUMP) != trip)
        acc["near_limit_cost_exceeds_time"] += int(not -TOL <= rest <= near_h * hours + TOL)
        acc["service_cost_exceeds_time"] += int(c["service"] > service_h * hours + TOL)
        acc["trip_time_inconsistent"] += int((tts is not None) != trip or (trip and tts > r["horizon_s"] + 1e-6))
        if trip:
            lo, hi = acc["physical_tripped"]
            acc["physical_tripped"] = [min(lo, c["physical"]), max(hi, c["physical"])]
        else:
            acc["physical_untripped_max"] = max(acc["physical_untripped_max"], c["physical"])


def raw_check(root, data, fractions, log):
    costs = load_config("sim/costs.yaml")
    counts = {"primary": new_counts(), "twin": new_counts()}
    stored = dict.fromkeys(("J", "p_trip", "J_belief", "p_trip_belief"), 0.0)
    identity = {L: {"J": 0.0, "J_belief": 0.0} for L in LUMPS}
    episodes = load_episodes(root, DEV_SPLITS, fractions)
    n = 0
    for name, eps in episodes.items():
        split = data[name]
        decs = [(e, d) for e in sorted(eps, key=lambda e: e["seed"]) for d in e["decisions"]]
        if not (np.array_equal([e["seed"] for e, _ in decs], split["dec"]["episode_seed"])
                and np.array_equal([d["tau"] for _, d in decs], split["dec"]["tau"])):
            raise ValueError(f"{name}: the arrays' rows do not follow the raw episodes' decisions")
        cand = split["cand"]
        start = np.r_[0, np.cumsum(np.bincount(cand["decision"], minlength=split["n"]))]
        for i, (e, d) in enumerate(decs):
            rows, family = slice(start[i], start[i + 1]), in_family(e)
            replicate_checks(d["results"], costs, counts["primary"])
            if (d.get("twin") or {}).get("results") is not None:
                replicate_checks(d["twin"]["results"], costs, counts["twin"])
            for L in LUMPS:
                dL = shifted_decision(d, L)
                lab, bl = candidate_labels(dL), belief_labels(dL, family)
                got = {"J": lab["J"], "p_trip": lab["p_trip"], "J_belief": bl["J_belief"],
                       "p_trip_belief": bl["p_trip_belief"]}
                for J, p in (("J", "p_trip"), ("J_belief", "p_trip_belief")):
                    want = cand[J][rows] + (L - LUMP) * cand[p][rows]
                    identity[L][J] = max(identity[L][J], float(np.abs(got[J] - want).max()))
                if L == LUMP:
                    for k in stored:
                        stored[k] = max(stored[k], float(np.abs(got[k] - cand[k][rows]).max()))
        n += len(decs)
        log(f"  raw {name}: {len(eps)} episodes, {len(decs)} decisions")
    worst = max(v for e in identity.values() for v in e.values())
    tallies = {w: {k: v for k, v in c.items() if isinstance(v, int)} for w, c in counts.items()}
    log(f"  raw: {n} decisions; labels vs arrays max abs error {max(stored.values()):.2e}; J_L vs formula "
        f"{worst:.2e}; {tallies}")
    return {"lump": LUMP, "lump_in_config": costs["physical"]["shutdown"], "decisions": n, "replicates": counts,
            "labels_vs_arrays_max_abs_error": stored, "J_L_vs_formula_max_abs_error": identity,
            "J_L_vs_formula_max_abs_error_overall": worst}


def rounded(obj, digits=4):
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, digits) for v in obj]
    if isinstance(obj, float):
        return round(obj, digits) if obj == 0 or abs(obj) >= 10 ** -digits else float(f"{obj:.3g}")
    return obj


def percent(c):
    return "-" if c is None or not c["n"] else f"{100 * c['k'] / c['n']:.2f}%"


def pct(c):
    return "-" if c is None or not c["n"] else f"{percent(c)} ({c['k']}/{c['n']})"


def diff_cell(d):
    cell = f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] p={d['p_two_sided']:.3f}"
    return f"**{cell}**" if d["lo"] > 0 or d["hi"] < 0 else cell


def summary_markdown(results, name):
    lines = [f"# Trip penalty sensitivity ({name}): train and val only", "",
             "Labels under trip lump L: J_L = J + (L - 100) p_trip (hindsight and belief). Definitions in "
             "results.json. E1 numbers are validation regrets of models selected on validation (optimistic).", ""]
    for set_name, r in results["sets"].items():
        info = r["data"]
        lines += [f"## {set_name}", "",
                  f"{info['decisions']} development decisions, {info['ambiguous']} ambiguous "
                  f"({info['ambiguous_with_belief_fallback']} with belief fallback).", "",
                  "| L | gather, belief, ambiguous | by backup (available / unavailable) | gather, hindsight, all | "
                  "belief regret of hindsight-best (mean; trip part; > 0; q95) | belief-best set changed | "
                  "gather/act flipped |", "|---" * 7 + "|"]
        for L, s in r["labels"].items():
            g = s["gather_belief_ambiguous_by_backup"]
            reg = s["hindsight_best_belief_regret_ambiguous"]
            backup = " / ".join(f"{100 * g[k]['share']:.2f}%" if k in g else "-" for k in ("available", "unavailable"))
            lines.append(f"| {L} | {pct(s['gather_belief_ambiguous'])} | {backup} | {pct(s['gather_hindsight_all'])} | "
                         f"{reg['mean']:.3f}; {reg['trip_part_mean']:.3f}; {percent(reg['positive'])}; "
                         f"{reg['q95']:.2f} | {pct(s['belief_best_changed_ambiguous'])} | "
                         f"{pct(s['gather_flipped_ambiguous'])} |")
        s = r["labels"][str(int(LUMP))]
        lines += ["", f"At L = 100 the trip lump is {s['mean_J_trip_part']:.3f} of the mean J {s['mean_J']:.3f}.", ""]
        h = r["horizon"]
        t = h["trip_time_min"]
        lines += ["Horizon (primary replicates):", "",
                  f"- trips within 1 h: {pct({'k': h['trips_within_1h'], 'n': h['trips']})}; "
                  f"time to trip median {t['median']:.2f} min (q25 {t['q25']:.2f}, q75 {t['q75']:.2f})",
                  f"- high-stakes decisions: {pct(h['high_stakes'])} at 3 h, {pct(h['high_stakes_1h'])} counting only "
                  f"trips within 1 h; {pct(h['high_stakes_still_at_1h'])} of the 3 h ones stay high-stakes", ""]
        if "e1" in r:
            views = [v for v in r["e1"]["reselected"][str(int(LUMP))]["views"] if v.startswith("belief/")]
            for variant, entries in r["e1"].items():
                lines += [f"E1 on validation, {variant} (belief-trained minus hindsight-trained regret, belief scoring):",
                          "", "| L | chosen (belief; hindsight) | " + " | ".join(views) + " |",
                          "|---" * (len(views) + 2) + "|"]
                for L, e in entries.items():
                    chosen = "; ".join(f"{'c' if p['center'] else 'u'}/{p['learning_rate']}/{p['max_leaf_nodes']}"
                                       for p in e["chosen"].values())
                    lines.append(f"| {L} | {chosen} | "
                                 + " | ".join(diff_cell(e["views"][v]["diff"]["regret"]) for v in views) + " |")
                lines.append("")
        if "raw" in r:
            raw = r["raw"]
            lines += ["Raw episodes (train and val):", "",
                      f"- {raw['decisions']} decisions; stored labels reproduced to "
                      f"{max(raw['labels_vs_arrays_max_abs_error'].values()):.2e}; J_L from shifted replicate totals "
                      f"vs the formula on the arrays: max abs error {raw['J_L_vs_formula_max_abs_error_overall']:.2e}"]
            for world, c in raw["replicates"].items():
                tally = ("replicates", "trips", "trips_within_1h")
                bad = {k: v for k, v in c.items() if isinstance(v, int) and k not in tally}
                lines.append(f"- {world}: {c['replicates']} replicates, {c['trips']} tripped ({c['trips_within_1h']} "
                             f"within 1 h); physical cost of tripped {c['physical_tripped'][0]:.2f}-"
                             f"{c['physical_tripped'][1]:.2f}, untripped max {c['physical_untripped_max']:.2f}; "
                             f"violations {bad}")
            lines.append("")
    return "\n".join(lines)


def metrics(results):
    out = {}
    for set_name, r in results["sets"].items():
        out[set_name] = {"gather_belief_ambiguous": {L: s["gather_belief_ambiguous"]["share"]
                                                     for L, s in r["labels"].items()}}
        if "e1" in r:
            out[set_name]["e1_belief_all"] = {variant: {L: e["views"]["belief/all"]["diff"]["regret"]["point"]
                                                        for L, e in entries.items()}
                                              for variant, entries in r["e1"].items()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    parser.add_argument("--check-raw", action="store_true", help="check the identity on the train / val raw episodes")
    parser.add_argument("--no-e1", action="store_true", help="skip the GBM refits (labels and horizon only)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--material-gap", type=float, default=0.5)
    parser.add_argument("--name", default="dev")
    args = parser.parse_args()
    t0 = time.perf_counter()
    run = start_run("trip_penalty", {"sets": {s: SETS[s] for s in args.sets}, "splits": list(DEV_SPLITS),
                                     "lump": LUMP, "lumps": list(LUMPS), "e1_lumps": list(E1_LUMPS), "tol": TOL,
                                     "short_horizon_s": SHORT_S, "check_raw": args.check_raw, "e1": not args.no_e1,
                                     "n_boot": args.n_boot, "bootstrap_seed": 0, "material_gap": args.material_gap,
                                     "flat_grid": FLAT_GRID},
                    seed=0, name=args.name)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    results = {"definitions": DEFINITIONS, "sets": {}}
    for name in args.sets:
        root = ROOT / SETS[name]
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        data = {s: dev_split(root, s) for s in DEV_SPLITS}
        splits = list(data.values())
        fam = np.concatenate([d["dec"]["family"] for d in splits]).astype(bool)
        fallback = np.concatenate([d["dec"]["belief_fallback"] for d in splits]).astype(bool)
        out = {"data": {"arrays": SETS[name], "decisions": int(fam.size), "ambiguous": int(fam.sum()),
                        "ambiguous_with_belief_fallback": int((fam & fallback).sum()),
                        "candidates": int(sum(d["cand"]["decision"].size for d in splits))}}
        log(f"{name}: {out['data']}")
        if args.check_raw:
            out["raw"] = raw_check(ROOT / meta["root"].replace("\\", "/"), data, tuple(meta["fractions"]), log)
        out["labels"] = {L: label_stats(splits, L) for L in LUMPS}
        for L, s in out["labels"].items():
            g, reg = s["gather_belief_ambiguous"], s["hindsight_best_belief_regret_ambiguous"]
            log(f"  L={L}: gather belief ambiguous {100 * g['share']:.2f}% ({g['k']}/{g['n']}), hindsight all "
                f"{100 * s['gather_hindsight_all']['share']:.2f}%; belief regret of hindsight-best {reg['mean']:.3f} "
                f"(trip part {reg['trip_part_mean']:.3f})")
        out["horizon"] = horizon_stats(splits)
        h = out["horizon"]
        log(f"  horizon: {h['trips_within_1h']}/{h['trips']} trips within 1 h; high-stakes "
            f"{h['high_stakes']['k']} -> {h['high_stakes_1h']['k']} ({h['high_stakes_still_at_1h']['k']} stay)")
        if not args.no_e1:
            out["e1"] = e1(data, meta["vocab"], args.n_boot, args.material_gap, log)
        results["sets"][name] = out
        del data, splits
    results = jsonable(results)
    run.save_json("results.json", {**rounded(results), "log": lines, "wall_s": round(time.perf_counter() - t0, 1)})
    (run.dir / "summary.md").write_text(summary_markdown(results, args.name), encoding="utf-8")
    run.finish(rounded(metrics(results)))
    print(f"-> {run.dir}")

if __name__ == "__main__":
    main()
