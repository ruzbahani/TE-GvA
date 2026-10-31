import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpsdefense.agents.flat_baseline import FlatBaseline, evidence_summary, feature_matrix
from cpsdefense.eval.metrics import cluster_bootstrap_ci, paired_cluster_bootstrap_diff
from cpsdefense.experience.belief import LINK, MODULE, pair_prior
from cpsdefense.experience.dataset import KINDS
from cpsdefense.model.direct import pad_candidates
from cpsdefense.paths import ROOT
from cpsdefense.utils.config import load_config
from cpsdefense.utils.runs import start_run
from train_direct import context, load_split, rows_for_choice, subsets, take

TEMPERATURES = (0.0, 0.5, 1.0, 2.0, np.inf)
SHIFTS = (-1.0, 0.0, 1.0)
SETTLED = 1e-12


def logit(p):
    return np.log(p) - np.log1p(-p)


def family_cause(split):
    dec = split["dec"]
    kind = dec["kind"]
    link, module = KINDS.index("link_fault"), KINDS.index("io_module_fault")
    fam = dec["family"].astype(bool)
    by_episode = {}
    for ep, k in zip(dec["episode_seed"][fam], kind[fam]):
        if k in (link, module):
            if by_episode.setdefault(int(ep), int(k)) != int(k):
                raise ValueError(f"episode {ep}: family decisions of both kinds")
    out = np.zeros(split["n"], np.int64)
    for i in np.flatnonzero(fam):
        k = by_episode.get(int(dec["episode_seed"][i]))
        if k is None:
            raise ValueError(f"decision {i}: family episode without a post-onset decision")
        out[i] = 1 if k == link else -1
    return out


def perturbed_posterior(p_true, cause, prior_link, T, delta):
    p_true = np.asarray(p_true, np.float64)
    fam = cause != 0
    settled = ~fam | (1.0 - p_true < SETTLED)
    out = np.ones_like(p_true)
    sel = fam & ~settled
    p_link = np.where(cause[sel] > 0, p_true[sel], 1.0 - p_true[sel])
    L0 = logit(prior_link)
    lam = logit(p_link) - L0
    if np.isinf(T):
        side = np.where(lam != 0, np.sign(lam), np.sign(L0 + delta))
        pl = np.where(side > 0, 1.0, np.where(side < 0, 0.0, 0.5))
    else:
        pl = 1.0 / (1.0 + np.exp(-(L0 + delta + T * lam)))
    out[sel] = np.where(cause[sel] > 0, pl, 1.0 - pl)
    return out


def twin_labels(split):
    dec_of = split["cand"]["decision"]
    p = split["dec"]["posterior"][dec_of].astype(np.float64)
    out = []
    for h, b in (("J", "J_belief"), ("p_trip", "p_trip_belief")):
        Jh, Jb = split["cand"][h].astype(np.float64), split["cand"][b].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(1.0 - p >= SETTLED, (Jb - p * Jh) / (1.0 - p), np.nan))
    return out


def perturbed_labels(split, cause, prior_link, T, delta):
    dec_of = split["cand"]["decision"]
    q = perturbed_posterior(split["dec"]["posterior"], cause, prior_link, T, delta)[dec_of]
    Jt, pt = twin_labels(split)
    J, pr = split["cand"]["J"].astype(np.float64), split["cand"]["p_trip"].astype(np.float64)
    mix = q < 1.0
    return (np.where(mix, q * J + (1 - q) * np.nan_to_num(Jt), J),
            np.where(mix, q * pr + (1 - q) * np.nan_to_num(pt), pr))


def m_mean(x):
    return float(np.mean(x)) if x.size else np.nan


def score_views(split, vocab, J_pred, material_gap, n_boot):
    out = {}
    for labels in ("belief", "hindsight"):
        for sub, sel in subsets(split):
            ctx = context(take(split, sel), vocab, material_gap, labels)
            choice = np.where(ctx["mask"], J_pred[sel], np.inf).argmin(1)
            out[f"{labels}/{sub}"] = (rows_for_choice(ctx, choice)["regret"], ctx["cluster"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--incidents", required=True, help="incidents config the set was generated with")
    parser.add_argument("--params-from", required=True, help="frozen comparison run (results.json flat.chosen)")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--material-gap", type=float, default=0.5)
    parser.add_argument("--name", default="sens")
    args = parser.parse_args()
    t0 = time.perf_counter()
    root = ROOT / args.data
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    incidents = load_config(args.incidents)
    priors = {pair_prior(incidents, g)[LINK] for g in vocab["groups"]
              if pair_prior(incidents, g)[MODULE] > 0 and pair_prior(incidents, g)[LINK] > 0}
    if len({round(p, 12) for p in priors}) != 1:
        raise ValueError(f"the pair prior differs between groups: {sorted(priors)}")
    prior_link = priors.pop()
    flat = json.loads((ROOT / args.params_from / "results.json").read_text(encoding="utf-8"))["flat"]
    params = {k: flat[k]["chosen"] for k in ("belief", "hindsight")}
    eval_splits = ["val"] + (["test"] if args.eval_test else [])
    data = {s: load_split(root, s) for s in ["train", *eval_splits]}
    cause = {s: family_cause(d) for s, d in data.items()}
    X = {s: feature_matrix(evidence_summary(d["batch"], vocab), d["cand"]["features"], d["cand"]["decision"],
                           vocab)[0] for s, d in data.items()}
    run = start_run("belief_sensitivity", {"data": args.data, "incidents": args.incidents, "prior_link": prior_link,
                                           "params": params, "temperatures": [str(t) for t in TEMPERATURES],
                                           "shifts": list(SHIFTS), "eval_splits": eval_splits,
                                           "n_boot": args.n_boot, "params_from": args.params_from},
                    seed=None, name=args.name)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    check = max(float(np.max(np.abs(perturbed_labels(d, cause[s], prior_link, 1.0, 0.0)[0] - d["cand"]["J_belief"])))
                for s, d in data.items())
    log(f"prior P(link) = {prior_link:.4f}; reproduction of the stored belief J at T=1, delta=0: max abs diff "
        f"{check:.2e}")
    if check > 1e-6:
        raise AssertionError("the perturbation does not reproduce the stored belief labels")

    def fit_predict(labels_train, prm):
        fb = FlatBaseline(**prm).fit(X["train"], labels_train[0], labels_train[1], data["train"]["cand"]["decision"],
                                     trip=False)
        return {s: pad_candidates(data[s]["cand"]["decision"], data[s]["n"], {"J": fb.predict(X[s])[0]})[1]["J"]
                for s in eval_splits}

    tr = data["train"]["cand"]
    refs = {"hindsight": fit_predict((tr["J"], tr["p_trip"]), params["hindsight"]),
            "belief": fit_predict((tr["J_belief"], tr["p_trip_belief"]), params["belief"])}
    ref_rows = {k: {s: score_views(data[s], vocab, v[s], args.material_gap, args.n_boot) for s in eval_splits}
                for k, v in refs.items()}
    results = {"prior_link": prior_link, "settled_share": {}, "configs": []}
    for s, d in data.items():
        fam = cause[s] != 0
        results["settled_share"][s] = float(np.mean(1.0 - d["dec"]["posterior"][fam] < SETTLED))
    for T in TEMPERATURES:
        for delta in SHIFTS:
            lab = perturbed_labels(data["train"], cause["train"], prior_link, T, delta)
            pred = fit_predict(lab, params["belief"])
            entry = {"T": "inf" if np.isinf(T) else T, "delta": delta, "splits": {}}
            for s in eval_splits:
                q = perturbed_posterior(data[s]["dec"]["posterior"], cause[s], prior_link, T, delta)
                fam = cause[s] != 0
                Jq = perturbed_labels(data[s], cause[s], prior_link, T, delta)[0]
                label_shift = float(np.mean(np.abs(Jq - data[s]["cand"]["J_belief"])[fam[data[s]["cand"]["decision"]]]))
                rows = score_views(data[s], vocab, pred[s], args.material_gap, args.n_boot)
                views = {}
                for view, (regret, cl) in rows.items():
                    views[view] = {
                        "regret": cluster_bootstrap_ci(m_mean, [regret], cl, n_boot=args.n_boot),
                        "minus_gbm_hindsight": paired_cluster_bootstrap_diff(
                            m_mean, [regret], [ref_rows["hindsight"][s][view][0]], cl, n_boot=args.n_boot),
                        "minus_gbm_belief": paired_cluster_bootstrap_diff(
                            m_mean, [regret], [ref_rows["belief"][s][view][0]], cl, n_boot=args.n_boot)}
                entry["splits"][s] = {"mean_abs_posterior_shift_family": float(np.mean(np.abs(
                    q - data[s]["dec"]["posterior"])[fam])), "mean_abs_label_shift_family": label_shift,
                    "views": views}
                v = views["belief/all"]
                h = views["hindsight/all"]
                log(f"{s} T={entry['T']} delta={delta:+.0f}: belief/all regret {v['regret']['point']:.3f}, minus GBM "
                    f"(hindsight) {v['minus_gbm_hindsight']['point']:+.3f} [{v['minus_gbm_hindsight']['lo']:+.3f}, "
                    f"{v['minus_gbm_hindsight']['hi']:+.3f}]; hindsight/all minus GBM (hindsight) "
                    f"{h['minus_gbm_hindsight']['point']:+.3f} [{h['minus_gbm_hindsight']['lo']:+.3f}, "
                    f"{h['minus_gbm_hindsight']['hi']:+.3f}]")
            results["configs"].append(entry)
    results["references"] = {k: {s: {view: cluster_bootstrap_ci(m_mean, [r], cl, n_boot=args.n_boot)
                                     for view, (r, cl) in v[s].items()} for s in eval_splits}
                             for k, v in ref_rows.items()}
    run.save_json("results.json", {**results, "log": lines, "wall_s": time.perf_counter() - t0,
                                   "eval_test": args.eval_test})
    run.finish({s: {f"T={c['T']},delta={c['delta']}": c["splits"][s]["views"]["belief/all"]["regret"]["point"]
                    for c in results["configs"]} for s in eval_splits})

if __name__ == "__main__":
    main()
