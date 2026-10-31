import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from belief_sensitivity import family_cause
from cpsdefense.agents.flat_baseline import (
    GROUP_KEYS, LINK_KEYS, FlatBaseline, evidence_summary, feature_matrix)
from cpsdefense.agents.playbook import Playbook
from cpsdefense.eval.metrics import cluster_bootstrap_ci, paired_cluster_bootstrap_diff
from cpsdefense.model.direct import pad_candidates
from cpsdefense.paths import ROOT
from cpsdefense.utils.runs import start_run
from dataset_stats import DEV_SPLITS, jsonable
from train_direct import (
    context, flat_predictions, fmt, label_view, load_split, m_mean, rows_for_choice, subsets, take)

DATA = ("data/processed/tep_decisions/v2-natural", "data/processed/tep_decisions/v2-stress")
THETAS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01)
MARGINS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0)
VAL_MARGINS = (0.0, 0.25, 0.5, 1.0)
FOLDS = 5
SPLIT_SEEDS = (1, 2, 3, 4, 5)
FOLD_SPLITS = ("group_kfold", *(f"seed{s}" for s in SPLIT_SEEDS))
COSTLY_GAP = 10.0
VIEWS = ("all", "family", "backup_unavailable", "family_backup_unavailable")
REFS = ("gbm_belief", "playbook_investigate_first")
STATUS_TOOLS = (("trace_link", "sensor"), ("switch_to_backup", "sensor"), ("restore_primary", "sensor"),
                ("repair_link", "sensor"), ("replace_io_module", "source"), ("verify_sensor", "source"),
                ("query_historian", "source"))
DEFINITIONS = {
    "splits": "train fits and selects everything (GBM cross-fitting, logistic posterior, theta, margin curves); "
              "val is scored once per rule; the test split is never read",
    "regret": "J of the chosen candidate minus the lowest J among the decision's candidates, under the scoring "
              "labels (belief: J_belief, the posterior mix over the link-vs-module pair; hindsight: J under the "
              "episode's true cause); mean over decisions",
    "views": "<scoring>/<subset>; all: every decision; family: the ambiguous decisions of the paper (dec_family: "
             "the episode's incident is a link fault on sensor:<g> or an I/O-module fault on <g>); "
             "backup_unavailable: scenarios without a backup channel (dec_backup_available false); "
             "family_backup_unavailable: both",
    "intervals": "regret: point, 95% percentile cluster bootstrap over episodes (n_boot replicates, seed 0); "
                 "minus_<ref>: method minus ref, paired cluster bootstrap on shared resamples, p_two_sided = "
                 "2 min(P(diff <= 0), P(diff >= 0)) over the replicates (0 means below 2 / n_boot)",
    "gather_rate": "share of the decisions with both observe and non-observe candidates in which the chosen "
                   "candidate is an observe action",
    "costly_gather_share": "the same share among the decisions with both whose labelled gap (lowest observe J "
                           "minus lowest non-observe J, scoring labels) is at least 10 cost points (costly_n)",
    "gbm": "flat GBM (agents/flat_baseline.py) through train_direct.flat_predictions: fitted on train for each "
           "of the 8 grid points, the lowest val regret under its own training labels kept (as in the frozen "
           "evaluation); gbm_belief / gbm_hindsight: fitted on belief / hindsight labels",
    "playbooks": "the playbooks' recorded first actions (dec_baseline_act_first, dec_baseline_investigate_first)",
    "held_situation": "a sensor group g whose valid stale alerts cover >= max(2, 0.5 x its channels) distinct "
                      "channels (agents/playbook.py group-wide staleness test); with several, the largest share",
    "p_link_exact": "P(link fault | E) from dec_posterior, the posterior of the episode's own cause "
                    "(belief_sensitivity.family_cause gives the cause, also for pre-onset decisions); defined in "
                    "the family only; privileged information (idealized observer)",
    "p_link_lr": "soft-label logistic regression (C=1, standardised features, each row as a link row with "
                 "weight p and a module row with weight 1 - p) on posterior_features at g, fitted on the held "
                 "train decisions with the exact posterior as target; train values cross-fitted over 5 "
                 "episode-grouped folds (GroupKFold over the held decisions), val values from the fit on all "
                 "of train",
    "bt": "BT(theta) in held situations (exact: held and in the family): max(p, 1 - p) < theta and no trace of "
          "sensor:g accepted or running -> trace_link sensor:g; such a trace pending -> monitor(300); "
          "otherwise p >= 0.5 -> the act-first playbook's choice, p < 0.5 -> replace_io_module source:g unless "
          "one is accepted, running or succeeded (then monitor(300)); outside held situations the act-first "
          "playbook (bt_<src>) or the GBM belief argmin (bt_<src>_gbm_fallback, same theta)",
    "gate": "gated GBM: in held situations (exact: held and in the family, as in bt) gather (argmin of the GBM "
            "over observe candidates) if max(p, 1 - p) < theta, else the GBM argmin over non-observe candidates; "
            "elsewhere the GBM argmin; theta chosen on train with GBM predictions cross-fitted over the "
            "group_kfold split",
    "theta_selection": "theta in THETAS with the lowest mean train regret under belief labels (the smallest "
                       "theta on ties); the lr posterior enters cross-fitted; gather_rate_held: share of the "
                       "source's held decisions in which the rule chooses an observe candidate",
    "held_detector": "decision counts per split: family decisions, held situations, held in / outside the family, "
                     "family decisions not held, decisions with several held groups, held decisions without a "
                     "trace_link sensor:g or a monitor candidate, held family decisions without an exact posterior",
    "posterior_lr_quality": "on the held family decisions (train: cross-fitted values; val: the fit on all of "
                            "train): mae_vs_exact = mean |p_lr - p_exact|; brier_vs_cause = mean (p - 1[link "
                            "fault])^2 for p_lr, brier_exact_vs_cause the same for p_exact; "
                            "map_agreement_with_exact = share in which p_lr and p_exact name the same likelier "
                            "cause",
    "train_references": "train regret under belief labels of the playbooks' recorded choices and of the GBM "
                        "argmin cross-fitted over the group_kfold split (the margin rule at m = 0 up to ties)",
    "gbm_grid_points_below": "number of the 8 GBM grid points (belief labels, fitted on all of train) whose val "
                             "regret (belief/all) is below the rule's",
    "margin": "gather only if (lowest predicted J among observe candidates) - (lowest among the others) < -m, "
              "else the GBM argmin among the others; decisions without both families: plain argmin",
    "margin_train": "per fold split: 5-fold episode-grouped cross-fitted GBM (belief labels, the chosen belief "
                    "grid point) on train; group_kfold = sklearn GroupKFold over candidate rows grouped by "
                    "episode; seed<s> = the episodes in a random order of numpy seed s dealt round-robin to the "
                    "folds; chosen_m = lowest regret (smallest m on ties); mean_curve = mean over the splits",
    "margin_val": "descriptive val values at VAL_MARGINS with the GBM fitted on all of train; minus_margin_0 = "
                  "paired difference against m = 0",
}


def load_dev(root, split):
    if split not in DEV_SPLITS:
        raise AssertionError(f"only the development splits {DEV_SPLITS} are read, not {split!r}")
    return load_split(root, split)


def held_groups(stale, vocab, stale_fraction=Playbook.stale_fraction):
    names = vocab["measurements"]
    stale = np.asarray(stale) > 0
    members = [[names.index(m) for m in vocab["group_measurements"][g]] for g in vocab["groups"]]
    n_stale = np.stack([stale[:, idx].sum(1) for idx in members], 1)
    size = np.array([len(idx) for idx in members])
    return n_stale >= np.maximum(2, stale_fraction * size), n_stale / size


def candidate_codes(split, vocab):
    f, mask = split["pad"]["features"], split["mask"]
    lo, hi = vocab["action_blocks"]["tool"]
    tool = np.where(mask, f[..., lo:hi].argmax(-1), -1)
    lo, hi = vocab["action_blocks"]["target"]
    names = np.array([t for k in vocab["target_kinds"] for t in vocab["targets"][k]])
    return tool, np.where(mask, names[f[..., lo:hi].argmax(-1)], "")


def find(tool, target, tool_id, wanted):
    hit = (tool == tool_id) & (target == np.asarray(wanted)[:, None])
    return np.where(hit.any(1), hit.argmax(1), -1)


def situation(split, vocab, summ, stale_fraction=Playbook.stale_fraction):
    n, tools, groups = split["n"], vocab["tools"], vocab["groups"]
    held, share = held_groups(summ["measurement"]["stale"], vocab, stale_fraction)
    g = np.where(held, share, -1.0).argmax(1)
    names = {"sensor": np.array([f"sensor:{groups[x]}" for x in g]),
             "source": np.array([f"source:{groups[x]}" for x in g])}
    tool, target = candidate_codes(split, vocab)
    s = {"held": held.any(1), "group": g, "multi": held.sum(1) > 1}
    for key, name, kind in (("trace", "trace_link", "sensor"), ("switch", "switch_to_backup", "sensor"),
                            ("restore", "restore_primary", "sensor"), ("module", "replace_io_module", "source")):
        s[key] = find(tool, target, tools.index(name), names[kind])
    review = split["pad"]["features"][..., vocab["action_columns"].index("arg:monitor.review_after_s")]
    is_monitor = tool == tools.index("monitor")
    s["monitor"] = np.where(is_monitor.any(1), np.where(is_monitor, review, np.inf).argmin(1), -1)
    subjects = next(b for b in vocab["subject_blocks"]["receipt"] if b["key"] == "subject")["ids"]
    subject = {kind: np.array([subjects.index(x) for x in v]) for kind, v in names.items()}
    at, rows = summ["receipt"]["at_status"], np.arange(n)
    s["status"] = {t: at[rows, tools.index(t), subject[kind]] for t, kind in STATUS_TOOLS}
    st = vocab["receipt_statuses"]
    pending = [st.index("accepted"), st.index("running")]
    s["trace_pending"] = np.isin(s["status"]["trace_link"], pending)
    s["module_busy"] = np.isin(s["status"]["replace_io_module"], pending + [st.index("succeeded")])
    return s


def held_counts(s, family, p_link):
    held = s["held"]
    return {"decisions": int(held.size), "family": int(family.sum()), "held": int(held.sum()),
            "held_in_family": int((held & family).sum()), "held_not_family": int((held & ~family).sum()),
            "family_not_held": int((family & ~held).sum()), "multi_group": int(s["multi"].sum()),
            "held_without_trace_candidate": int((held & (s["trace"] < 0)).sum()),
            "held_without_monitor": int((held & (s["monitor"] < 0)).sum()),
            "held_family_without_posterior": int((held & family & ~np.isfinite(p_link)).sum())}


def link_posterior(split):
    cause = family_cause(split)
    p = np.asarray(split["dec"]["posterior"], np.float64)
    return np.where(cause > 0, p, np.where(cause < 0, 1.0 - p, np.nan))


def posterior_features(summ, s, vocab):
    g = s["group"]
    rows = np.arange(g.size)
    link = np.array([vocab["links"].index(f"sensor:{vocab['groups'][x]}") for x in g])
    cols, names = [], []

    def add(values, name, missing=True):
        values = np.asarray(values, np.float64)
        cols.append(np.nan_to_num(values, nan=0.0))
        names.append(name)
        if missing:
            cols.append(np.isnan(values).astype(np.float64))
            names.append(f"{name}:missing")

    for key in LINK_KEYS:
        add(summ["link"][key][rows, link], f"link_g:{key}")
    for key in GROUP_KEYS:
        add(summ["group"][key][rows, g], f"group_g:{key}")
    for j, cause in enumerate(vocab["payload_layout"]["operator_report"]):
        add(summ["operator"][:, j], f"operator:{cause}", False)
    add(summ["operator_any"], "operator_any", False)
    add(summ["since_first_alert"], "since_first_alert")
    add(summ["since_last_alert"], "since_last_alert")
    add(summ["alerts"], "alerts", False)
    add(summ["n_valid"], "n_valid", False)
    for tool, status in s["status"].items():
        for k, name in enumerate(vocab["receipt_statuses"]):
            add(status == k, f"status_g:{tool}={name}", False)
    add(s["switch"] >= 0, "candidate:switch_to_backup", False)
    add(s["restore"] >= 0, "candidate:restore_primary", False)
    return np.column_stack(cols), names


def fit_lr(X, p, C=1.0):
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    model = LogisticRegression(C=C, max_iter=5000).fit(
        np.vstack([Z, Z]), np.r_[np.ones(len(p)), np.zeros(len(p))], sample_weight=np.r_[p, 1 - p])
    return lambda X_new: model.predict_proba(scaler.transform(X_new))[:, 1]


def posterior_quality(p_hat, p_exact, cause, rows):
    y = (cause[rows] > 0).astype(np.float64)
    a, b = p_hat[rows], p_exact[rows]
    return {"n": int(rows.sum()), "mae_vs_exact": m_mean(np.abs(a - b)), "brier_vs_cause": m_mean((a - y) ** 2),
            "brier_exact_vs_cause": m_mean((b - y) ** 2), "map_agreement_with_exact": m_mean(((a >= .5) == (b >= .5)) * 1.)}


def episode_folds(groups, key, folds=FOLDS):
    groups = np.asarray(groups)
    if key == "group_kfold":
        return [test for _, test in GroupKFold(folds).split(np.zeros(groups.size), groups=groups)]
    if not key.startswith("seed"):
        raise ValueError(f"unknown fold split {key!r}")
    order = np.random.default_rng(int(key[4:])).permutation(np.unique(groups))
    fold_of = dict(zip(order.tolist(), (np.arange(order.size) % folds).tolist()))
    fold = np.array([fold_of[e] for e in groups.tolist()])
    return [np.flatnonzero(fold == k) for k in range(folds)]


def padded(split, values, fill=np.inf):
    mask, pad = pad_candidates(split["cand"]["decision"], split["n"], {"x": np.asarray(values, np.float64)})
    return np.where(mask, pad["x"], fill)


def crossfit_gbm(X, split, params, tests):
    cand = split["cand"]
    oof = np.full(len(cand["decision"]), np.nan)
    for test in tests:
        fit = np.ones(oof.size, bool)
        fit[test] = False
        fb = FlatBaseline(**params).fit(X[fit], cand["J_belief"][fit], cand["p_trip_belief"][fit],
                                        cand["decision"][fit], trip=False)
        oof[test] = fb.predict(X[test])[0]
    if np.isnan(oof).any():
        raise AssertionError("the folds do not cover every candidate")
    return padded(split, oof)


def bt_choice(s, held, p_link, theta, fallback, playbook):
    choice = np.array(fallback, copy=True)
    monitor = s["monitor"]
    unsure = held & (np.maximum(p_link, 1 - p_link) < theta)
    trace = unsure & (s["trace"] >= 0) & ~s["trace_pending"]
    wait = unsure & s["trace_pending"] & (monitor >= 0)
    act = held & ~trace & ~wait
    act_link, act_module = act & (p_link >= 0.5), act & (p_link < 0.5)
    choice[trace] = s["trace"][trace]
    choice[wait] = monitor[wait]
    choice[act_link] = playbook[act_link]
    replace = act_module & (s["module"] >= 0) & ~s["module_busy"]
    choice[replace] = s["module"][replace]
    other = act_module & ~replace
    choice[other] = np.where(monitor[other] >= 0, monitor[other], playbook[other])
    return choice


def gate_choice(held, p_link, theta, J_pred, observe, mask):
    Jp = np.where(mask, J_pred, np.inf)
    obs, act = mask & observe, mask & ~observe
    gather = held & (np.maximum(p_link, 1 - p_link) < theta) & obs.any(1)
    acting = held & ~gather & act.any(1)
    allowed = np.where(gather[:, None], obs, np.where(acting[:, None], act, mask))
    return np.where(allowed, Jp, np.inf).argmin(1)


def margin_choice(J_pred, observe, mask, m):
    Jp = np.where(mask, J_pred, np.inf)
    obs, act = mask & observe, mask & ~observe
    both = obs.any(1) & act.any(1)
    with np.errstate(invalid="ignore"):
        gap = np.where(obs, Jp, np.inf).min(1) - np.where(act, Jp, np.inf).min(1)
    gather = both & (gap < -m)
    allowed = np.where(gather[:, None], obs, np.where(both[:, None], act, mask))
    return np.where(allowed, Jp, np.inf).argmin(1)


def first_min(curve, key, tol=1e-12):
    best = min(c[key] for c in curve)
    return next(c for c in curve if c[key] <= best + tol)


def argmin_masked(split, J_pred):
    return np.where(split["mask"], J_pred, np.inf).argmin(1)


def costly(ctx):
    return ctx["has_both"] & (np.nan_to_num(ctx["gap"]) >= COSTLY_GAP)


def curve_point(ctx, choice):
    r = rows_for_choice(ctx, choice)
    return {"regret": m_mean(r["regret"]), "gather_rate": m_mean(r["gather"][ctx["has_both"]]),
            "costly_gather_share": m_mean(r["gather"][costly(ctx)])}


def interval(ci):
    return {k: ci[k] for k in ("point", "lo", "hi", "p_two_sided") if k in ci}


def scored(ctx, choices, refs, n_boot):
    rows = {m: rows_for_choice(ctx, c) for m, c in choices.items()}
    hard = costly(ctx)
    out = {"decisions": int(ctx["n"]), "episodes": int(np.unique(ctx["cluster"]).size),
           "with_both": int(ctx["has_both"].sum()), "costly_n": int(hard.sum()), "methods": {}}
    for m, r in rows.items():
        e = {"regret": interval(cluster_bootstrap_ci(m_mean, [r["regret"]], ctx["cluster"], n_boot=n_boot)),
             "gather_rate": m_mean(r["gather"][ctx["has_both"]]), "costly_gather_share": m_mean(r["gather"][hard])}
        for ref in refs:
            if ref != m:
                e[f"minus_{ref}"] = interval(paired_cluster_bootstrap_diff(
                    m_mean, [r["regret"]], [rows[ref]["regret"]], ctx["cluster"], n_boot=n_boot))
        out["methods"][m] = e
    return out


def run_set(root, n_boot, material_gap, log):
    t0 = time.perf_counter()
    vocab = json.loads((root / "meta.json").read_text(encoding="utf-8"))["vocab"]
    data = {s: load_dev(root, s) for s in DEV_SPLITS}
    tr, va = data["train"], data["val"]
    for s, d in data.items():
        for mode in ("act_first", "investigate_first"):
            if (d["dec"][f"baseline_{mode}"] < 0).any():
                raise ValueError(f"{root.name} {s}: the {mode} playbook's choice is missing at some decisions")
    out = {"decisions": {s: d["n"] for s, d in data.items()}, "gbm": {}}
    log(f"{root.name}: " + ", ".join(f"{s} {d['n']} decisions" for s, d in data.items()))

    gbm_val = {}
    for labels in ("belief", "hindsight"):
        flat, table, params = flat_predictions({s: label_view(d, labels) for s, d in data.items()}, vocab, log)
        gbm_val[labels] = flat["flat_gbm"]["val"][0][0]
        out["gbm"][labels] = {"grid": table, "chosen": params}
    params = out["gbm"]["belief"]["chosen"]

    summ = {s: evidence_summary(d["batch"], vocab) for s, d in data.items()}
    sit = {s: situation(d, vocab, summ[s]) for s, d in data.items()}
    family = {s: d["dec"]["family"].astype(bool) for s, d in data.items()}
    cause = {s: family_cause(d) for s, d in data.items()}
    p_exact = {s: link_posterior(d) for s, d in data.items()}
    held_exact = {s: sit[s]["held"] & family[s] & np.isfinite(p_exact[s]) for s in data}
    out["held_detector"] = {s: held_counts(sit[s], family[s], p_exact[s]) for s in data}
    log(f"{root.name} held situations: {json.dumps(out['held_detector'])}")
    feats = {}
    for s in data:
        feats[s], names = posterior_features(summ[s], sit[s], vocab)
    fit_rows = sit["train"]["held"] & np.isfinite(p_exact["train"])
    idx = np.flatnonzero(fit_rows)
    p_lr = {s: np.full(d["n"], np.nan) for s, d in data.items()}
    for test in episode_folds(tr["dec"]["episode_seed"][idx], "group_kfold"):
        fit = np.setdiff1d(np.arange(idx.size), test)
        p_lr["train"][idx[test]] = fit_lr(feats["train"][idx[fit]], p_exact["train"][idx[fit]])(feats["train"][idx[test]])
    lr_full = fit_lr(feats["train"][idx], p_exact["train"][idx])
    for s, rest in (("train", sit["train"]["held"] & ~fit_rows), ("val", sit["val"]["held"])):
        if rest.any():
            p_lr[s][rest] = lr_full(feats[s][rest])
    out["posterior_lr"] = {"features": len(names), "fit_rows_train": int(idx.size),
                           **{s: posterior_quality(p_lr[s], p_exact[s], cause[s], held_exact[s]) for s in data}}
    log(f"{root.name} lr posterior: {json.dumps(out['posterior_lr'])}")
    sources = {"exact": (held_exact, p_exact), "lr": ({s: sit[s]["held"] for s in data}, p_lr)}

    ctx_tr = {lab: context(tr, vocab, material_gap, lab) for lab in ("belief", "hindsight")}
    pb_tr = tr["dec"]["baseline_act_first"]
    out["theta_selection"] = {}
    for src, (held, p) in sources.items():
        curve = []
        for theta in THETAS:
            choice = bt_choice(sit["train"], held["train"], p["train"], theta, pb_tr, pb_tr)
            rb = rows_for_choice(ctx_tr["belief"], choice)
            curve.append({"theta": theta, "regret_belief": m_mean(rb["regret"]),
                          "regret_hindsight": m_mean(rows_for_choice(ctx_tr["hindsight"], choice)["regret"]),
                          "gather_rate_held": m_mean(rb["gather"][held["train"]])})
        out["theta_selection"][src] = {"curve": curve, "chosen_theta": first_min(curve, "regret_belief")["theta"]}
        log(f"{root.name} BT {src}: theta {out['theta_selection'][src]['chosen_theta']}, train curve "
            + ", ".join(f"{c['theta']}: {c['regret_belief']:.4f}" for c in curve))

    X_tr = feature_matrix(summ["train"], tr["cand"]["features"], tr["cand"]["decision"], vocab)[0]
    ep_cand = tr["dec"]["episode_seed"][tr["cand"]["decision"]]
    splits, J_oof = {}, None
    for key in FOLD_SPLITS:
        J = crossfit_gbm(X_tr, tr, params, episode_folds(ep_cand, key))
        curve = [{"m": m, **curve_point(ctx_tr["belief"], margin_choice(J, ctx_tr["belief"]["observe"], tr["mask"], m))}
                 for m in MARGINS]
        splits[key] = {"curve": curve, "chosen_m": first_min(curve, "regret")["m"],
                       "plain_argmin_regret": m_mean(rows_for_choice(ctx_tr["belief"], argmin_masked(tr, J))["regret"])}
        J_oof = J if key == "group_kfold" else J_oof
        log(f"{root.name} margin, train cross-fitted ({key}): chosen m {splits[key]['chosen_m']}; "
            + ", ".join(f"{c['m']}: {c['regret']:.4f}" for c in curve))
    mean_curve = [{"m": m, **{k: float(np.mean([splits[s]["curve"][i][k] for s in splits]))
                              for k in ("regret", "gather_rate", "costly_gather_share")}}
                  for i, m in enumerate(MARGINS)]
    out["margin"] = {"train_crossfit": {"splits": splits, "mean_curve": mean_curve,
                                        "chosen_m_mean_curve": first_min(mean_curve, "regret")["m"]}}
    log(f"{root.name} margin, mean train curve: " + ", ".join(f"{c['m']}: {c['regret']:.4f}" for c in mean_curve))

    obs_tr = ctx_tr["belief"]["observe"]
    out["gate_selection"] = {}
    for src, (held, p) in sources.items():
        curve = []
        for theta in THETAS:
            choice = gate_choice(held["train"], p["train"], theta, J_oof, obs_tr, tr["mask"])
            r = rows_for_choice(ctx_tr["belief"], choice)
            curve.append({"theta": theta, "regret_belief": m_mean(r["regret"]),
                          "gather_rate_held": m_mean(r["gather"][held["train"]])})
        out["gate_selection"][src] = {"curve": curve, "chosen_theta": first_min(curve, "regret_belief")["theta"]}
        log(f"{root.name} gate {src}: theta {out['gate_selection'][src]['chosen_theta']}")
    out["train_references"] = {
        "playbook_act_first": m_mean(rows_for_choice(ctx_tr["belief"], pb_tr)["regret"]),
        "playbook_investigate_first": m_mean(rows_for_choice(ctx_tr["belief"],
                                                             tr["dec"]["baseline_investigate_first"])["regret"]),
        "gbm_belief_crossfit_group_kfold": splits["group_kfold"]["plain_argmin_regret"]}

    pb_v, obs_v = va["dec"]["baseline_act_first"], va["mask"] & va["pad"]["observe"].astype(bool)
    gbm_v = argmin_masked(va, gbm_val["belief"])
    choices = {"gbm_belief": gbm_v, "gbm_hindsight": argmin_masked(va, gbm_val["hindsight"]),
               "playbook_act_first": pb_v, "playbook_investigate_first": va["dec"]["baseline_investigate_first"]}
    for src, (held, p) in sources.items():
        theta = out["theta_selection"][src]["chosen_theta"]
        choices[f"bt_{src}"] = bt_choice(sit["val"], held["val"], p["val"], theta, pb_v, pb_v)
        choices[f"bt_{src}_gbm_fallback"] = bt_choice(sit["val"], held["val"], p["val"], theta, gbm_v, pb_v)
    for src, (held, p) in sources.items():
        choices[f"gate_{src}"] = gate_choice(held["val"], p["val"], out["gate_selection"][src]["chosen_theta"],
                                             gbm_val["belief"], obs_v, va["mask"])
    margins = {f"margin_{m:g}": margin_choice(gbm_val["belief"], obs_v, va["mask"], m) for m in VAL_MARGINS}
    out["val"], out["margin"]["val"] = {}, {}
    for labels in ("belief", "hindsight"):
        for sub, sel in subsets(va):
            if sub in VIEWS:
                ctx = context(take(va, sel), vocab, material_gap, labels)
                view = f"{labels}/{sub}"
                out["val"][view] = scored(ctx, {k: c[sel] for k, c in choices.items()}, REFS, n_boot)
                out["margin"]["val"][view] = scored(ctx, {k: c[sel] for k, c in margins.items()}, ("margin_0",),
                                                    n_boot)
    main_view = out["val"]["belief/all"]["methods"]
    out["gbm_grid_points_below"] = {
        m: sum(g["val_regret"] < main_view[m]["regret"]["point"] for g in out["gbm"]["belief"]["grid"])
        for m in main_view if m.startswith(("bt_", "gate_"))}
    for m, e in main_view.items():
        log(f"{root.name} val belief/all {m}: {fmt(e['regret'])}"
            + "".join(f"; minus {r} {fmt(e[f'minus_{r}'])} p={e[f'minus_{r}']['p_two_sided']:.3f}"
                      for r in REFS if f"minus_{r}" in e))
    out["wall_s"] = time.perf_counter() - t0
    return out


def rounded(obj, digits=4):
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return round(x, digits) if isinstance(x, float) else x
    return walk(jsonable(obj))


def cell(ci, p=False):
    if ci is None or ci.get("point") is None:
        return "-"
    text = fmt(ci)
    return f"{text}, p={ci['p_two_sided']:.3f}" if p and ci.get("p_two_sided") is not None else text


def share(x):
    return "-" if x is None else f"{x:.3f}"


def summary_markdown(results):
    n_boot = results["config"]["n_boot"]
    lines = ["# Belief-threshold baseline and GBM margin (train and val only)", "",
             "Val regret [95% cluster-bootstrap CI over episodes]; minus X: paired difference method minus X "
             f"[95% CI], two-sided bootstrap p ({n_boot} replicates; p=0.000 means p < {2 / n_boot:g}). Fitted "
             "and selected on train, val scored once per rule; the test split is not read. Definitions in "
             "results.json.", ""]
    for name, r in results["sets"].items():
        hd, g = r["held_detector"], r["gbm"]
        lines += [f"## {name}", "",
                  f"- GBM grid points (chosen on val regret under the training labels): belief {g['belief']['chosen']}, "
                  f"hindsight {g['hindsight']['chosen']}",
                  "- held situations: " + "; ".join(f"{s} {h['held']} held, {h['family']} family, "
                                                    f"{h['held_not_family']} held outside the family"
                                                    for s, h in hd.items()),
                  "- theta chosen on train: " + ", ".join(
                      [f"bt_{s} {v['chosen_theta']}" for s, v in r["theta_selection"].items()]
                      + [f"gate_{s} {v['chosen_theta']}" for s, v in r["gate_selection"].items()]),
                  "- train references (belief regret): " + ", ".join(
                      f"{k} {v:.3f}" for k, v in r["train_references"].items()),
                  "- GBM grid points (of 8) with lower val regret than the rule: " + ", ".join(
                      f"{k} {v}" for k, v in r["gbm_grid_points_below"].items())]
        for view, v in r["val"].items():
            lines += ["", f"### {view}: {v['decisions']} decisions, {v['episodes']} episodes, {v['with_both']} "
                          f"with both families, {v['costly_n']} costly", "",
                      "| method | regret | minus gbm_belief | minus investigate-first | gather rate | "
                      "costly-gather share |", "|---|---|---|---|---|---|"]
            for m, e in v["methods"].items():
                lines.append(f"| {m} | {cell(e['regret'])} | {cell(e.get('minus_gbm_belief'), True)} | "
                             f"{cell(e.get('minus_playbook_investigate_first'), True)} | {share(e['gather_rate'])} | "
                             f"{share(e['costly_gather_share'])} |")
        mt = r["margin"]["train_crossfit"]
        lines += ["", "### Margin rule, train regret (belief labels) with cross-fitted GBM predictions", "",
                  "| fold split | " + " | ".join(f"m={c['m']:g}" for c in mt["mean_curve"]) + " | chosen m |",
                  "|---" * (len(MARGINS) + 2) + "|"]
        for key, sp in mt["splits"].items():
            lines.append(f"| {key} | " + " | ".join(f"{c['regret']:.4f}" for c in sp["curve"])
                         + f" | {sp['chosen_m']:g} |")
        lines.append("| mean | " + " | ".join(f"{c['regret']:.4f}" for c in mt["mean_curve"])
                     + f" | {mt['chosen_m_mean_curve']:g} |")
        for view in ("belief/all", "hindsight/all"):
            v = r["margin"]["val"][view]
            lines += ["", f"### Margin rule, val {view} (descriptive; costly_n {v['costly_n']})", "",
                      "| m | regret | minus m=0 | gather rate | costly-gather share |", "|---|---|---|---|---|"]
            for m, e in v["methods"].items():
                lines.append(f"| {m[7:]} | {cell(e['regret'])} | {cell(e.get('minus_margin_0'), True)} | "
                             f"{share(e['gather_rate'])} | {share(e['costly_gather_share'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=list(DATA))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--material-gap", type=float, default=0.5)
    parser.add_argument("--name", default="val")
    args = parser.parse_args()
    t0 = time.perf_counter()
    config = {"data": args.data, "splits": list(DEV_SPLITS), "n_boot": args.n_boot, "bootstrap_seed": 0,
              "material_gap": args.material_gap, "thetas": list(THETAS), "margins": list(MARGINS),
              "val_margins": list(VAL_MARGINS), "folds": FOLDS, "fold_splits": list(FOLD_SPLITS),
              "costly_gap": COSTLY_GAP, "stale_fraction": Playbook.stale_fraction, "lr_C": 1.0,
              "views": list(VIEWS), "references": list(REFS)}
    run = start_run("threshold_baseline", config, seed=0, name=args.name)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    sets = {}
    for path in args.data:
        root = ROOT / path
        sets[root.name] = run_set(root, args.n_boot, args.material_gap, log)
    plain = jsonable({"definitions": DEFINITIONS, "config": config, "sets": sets, "log": lines,
                      "wall_s": time.perf_counter() - t0})
    results = rounded(plain)
    run.save_json("results.json", results)
    (run.dir / "summary.md").write_text(summary_markdown(plain), encoding="utf-8")
    run.finish({name: {m: e["regret"]["point"] for m, e in r["val"]["belief/all"]["methods"].items()}
                for name, r in results["sets"].items()})
    print(f"wrote {run.dir}")

if __name__ == "__main__":
    main()
