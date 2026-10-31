import math

import numpy as np
import pandas as pd

from cpsdefense.detect.baselines import PCAReconstruction, RobustZScore
from cpsdefense.detect.forecast import GRUForecaster
from cpsdefense.detect.scoring import (
    MIN_CI_CLUSTERS,
    alarm_report,
    apply_warmup,
    array_digest,
    evaluate,
    fit_thresholds,
    normal_series,
    normal_thresholds,
    seed_mean_intervals,
    smooth,
)
from cpsdefense.eval.metrics import average_precision, precision_recall_f1
from cpsdefense.utils.runs import start_run

DETECTORS = ("zscore", "pca", "gru")
SPLITS = ("dev_test", "test")
NAN = float("nan")
VAL_CHECK_KEYS = ("n_false_alarms", "normal_hours", "fa_per_hour", "alarm_fraction", "n_false_alarms_duration",
                  "fa_per_hour_duration", "normal_alarm_fraction")


def build_detector(name, config, seed=0, log=None):
    cfg = dict(config.get(name) or {})
    cfg.pop("seeds", None)
    if name == "zscore":
        return RobustZScore(**cfg)
    if name == "pca":
        return PCAReconstruction(**cfg)
    if name == "gru":
        return GRUForecaster(**cfg, seed=seed, log=log)
    raise ValueError(f"unknown detector {name!r}; expected one of {DETECTORS}")


def warmup_rows(config):
    return int(config["gru"]["window"]) - 1


def masked_scores(detector, split, warmup):
    return {k: apply_warmup(v, split["segment"], warmup)
            for k, v in detector.scores(split["X"], split["segment"]).items()}


def _eval_args(config):
    ev = config["eval"]
    return {"period_s": float(config.get("period_s", 1)), "merge_gap": int(ev["merge_gap"]),
            "grace_s": float(ev["grace_s"]), "pre_tolerance_s": float(ev["pre_tolerance_s"]),
            "fa_unit_s": float(ev["fa_unit_s"])}


def _boot_args(config):
    ev = config["eval"]
    return {"block_s": ev["block_s"], "n_boot": ev["n_boot"], "alpha": ev["alpha"], "seed": ev["boot_seed"],
            "min_clusters": int(ev.get("min_ci_clusters", MIN_CI_CLUSTERS))}


def _threshold_args(config):
    fa_s = config["eval"].get("max_false_alarm_s")
    return {"period_s": float(config.get("period_s", 1)), "merge_gap": int(config["eval"]["merge_gap"]),
            "max_false_alarm_s": None if fa_s is None else float(fa_s)}


def dev_criterion(dev_scores, dev, targets, config):
    args = _eval_args(config)
    parts = normal_series(dev_scores, dev["y"], dev["segment"], args["grace_s"], args["period_s"])
    ok = ~np.isnan(dev_scores)
    y, s = dev["y"][ok], dev_scores[ok]
    thresholds = normal_thresholds(parts, targets, **_threshold_args(config))
    recalls = [precision_recall_f1(y, s, thr)["recall"] for thr in thresholds.values()]
    return float(np.mean(recalls)), recalls, average_precision(y, s)


def select_setting(dev_scores_per_seed, dev, config):
    targets = config["eval"]["fa_targets_per_hour"]
    table = []
    for setting in dev_scores_per_seed[0]:
        for window in config["eval"]["smoothing"]:
            per_seed = [dev_criterion(smooth(scores[setting], dev["segment"], window), dev, targets, config)
                        for scores in dev_scores_per_seed]
            table.append({"setting": setting, "smoothing": int(window),
                          "criterion": float(np.mean([c for c, _, _ in per_seed])),
                          "per_seed": [c for c, _, _ in per_seed],
                          "recall_per_target": np.mean([r for _, r, _ in per_seed], axis=0).tolist(),
                          "average_precision": float(np.mean([ap for _, _, ap in per_seed]))})
    best = max(table, key=lambda row: row["criterion"])
    return {"setting": best["setting"], "smoothing": best["smoothing"], "criterion": best["criterion"],
            "rule": "max mean point recall over FA targets on dev_test, thresholds calibrated on dev_test "
                    "normal time (first in grid order on ties)",
            "table": table}


def save_scores(path, split, raw, score, thresholds):
    frame = pd.DataFrame({
        "timestamp": split["timestamp"],
        "file": split["file"],
        "segment": split["segment"],
        "label": split["y"].astype(np.int8),
        "scored": ~np.isnan(score),
        "score_raw": raw,
        "score": score,
    })
    for target, thr in thresholds.items():
        frame[f"alarm_fa{target:g}"] = np.nan_to_num(score, nan=-np.inf) >= thr
    frame.to_parquet(path, index=False)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _split_info(split):
    return {"rows": int(len(split["y"])), "attack_rows": int(split["y"].sum()),
            "segments": int(np.unique(split["segment"]).size),
            "files": list(dict.fromkeys(str(f) for f in split["file"]))}


def run_detector(name, release, load, config, skip_test=False, run_root=None, log=print):
    ev, args = config["eval"], _eval_args(config)
    targets = [float(t) for t in ev["fa_targets_per_hour"]]
    warmup = warmup_rows(config)
    train, val, dev = load("train"), load("val"), load("dev_test")
    if train["y"].any() or val["y"].any():
        raise ValueError("train and val must be normal-only (all labels 0)")
    if not (train["columns"] == val["columns"] == dev["columns"]):
        raise ValueError("feature columns differ between splits")
    train_digest = array_digest(np.asarray(train["X"], np.float32))
    seeds = list((config.get(name) or {}).get("seeds", [0]))
    run_config = {"release": release, "detector": name, "detector_config": config.get(name),
                  "eval": ev, "period_s": args["period_s"], "warmup_rows": warmup,
                  "n_features": len(train["columns"]), "feature_columns": list(train["columns"])}

    fitted = []
    for seed in seeds:
        run = start_run("detect", run_config, seed, name=f"hai{release}-{name}-s{seed}", root=run_root)
        log(f"[{release} {name} seed {seed}] fitting -> {run.dir}")
        det = build_detector(name, config, seed, log=log).fit(train, val)
        if det.scaler.fit_sha256 != train_digest or det.scaler.n_fit != len(train["y"]):
            raise AssertionError("feature scaling was not fitted on exactly the train split")
        if getattr(det, "floored_columns", None):
            log(f"[{release} {name} seed {seed}] robust scale floored at {det.scaler.std_floor:g} x std for "
                f"{len(det.floored_columns)} features: {', '.join(det.floored_columns)}")
        fitted.append((seed, run, det, masked_scores(det, val, warmup), masked_scores(det, dev, warmup)))

    selection = select_setting([d for _, _, _, _, d in fitted], dev, config)
    setting, window = selection["setting"], selection["smoothing"]
    log(f"[{release} {name}] selected {setting}, smoothing {window} (dev_test criterion "
        f"{selection['criterion']:.4f})")
    test = None if skip_test else load("test")
    if test is not None and test["columns"] != train["columns"]:
        raise ValueError("feature columns differ between train and test")

    boot = _boot_args(config)
    evaluated = []
    for seed, run, det, val_scores, dev_scores in fitted:
        val_final = smooth(val_scores[setting], val["segment"], window)
        thresholds = fit_thresholds(val_final, val["segment"], targets, **_threshold_args(config))
        val_check = {str(t): {k: v for k, v in alarm_report(val_final, val["y"], val["segment"], thr, **args).items()
                              if k in VAL_CHECK_KEYS}
                     for t, thr in thresholds.items()}
        metrics = {"release": release, "detector": name, "seed": seed, "setting": setting, "smoothing": window,
                   "run_dir": str(run.dir), "thresholds": {str(t): v for t, v in thresholds.items()}, "val": val_check,
                   "selection": selection, "data": {"train_sha256": train_digest, "warmup_rows": warmup,
                                                    "train": _split_info(train), "val": _split_info(val),
                                                    "dev_test": _split_info(dev)},
                   "model": det.describe()}
        final = {"dev_test": smooth(dev_scores[setting], dev["segment"], window)}
        metrics["dev_test"] = evaluate(final["dev_test"], dev["y"], dev["segment"], thresholds, **args, **boot)
        save_scores(run.dir / "scores_dev_test.parquet", dev, dev_scores[setting], final["dev_test"], thresholds)
        if test is not None:
            raw = masked_scores(det, test, warmup)[setting]
            final["test"] = smooth(raw, test["segment"], window)
            metrics["test"] = evaluate(final["test"], test["y"], test["segment"], thresholds, **args, **boot)
            metrics["data"]["test"] = _split_info(test)
            save_scores(run.dir / "scores_test.parquet", test, raw, final["test"], thresholds)
        evaluated.append((run, metrics, final, thresholds))

    if len(evaluated) > 1:
        data = {"dev_test": dev, "test": test}
        seed_mean = {"seeds": seeds, "rule": "mean over seeds; CIs resample the same blocks / events for every seed"}
        for split in evaluated[0][2]:
            seed_mean[split] = seed_mean_intervals([e[2][split] for e in evaluated], data[split]["y"],
                                                   data[split]["segment"], [e[3] for e in evaluated], **args, **boot)
        for _, metrics, _, _ in evaluated:
            metrics["seed_mean"] = seed_mean

    results = []
    for run, metrics, _, _ in evaluated:
        metrics = jsonable(metrics)
        run.finish(metrics)
        results.append(metrics)
        log(f"[{release} {name} seed {metrics['seed']}] done: " + "; ".join(
            f"{s} AP {metrics[s]['average_precision']['point']:.3f}" for s in SPLITS if s in metrics))
    return results

INTERVAL_FIELDS = ("lo", "hi", "n_clusters")


def _num(value):
    return NAN if value is None else value


def _intervals_of(m):
    out = {"ap": m["average_precision"], "auc": m["roc_auc"]}
    for t, r in m["targets"].items():
        tag = f"@{float(t):g}"
        out.update({f"recall{tag}": r["point_recall"], f"event_recall{tag}": r["event_recall"],
                    f"strict_recall{tag}": r["strict_event_recall"], f"mean_delay{tag}": r["mean_delay_s"],
                    f"strict_delay{tag}": r["strict_mean_delay_s"]})
    return out


def _interval_columns(m):
    return {f"{key}_{field}": _num(ci.get(field)) for key, ci in _intervals_of(m).items() for field in INTERVAL_FIELDS}


def result_row(metrics, split):
    m = metrics[split]
    row = {"release": metrics["release"], "detector": metrics["detector"], "split": split,
           "seed": metrics["seed"], "setting": metrics["setting"], "smoothing": metrics["smoothing"]}
    row.update({key: _num(ci["point"]) for key, ci in _intervals_of(m).items()})
    for t, r in m["targets"].items():
        tag = f"@{float(t):g}"
        row.update({
            f"precision{tag}": _num(r["point_precision"]),
            f"n_events{tag}": r["n_events"], f"n_detected{tag}": r["n_detected"],
            f"n_strict_detected{tag}": r["n_strict_detected"], f"n_pre_alarmed{tag}": r["n_pre_alarmed"],
            f"median_delay{tag}": _num(r["median_delay_s"]),
            f"strict_median_delay{tag}": _num(r["strict_median_delay_s"]),
            f"fa_per_hour{tag}": _num(r["fa_per_hour"]), f"n_false_alarms{tag}": r["n_false_alarms"],
            f"fa_per_hour_duration{tag}": _num(r["fa_per_hour_duration"]),
            f"n_false_alarms_duration{tag}": r["n_false_alarms_duration"],
            f"false_alarm_s{tag}": _num(r["false_alarm_s"]),
            f"normal_alarm_pct{tag}": 100 * _num(r["normal_alarm_fraction"]),
            f"alarm_pct{tag}": 100 * _num(r["alarm_fraction"]), f"normal_hours{tag}": _num(r["normal_hours"]),
        })
    row.update(_interval_columns(m))
    return row


def summarize(results):
    rows = pd.DataFrame([result_row(m, s) for m in results for s in SPLITS if s in m])
    if rows.empty:
        return rows, rows
    keys = ["release", "detector", "split"]
    interval = [c for c in rows.columns if c.endswith(tuple(f"_{f}" for f in INTERVAL_FIELDS))]
    numeric = [c for c in rows.columns
               if c not in keys + interval + ["setting", "seed", "smoothing"] and rows[c].dtype.kind in "fi"]
    grouped = rows.groupby(keys, sort=False)
    agg = grouped[numeric].mean().join(grouped[numeric].std(ddof=1).add_suffix("_std"))
    agg = agg.join(grouped["seed"].count().rename("n_seeds"))
    agg = agg.join(grouped[["setting", "smoothing"]].first()).reset_index()
    ci_rows = []
    for _, r in agg.iterrows():
        members = [m for m in results if (m["release"], m["detector"]) == (r["release"], r["detector"])
                   and r["split"] in m]
        if len(members) == 1:
            ci_rows.append(_interval_columns(members[0][r["split"]]))
            continue
        joint = members[0].get("seed_mean") or {}
        same_seeds = list(joint.get("seeds", [])) == [m["seed"] for m in members]
        ci_rows.append(_interval_columns(joint[r["split"]]) if same_seeds and r["split"] in joint else {})
    agg = agg.join(pd.DataFrame(ci_rows, index=agg.index, columns=interval))
    order = keys + ["n_seeds", "setting", "smoothing"]
    for c in numeric:
        order += [c] + [f"{c}_{f}" for f in INTERVAL_FIELDS if f"{c}_{f}" in agg.columns] + [f"{c}_std"]
    return rows, agg[order]


def _fmt(value, digits=3):
    return "-" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _count(value):
    return "-" if pd.isna(value) else (f"{int(value)}" if float(value).is_integer() else f"{value:.1f}")


def _with_ci(r, key, digits=3):
    lo, hi = r.get(f"{key}_lo"), r.get(f"{key}_hi")
    if lo is None or pd.isna(lo):
        return _fmt(r[key], digits)
    return f"{_fmt(r[key], digits)} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def format_table(agg, targets):
    head = ["release", "detector", "split", "setting", "AP [95% CI]", "ROC-AUC [95% CI]", "seed sd AP / AUC"]
    lines = ["**Ranking (threshold-free)**", "", _row(head), "|" + "---|" * len(head)]
    for _, r in agg.iterrows():
        seeded = r["n_seeds"] > 1
        lines.append(_row([r["release"], r["detector"] + (f" (n={r['n_seeds']})" if seeded else ""), r["split"],
                           f"{r['setting']}/w{int(r['smoothing'])}", _with_ci(r, "ap"), _with_ci(r, "auc"),
                           f"{_fmt(r['ap_std'])} / {_fmt(r['auc_std'])}" if seeded else "-"]))
    head = ["release", "detector", "split", "FA target /h", "point recall [95% CI]", "point precision",
            "strict event recall [95% CI] (k/n)", "pre-alarmed", "lenient event recall [95% CI] (k/n)",
            "strict delay s: mean [95% CI] / median (n)", "FA/h segment", "FA/h duration", "normal alarm %",
            "seed sd strict recall / FA/h duration"]
    lines += ["", "**Alarms at the val thresholds**", "", _row(head), "|" + "---|" * len(head)]
    for _, r in agg.iterrows():
        seeded = r["n_seeds"] > 1
        for t in targets:
            tag = f"@{t:g}"
            n_events = _count(r[f"n_events{tag}"])
            delay = (f"{_with_ci(r, f'strict_delay{tag}', 0)} / {_fmt(r[f'strict_median_delay{tag}'], 0)} "
                     f"(n={_count(r[f'strict_delay{tag}_n_clusters'])})")
            lines.append(_row([
                r["release"], r["detector"] + (f" (n={r['n_seeds']})" if seeded else ""), r["split"], f"{t:g}",
                _with_ci(r, f"recall{tag}"), _fmt(r[f"precision{tag}"]),
                f"{_with_ci(r, f'strict_recall{tag}')} ({_count(r[f'n_strict_detected{tag}'])}/{n_events})",
                _count(r[f"n_pre_alarmed{tag}"]),
                f"{_with_ci(r, f'event_recall{tag}')} ({_count(r[f'n_detected{tag}'])}/{n_events})",
                delay, _fmt(r[f"fa_per_hour{tag}"], 2), _fmt(r[f"fa_per_hour_duration{tag}"], 2),
                _fmt(r[f"normal_alarm_pct{tag}"], 2),
                f"{_fmt(r[f'strict_recall{tag}_std'])} / {_fmt(r[f'fa_per_hour_duration{tag}_std'], 2)}"
                if seeded else "-"]))
    return "\n".join(lines)


def legend(config):
    ev = config["eval"]
    grace, gap, pre, unit = ev["grace_s"], ev["merge_gap"], ev["pre_tolerance_s"], ev["fa_unit_s"]
    window = f"[start - {pre:g} s, end + {grace:g} s)"
    return [
        "- Rows: one detector on one split. setting/wN = detector setting and causal smoothing window "
        "(N samples), chosen on dev_test (mean point recall over the FA targets at dev-calibrated thresholds) "
        "before the held-out test split was loaded.",
        f"- Thresholds: fitted on normal val scores so that alarm runs (merged over gaps <= {gap} samples) occur "
        f"at most at the FA target per hour and alarm time stays <= target x {ev.get('max_false_alarm_s')} s "
        "per hour; dev_test and test are scored at these fixed thresholds.",
        f"- [lo, hi]: 95% percentile cluster-bootstrap CI ({ev['n_boot']} resamples, alpha {ev['alpha']}, "
        f"seed {ev['boot_seed']}). AP, ROC-AUC and point recall resample {ev['block_s']} s blocks inside "
        "segments, event recall resamples events, delays resample detected events. A CI is shown only with "
        f">= {ev.get('min_ci_clusters', MIN_CI_CLUSTERS)} clusters, otherwise the point value alone (delay "
        "cells always give n). Thresholds are fixed, so a CI covers test sampling variability only.",
        "- Seeded detectors (\"(n=3)\"): values are means over seeds. Their CI resamples the same blocks / events "
        "for every seed and is the CI of the seed mean; the \"seed sd\" columns are the std over seeds (ddof 1), "
        "which measures training-seed variability only. zscore and pca are deterministic (one run, no seed sd).",
        "- point recall / precision: share of attack rows in alarm / share of alarm rows that are attack rows.",
        f"- lenient event recall (v1 \"event recall\"): an event counts as detected if any alarm is on between "
        f"its start and end + {grace:g} s, with delay 0 if an alarm is already on, however long before it began.",
        f"- strict event recall: an event counts only if an alarm onset (start of an alarm run merged over gaps "
        f"<= {gap} samples) falls in {window}; strict delay = max(0, onset - start) over strict detections. "
        f"pre-alarmed: events the lenient rule credits to an alarm that began before start - {pre:g} s "
        "(not strict detections).",
        f"- FA/h segment (v1 \"FA/h\"): alarm runs that touch no event or {grace:g} s grace window, per hour of "
        "normal time; a run that touches an event is never counted, however long it lasts.",
        f"- FA/h duration: each part of an alarm run outside every window {window} counts ceil(d / {unit:g} s) "
        "false alarms, per hour of normal time. normal alarm %: share of normal time (outside events and their "
        "grace windows) spent in alarm.",
        f"- pre_tolerance_s = {pre:g} and fa_unit_s = {unit:g} were fixed a priori; they are used only for reporting "
        "and play no part in setting selection or thresholds.",
        "- GRU use of val: the normal val split drives the GRU's early stopping, learning-rate schedule and "
        "per-feature residual scale (its score normalisation), and its thresholds are fitted on the same val "
        "scores. zscore and pca fit everything on train and use val only for thresholds. No separate "
        "calibration split is used; the dev_test FA rates show how the thresholds hold on unseen normal time.",
    ]


def format_results(agg, config, note=None):
    targets = [float(t) for t in config["eval"]["fa_targets_per_hour"]]
    parts = ["# Stage A detectors on HAI", ""]
    if note:
        parts += [note, ""]
    parts += ["## Legend", "", *legend(config), "", "## Results", "", format_table(agg, targets)]
    return "\n".join(parts) + "\n"
