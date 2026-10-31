import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cpsdefense.data import hai_tasks
from cpsdefense.detect.train import jsonable
from cpsdefense.model import experiments as X
from cpsdefense.model import train as T
from cpsdefense.paths import RUNS_DIR
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_json
from cpsdefense.utils.runs import start_run

CONFIG = "model/hai_backbone.yaml"
TASK = "hai_backbone"
EVAL_SPLITS = ("val", "dev_test")
SHARED_KEYS = ("questions", "model", "train", "baselines", "invariance")


def log(message):
    print(message, flush=True)


def shared_config(config):
    shared = {k: config[k] for k in SHARED_KEYS}
    shared["variants"] = T.VARIANTS
    shared["eval_rule"] = [config["eval"]["rule"], config["eval"]["tail"]]
    return shared


def question_spec(config, version):
    meta = hai_tasks.load_meta(version)
    n_levels = int(meta["score"]["n_bins"]) if meta.get("score") else None
    return meta, T.QuestionSpec.from_config(config, meta["question_names"], n_levels)


def data_info(meta, splits):
    return {"hai_tasks_prepared_utc": meta["prepared_utc"], "hai_tasks_code_sha256": meta["code_sha256"],
            "semi_synthetic": meta["semi_synthetic"], "heldout_test_used": False,
            "n_decisions": {name: int(s.n) for name, s in splits.items()}}


def train_baselines(version, splits, meta, spec, config, root=None):
    b = config["baselines"]
    run_config = {"task": TASK, "version": version, "variant": "km", "shared": shared_config(config),
                  "shared_sha256": sha256_json(shared_config(config)), "data": data_info(meta, splits)}
    run = start_run("model", run_config, seed=None, name=f"hai{version}-km", root=root)
    km = T.KMPredictor(spec, b["min_events"], b["age_bins_s"], b["shrink_events"]).fit(splits["train"])
    for split in EVAL_SPLITS:
        for kind in T.BASELINES:
            T.save_predictions(run.dir / f"pred_{split}_{kind}.npz", km.predict(splits[split], kind))
    run.finish({"baselines": list(T.BASELINES), "fitted_on": "train"})
    log(f"[{version}] KM baselines -> {run.dir}")


def train_variant(version, variant, seed, splits, meta, spec, config, device, root=None):
    run_config = {"task": TASK, "version": version, "variant": variant, "seed": seed,
                  "shared": shared_config(config), "shared_sha256": sha256_json(shared_config(config)),
                  "data": data_info(meta, splits)}
    run = start_run("model", run_config, seed=seed, name=f"hai{version}-{variant}-s{seed}", root=root)
    log(f"[{version} {variant} seed {seed}] training -> {run.dir}")
    model = T.build_model(config, meta, variant, spec, seed, device)
    history = T.train_model(model, splits["train"], splits["val"], spec, config, seed, device, log=log)
    T.save_model(run.dir / "model.pt", model, spec, {"version": version, "variant": variant, "seed": seed})
    metrics = {"version": version, "variant": variant, "seed": seed, "training": history, "invariance": {},
               "model_config": model.config.to_dict()}
    questions = spec.questions()
    for split in EVAL_SPLITS:
        gpu = splits[split].batch.to(device)
        raw = T.predict(model, gpu, questions, int(config["train"]["eval_chunk"]))
        T.save_predictions(run.dir / f"pred_{split}.npz", T.to_predictions(raw, spec))
        metrics["invariance"][split] = X.invariance_audit(model, gpu, splits[split], spec, config, base=raw)
        fills = metrics["invariance"][split]["fills"]
        log(f"  {split} audit: " + ", ".join(f"{f} bitwise {r['bitwise_equal_rate']:.4f}" for f, r in fills.items()))
        del gpu
    run.finish(jsonable(metrics))
    log(f"[{version} {variant} seed {seed}] done: best step {history['best_step']}, stop {history['stop']}, "
        f"{history['train_seconds'] / 60:.1f} min, val total {history['best_val']['total']:.4f}")
    del model
    torch.cuda.empty_cache()


def train_all(config, versions, variants, seeds, baselines, device, root=None):
    for version in versions:
        meta, spec = question_spec(config, version)
        splits = {s: T.load_split(version, s) for s in ("train", *EVAL_SPLITS)}
        log(f"[{version}] loaded " + ", ".join(f"{k} {v.n}" for k, v in splits.items()))
        if baselines:
            train_baselines(version, splits, meta, spec, config, root)
        for variant in variants:
            for seed in seeds:
                train_variant(version, variant, seed, splits, meta, spec, config, device, root)
        del splits


def find_runs(version, shared_sha, root=None):
    found = {}
    for path in sorted((Path(root or RUNS_DIR) / "model").glob("*/run.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        cfg = rec.get("config") or {}
        if (cfg.get("task") != TASK or cfg.get("version") != version or cfg.get("shared_sha256") != shared_sha
                or "finished_utc" not in rec):
            continue
        found[(cfg["variant"], cfg.get("seed"))] = path.parent
    return found


def training_summary(metrics_by_variant):
    out = {}
    for variant, ms in metrics_by_variant.items():
        tr = [m["training"] for m in ms]
        out[variant] = {"n_seeds": len(ms), "best_steps": [t["best_step"] for t in tr], "stops": sorted({t["stop"] for t in tr}),
                        "train_minutes_mean": float(np.mean([t["train_seconds"] for t in tr]) / 60),
                        "best_val_total_mean": float(np.mean([t["best_val"]["total"] for t in tr])),
                        "n_parameters": tr[0]["n_parameters"]}
    return out


def evaluate_version(version, config, variants, seeds, root=None):
    meta, spec = question_spec(config, version)
    runs = find_runs(version, sha256_json(shared_config(config)), root)
    if ("km", None) not in runs:
        raise FileNotFoundError(f"no finished KM baseline run for {version} with this config")
    splits = {s: T.load_split(version, s) for s in EVAL_SPLITS}
    preds = {s: {k: [T.load_predictions(runs[("km", None)] / f"pred_{s}_{k}.npz")] for k in T.BASELINES}
             for s in EVAL_SPLITS}
    metrics, used = {}, {"km": str(runs[("km", None)])}
    for variant in variants:
        dirs = [runs.get((variant, seed)) for seed in seeds]
        if any(d is None for d in dirs):
            log(f"[{version}] {variant}: missing seeds {[s for s, d in zip(seeds, dirs) if d is None]}; skipped")
            continue
        for s in EVAL_SPLITS:
            preds[s][variant] = [T.load_predictions(d / f"pred_{s}.npz") for d in dirs]
        metrics[variant] = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in dirs]
        used[variant] = [str(d) for d in dirs]
    res = {"runs": used, "n_seeds": {v: len(m) for v, m in metrics.items()}, "splits": {},
           "training": training_summary(metrics),
           "invariance": {s: {v: X.aggregate_audits([m["invariance"][s] for m in ms]) for v, ms in metrics.items()}
                          for s in EVAL_SPLITS},
           "revalidation": {s: X.revalidation_stats(splits[s].batch, config["revalidation"]["deltas_s"])
                            for s in EVAL_SPLITS}}
    for s in EVAL_SPLITS:
        res["splits"][s] = X.evaluate_split(splits[s], preds[s], spec, config, log=log)
    return res, splits, runs, spec


def run_latency(config, version, splits, runs, spec):
    lc = config["latency"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for variant in ("masked", "block_local"):
        if (variant, 0) in runs:
            models[variant], _ = T.load_model(runs[(variant, 0)] / "model.pt", device)
    if not models:
        return None
    out = X.latency_benchmark(models, splits[lc["split"]].batch, spec, config, log=log)
    out.update(version=version, split=lc["split"], models={v: str(runs[(v, 0)]) for v in models})
    return out


def compact_summary(results):
    lines = []
    for version, res in results["versions"].items():
        for split, r in res["splits"].items():
            lines.append(f"HAI {version} {split}: NLL / IBS / C-td")
            for name, by_v in r["questions"].items():
                cells = [f"{v} {m['nll']['point']:.3f}/{m['ibs']['point']:.4f}/{m['c_td']['point']:.3f}"
                         for v, m in by_v.items()]
                lines.append(f"  {name}: " + "; ".join(cells))
    return "\n".join(lines)


def evaluate_all(config, versions, variants, seeds, note, skip_latency, root=None):
    results = {"versions": {}, "latency": None}
    all_runs = {}
    for version in versions:
        log(f"[{version}] evaluating")
        res, splits, runs, spec = evaluate_version(version, config, variants, seeds, root)
        results["versions"][version] = res
        all_runs[version] = res["runs"]
        if not skip_latency and version == str(config["latency"]["version"]):
            log(f"[{version}] latency benchmark")
            results["latency"] = run_latency(config, version, splits, runs, spec)
    summary = start_run("model", {"task": TASK, "stage": "summary", "config": config, "versions": versions,
                                  "variants": variants, "seeds": seeds, "note": note, "runs": all_runs,
                                  "shared_sha256": sha256_json(shared_config(config))},
                        seed=None, name="hai-backbone-summary", root=root)
    results = jsonable(results)
    summary.save_json("results.json", results)
    (summary.dir / "results.md").write_text(X.format_results(results, config, note), encoding="utf-8")
    summary.finish({"results": "results.json", "runs": all_runs})
    log("\n" + compact_summary(results))
    log(f"\nsummary run: {summary.dir}")
    return summary.dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("train", "evaluate", "all"))
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--versions", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--no-baselines", action="store_true", help="train: skip the KM baseline run")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.max_minutes=1")
    parser.add_argument("--note", default=None, help="provenance note for results.md")
    parser.add_argument("--runs-root", default=None, help="runs directory (default runs/); use a scratch dir for smoke runs")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    versions = [str(v) for v in (args.versions or config["versions"])]
    variants = args.variants or config["variants"]
    seeds = args.seeds if args.seeds is not None else [int(s) for s in config["seeds"]]
    device = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    if args.stage in ("train", "all"):
        train_all(config, versions, variants, seeds, not args.no_baselines, device, args.runs_root)
    if args.stage in ("evaluate", "all"):
        evaluate_all(config, versions, variants, seeds, args.note, args.skip_latency, args.runs_root)

if __name__ == "__main__":
    main()
