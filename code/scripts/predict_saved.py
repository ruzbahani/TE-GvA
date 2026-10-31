import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpsdefense.model.direct import DirectConfig, DirectDecisionModel
from cpsdefense.model.summary import SummaryDescriptor
from cpsdefense.paths import ROOT
from train_direct import add_summary_descriptor, label_view, load_split, predict, to_device


def run_dir_of(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def model_config(run_dir, cfg, vocab, split_data):
    path = run_dir / "model_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    model = json.loads(json.dumps(cfg["model"]))
    model["backbone"] = {**model["backbone"], "payload_dim": [int(x) for x in vocab["payload_dim"]],
                         "n_kinds": len(vocab["kinds"]),
                         "descriptor_dim": int(split_data["pad"]["features"].shape[-1])}
    return model


def seeds_of(run_dir, cfg):
    ens = run_dir / "ensemble.json"
    if ens.exists():
        members = json.loads(ens.read_text(encoding="utf-8"))["members"]
        return [int(Path(m).stem.split("_s")[-1]) for m in members]
    return [int(s) for s in cfg["seeds"]]


def predict_run(run_dir, split, device):
    cfg = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["config"]
    if cfg.get("train_fraction", 1.0) != 1.0:
        raise ValueError(f"{run_dir.name}: learning-curve runs are not predicted here")
    meta = json.loads((ROOT / cfg["data"] / "meta.json").read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    data = {split: load_split(ROOT / cfg["data"], split)}
    if cfg["model"].get("summary_descriptor"):
        desc = SummaryDescriptor.load(run_dir / "summary_descriptor.npz", vocab)
        add_summary_descriptor(data, vocab, log=lambda m: None, desc=desc)
    d = data[split]
    mcfg = model_config(run_dir, cfg, vocab, d)
    dev = to_device(label_view(d, cfg.get("labels", "hindsight")), device)
    seeds, J, trip = seeds_of(run_dir, cfg), [], []
    for seed in seeds:
        model = DirectDecisionModel(DirectConfig(**mcfg)).to(device)
        model.load_state_dict(torch.load(run_dir / f"model_s{seed}.pt", map_location=device, weights_only=True))
        Jp, tp = predict(model, dev)
        J.append(Jp)
        trip.append(tp)
    return np.stack(J), np.stack(trip), np.asarray(seeds), d["dec"]["episode_seed"], d["mask"]


def check(run_dir, split, device, tol, min_agree):
    saved = np.load(run_dir / f"predictions_{split}.npz")
    J, trip, seeds, episodes, mask = predict_run(run_dir, split, device)
    if not (np.array_equal(saved["seeds"], seeds) and np.array_equal(saved["episode_seed"], episodes)):
        return False, {"error": "seeds or decisions differ from the saved predictions"}
    m = np.broadcast_to(mask[None], J.shape)
    diff = float(np.max(np.abs(J[m] - saved["J_pred"][m])))
    a = np.where(m, J, np.inf).argmin(-1)
    b = np.where(m, saved["J_pred"], np.inf).argmin(-1)
    agree = float((a == b).mean())
    trip_diff = float(np.max(np.abs(trip[m] - saved["trip"][m])))
    ok = diff <= tol and agree >= min_agree
    return ok, {"max_abs_J_diff": diff, "argmin_agreement": agree, "max_abs_trip_diff": trip_diff}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default=None, choices=("val", "test"))
    parser.add_argument("--check", default=None, choices=("val",))
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--min-agree", type=float, default=0.995)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    for r in args.runs:
        run_dir = run_dir_of(r)
        entry = {}
        if args.check:
            ok, stats = check(run_dir, args.check, device, args.tol, args.min_agree)
            entry["check"] = {"split": args.check, "ok": ok, **stats}
            print(f"{run_dir.name}: check {args.check} {'OK' if ok else 'FAILED'} {stats}", flush=True)
            if not ok:
                report[run_dir.name] = entry
                continue
        if args.split:
            out = run_dir / f"predictions_{args.split}.npz"
            if out.exists():
                raise FileExistsError(f"{out} exists; predictions are never overwritten")
            J, trip, seeds, episodes, _ = predict_run(run_dir, args.split, device)
            np.savez_compressed(out, J_pred=J, trip=trip, seeds=seeds, episode_seed=episodes)
            entry["wrote"] = str(out.relative_to(ROOT))
            print(f"{run_dir.name}: wrote {out.name} {J.shape}", flush=True)
        report[run_dir.name] = entry
    print(json.dumps(report, indent=1))
    if any(not e.get("check", {"ok": True})["ok"] for e in report.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
