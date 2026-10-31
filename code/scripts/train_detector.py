import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.data import hai
from cpsdefense.detect.train import format_results, format_table, jsonable, run_detector, summarize
from cpsdefense.utils.config import load_config
from cpsdefense.utils.runs import start_run

CONFIG = "detect/hai.yaml"
DEV_NOTE = "Development run (--skip-test): the held-out test split was not loaded."


def log(message):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--versions", nargs="*", default=None, help="default: releases in the config")
    parser.add_argument("--detectors", nargs="*", default=None, help="default: detectors in the config")
    parser.add_argument("--skip-test", action="store_true", help="development mode: never load the held-out test")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. gru.max_epochs=5")
    parser.add_argument("--note", default=None, help="provenance note written at the top of results.md")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    versions = args.versions or config["releases"]
    detectors = args.detectors or config["detectors"]
    note = " ".join(n for n in (DEV_NOTE if args.skip_test else None, args.note) if n) or None
    results = []
    for version in versions:
        def load(split, version=version):
            return hai.load_split(version, split)

        for name in detectors:
            results += run_detector(name, version, load, config, skip_test=args.skip_test, log=log)

    rows, agg = summarize(results)
    summary = start_run("detect", {**config, "versions": versions, "detectors": detectors,
                                   "skip_test": args.skip_test, "note": note}, seed=None, name="hai-summary")
    rows.to_csv(summary.dir / "results_per_seed.csv", index=False)
    agg.to_csv(summary.dir / "results.csv", index=False)
    (summary.dir / "results.md").write_text(format_results(agg, config, note), encoding="utf-8")
    summary.finish(jsonable({"table": agg.to_dict(orient="records"),
                             "runs": [m["run_dir"] for m in results]}))
    log("\n" + format_table(agg, [float(t) for t in config["eval"]["fa_targets_per_hour"]]))
    log(f"\nsummary run: {summary.dir}")

if __name__ == "__main__":
    main()
