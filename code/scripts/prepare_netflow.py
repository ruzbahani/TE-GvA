import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.data.netflow import prepare_netflow
from cpsdefense.paths import CONFIG_DIR, ROOT
from cpsdefense.utils.config import load_config
from cpsdefense.utils.runs import start_run


def summarize(meta):
    temporal = meta["split"]["temporal"]
    random = meta["split"]["random"]
    return {
        "rows": meta["rows"],
        "start_utc": [meta["time_range"]["start_min_utc"], meta["time_range"]["start_max_utc"]],
        "end_before_start_fraction": meta["end_before_start"]["fraction"],
        "duplicate_rows_all_columns": meta["duplicates"]["all_columns"]["duplicate_rows"],
        "duplicate_rows_flow_features": meta["duplicates"]["flow_features"]["duplicate_rows"],
        "label_attack_consistent": meta["label_attack_consistency"]["consistent"],
        "thresholds_utc": temporal["thresholds_utc"],
        "temporal_rows": {k: v["rows"] for k, v in temporal["splits"].items()},
        "temporal_dropped": temporal["dropped"],
        "absent_from_train": temporal["absent_from_train"],
        "absent_from_val": temporal["absent_from_val"],
        "absent_from_test": temporal["absent_from_test"],
        "test_rows_with_feature_duplicate_in_train": {
            "temporal": temporal["test_rows_with_feature_duplicate_in_train"],
            "random": random["test_rows_with_feature_duplicate_in_train"],
        },
        "seconds": meta["seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "data" / "netflow.yaml"))
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    if args.datasets:
        config["datasets"] = args.datasets
    run = start_run("data", config, seed=config["seed"], name="netflow")
    io = config["io"]
    summary = {}
    for name in config["datasets"]:
        print(f"[{name}] converting", flush=True)
        meta = prepare_netflow(
            ROOT / config["raw_dir"] / f"{name}.zip",
            ROOT / config["out_dir"],
            name=name,
            fractions=config["split"]["fractions"],
            purge_gap_s=config["split"]["purge_gap_s"],
            seed=config["seed"],
            block_size=int(io["block_size_mb"] * (1 << 20)),
            rows_per_part=io["rows_per_part"],
            row_group_rows=io["row_group_rows"],
            compression=io["compression"],
            log=lambda message: print(message, flush=True),
        )
        run.save_json(f"meta-{name}.json", meta)
        summary[name] = summarize(meta)
        for key, value in summary[name].items():
            print(f"  {key}: {value}", flush=True)
    run.finish(summary)
    print(f"run record: {run.dir}")

if __name__ == "__main__":
    main()
