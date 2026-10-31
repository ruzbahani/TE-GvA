import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.data import hai_tasks
from cpsdefense.utils.config import load_config
from cpsdefense.utils.runs import start_run


def summarize(meta):
    return {
        "sources": dict(zip(meta["sources"], meta["payload_dim"])),
        "questions": {q: {k: v for k, v in p.items() if k in ("type", "source", "columns", "lo", "hi")}
                      for q, p in meta["questions"].items()},
        "score": meta["score"],
        "splits": {
            split: {
                "n_decisions": r["n_decisions"], "M": r["M"], "P": r["P"], "file_bytes": r["file_bytes"],
                "invalid_at_tau": r["invalid_at_tau"], "valid_per_decision": r["valid_per_decision"],
                "questions": {q: {k: v[k] for k in ("events_in_split", "observed_rate", "censoring_rate",
                                                    "censored_at_segment_end")}
                              for q, v in r["questions"].items()},
                "score": r["score"],
            }
            for split, r in meta["reports"].items()
        },
        "warnings": meta["warnings"],
    }


def print_summary(meta):
    print(f"\n== HAI {meta['version']} tasks ({meta['semi_synthetic_note']})")
    print("sources / payload dims:", ", ".join(f"{s}={d}" for s, d in zip(meta["sources"], meta["payload_dim"])))
    for name, q in meta["questions"].items():
        band = f" band [{q['lo']:.4g}, {q['hi']:.4g}]" if "lo" in q else ""
        print(f"question {name}: {q['type']} on {q['columns']} (source {q['source']}){band}")
    print(f"{'split':<9}{'N':>8}{'M':>4}{'P':>5}{'invalid@tau':>12}{'valid/dec':>10}{'MB':>8}")
    for split, r in meta["reports"].items():
        print(f"{split:<9}{r['n_decisions']:>8}{r['M']:>4}{r['P']:>5}{r['invalid_at_tau']:>12.3f}"
              f"{r['valid_per_decision']['mean']:>10.1f}{r['file_bytes'] / 1e6:>8.0f}")
    print(f"{'question':<14}{'split':<9}{'events':>7}{'observed':>9}{'cens.H':>8}{'cens.end':>9}{'med tte':>8}")
    for split, r in meta["reports"].items():
        for q, v in r["questions"].items():
            med = v["median_time_to_event_observed_s"]
            print(f"{q:<14}{split:<9}{v['events_in_split']:>7}{v['observed_rate']:>9.3f}"
                  f"{v['censored_at_horizon']:>8.3f}{v['censored_at_segment_end']:>9.3f}"
                  f"{'-' if med is None else f'{med:.0f}':>8}")
    for split, r in meta["reports"].items():
        if r["score"]:
            print(f"score {split}: classes {[round(x, 3) for x in r['score']['class_fraction']]}, "
                  f"persistence acc {r['score']['persistence_accuracy']:.3f}, missing {r['score']['missing']}")
    for warning in meta["warnings"]:
        print("  warning:", warning)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="*", default=["23.05", "22.04"])
    parser.add_argument("--config", default=hai_tasks.CONFIG)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. decisions.stride_s=20")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    for split in config.get("splits") or hai_tasks.DEV_SPLITS:
        hai_tasks.check_split(split)
    run = start_run("data", config, seed=config.get("seed", 0), name="prepare_hai_tasks")
    summaries = {}
    for version in args.versions:
        meta = hai_tasks.prepare_version(version, config=config)
        summaries[version] = summarize(meta)
        print_summary(meta)
    run.finish(summaries)
    print(f"\nrun record: {run.dir}")

if __name__ == "__main__":
    main()
