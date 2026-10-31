import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.data import hai
from cpsdefense.utils.config import load_config
from cpsdefense.utils.runs import start_run


def print_summary(summary):
    print(f"\n== HAI {summary['version']}: {summary['n_features']} features")
    print(f"{'split':<9}{'files':>6}{'rows':>10}{'segments':>10}{'events':>8}{'label %':>9}")
    for name, s in summary["splits"].items():
        frac = "-" if s["label_fraction"] is None else f"{100 * s['label_fraction']:.3f}"
        print(f"{name:<9}{s['files']:>6}{s['rows']:>10}{s['n_segments']:>10}{s['n_events']:>8}{frac:>9}")
    print(f"constant on train, dropped ({len(summary['constant_columns'])}): {', '.join(summary['constant_columns'])}")
    for column, splits in summary["constant_but_vary_outside_train"].items():
        print(f"  {column} is constant on train but varies in {', '.join(splits)}")
    gaps = {stem: n for stem, n in summary["gaps"].items() if n}
    print(f"gaps: {gaps or 'none'}; irregular cadence: {summary['irregular_cadence'] or 'none'}")
    purged = {stem: n for stem, n in summary["purged_rows"].items() if n}
    print(f"purged rows: {purged}")
    for contact in summary["adjacent_files"]:
        print(f"  adjacent: {contact['before']} -> {contact['after']} gap {contact['gap_s']:g} s "
              f"({' / '.join(contact['splits'])}), purged {contact['purged']}")
    for issue in summary["discrepancies"]:
        print(f"  summary discrepancy: {issue}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="*", default=["22.04", "23.05"])
    parser.add_argument("--config", default=hai.CONFIG)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. purge_s=600")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    run = start_run("data", config, name="prepare_hai")
    summaries = {}
    for version in args.versions:
        meta = hai.prepare_release(version, config=config)
        summaries[version] = hai.summarize(meta)
        print_summary(summaries[version])
    run.finish(summaries)
    print(f"\nrun record: {run.dir}")

if __name__ == "__main__":
    main()
