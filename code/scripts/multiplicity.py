import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.paths import ROOT
from cpsdefense.utils.runs import start_run

ALPHA = 0.05
RESOLUTION = 0.001
SUBSETS = {"natural": ("all", "family"),
           "stress": ("all", "family", "backup_available", "family_backup_available", "backup_unavailable",
                      "family_backup_unavailable")}
PRESPECIFIED_E1 = {"natural": ("all", "family"), "stress": ("all", "family", "family_backup_unavailable")}
E2_PAIRS = ("summary - base", "summary_ensemble - base", "summary_items - summary")


def holm(tests, alpha=ALPHA, resolution=RESOLUTION):
    order = sorted(tests, key=lambda t: max(t[1], resolution))
    m, rows, rejecting = len(order), [], True
    for i, (label, p) in enumerate(order):
        threshold = alpha / (m - i)
        rejecting = rejecting and max(p, resolution) <= threshold
        rows.append({"test": label, "p": p, "threshold": round(threshold, 6), "rejected": rejecting})
    return rows


def e1_tests(numbers):
    out = {}
    for set_name, subs in SUBSETS.items():
        for sub in subs:
            for scoring in ("belief", "hindsight"):
                view = numbers[f"{set_name}_main"]["test"][f"{scoring}/{sub}"]
                assert view["reference"] == "flat_gbm_belief"
                out[f"E1 {set_name} {scoring}/{sub}"] = view["paired"]["flat_gbm_hindsight"]["regret"]["p_two_sided"]
    return out


def e2_e3_tests(numbers):
    out = {}
    for set_name in SUBSETS:
        pairs = numbers[f"{set_name}_extra_pairs"]["test"]["belief/all"]
        for pair in E2_PAIRS:
            out[f"E2 {set_name} {pair}"] = pairs[pair]["regret"]["p_two_sided"]
        main = numbers[f"{set_name}_main"]["test"]["belief/all"]
        out[f"E2 {set_name} summary_ensemble - flat_gbm_belief"] = \
            main["paired"]["summary_ensemble"]["regret"]["p_two_sided"]
        labels = numbers[f"{set_name}_labels"]["test"]["belief/all"]
        assert labels["reference"] == "direct_belief"
        out[f"E3 {set_name} direct_belief - direct_hindsight"] = \
            labels["paired"]["direct_hindsight"]["regret"]["p_two_sided"]
    return out


def grid_tests(numbers):
    out = {}
    for set_name in SUBSETS:
        for c in numbers[f"{set_name}_sensitivity"]["configs"]:
            for view in ("belief/all", "hindsight/all"):
                d = c["splits"]["test"]["views"][view]["minus_gbm_hindsight"]
                out[f"S {set_name} T={c['T']} delta={c['delta']} {view}"] = d["p_two_sided"]
    return out


def families(numbers):
    e1 = e1_tests(numbers)
    keys = [f"E1 {s} {sc}/{sub}" for s, subs in PRESPECIFIED_E1.items() for sub in subs for sc in ("belief", "hindsight")]
    pre = {k: e1[k] for k in keys}
    pre.update(e2_e3_tests(numbers))
    grid = dict(pre)
    grid.update(grid_tests(numbers))
    return {"primary": {k: e1[k] for k in ("E1 natural belief/all", "E1 stress belief/all")},
            "e1_views": e1, "prespecified": pre, "with_grid": grid}


def main():
    numbers = json.loads((ROOT / "paper" / "numbers.json").read_text(encoding="utf-8"))
    run = start_run("multiplicity", {"alpha": ALPHA, "resolution": RESOLUTION, "source": "paper/numbers.json"},
                    name="test-recorded")
    results, lines = {}, ["# Holm adjustment of the recorded test comparisons", ""]
    for name, tests in families(numbers).items():
        rows = holm(list(tests.items()))
        smallest = ALPHA / len(tests)
        results[name] = {"m": len(tests), "smallest_threshold": round(smallest, 6),
                         "resolvable": RESOLUTION <= smallest, "rows": rows,
                         "rejected": [r["test"] for r in rows if r["rejected"]]}
        lines.append(f"## {name} (m = {len(tests)}, smallest threshold {smallest:.6f}, "
                     f"resolvable at {RESOLUTION}: {RESOLUTION <= smallest})")
        lines += [f"- {r['test']}: p = {r['p']:.3f}, threshold {r['threshold']:.5f}" for r in rows if r["rejected"]]
        if not results[name]["rejected"]:
            lines.append("- none rejected")
        lines.append("")
    run.save_json("results.json", results)
    (run.dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    run.finish({name: len(r["rejected"]) for name, r in results.items()})
    print("\n".join(lines))
    print(f"wrote {run.dir}")

if __name__ == "__main__":
    sys.exit(main())
