import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strip_comments import strip_tree

ROOT = Path(__file__).resolve().parents[1]
SETS = ("v2-natural", "v2-stress")
EXPECTED = {
    ("v2-natural", "train.npz"): "e406d338a77466591a5cfdaa3cfefdb40a986b71b154a6805b723a2e9e17a1d5",
    ("v2-natural", "val.npz"): "05c83ac63a1b79d1a0be09c249c5e8a241600871fe270c75513be35c838cfdbf",
    ("v2-natural", "test.npz"): "74760c93c62f10dda40c1edada45caf8ed70ad062b18153fb8b50fc8b1bef36f",
    ("v2-natural", "meta.json"): "7e30201393b5838a638c0302cfe2dc05d6c7192c791fa6cfe5646f8fe6ade836",
    ("v2-stress", "train.npz"): "fdc5b755d58bb49cff34fb75ab3f340ef15ea8fffeabfcad539024a66bc5418b",
    ("v2-stress", "val.npz"): "3c21f5d0da7d2ffc6a4a2f3e9870fc5d026e19bb7fc1cf8d70aee99b734ff6c1",
    ("v2-stress", "test.npz"): "13408ae90c735beac191406de85dbaa0e9fb0409a6965480553d2fcd096bce6d",
    ("v2-stress", "meta.json"): "222bbc7d3520eebd7e9229a840a74aaa958070b3de58d1e811c05af5a1e5206d",
}
CODE_DIRS = ("cpsdefense", "scripts", "configs", "tests")
RESULT_RUNS = ("runs/model_direct/*-test-natural-main", "runs/model_direct/*-test-stress-main",
               "runs/model_direct/*-test-natural-labels", "runs/model_direct/*-test-stress-labels",
               "runs/belief_sensitivity/*-sens-natural-test", "runs/belief_sensitivity/*-sens-stress-test",
               "runs/multiplicity/*-test-recorded", "runs/label_diagnostics/*-dev", "runs/replicate_diagnostics/*-dev",
               "runs/trip_penalty/*-dev", "runs/threshold_baseline/*-val", "runs/backup_strata/*-val")
FORBIDDEN = (".pkl", ".pickle", ".pt", ".pth")
DOC_TESTS = (("tests/test_experience_tensors.py", "test_the_tensor_config_states_the_receipt_validity_rule"),)


def drop_function(path, name):
    import ast
    text = path.read_text(encoding="utf-8")
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise AssertionError(f"{path.name}: function {name} not found once")
    node = nodes[0]
    first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    lines = text.splitlines(keepends=True)
    rest = lines[:first] + lines[node.end_lineno:]
    while first < len(rest) and not rest[first].strip():
        del rest[first]
    new = "".join(rest)
    expected = ast.parse(text)
    expected.body = [n for n in expected.body if n is not None and not (isinstance(n, ast.FunctionDef) and n.name == name)]
    if ast.dump(ast.parse(new)) != ast.dump(expected):
        raise AssertionError(f"{path.name}: removing {name} changed other code")
    path.write_text(new, encoding="utf-8", newline="\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"))

README = """# TE-GvA

[![arXiv](https://img.shields.io/badge/arXiv-2611.03974-b31b1b.svg)](https://arxiv.org/abs/2611.03974)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23830595.svg)](https://doi.org/10.5281/zenodo.23830595)

Counterfactual gather-or-act decisions for fault-driven incident response on the Tennessee Eastman process.
Data, code and results of the paper "Gather or Act? Belief-Level Counterfactual Labels for Deadline-Aware
Incident Response in Cyber-Physical Systems" (A. Mohammadi Ruzbahani, arXiv:2611.03974,
https://arxiv.org/abs/2611.03974). The data and the datasheet are archived on Zenodo:
https://doi.org/10.5281/zenodo.23830595.

## Contents

| Path | What |
|---|---|
| `data/natural/`, `data/stress/` | `{train,val,test}.npz` (load with `numpy.load(..., allow_pickle=False)`), `meta.json` (vocabulary, scaler, split seeds, label keys, provenance), `*_item_ids.json` |
| `code/` | generation, labelling, models and evaluation (`cpsdefense/`, `scripts/`, `configs/`, `tests/`) |
| `results/` | `numbers.json` and the JSON results of the test evaluation and of the development analyses |
| `DATASHEET.md` | what the dataset contains, how it was made, what each field means, and its limitations |
| `CITATION.cff` | how to cite the dataset |
| `SHA256SUMS` | checksums of every file |

## Loading

```python
import numpy as np, json
a = np.load("data/stress/train.npz", allow_pickle=False)
meta = json.load(open("data/stress/meta.json"))
J_hindsight, J_belief = a["y_J"], a["y_J_belief"]
decision_of_candidate = a["cand_decision"]
```

`DATASHEET.md` describes every field. The fields `dec_kind`, `dec_onset`, `dec_oracle_tts`, `dec_family`, `dec_posterior`, `dec_twin_ok` and
`dec_belief_fallback` are hidden judging fields: use them for scoring only, never as model inputs. Report the
natural and stress sets separately. The analysis scripts read the arrays from the directories named in their
settings (the `SETS` constants in `code/scripts` and the `data` entry in `code/configs/model`); copy the two sets
there or change the settings.

## Not included

- The Tennessee Eastman kernel and TEP Studio. Regenerating the data needs TEP Studio at commit
  `852f0c69ab974540750d61bd1dbfafd56bfd80bb` and the revised kernel of Bathelt, Ricker and Jelali
  (`temexd_mod.c`, sha256 `eff57b75700e7472005785150c21f82b0c3d178ebceebe37e532794f23785aba`).
- The raw episode records and model weights.

## Licence

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use the dataset or the code, please cite the paper and the dataset:

```bibtex
@article{mohammadiruzbahani2026gather,
  title   = {Gather or Act? Belief-Level Counterfactual Labels for Deadline-Aware Incident Response in
             Cyber-Physical Systems},
  author  = {Mohammadi Ruzbahani, Ali},
  journal = {arXiv preprint arXiv:2611.03974},
  year    = {2026},
  doi     = {10.48550/arXiv.2611.03974}
}

@misc{mohammadiruzbahani2026tegva,
  title     = {{TE-GvA}: A Counterfactual Gather-or-Act Decision Dataset for Fault-Driven Incident Response
               on the {Tennessee Eastman} Process},
  author    = {Mohammadi Ruzbahani, Ali},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23830595}
}
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="release/te-gva")
    parser.add_argument("--no-data", action="store_true", help="skip the arrays (for a quick dry run)")
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it first")
    out.mkdir(parents=True)
    if not args.no_data:
        for s in SETS:
            src = ROOT / "data" / "processed" / "tep_decisions" / s
            dst = out / "data" / s.removeprefix("v2-")
            dst.mkdir(parents=True)
            for f in sorted(src.iterdir()):
                if f.suffix in (".npz", ".json"):
                    shutil.copy2(f, dst / f.name)
                    want = EXPECTED.get((s, f.name))
                    if want and sha256(dst / f.name) != want:
                        raise AssertionError(f"{s}/{f.name}: checksum differs from the datasheet")
    (out / "code").mkdir()
    for d in CODE_DIRS:
        copy_tree(ROOT / d, out / "code" / d)
    shutil.copy2(ROOT / "pyproject.toml", out / "code" / "pyproject.toml")
    stripped = strip_tree(out / "code")
    test = out / "code" / "tests" / "test_netflow.py"
    text = test.read_text(encoding="utf-8")
    if text.count("nf.__doc__.split()") != 1:
        raise AssertionError("tests/test_netflow.py: the docstring assertion was not found once")
    test.write_text(text.replace("nf.__doc__.split()", '(nf.__doc__ or "").split()'), encoding="utf-8", newline="\n")
    for rel, name in DOC_TESTS:
        drop_function(out / "code" / rel, name)
    res = out / "results"
    res.mkdir()
    shutil.copy2(ROOT / "paper" / "numbers.json", res / "numbers.json")
    for pattern in RESULT_RUNS:
        hits = sorted(ROOT.glob(pattern))
        if len(hits) != 1:
            raise FileNotFoundError(f"{pattern}: {len(hits)} matches")
        dst = res / hits[0].parent.name / hits[0].name
        dst.mkdir(parents=True)
        for f in hits[0].iterdir():
            if f.suffix == ".json":
                shutil.copy2(f, dst / f.name)
    (out / "README.md").write_text(README, encoding="utf-8")
    shutil.copy2(ROOT / "docs" / "datasets" / "TE-GvA-datasheet.md", out / "DATASHEET.md")
    shutil.copy2(ROOT / "docs" / "release" / "CITATION.cff", out / "CITATION.cff")
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    bad = [p for p in out.rglob("*") if p.is_file() and (p.suffix in FORBIDDEN or "temexd" in p.name.lower()
                                                          or "third_party" in p.parts)]
    if bad:
        raise AssertionError(f"forbidden files in the release: {bad[:5]}")
    lines = [f"{sha256(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file()]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"out": str(out), "files": len(lines) + 1, "megabytes": round(size / 1e6, 1),
                      "stripped": stripped}))

if __name__ == "__main__":
    sys.exit(main())
