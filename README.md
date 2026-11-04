# TE-GvA

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
