import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backup_strata import (
    DEFINITIONS, analyse, between_cause, bin_of, cell, contrast, first_actions, load_dev, rounded, summary_markdown,
    tercile_cuts)
from dataset_stats import jsonable

SWITCH = 3


def toy_split():
    p = np.array([0.8, 1.0, 1.0])
    dec = np.repeat(np.arange(3), 2)
    J = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 6.0])
    J_twin = np.array([6.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    cand = {"decision": dec, "J": J, "J_belief": p[dec] * J + (1 - p[dec]) * J_twin,
            "p_trip": np.zeros(6), "p_trip_belief": np.zeros(6)}
    return {"n": 3, "dec": {"posterior": p}, "cand": cand}


def toy_decisions():
    reg_b = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 0.0])
    reg_h = np.array([0.0, 3.0, 1.0, 2.0, 0.0, 4.0])
    regret = {"belief": {"belief": reg_b, "hindsight": reg_h},
              "hindsight": {"belief": reg_b + 1.0, "hindsight": reg_h + 1.0}}
    return {"n": 6, "cluster": np.array([1, 1, 2, 2, 3, 3]), "family": np.array([1, 1, 0, 0, 1, 1], bool),
            "backup": np.array([1, 1, 0, 0, 0, 1], bool), "v": np.array([0.0, 2.0, 0.0, 1.0, 5.0, 3.0]),
            "a": np.zeros(6), "regret": regret,
            "e1": {sc: regret[sc]["belief"] - regret[sc]["hindsight"] for sc in regret},
            "choice": {"belief": np.array([0, 1, 0, 1, 0, 0]), "hindsight": np.array([0, 0, 1, 1, 0, 1])},
            "has_both": np.array([1, 1, 1, 0, 1, 1], bool), "gather_optimal": np.array([0, 1, 0, 0, 0, 0], bool),
            "high_stakes": np.array([0, 1, 1, 0, 0, 1], bool), "high_stakes_belief": np.array([0, 1, 1, 0, 0, 1], bool),
            "mask": np.ones((6, 2), bool),
            "J_belief": np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1.0, 1.0]]),
            "tool": np.array([[SWITCH, 0], [0, 1], [SWITCH, 9], [0, 1], [0, 1], [0, SWITCH]]), "switch": SWITCH}


def test_load_dev_refuses_the_test_split(tmp_path):
    with pytest.raises(AssertionError):
        load_dev(tmp_path, ("train", "test"))
    with pytest.raises(AssertionError):
        load_dev(tmp_path, ("val", "all"))


def test_between_cause_term_and_ambiguity():
    v, a = between_cause(toy_split())
    np.testing.assert_allclose(v, [0.16 * 12.5, 0.0, 0.0])
    np.testing.assert_allclose(a, [0.16, 0.0, 0.0])


def test_bins_put_zero_apart_and_split_the_rest_in_terciles():
    score = np.array([0.0, 1e-12, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cuts = tercile_cuts(score)
    np.testing.assert_allclose(cuts, np.quantile(np.arange(1.0, 7.0), [1 / 3, 2 / 3]))
    assert bin_of(score, cuts).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert bin_of(np.array([2.0, 3.0]), [2.0, 3.0]).tolist() == [2, 3]


@pytest.mark.filterwarnings("ignore:only 3 clusters")
@pytest.mark.filterwarnings("ignore:.*bootstrap replicates were undefined")
def test_cell_and_contrast():
    d = toy_decisions()
    c = cell(d, np.ones(6, bool), n_boot=200)
    assert (c["n"], c["episodes"]) == (6, 3)
    assert c["E1"]["belief_scoring"]["point"] == pytest.approx(-7 / 6)
    assert c["E1"]["hindsight_scoring"]["point"] == pytest.approx(-7 / 6)
    assert c["E1"]["belief_scoring"]["lo"] <= -7 / 6 <= c["E1"]["belief_scoring"]["hi"]
    assert c["regret"]["belief_scoring"] == {"gbm_belief": pytest.approx(0.5), "gbm_hindsight": pytest.approx(5 / 3)}
    assert c["gather_optimal_belief_share"] == pytest.approx(1 / 5)
    assert c["high_stakes_share"] == pytest.approx(0.5)
    assert cell(d, np.zeros(6, bool), n_boot=200) == {"n": 0, "episodes": 0}
    k = contrast(d, ~d["backup"], d["backup"], n_boot=200)
    assert k["belief_scoring"]["point"] == pytest.approx(-1 / 3 + 2)
    assert set(k["belief_scoring"]) >= {"point", "lo", "hi", "p_two_sided"}


def test_first_actions():
    d = toy_decisions()
    f = first_actions(d, np.ones(6, bool))
    assert f["differ_share"] == pytest.approx(0.5)
    assert f["E1_given_differ"]["belief_scoring"] == pytest.approx(-7 / 3)
    assert f["E1_zero_where_same"]
    b = first_actions(d, d["backup"])
    assert b["switch_belief_optimal_n"] == 1 and b["switch_belief_optimal_share"] == pytest.approx(1 / 3)
    assert b["switch_belief_co_optimal_share"] == pytest.approx(2 / 3)
    assert b["gbm_switch_share"] == {"gbm_belief": pytest.approx(1 / 3), "gbm_hindsight": pytest.approx(2 / 3)}


@pytest.mark.filterwarnings("ignore")
def test_analyse_and_summary():
    d = toy_decisions()
    d["a"] = d["v"] / 10
    d["subsets"] = [("all", np.ones(6, bool)), ("backup_available", d["backup"]), ("backup_unavailable", ~d["backup"])]
    res = analyse(d, np.array([1.5, 2.5]), threshold=1.5, n_boot=50)
    assert [c["n"] for c in res["v_bins"]["bins"].values()] == [2, 1, 1, 2]
    assert [c["n"] for c in res["v_bins_reference_cuts"]["bins"].values()] == [2, 1, 1, 2]
    assert res["first_actions"]["v_above_median/backup_available"]["n"] == 2
    flat = {m: {"chosen": {"center": True}} for m in ("belief", "hindsight")}
    res = jsonable({"sets": {"toy": {**res, "flat": flat}}, "caveats": ["c"], "v_median_threshold": 1.5})
    md = summary_markdown(res)
    assert "| toy | no backup minus backup (joint) |" in md and "v_above_median/backup_unavailable" in md


def test_rounded():
    out = rounded({"a": np.float64(1.234567), "b": [np.nan, 2, True], "c": np.array([0.123456]), 3: (np.inf,)})
    assert out == {"a": 1.2346, "b": [None, 2, True], "c": [0.1235], "3": [None]}
    assert rounded(DEFINITIONS) == DEFINITIONS
