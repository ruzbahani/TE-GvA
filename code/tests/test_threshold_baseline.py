import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from threshold_baseline import (
    bt_choice, episode_folds, find, first_min, fit_lr, gate_choice, held_groups, link_posterior, load_dev,
    margin_choice, rounded)
from cpsdefense.experience.dataset import KINDS

LINK, MODULE, NONE = KINDS.index("link_fault"), KINDS.index("io_module_fault"), KINDS.index("none")


def test_load_dev_refuses_the_test_split_before_opening_anything(tmp_path):
    (tmp_path / "test.npz").write_bytes(b"not an npz")
    for split in ("test", "TEST", "other"):
        with pytest.raises(AssertionError):
            load_dev(tmp_path, split)


def test_held_groups_follow_the_playbook_test_on_distinct_channels():
    vocab = {"measurements": ["a1", "a2", "a3", "a4", "b1", "b2", "c1", "c2", "c3"],
             "groups": ["a", "b", "c"],
             "group_measurements": {"a": ["a1", "a2", "a3", "a4"], "b": ["b1", "b2"], "c": ["c1", "c2", "c3"]}}
    stale = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
                      [5, 0, 0, 0, 3, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    held, share = held_groups(stale, vocab, 0.5)
    assert held.tolist() == [[True, False, False], [False, False, False], [False, True, True],
                             [False, False, False]]
    np.testing.assert_allclose(share[2], [0.0, 1.0, 2 / 3])


def test_find_returns_the_first_matching_candidate_or_minus_one():
    tool = np.array([[0, 1, 1, -1], [2, 2, 0, 1]])
    target = np.array([["x", "a", "b", ""], ["a", "b", "b", "c"]])
    assert find(tool, target, 1, ["b", "c"]).tolist() == [2, 3]
    assert find(tool, target, 2, ["a", "x"]).tolist() == [-1, -1]


def toy_situation():
    n = 6
    return {"trace": np.zeros(n, int), "module": np.ones(n, int), "monitor": np.full(n, 2),
            "trace_pending": np.array([False, True, False, False, False, False]),
            "module_busy": np.array([False, False, False, False, True, False])}


def test_bt_choice_gathers_when_unsure_and_acts_on_the_more_likely_cause():
    s = toy_situation()
    held = np.array([True, True, True, True, True, False])
    p_link = np.array([0.6, 0.6, 0.9, 0.1, 0.1, 0.5])
    fallback, playbook = np.full(6, 4), np.full(6, 3)
    choice = bt_choice(s, held, p_link, 0.8, fallback, playbook)
    assert choice.tolist() == [0, 2, 3, 1, 2, 4]
    assert bt_choice(s, held, p_link, 0.5, fallback, playbook).tolist() == [3, 3, 3, 1, 2, 4]
    assert bt_choice(s, held, p_link, 1.01, fallback, playbook).tolist() == [0, 2, 0, 0, 0, 4]


def test_gate_choice_restricts_the_gbm_argmin_to_the_posterior_family():
    J = np.array([[3.0, 1.0, 2.0, 0.0]] * 3)
    observe = np.array([[True, True, False, False]] * 3)
    mask = np.array([[True, True, True, True]] * 3)
    held = np.array([True, True, False])
    p_link = np.array([0.6, 0.95, 0.6])
    assert gate_choice(held, p_link, 0.8, J, observe, mask).tolist() == [1, 3, 3]


def test_margin_choice_gathers_only_beyond_the_margin():
    J = np.array([[0.0, 2.0, 1.0], [1.0, 1.2, 1.5], [0.0, 5.0, 9.0], [4.0, 3.0, 0.0]])
    observe = np.array([[True, False, False], [True, False, False], [True, True, False], [True, True, False]])
    mask = np.array([[True, True, True], [True, True, True], [True, True, False], [True, True, False]])
    assert margin_choice(J, observe, mask, 0.0).tolist() == [0, 0, 0, 1]
    assert margin_choice(J, observe, mask, 0.5).tolist() == [0, 1, 0, 1]
    assert margin_choice(J, observe, mask, 5.0).tolist() == [2, 1, 0, 1]
    Jp = np.where(mask, J, np.inf)
    assert margin_choice(J, observe, mask, 0.0).tolist() == Jp.argmin(1).tolist()


def test_episode_folds_keep_episodes_whole_and_are_reproducible():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, size=500)
    for key in ("group_kfold", "seed1", "seed5"):
        tests = episode_folds(groups, key)
        assert len(tests) == 5
        rows = np.sort(np.concatenate(tests))
        assert rows.tolist() == list(range(500))
        owners = [set(groups[t].tolist()) for t in tests]
        assert all(not (a & b) for i, a in enumerate(owners) for b in owners[i + 1:])
        assert all(np.array_equal(a, b) for a, b in zip(tests, episode_folds(groups, key)))
    assert not all(np.array_equal(a, b) for a, b in zip(episode_folds(groups, "seed1"), episode_folds(groups, "seed2")))
    with pytest.raises(ValueError):
        episode_folds(groups, "random")


def test_fit_lr_recovers_a_soft_posterior():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 0.5)))
    predict = fit_lr(X, p)
    assert np.mean(np.abs(predict(X) - p)) < 0.02


def test_link_posterior_uses_the_episode_cause():
    dec = {"episode_seed": np.array([1, 1, 2, 3]), "kind": np.array([NONE, LINK, MODULE, KINDS.index("disturbance")]),
           "family": np.array([True, True, True, False]), "posterior": np.array([0.7, 0.9, 0.8, 1.0])}
    p = link_posterior({"n": 4, "dec": dec})
    np.testing.assert_allclose(p[:3], [0.7, 0.9, 0.2])
    assert np.isnan(p[3])


def test_first_min_takes_the_smallest_value_on_ties():
    curve = [{"t": 0.5, "r": 2.0}, {"t": 0.8, "r": 1.0}, {"t": 0.85, "r": 1.0}, {"t": 0.9, "r": 1.5}]
    assert first_min(curve, "r")["t"] == 0.8


def test_rounded_gives_plain_json_with_four_decimals():
    out = rounded({"a": np.float64(1.234567), "b": [np.nan, np.inf, 2], "c": np.array([0.11111, 1.0]),
                   "d": (True, np.int64(3))})
    assert out == {"a": 1.2346, "b": [None, None, 2], "c": [0.1111, 1.0], "d": [True, 3]}
