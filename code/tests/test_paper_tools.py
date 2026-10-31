import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from belief_sensitivity import (
    family_cause, logit, perturbed_labels, perturbed_posterior, twin_labels)
from cpsdefense.experience.dataset import KINDS

LINK, MODULE, NONE = KINDS.index("link_fault"), KINDS.index("io_module_fault"), KINDS.index("none")


def toy_split():
    dec = {"episode_seed": np.array([1, 1, 2, 3]), "kind": np.array([NONE, LINK, MODULE, KINDS.index("disturbance")]),
           "family": np.array([True, True, True, False]), "posterior": np.array([0.6, 0.9, 1.0, 1.0])}
    cand_decision = np.repeat(np.arange(4), 2)
    J = np.array([1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0])
    J_twin = np.array([10.0, 0.0, 20.0, 1.0, 30.0, 2.0, 0.0, 0.0])
    p_trip = np.array([0.0, 0.5, 0.25, 0.75, 0.0, 1.0, 0.0, 0.0])
    p_twin = np.array([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    p = dec["posterior"][cand_decision]
    cand = {"decision": cand_decision, "J": J, "p_trip": p_trip, "J_belief": p * J + (1 - p) * J_twin,
            "p_trip_belief": p * p_trip + (1 - p) * p_twin}
    return {"n": 4, "dec": dec, "cand": cand}, J_twin, p_twin


def test_family_cause_uses_the_episode_kind_for_pre_onset_decisions():
    split, _, _ = toy_split()
    assert family_cause(split).tolist() == [1, 1, -1, 0]


def test_exact_posterior_is_reproduced_and_twin_labels_recovered():
    split, J_twin, p_twin = toy_split()
    cause = family_cause(split)
    q = perturbed_posterior(split["dec"]["posterior"], cause, 0.5, 1.0, 0.0)
    np.testing.assert_allclose(q, split["dec"]["posterior"])
    Jt, pt = twin_labels(split)
    mixed = split["dec"]["posterior"][split["cand"]["decision"]] < 1
    np.testing.assert_allclose(Jt[mixed], J_twin[mixed])
    np.testing.assert_allclose(pt[mixed], p_twin[mixed])
    assert np.isnan(Jt[~mixed]).all()
    J_b, p_b = perturbed_labels(split, cause, 0.5, 1.0, 0.0)
    np.testing.assert_allclose(J_b, split["cand"]["J_belief"])
    np.testing.assert_allclose(p_b, split["cand"]["p_trip_belief"])


def test_perturbations_move_the_posterior_as_specified():
    split, _, _ = toy_split()
    cause = family_cause(split)
    p = split["dec"]["posterior"]
    prior = 0.5
    q0 = perturbed_posterior(p, cause, prior, 0.0, 1.0)
    shifted = 1 / (1 + np.exp(-(logit(prior) + 1.0)))
    np.testing.assert_allclose(q0[:2], [shifted, shifted])
    np.testing.assert_allclose(perturbed_posterior(p, cause, prior, np.inf, 0.0)[:2], [1.0, 1.0])
    q2 = perturbed_posterior(p, cause, prior, 2.0, 0.0)
    assert q2[1] > p[1] and q2[0] > p[0]
    for T in (0.0, 0.5, 2.0, np.inf):
        for d in (-1.0, 0.0, 1.0):
            assert perturbed_posterior(p, cause, prior, T, d)[2:].tolist() == [1.0, 1.0]


def test_module_cause_uses_the_complement_of_the_link_posterior():
    dec = {"posterior": np.array([0.7]), "kind": np.array([MODULE]), "family": np.array([True]),
           "episode_seed": np.array([5])}
    cause = family_cause({"n": 1, "dec": dec})
    assert perturbed_posterior(dec["posterior"], cause, 0.5, 1.0, 1.0)[0] < 0.7
    assert perturbed_posterior(dec["posterior"], cause, 0.5, 1.0, -1.0)[0] > 0.7
    assert perturbed_posterior(dec["posterior"], cause, 0.5, np.inf, 0.0)[0] == 1.0


def test_predict_saved_refuses_to_overwrite(tmp_path, monkeypatch):
    import predict_saved
    run = tmp_path / "run"
    run.mkdir()
    (run / "predictions_test.npz").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["predict_saved.py", str(run), "--split", "test"])
    with pytest.raises(FileExistsError):
        predict_saved.main()


def test_strip_python_removes_comments_and_docstrings_only():
    from strip_comments import strip_python
    src = ('"""Module doc."""\n\nimport os  # trailing\n\n# full line\nX = "a # not a comment"\n'
           'Y = """keep\n\n# this line\n"""\n\n\ndef f():\n    """Only a docstring."""\n\n\n'
           'class A:\n    """Doc."""\n\n    def g(self):\n        # c\n        return f"{X} # kept"\n')
    out = strip_python(src)
    assert "Module doc" not in out and "Only a docstring" not in out and "Doc." not in out
    assert "# full line" not in out and "# trailing" not in out and "# c" not in out
    assert '"a # not a comment"' in out and 'f"{X} # kept"' in out
    assert 'Y = """keep\n\n# this line\n"""' in out
    assert "def f():\n    pass\n" in out
    ns = {}
    exec(compile(out, "<stripped>", "exec"), ns)
    assert ns["Y"] == "keep\n\n# this line\n" and ns["A"]().g() == "a # not a comment # kept"


def test_strip_yaml_keeps_the_document():
    import yaml
    from strip_comments import strip_yaml
    text = "# head\na: 1  # one\nb: 'x # y'\nc: \"p#q\"\nd:\n  - 2 # two\n\n\n# tail\ne: url#frag\n"
    out = strip_yaml(text)
    assert yaml.safe_load(out) == yaml.safe_load(text)
    assert "head" not in out and "one" not in out and "two" not in out and "tail" not in out
