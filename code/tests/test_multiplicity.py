import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from multiplicity import holm


def test_holm_steps_down_and_stops_at_the_first_failure():
    rows = holm([("a", 0.04), ("b", 0.001), ("c", 0.02), ("d", 0.012)])
    assert [r["test"] for r in rows] == ["b", "d", "c", "a"]
    assert [r["threshold"] for r in rows] == [0.0125, 0.016667, 0.025, 0.05]
    assert [r["rejected"] for r in rows] == [True, True, True, True]
    rows = holm([("a", 0.04), ("b", 0.001), ("c", 0.03), ("d", 0.012)])
    assert [r["rejected"] for r in rows] == [True, True, False, False]


def test_holm_enters_a_recorded_zero_at_the_bootstrap_resolution():
    rows = holm([(f"t{i}", 0.0) for i in range(60)])
    assert not any(r["rejected"] for r in rows)
    assert holm([("x", 0.0)], resolution=0.001)[0]["rejected"]
