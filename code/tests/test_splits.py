import numpy as np
import pytest

from cpsdefense.data.splits import (
    events_from_labels,
    grouped_folds,
    run_bounds,
    segment_by_gaps,
    time_split_with_purge,
    window_starts,
)


def test_run_bounds():
    starts, stops = run_bounds([3, 3, 1, 1, 1, 3])
    assert starts.tolist() == [0, 2, 5]
    assert stops.tolist() == [2, 5, 6]
    starts, stops = run_bounds([])
    assert starts.size == 0 and stops.size == 0


def test_segment_by_gaps_basic():
    t = [0, 1, 2, 10, 11, 12, 13, 20]
    assert segment_by_gaps(t, 1).tolist() == [0, 0, 0, 1, 1, 1, 1, 2]


def test_segment_by_gaps_step_equal_to_max_gap_is_not_a_gap():
    assert segment_by_gaps([0, 2, 4, 7], 2).tolist() == [0, 0, 0, 1]


def test_segment_by_gaps_edge_cases():
    assert segment_by_gaps([], 1).tolist() == []
    assert segment_by_gaps([5.0], 1).tolist() == [0]
    assert segment_by_gaps([0, 0, 1], 1).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        segment_by_gaps([0, 2, 1], 1)
    with pytest.raises(ValueError):
        segment_by_gaps([0, np.nan, 2], 1)
    with pytest.raises(ValueError):
        segment_by_gaps(np.zeros((2, 2)), 1)


def test_events_from_labels_basic():
    y = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    assert events_from_labels(y) == [(1, 3), (5, 6), (7, 10)]


def test_events_from_labels_edges():
    assert events_from_labels([]) == []
    assert events_from_labels([0, 0, 0]) == []
    assert events_from_labels([1, 1, 1]) == [(0, 3)]
    assert events_from_labels([1, 0, 1]) == [(0, 1), (2, 3)]
    assert events_from_labels([0, 2, 5, 0]) == [(1, 3)]


def test_events_from_labels_cut_at_segments():
    y = [0, 1, 1, 1, 1, 0]
    seg = [0, 0, 0, 1, 1, 1]
    assert events_from_labels(y, seg) == [(1, 3), (3, 5)]
    with pytest.raises(ValueError):
        events_from_labels(y, seg[:-1])


def test_time_split_sizes_and_purge_gap():
    t = np.arange(1000.0)
    codes = time_split_with_purge(t, (0.8, 0.2), purge=50)
    val = np.flatnonzero(codes == 1)
    train = np.flatnonzero(codes == 0)
    assert val[0] == 800 and val[-1] == 999 and val.size == 200
    assert t[val[0]] - t[train[-1]] >= 50
    assert t[val[0]] - t[train[-1]] == 50
    assert np.all(codes[train[-1] + 1:val[0]] == -1)
    assert train[0] == 0 and np.all(np.diff(train) == 1)


def test_time_split_without_purge_and_three_way():
    codes = time_split_with_purge(np.arange(10.0), (0.8, 0.2))
    assert codes.tolist() == [0] * 8 + [1] * 2
    codes = time_split_with_purge(np.arange(100.0), (0.6, 0.2, 0.2), purge=5)
    for k in (1, 2):
        first = np.flatnonzero(codes == k)[0]
        last_prev = np.flatnonzero(codes == k - 1)[-1]
        assert first - last_prev >= 5
    assert set(np.unique(codes)) == {-1, 0, 1, 2}


def test_time_split_purge_is_in_time_units_not_rows():
    t = np.r_[np.arange(80.0), 1000 + np.arange(20.0)]
    codes = time_split_with_purge(t, (0.8, 0.2), purge=50)
    assert np.all(codes[:80] == 0)
    assert np.all(codes[80:] == 1)


def test_time_split_errors():
    t = np.arange(100.0)
    with pytest.raises(ValueError):
        time_split_with_purge(t, (0.7, 0.2))
    with pytest.raises(ValueError):
        time_split_with_purge(t, (1.2, -0.2))
    with pytest.raises(ValueError):
        time_split_with_purge(t, (0.8, 0.2), purge=-1)
    with pytest.raises(ValueError):
        time_split_with_purge(np.arange(2.0), (0.9, 0.1))
    with pytest.raises(ValueError):
        time_split_with_purge(t, (0.8, 0.2), purge=500)
    with pytest.raises(ValueError):
        time_split_with_purge(t[::-1], (0.8, 0.2))


def test_window_starts_respects_segments():
    seg = [0] * 5 + [1] * 3 + [2] * 7
    starts = window_starts(seg, 3, 1)
    assert starts.tolist() == [0, 1, 2, 5, 8, 9, 10, 11, 12]
    assert window_starts(seg, 3, 2).tolist() == [0, 2, 5, 8, 10, 12]
    assert window_starts(seg, 6, 1).tolist() == [8, 9]
    assert window_starts(seg, 8, 1).size == 0


def test_window_starts_repeated_id_is_a_new_segment():
    seg = [0, 0, 0, 1, 0, 0, 0]
    assert window_starts(seg, 3, 1).tolist() == [0, 4]


def test_window_starts_never_cross_random_boundaries():
    rng = np.random.default_rng(0)
    seg = np.repeat(np.arange(40), rng.integers(1, 30, size=40))
    for length, stride in [(1, 1), (5, 1), (7, 3), (25, 4)]:
        starts = window_starts(seg, length, stride)
        assert np.all(starts + length <= seg.size)
        for s in starts:
            assert np.all(seg[s:s + length] == seg[s])
        expected = sum(max(0, (n - length) // stride + 1) for n in np.bincount(seg))
        assert starts.size == expected


def test_window_starts_errors_and_empty():
    assert window_starts([], 3).size == 0
    with pytest.raises(ValueError):
        window_starts([0, 0], 0)
    with pytest.raises(ValueError):
        window_starts([0, 0], 1, stride=0)


def _check_partition(groups, folds):
    groups = np.asarray(groups)
    seen = np.zeros(groups.size, int)
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        assert train.size + test.size == groups.size
        assert np.intersect1d(np.unique(groups[train]), np.unique(groups[test])).size == 0
        seen[test] += 1
    assert np.all(seen == 1)


def test_grouped_folds_ordered_blocks():
    groups = np.repeat(["e3", "e1", "e2", "e5", "e4", "e6"], 10)
    folds = grouped_folds(groups, 3)
    _check_partition(groups, folds)
    test_groups = [list(dict.fromkeys(groups[test])) for _, test in folds]
    assert test_groups == [["e3", "e1"], ["e2", "e5"], ["e4", "e6"]]


def test_grouped_folds_shuffled_and_reproducible():
    groups = np.repeat(np.arange(12), 5)
    a = grouped_folds(groups, 4, seed=1)
    b = grouped_folds(groups, 4, seed=1)
    c = grouped_folds(groups, 4, seed=2)
    _check_partition(groups, a)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a, c))
    assert all(test.size == 15 for _, test in a)


def test_grouped_folds_dominant_group_keeps_folds_non_empty():
    groups = np.r_[np.zeros(1000, int), np.arange(1, 4)]
    folds = grouped_folds(groups, 3)
    _check_partition(groups, folds)
    assert all(test.size > 0 for _, test in folds)


def test_grouped_folds_errors():
    with pytest.raises(ValueError):
        grouped_folds([0, 0, 1], 3)
    with pytest.raises(ValueError):
        grouped_folds([0, 1, 2], 1)
