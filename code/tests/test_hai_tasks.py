import copy
import json

import numpy as np
import pytest

from cpsdefense.data import hai, hai_tasks
from cpsdefense.data.hai_tasks import CONTRACT_FIELDS, TIME_FIELDS

COLS = ["A_x", "A_d", "B_lvl", "B_y", "C_p"]
CFG = {
    "splits": ["train", "val", "dev_test"],
    "seed": 3,
    "sources": {"A": ["A_"], "B": ["B_"], "C": ["C_"]},
    "windows": {"window_s": 5, "sample_period_s": 1, "stats": ["last", "mean", "std", "slope"]},
    "decisions": {"stride_s": 5, "recent_windows": 3, "stale_lags": [8, 12]},
    "scaling": {"floor_frac": 0.01, "clip": 20},
    "timing": {
        "default": {"delay_median_s": 3, "delay_sigma": 0.6, "lifetime_s": 20, "lifetime_jitter": 0.2,
                    "outage_p_start": 0.02, "outage_mean_windows": 4},
        "sources": {"C": {"delay_median_s": 6, "lifetime_s": 30}},
        "retraction": {"rate": 0.1, "delay_median_s": 10, "delay_sigma": 0.5, "replay_payload": True,
                       "replay_lag_windows": [2, 6]},
    },
    "derived": {"enabled": True, "name": "plant", "stats": ["mean", "slope"], "delay_median_s": 1, "delay_sigma": 0.3},
    "targets": {
        "horizon_s": 120,
        "questions": {
            "a_switch": {"source": "A", "all": {"type": "discrete_change", "columns": ["A_d"]}},
            "b_level": {"source": "B", "all": {"type": "hysteresis", "column": "B_lvl", "half_width_iqr": 0.25}},
        },
        "score": {"name": "b_bin", "column": "B_lvl", "lead_s": 20, "n_bins": 4},
    },
    "discovery": {"max_levels": 20},
    "min_events": {"train": 5, "val": 2},
    "save": {"compress": False},
}
VERSION = "23.05"


def _synthetic(lengths=(1302, 1100), seed=0, attack=((700, 760),)):
    rng = np.random.default_rng(seed)
    X, ts, seg = [], [], []
    t0 = np.datetime64("2022-01-01T00:00:00", "s")
    offset = 0
    for s, n in enumerate(lengths):
        tt = np.arange(n)
        toggle = (np.cumsum(rng.random(n) < 1 / 40) % 2).astype(np.float64)
        level = 50 + 20 * np.sin(2 * np.pi * tt / 180 + s) + rng.normal(0, 2, n)
        cols = [rng.normal(size=n), toggle, level, rng.normal(0, 1, n).cumsum(), rng.normal(5, 1, n)]
        X.append(np.stack(cols, 1).astype(np.float32))
        ts.append(t0 + np.timedelta64(offset, "s") + tt.astype("timedelta64[s]"))
        seg.append(np.full(n, s))
        offset += n + 100
    y = np.zeros(sum(lengths), np.int8)
    for a, b in attack:
        y[a:b] = 1
    return {"X": np.concatenate(X), "y": y, "timestamp": np.concatenate(ts).astype("datetime64[ns]"),
            "segment": np.concatenate(seg).astype(np.int64), "columns": list(COLS)}


@pytest.fixture(scope="module")
def built():
    train = _synthetic(seed=0)
    spec = hai_tasks.resolve_spec(CFG, VERSION, COLS)
    params = hai_tasks.fit_params(train, spec)
    data = _synthetic(seed=1)
    entropy = hai_tasks.seed_entropy(CFG["seed"], VERSION, "val")
    arrays, report = hai_tasks.build_split(data, spec, params, entropy)
    return {"train": train, "spec": spec, "params": params, "data": data, "entropy": entropy,
            "arrays": arrays, "report": report}


def _hyst_loop(x, lo, hi, seg):
    out = np.zeros(len(x), np.int64)
    s = 0
    for i in range(len(x)):
        if i == 0 or seg[i] != seg[i - 1]:
            s = 1 if x[i] >= hi else 0 if x[i] <= lo else int(x[i] >= (lo + hi) / 2)
        elif x[i] >= hi:
            s = 1
        elif x[i] <= lo:
            s = 0
        out[i] = s
    return out


def _brute_event(values, t, seg, r, horizon):
    n = len(values)
    changed = lambda j: np.any(values[j] != values[j - 1])
    j = r + 1
    while j < n and seg[j] == seg[r]:
        if changed(j):
            break
        j += 1
    last = j - 1 if j >= n or seg[j] != seg[r] else None
    if last is None and t[j] - t[r] <= horizon:
        tte, obs = t[j] - t[r], True
    else:
        if last is None:
            while j < n and seg[j] == seg[r]:
                j += 1
            last = j - 1
        tte, obs = min(horizon, t[last] - t[r]), False
    since = np.nan
    j = r
    while j > 0 and seg[j - 1] == seg[r]:
        if changed(j):
            since = t[r] - t[j]
            break
        j -= 1
    return tte, obs, since


def _equal(a, b):
    return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def test_window_stats_match_brute_force():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3)).astype(np.float32)
    row0 = np.array([0, 10, 25])
    out = hai_tasks.window_stats(X, row0, 5, ["first", "last", "mean", "std", "min", "max", "slope"])
    assert out.shape == (3, 21)
    for w, a in enumerate(row0):
        block = X[a:a + 5].astype(np.float64)
        for j in range(3):
            slope = np.polyfit(np.arange(5), block[:, j], 1)[0]
            expected = [block[0, j], block[-1, j], block[:, j].mean(), block[:, j].std(),
                        block[:, j].min(), block[:, j].max(), slope]
            np.testing.assert_allclose(out[w, j::3], expected, rtol=1e-10, atol=1e-12)


def test_windows_stay_inside_segments_and_check_cadence():
    t = np.r_[np.arange(12.0), 100 + np.arange(7.0)]
    seg = np.r_[np.zeros(12), np.ones(7)].astype(int)
    win = hai_tasks.windows(t, seg, 5)
    assert win["row0"].tolist() == [0, 5, 12]
    assert win["k"].tolist() == [0, 1, 0] and win["first"].tolist() == [0, 0, 2]
    assert win["seg_stop"].tolist() == [12, 12, 19]
    t_bad = t.copy()
    t_bad[3:] += 0.5
    with pytest.raises(ValueError, match="cadence"):
        hai_tasks.windows(t_bad, seg, 5)


def test_group_sources_requires_exactly_one_source():
    assert hai_tasks.group_sources(COLS, CFG["sources"]) == {"A": [0, 1], "B": [2, 3], "C": [4]}
    with pytest.raises(ValueError, match="exactly one"):
        hai_tasks.group_sources(COLS + ["D_q"], CFG["sources"])
    with pytest.raises(ValueError, match="exactly one"):
        hai_tasks.group_sources(COLS, {**CFG["sources"], "AA": ["A_x"]})


def test_resolve_spec_rejects_bad_settings():
    for key, value in (("decisions.stride_s", 7), ("decisions.stale_lags", [2]), ("windows.stats", ["median"])):
        cfg = copy.deepcopy(CFG)
        section, name = key.split(".")
        cfg[section][name] = value
        with pytest.raises(ValueError):
            hai_tasks.resolve_spec(cfg, VERSION, COLS)
    cfg = copy.deepcopy(CFG)
    cfg["targets"]["questions"]["a_switch"] = {"source": "A", "22.04": {"type": "discrete_change", "columns": ["A_d"]}}
    with pytest.raises(ValueError, match="no definition"):
        hai_tasks.resolve_spec(cfg, VERSION, COLS)


def test_heldout_test_split_is_refused(tmp_path):
    for call in (lambda: hai_tasks.check_split("test"),
                 lambda: hai_tasks.seed_entropy(0, VERSION, "test"),
                 lambda: hai_tasks.load_tasks(VERSION, "test", root=tmp_path)):
        with pytest.raises(ValueError, match="held-out"):
            call()
    cfg = {**copy.deepcopy(CFG), "splits": ["train", "test"]}
    with pytest.raises(ValueError, match="held-out"):
        hai_tasks.prepare_version(VERSION, config=cfg, out_root=tmp_path, hai_root=tmp_path / "missing")


def test_contract_fields_shapes_and_padding(built):
    a, spec = built["arrays"], built["spec"]
    n, m = a["present"].shape
    dims = hai_tasks.payload_dims(spec)
    assert m == len(hai_tasks.source_names(spec)) * len(spec["lags"])
    assert a["payload"].shape == (n, m, max(dims)) and a["payload"].dtype == np.float32
    assert a["decision_time"].dtype == np.float64 and a["decision_time"].shape == (n,)
    assert a["segment"].dtype == np.int64
    for field in ("item_id", "source", "kind"):
        assert a[field].dtype == np.int64 and a[field].shape == (n, m)
    assert a["present"].dtype == bool
    for field in TIME_FIELDS:
        assert a[field].dtype == np.float64 and a[field].shape == (n, m)
    pad = ~a["present"]
    assert pad.any() and a["present"].any()
    assert (a["item_id"][pad] == -1).all() and (a["source"][pad] == -1).all() and (a["kind"][pad] == -1).all()
    assert all(np.isnan(a[f][pad]).all() for f in TIME_FIELDS)
    assert (a["payload"][pad] == 0).all()
    for s, dim in enumerate(dims):
        rows = a["source"] == s
        assert (a["payload"][rows][:, dim:] == 0).all()
        assert np.abs(a["payload"][rows][:, :dim]).sum() > 0
    ids = a["item_id"][a["present"]]
    assert (ids >= 0).all() and (ids < a["items_delivered"].size).all() and a["items_delivered"][ids].all()
    assert not hai_tasks.valid_at(a, a["decision_time"])[pad].any()


def test_decisions_skip_context_and_items_never_after_tau(built):
    a, spec, data = built["arrays"], built["spec"], built["data"]
    t = hai.to_seconds(data["timestamp"])
    starts = {s: t[data["segment"] == s][0] for s in np.unique(data["segment"])}
    context = (max(spec["lags"]) + 1) * spec["window_s"]
    seg_start = np.array([starts[s] for s in a["segment"]])
    assert (a["decision_time"] - seg_start >= context - 1).all()
    assert np.allclose(np.diff(a["decision_time"])[np.diff(a["segment"]) == 0], CFG["decisions"]["stride_s"])
    tau = np.broadcast_to(a["decision_time"][:, None], a["present"].shape)
    p = a["present"]
    assert (a["event_time"][p] <= tau[p]).all()
    assert (a["event_start"][p] >= seg_start[:, None].repeat(p.shape[1], 1)[p]).all()
    assert (a["arrival_time"][p] > a["event_time"][p]).all()
    lag = np.tile(spec["lags"], len(hai_tasks.source_names(spec)))
    valid = hai_tasks.valid_at(a, a["decision_time"])
    assert not valid[:, lag == 0].any()
    assert (a["arrival_time"][:, lag == 0][p[:, lag == 0]] > tau[:, lag == 0][p[:, lag == 0]]).all()
    assert spec["stale_lags_always_stale"]
    assert not valid[:, lag >= min(CFG["decisions"]["stale_lags"])].any()
    assert valid.any(axis=1).all()


def test_same_item_is_identical_in_every_decision(built):
    a = built["arrays"]
    p = a["present"]
    ids = a["item_id"][p]
    for field, table in (("arrival_time", "items_arrival_time"), ("valid_until", "items_valid_until"),
                         ("retracted_at", "items_retracted_at"), ("event_time", "items_event_time"),
                         ("event_start", "items_event_start"), ("source", "items_source"), ("kind", "items_kind")):
        assert _equal(a[field][p], a[table][ids]), field
    first = {}
    payload = a["payload"][p]
    for i, item in enumerate(ids):
        j = first.setdefault(item, i)
        assert np.array_equal(payload[i], payload[j])


def test_timing_is_deterministic_and_seeded(built):
    again, _ = hai_tasks.build_split(built["data"], built["spec"], built["params"], built["entropy"])
    for key, value in built["arrays"].items():
        assert _equal(value, again[key]), key
    other, _ = hai_tasks.build_split(built["data"], built["spec"], built["params"],
                                     hai_tasks.seed_entropy(CFG["seed"] + 1, VERSION, "val"))
    assert not _equal(other["arrival_time"], built["arrays"]["arrival_time"])
    assert _equal(other["time_to_event"], built["arrays"]["time_to_event"])


def test_markov_outage_matches_its_parameters():
    rng = np.random.default_rng(0)
    n = 400_000
    first = np.zeros(n, bool)
    first[0] = True
    down = hai_tasks.markov_outage(rng.random(n), first, 0.01, 0.25)
    assert abs(down.mean() - 0.01 / 0.26) < 0.004
    edges = np.flatnonzero(np.diff(np.r_[0, down.astype(int), 0]))
    assert abs(np.mean(edges[1::2] - edges[::2]) - 4) < 0.2
    reset = np.zeros(10, bool)
    reset[[0, 5]] = True
    assert not hai_tasks.markov_outage(np.full(10, 0.5), reset, 0.0, 0.5).any()


def test_dropped_windows_are_absent(built):
    a = built["arrays"]
    dropped = np.flatnonzero(~a["items_delivered"])
    assert dropped.size > 0
    assert not np.isin(a["item_id"], dropped).any()
    assert all(np.isnan(a[f][dropped]).all() for f in ("items_arrival_time", "items_valid_until"))


def test_derived_items_inherit_minimum_validity_and_retraction(built):
    a, spec = built["arrays"], built["spec"]
    S = len(spec["sources"])
    derived = np.flatnonzero((a["items_kind"] == 1) & a["items_delivered"])
    par = a["items_parents"][derived]
    assert (par >= 0).all() and a["items_delivered"][par].all()
    assert (a["items_source"][par] == np.arange(S)).all()
    assert (a["items_event_time"][par] == a["items_event_time"][derived][:, None]).all()
    arr, vu, rt = a["items_arrival_time"], a["items_valid_until"], a["items_retracted_at"]
    assert (arr[derived] >= arr[par].max(axis=1)).all()
    assert (vu[derived] == vu[par].min(axis=1)).all()
    assert (rt[derived] == rt[par].min(axis=1)).all()
    retracted_input = np.isfinite(rt[par]).any(axis=1)
    assert retracted_input.any() and np.isfinite(rt[derived][retracted_input]).all()
    assert not np.isfinite(rt[derived][~retracted_input]).any()
    lo, hi = np.nanmin(arr[derived]) - 10, np.nanmax(vu[derived]) + 10
    for t in np.linspace(lo, hi, 400):
        valid = lambda i: (arr[i] <= t) & (t < vu[i]) & (t < rt[i])
        vd = valid(derived)
        assert valid(par)[vd].all()
    windows_missing_input = ~a["items_delivered"][a["items_parents"][a["items_kind"] == 1]].all(axis=1)
    assert windows_missing_input.any()
    assert not a["items_delivered"][a["items_kind"] == 1][windows_missing_input].any()


def test_derived_payload_is_computed_from_delivered_inputs(built):
    a, spec, params = built["arrays"], built["spec"], built["params"]
    dims = hai_tasks.payload_dims(spec)
    S = len(spec["sources"])
    p = a["present"] & (a["kind"] == 1)
    n_idx, m_idx = np.nonzero(p)
    for n, m in list(zip(n_idx, m_idx))[:200]:
        parents = a["items_parents"][a["item_id"][n, m]]
        blocks = []
        for s in range(S):
            where = np.flatnonzero(a["item_id"][n] == parents[s])
            assert where.size == 1
            blocks.append(a["payload"][n, where[0], :dims[s]][None, :])
        raw = hai_tasks.derived_raw(blocks, spec)
        z, _ = hai_tasks.apply_scaler(raw, params["scalers"]["plant"], CFG["scaling"]["clip"])
        np.testing.assert_allclose(a["payload"][n, m, :dims[S]], z[0], rtol=1e-5, atol=1e-5)


def test_retracted_items_replay_an_earlier_window(built):
    a, spec, params, data = built["arrays"], built["spec"], built["params"], built["data"]
    rep = np.flatnonzero(a["items_replayed_from"] >= 0)
    assert rep.size > 0
    assert np.isfinite(a["items_retracted_at"][rep]).all()
    src = a["items_replayed_from"][rep]
    assert (a["items_source"][src] == a["items_source"][rep]).all()
    assert (a["items_event_time"][src] < a["items_event_time"][rep]).all()
    t = hai.to_seconds(data["timestamp"])
    L = spec["window_len"]
    dims = hai_tasks.payload_dims(spec)
    hits = 0
    for item, origin in zip(rep, src):
        where = np.argwhere(a["item_id"] == item)
        if not where.size:
            continue
        s = a["items_source"][item]
        name = spec["sources"][s]
        row0 = int(np.searchsorted(t, a["items_event_start"][origin]))
        raw = hai_tasks.window_stats(data["X"][:, spec["source_index"][name]], np.array([row0]), L, spec["stats"])
        z, _ = hai_tasks.apply_scaler(raw, params["scalers"][name], CFG["scaling"]["clip"])
        n, m = where[0]
        np.testing.assert_allclose(a["payload"][n, m, :dims[s]], z[0], rtol=1e-6, atol=1e-6)
        hits += 1
    assert hits > 0


def test_changing_future_data_leaves_evidence_unchanged(built):
    data, spec, params, entropy = built["data"], built["spec"], built["params"], built["entropy"]
    a = built["arrays"]
    t = hai.to_seconds(data["timestamp"])
    seg0 = a["decision_time"][a["segment"] == 0]
    cut = seg0[seg0.size // 2]
    future = t > cut
    rng = np.random.default_rng(5)
    for perturb in ("offset", "roll"):
        changed = copy.deepcopy(data)
        if perturb == "offset":
            changed["X"][future] += rng.normal(0, 50, size=changed["X"][future].shape).astype(np.float32)
        else:
            changed["X"][future] = np.roll(changed["X"][future], 37, axis=0)
        b, _ = hai_tasks.build_split(changed, spec, params, entropy)
        early = a["decision_time"] <= cut
        for field in CONTRACT_FIELDS:
            assert _equal(a[field][early], b[field][early]), (perturb, field)
        assert not np.array_equal(a["payload"][~early], b["payload"][~early])
        target_changed = ~np.isclose(a["time_to_event"][early], b["time_to_event"][early]) | (
            a["event_observed"][early] != b["event_observed"][early])
        assert target_changed.any()


def test_targets_match_brute_force(built):
    a, data, params, spec = built["arrays"], built["data"], built["params"], built["spec"]
    t = hai.to_seconds(data["timestamp"])
    seg = data["segment"]
    H = CFG["targets"]["horizon_s"]
    band = params["questions"]["b_level"]
    series = {
        "a_switch": data["X"][:, [1]].astype(np.float64),
        "b_level": _hyst_loop(data["X"][:, 2].astype(np.float64), band["lo"], band["hi"], seg)[:, None],
    }
    assert a["tau_row"].tolist() == [int(np.searchsorted(t, x)) for x in a["decision_time"]]
    for q, name in enumerate(spec["questions"]):
        values = series[name]
        for i, r in enumerate(a["tau_row"]):
            tte, obs, since = _brute_event(values, t, seg, r, H)
            assert a["time_to_event"][i, q] == tte and a["event_observed"][i, q] == obs, (name, i)
            assert _equal(np.array([a["time_since_event"][i, q]]), np.array([since])), (name, i)
        assert (a["time_to_event"][:, q] > 0).all()
        assert a["event_observed"][:, q].any() and (~a["event_observed"][:, q]).any()
    assert (a["state_at_tau"][:, 1] == series["b_level"][a["tau_row"], 0]).all()
    assert (a["state_at_tau"][:, 0] == data["X"][a["tau_row"], 1]).all()
    edges = params["score"]["edges"]
    x = data["X"][:, 2].astype(np.float64)
    for i, r in enumerate(a["tau_row"]):
        ahead = r + 20
        expected = -1 if ahead >= len(x) or seg[ahead] != seg[r] else sum(x[ahead] >= e for e in edges)
        assert a["score_bin"][i] == expected
        assert a["score_bin_now"][i] == sum(x[r] >= e for e in edges)
        stop = r + 1
        while stop < len(x) and seg[stop] == seg[r] and stop <= r + H:
            stop += 1
        assert a["attack_in_horizon"][i] == int(data["y"][r + 1:stop].any())
        assert a["attack_at_tau"][i] == data["y"][r]
    assert a["attack_in_horizon"].any()


def test_hysteresis_state_matches_loop():
    rng = np.random.default_rng(1)
    x = np.round(rng.normal(size=5000).cumsum() % 7, 1)
    seg = np.repeat(np.arange(5), 1000)
    for lo, hi in ((2.0, 4.0), (3.0, 3.0), (0.0, 6.9)):
        assert np.array_equal(hai_tasks.hysteresis_state(x, lo, hi, seg), _hyst_loop(x, lo, hi, seg))


def test_event_targets_edges():
    t = np.arange(20.0)
    seg_start, seg_stop = np.zeros(4, int), np.full(4, 20)
    events = np.array([5, 15])
    tte, obs, since = hai_tasks.event_targets(t, events, np.array([0, 4, 5, 16]), seg_start, seg_stop, 10)
    assert tte.tolist() == [5, 1, 10, 3] and obs.tolist() == [True, True, True, False]
    assert np.isnan(since[:2]).all() and since[2:].tolist() == [0, 1]
    tte, obs, _ = hai_tasks.event_targets(t, events, np.array([4]), seg_start[:1], seg_stop[:1], 0.5)
    assert tte.tolist() == [0.5] and not obs[0]
    tte, obs, _ = hai_tasks.event_targets(t, np.zeros(0, int), np.array([3]), seg_start[:1], seg_stop[:1], 100)
    assert tte.tolist() == [16] and not obs[0]


def test_scalers_are_fitted_on_train_only(built):
    train, spec, params = built["train"], built["spec"], built["params"]
    shifted = copy.deepcopy(built["data"])
    shifted["X"] *= 3
    again = hai_tasks.fit_params(train, spec)
    assert json.dumps(again, sort_keys=True) == json.dumps(params, sort_keys=True)
    t = hai.to_seconds(train["timestamp"])
    win = hai_tasks.windows(t, train["segment"], spec["window_len"])
    cols = spec["source_index"]["B"]
    raw = hai_tasks.window_stats(train["X"][:, cols], win["row0"], spec["window_len"], spec["stats"])
    z, _ = hai_tasks.apply_scaler(raw, params["scalers"]["B"], 1e9)
    floor = CFG["scaling"]["floor_frac"] * np.tile(train["X"][:, cols].astype(np.float64).std(axis=0), len(spec["stats"]))
    unfloored = raw.std(axis=0) > floor
    assert unfloored.any()
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0)[unfloored], 1, atol=1e-4)
    assert (z.std(axis=0) <= 1 + 1e-4).all()
    b, _ = hai_tasks.build_split(shifted, spec, params, built["entropy"])
    assert not np.array_equal(b["payload"], built["arrays"]["payload"])


def test_prepare_version_writes_contract_files(tmp_path, monkeypatch):
    splits = {"train": _synthetic(seed=0), "val": _synthetic(seed=1), "dev_test": _synthetic(seed=2)}
    calls = []

    def fake_load_split(version, split, root=None, columns=None):
        calls.append(split)
        return copy.deepcopy(splits[split])

    monkeypatch.setattr(hai, "load_split", fake_load_split)
    monkeypatch.setattr(hai, "load_meta", lambda version, root=None: {"prepared_utc": "synthetic"})
    meta = hai_tasks.prepare_version(VERSION, config=copy.deepcopy(CFG), out_root=tmp_path, verbose=False)
    assert "test" not in calls and set(calls) == {"train", "val", "dev_test"}
    assert meta["semi_synthetic"] and "SEMI-SYNTHETIC" in meta["semi_synthetic_note"]
    assert not meta["heldout_test_used"]
    assert meta["sources"] == ["A", "B", "C", "plant"] and meta["kinds"] == list(hai_tasks.KINDS)
    assert meta["payload_dim"] == [8, 8, 4, 6]
    assert meta["discrete_columns_train"]["A"][0]["column"] == "A_d"
    on_disk = hai_tasks.load_meta(VERSION, root=tmp_path)
    assert on_disk["reports"]["val"]["n_decisions"] == meta["reports"]["val"]["n_decisions"]
    spec = hai_tasks.resolve_spec(CFG, VERSION, COLS)
    params = hai_tasks.fit_params(splits["train"], spec)
    assert json.dumps(meta["scalers"], sort_keys=True) == json.dumps(params["scalers"], sort_keys=True)
    for split, data in splits.items():
        loaded = hai_tasks.load_tasks(VERSION, split, root=tmp_path)
        direct, _ = hai_tasks.build_split(data, spec, params, meta["seed_entropy"][split])
        assert set(CONTRACT_FIELDS) <= set(loaded)
        for key in direct:
            assert _equal(loaded[key], direct[key]), (split, key)
    assert sorted(p.name for p in (tmp_path / "hai_tasks" / VERSION).iterdir()) == [
        "dev_test.npz", "meta.json", "train.npz", "val.npz"]


def _prepared(version):
    try:
        return hai_tasks.load_meta(version)
    except FileNotFoundError:
        pytest.skip(f"HAI tasks {version} not prepared")


@pytest.mark.data
@pytest.mark.parametrize("version", ["23.05", "22.04"])
def test_prepared_meta_is_consistent(version):
    meta = _prepared(version)
    assert meta["semi_synthetic"] and not meta["heldout_test_used"]
    assert meta["splits"] == list(hai_tasks.DEV_SPLITS)
    files = sorted(p.name for p in hai_tasks.tasks_dir(version).iterdir())
    assert files == ["dev_test.npz", "meta.json", "train.npz", "val.npz"]
    assert meta["sources"] == ["P1", "P2", "P3", "P4", "plant"]
    hai_meta = hai.load_meta(version)
    assert sorted(c for cols in meta["source_columns"].values() for c in cols) == sorted(hai_meta["feature_columns"])
    if version == "23.05":
        assert all(c in meta["source_columns"]["P1"] for c in hai_meta["feature_columns"] if c.startswith("x100"))
        for q in meta["question_names"]:
            assert meta["reports"]["train"]["questions"][q]["events_in_split"] >= 200
            assert meta["reports"]["val"]["questions"][q]["events_in_split"] >= 50
    for report in meta["reports"].values():
        assert report["invalid_at_tau_by_lag"]["0"] == 1.0
        assert 0.3 < report["invalid_at_tau"] < 0.9


@pytest.mark.data
@pytest.mark.slow
def test_prepared_dev_test_is_reproducible_and_ignores_future_data():
    version = "23.05"
    meta = _prepared(version)
    train = hai.load_split(version, "train")
    spec = hai_tasks.resolve_spec(meta["config"], version, train["columns"])
    params = hai_tasks.fit_params(train, spec)
    assert json.dumps(params["scalers"], sort_keys=True) == json.dumps(meta["scalers"], sort_keys=True)
    data = hai.load_split(version, "dev_test")
    fresh, _ = hai_tasks.build_split(data, spec, params, meta["seed_entropy"]["dev_test"])
    saved = hai_tasks.load_tasks(version, "dev_test")
    assert set(saved) == set(fresh)
    for key in fresh:
        assert _equal(saved[key], fresh[key]), key
    t = hai.to_seconds(data["timestamp"])
    cut = fresh["decision_time"][fresh["decision_time"].size // 2]
    future = t > cut
    changed = {**data, "X": data["X"].copy()}
    changed["X"][future] = changed["X"][future][::-1]
    b, _ = hai_tasks.build_split(changed, spec, params, meta["seed_entropy"]["dev_test"])
    early = fresh["decision_time"] <= cut
    for field in CONTRACT_FIELDS:
        assert _equal(fresh[field][early], b[field][early]), field
    assert not np.array_equal(fresh["payload"][~early], b["payload"][~early])


@pytest.mark.data
@pytest.mark.slow
def test_prepared_val_matches_raw_data():
    version = "23.05"
    meta = _prepared(version)
    fields = ["decision_time", "tau_row", "segment", "time_to_event", "event_observed", "item_id", "present",
              "source", "kind", "event_start", "payload", "items_replayed_from", "score_bin"]
    a = hai_tasks.load_tasks(version, "val", fields=fields)
    data = hai.load_split(version, "val")
    t = hai.to_seconds(data["timestamp"])
    seg = data["segment"]
    assert np.array_equal(t[a["tau_row"]], a["decision_time"]) and np.array_equal(seg[a["tau_row"]], a["segment"])
    col = {c: j for j, c in enumerate(data["columns"])}
    rng = np.random.default_rng(0)
    pick = rng.choice(a["decision_time"].size, 150, replace=False)
    H = meta["horizon_s"]
    for q, name in enumerate(meta["question_names"]):
        spec_q = meta["questions"][name]
        x = data["X"][:, [col[c] for c in spec_q["columns"]]].astype(np.float64)
        values = x if spec_q["type"] == "discrete_change" else _hyst_loop(x[:, 0], spec_q["lo"], spec_q["hi"], seg)[:, None]
        for i in pick:
            tte, obs, _ = _brute_event(values, t, seg, a["tau_row"][i], H)
            assert a["time_to_event"][i, q] == tte and a["event_observed"][i, q] == obs
    dims = meta["payload_dim"]
    L = int(meta["window_s"])
    for i in pick[:40]:
        for m in np.flatnonzero(a["present"][i] & (a["kind"][i] == 0)):
            if a["items_replayed_from"][a["item_id"][i, m]] >= 0:
                continue
            s = a["source"][i, m]
            name = meta["sources"][s]
            row0 = int(np.searchsorted(t, a["event_start"][i, m]))
            cols = [col[c] for c in meta["source_columns"][name]]
            raw = hai_tasks.window_stats(data["X"][:, cols], np.array([row0]), L, meta["stats"])
            z, _ = hai_tasks.apply_scaler(raw, meta["scalers"][name], meta["config"]["scaling"]["clip"])
            np.testing.assert_allclose(a["payload"][i, m, :dims[s]], z[0], rtol=1e-5, atol=1e-5)
