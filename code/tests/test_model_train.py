import numpy as np
import pytest
import torch

from cpsdefense.eval import survival
from cpsdefense.model import experiments as X
from cpsdefense.model import timecalc
from cpsdefense.model import train as T
from cpsdefense.model.evidence import EvidenceBatch

T0 = 1.7e9
EDGES = [0, 5, 10, 20, 30, 60, 120, 300, 600]
PAYLOAD_DIM = (4, 2)
META = {"payload_dim": list(PAYLOAD_DIM), "kinds": ["window_summary"]}


def config(**train):
    tc = {"batch_size": 64, "lr": 3e-3, "weight_decay": 0.0, "grad_clip": 1.0, "max_steps": 300, "max_minutes": 2,
          "eval_every": 50, "patience": 100, "lr_patience": 100, "lr_factor": 0.5, "eval_chunk": 128}
    tc.update(train)
    return {
        "questions": {"hazard_edges": EDGES, "noul_within_s": 60, "score": True, "censor_rule": "start"},
        "model": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "n_pool": 2, "dropout": 0.0,
                  "branch_layers": 1, "hazard_hidden": 16},
        "train": tc,
        "baselines": {"min_events": 5, "age_bins_s": [30, 120], "shrink_events": 10},
        "eval": {"block_s": 600, "n_boot": 40, "alpha": 0.05, "boot_seed": 0, "min_ci_clusters": 3,
                 "rule": "linear", "tail": "inf", "ibs_grid_s": [5, 600, 5], "calib_horizons_s": [60, 300],
                 "calib_bins": 10, "dcal_bins": 10},
        "compare": [["masked", "km_state"], ["masked", "km_marginal"]],
        "invariance": {"fills": ["random", "huge", "nan", "inf", "hour_earlier"], "random_std": 3.0, "huge": 1e30,
                       "hour_s": 100},
        "revalidation": {"deltas_s": [1, 5, 15, 30]},
        "latency": {"delta_s": 15, "batch_sizes": [1, 4], "repeats": {"1": 3, "4": 2}, "warmup": 1,
                    "cpu_threads": 2, "devices": ["cpu"], "seed": 0},
        "deadline": {"predictors": ["km_state", "masked"], "deadlines_s": [120, 600], "deltas": [0.05, 0.1],
                     "n_grid": 512, "rule": "linear", "tail": "inf", "seed": 0, "grid_max_s": 120,
                     "grid_step_s": 0.5,
                     "pipeline": [{"name": "a", "type": "lognormal", "median": 4.0, "sigma": 0.6},
                                  {"name": "b", "type": "mixture", "weights": [0.96, 0.04],
                                   "parts": [{"type": "gamma", "shape": 2.0, "scale": 1.0},
                                             {"type": "uniform", "low": 30.0, "high": 60.0}]},
                                  {"name": "c", "type": "mixture", "weights": [0.96, 0.04],
                                   "parts": [{"type": "lognormal", "median": 2.0, "sigma": 0.5},
                                             {"type": "uniform", "low": 20.0, "high": 60.0}]}]},
    }


def make_split(seed=0, n=400, split="val", segments=2):
    rng = np.random.default_rng(seed)
    seg = np.repeat(np.arange(segments), n // segments)
    tau = T0 + 10.0 * np.arange(n) + 1e5 * seg
    x = rng.uniform(0, 1, n)
    m = 6
    t = tau[:, None]
    event_time = t - rng.uniform(5, 30, (n, m))
    arrival = event_time + rng.uniform(0, 4, (n, m))
    valid_until = t + rng.uniform(5, 60, (n, m))
    retracted = np.full((n, m), np.inf)
    valid_until[:, 3] = t[:, 0] - rng.uniform(0, 10, n)
    retracted[:, 4] = t[:, 0] - rng.uniform(0, 10, n)
    arrival[:, 5] = t[:, 0] + rng.uniform(1, 10, n)
    source = np.tile([0, 1, 0, 1, 0, 1], (n, 1))
    payload = rng.normal(0, 1, (n, m, max(PAYLOAD_DIM))).astype(np.float32)
    payload[:, :3, 0] = x[:, None]
    payload[..., 2:][source == 1] = 0
    item_id = np.arange(n * m).reshape(n, m)
    batch = EvidenceBatch.from_arrays(tau, item_id, source, event_time, arrival, valid_until, payload,
                                      segment=seg, retracted_at=retracted)
    names = ["qa", "qb"]
    rate = np.stack([1 / (10 + 400 * x), 1 / (400 - 350 * x)], 1)
    true_t = np.ceil(rng.exponential(1 / rate))
    cens = np.full((n, 2), 600.0)
    early = rng.random((n, 2)) < 0.05
    cens[early] = rng.uniform(1, 600, int(early.sum()))
    time = np.minimum(true_t, cens)
    event = true_t <= cens
    score = np.minimum((x * 3).astype(np.int64), 2)
    return T.TaskSplit(version="syn", split=split, names=names, batch=batch, time=time, event=event,
                       state=np.stack([(x > 0.5).astype(np.int64), np.zeros(n, np.int64)], 1),
                       since=np.where(rng.random((n, 2)) < 0.2, np.nan, rng.uniform(0, 400, (n, 2))),
                       score=score, score_now=score, attack_at_tau=np.zeros(n, np.int8),
                       attack_in_horizon=np.zeros(n, np.int8))


def spec_for(cfg=None):
    return T.QuestionSpec.from_config(cfg or config(), ["qa", "qb"], n_levels=3)


def test_within_target_hand_values():
    time = np.array([10, 60, 60, 61, 30, 600])
    event = np.array([True, True, False, True, False, False])
    out = T.within_target(time, event, 60)
    np.testing.assert_array_equal(out[[0, 1, 2, 3, 5]], [1, 1, 0, 0, 0])
    assert np.isnan(out[4])


def test_question_spec_and_device_targets():
    spec = spec_for()
    qs = spec.questions()
    assert [q.name for q in qs] == ["qa", "qb", "qa_within60", "qb_within60", "level_bin"]
    assert [q.type for q in qs] == ["hazard", "hazard", "noul", "noul", "score"]
    assert len({q.qid for q in qs}) == len(qs) and max(q.qid for q in qs) < spec.n_query_ids
    split = make_split()
    y = T.DeviceTargets(split, spec, "cpu")
    idx = torch.arange(10)
    tg = y(idx)
    np.testing.assert_array_equal(tg["qa"]["time"].numpy(), split.time[:10, 0])
    np.testing.assert_array_equal(tg["qb_within60"].numpy(),
                                  T.within_target(split.time[:10, 1], split.event[:10, 1], 60).astype(np.float32))
    assert y.kept(idx)["qa"] == 10
    with pytest.raises(ValueError):
        T.QuestionSpec.from_config({"questions": {"hazard_edges": [1, 5], "noul_within_s": 60}}, ["a"])


def test_load_split_refuses_heldout_test():
    with pytest.raises(ValueError, match="held-out"):
        T.load_split("23.05", "test")


def test_km_predictor_matches_hand_fits():
    train, val = make_split(0, split="train"), make_split(1)
    spec = spec_for()
    km = T.KMPredictor(spec, min_events=5, age_edges=(30, 120), shrink_events=10).fit(train)
    edges = spec.edges
    marg = km.predict(val, "km_marginal")
    ref = survival.km_pmf(survival.kaplan_meier(train.time[:, 0], train.event[:, 0]), edges)
    np.testing.assert_allclose(marg["pmf"]["qa"], np.tile(ref, (val.n, 1)))
    state = km.predict(val, "km_state")
    for s in (0, 1):
        sel = train.state[:, 0] == s
        ref_s = survival.km_pmf(survival.kaplan_meier(train.time[sel, 0], train.event[sel, 0]), edges)
        np.testing.assert_allclose(state["pmf"]["qa"][val.state[:, 0] == s], ref_s[None].repeat((val.state[:, 0] == s).sum(), 0))
    np.testing.assert_allclose(state["noul"]["qa"], X.cdf_at_edge(state["pmf"]["qa"], edges, 60.0))
    age = km.predict(val, "km_state_age")
    g_train = train.state[:, 0] * 100 + T.age_bin(train.since[:, 0], (30, 120))
    g_val = val.state[:, 0] * 100 + T.age_bin(val.since[:, 0], (30, 120))
    g = g_val[0]
    sel = g_train == g
    d = train.event[sel, 0].sum()
    grp = survival.km_pmf(survival.kaplan_meier(train.time[sel, 0], train.event[sel, 0]), edges)
    np.testing.assert_allclose(age["pmf"]["qa"][0], (d * grp + 10 * state["pmf"]["qa"][0]) / (d + 10))
    for p in (marg, state, age):
        np.testing.assert_allclose(p["pmf"]["qb"].sum(1), 1.0)
    np.testing.assert_allclose(marg["score"][0], np.bincount(train.score, minlength=3) / train.n)
    assert np.allclose(state["score"][val.score_now == 2][:, 2], 1.0)


def test_age_bin():
    np.testing.assert_array_equal(T.age_bin([0, 29.9, 30, 150, np.nan], (30, 120)), [0, 0, 1, 2, 3])


def test_training_learns_and_restores_best_checkpoint():
    cfg = config()
    train, val = make_split(0, n=600, split="train"), make_split(1, n=200)
    spec = spec_for(cfg)
    model = T.build_model(cfg, META, "masked", spec, seed=0)
    y_val = T.DeviceTargets(val, spec, "cpu")
    before = T.evaluate_loss(model, val.batch.to("cpu"), y_val, spec.questions())["total"]
    hist = T.train_model(model, train, val, spec, cfg, seed=0, device="cpu", log=lambda m: None)
    after = T.evaluate_loss(model, val.batch.to("cpu"), y_val, spec.questions())
    assert hist["steps"] == 300 and hist["stop"] == "max_steps"
    assert after["total"] < before - 0.3
    assert after["total"] == pytest.approx(hist["best_val"]["total"], abs=1e-6)
    assert min(r["val"]["total"] for r in hist["curve"]) == pytest.approx(after["total"], abs=1e-6)
    raw = T.predict(model, val.batch.to("cpu"), spec.questions(), chunk=64)
    preds = T.to_predictions(raw, spec)
    c = X.c_index_td_fast(preds["pmf"]["qa"], spec.edges, val.time[:, 0], val.event[:, 0])
    assert c > 0.65


def test_early_stopping_and_time_budget():
    cfg = config(patience=2, lr=0.0, max_steps=1000, eval_every=20)
    spec = spec_for(cfg)
    model = T.build_model(cfg, META, "masked", spec, seed=0)
    hist = T.train_model(model, make_split(0, split="train"), make_split(1), spec, cfg, 0, "cpu", log=lambda m: None)
    assert hist["stop"] == "early_stopping" and hist["steps"] == 60
    cfg = config(max_minutes=0.0, max_steps=1000, eval_every=10)
    model = T.build_model(cfg, META, "masked", spec, seed=0)
    hist = T.train_model(model, make_split(0, split="train"), make_split(1), spec, cfg, 0, "cpu", log=lambda m: None)
    assert hist["stop"] == "time_budget" and hist["steps"] == 10


def test_predict_is_deterministic_and_chunk_invariant(tmp_path):
    cfg = config()
    spec = spec_for(cfg)
    split = make_split(2)
    model = T.build_model(cfg, META, "block_local", spec, seed=1)
    batch = split.batch.to("cpu")
    a = T.predict(model, batch, spec.questions(), chunk=64)
    b = T.predict(model, batch, spec.questions(), chunk=64)
    c = T.predict(model, batch, spec.questions(), chunk=150)
    for k in a:
        assert np.array_equal(a[k].view(np.uint32), b[k].view(np.uint32))
        np.testing.assert_allclose(a[k], c[k], atol=1e-5)
    preds = T.to_predictions(a, spec)
    np.testing.assert_allclose(preds["pmf"]["qa"].sum(1), 1.0)
    timecalc.check_pmf(preds["pmf"]["qb"], spec.edges)
    assert preds["score"].shape == (split.n, 3)
    T.save_predictions(tmp_path / "p.npz", preds)
    back = T.load_predictions(tmp_path / "p.npz")
    np.testing.assert_array_equal(back["pmf"]["qa"], preds["pmf"]["qa"])
    np.testing.assert_array_equal(back["noul"]["qb"], preds["noul"]["qb"])
    np.testing.assert_array_equal(back["score"], preds["score"])
    T.save_model(tmp_path / "m.pt", model, spec)
    loaded, spec2 = T.load_model(tmp_path / "m.pt")
    assert spec2.to_dict() == spec.to_dict()
    d = T.predict(loaded, batch, spec.questions(), chunk=64)
    for k in a:
        assert np.array_equal(a[k].view(np.uint32), d[k].view(np.uint32))


def test_variants_configure_the_encoder():
    spec = spec_for()
    for variant, expected in T.VARIANTS.items():
        m = T.build_model(config(), META, variant, spec, seed=0)
        assert m.mode == expected["mode"] and m.encoder.hide_validity == expected["hide_validity"]
    with pytest.raises(ValueError):
        T.build_model(config(), META, "nope", spec, seed=0)
    a = T.build_model(config(), META, "masked", spec, seed=3).state_dict()
    b = T.build_model(config(), META, "masked", spec, seed=3).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_hour_partner():
    seg = np.array([0, 0, 0, 0, 1, 1])
    tau = np.array([0.0, 10, 20, 30, 0, 10])
    np.testing.assert_array_equal(X.hour_partner(seg, tau, hour_s=20), [2, 3, 0, 1, -1, -1])


def test_fill_transform_touches_only_invalid_items():
    split = make_split(3, n=40)
    batch = split.batch.to("cpu")
    valid = torch.as_tensor(split.batch.validity())
    partner = X.hour_partner(split.segment, split.decision_time, 100.0)
    rows = torch.arange(batch.n)
    for fill in ("random", "huge", "nan", "inf", "hour_earlier"):
        out = X.fill_transform(fill, batch, partner)(batch, rows)
        before, after = batch.payload, out.payload
        assert torch.equal(after[valid], before[valid])
        changed = ~((after == before) | (torch.isnan(after) & torch.isnan(before))).all(-1)
        assert not changed[valid].any()
        assert changed[~valid & batch.present].float().mean() > 0.5
        for name in ("item_id", "source", "present"):
            assert torch.equal(getattr(out, name), getattr(batch, name))
        for name in ("arrival_time", "valid_until", "retracted_at", "event_time"):
            torch.testing.assert_close(getattr(out, name), getattr(batch, name), rtol=0, atol=0, equal_nan=True)
    hour = X.fill_transform("hour_earlier", batch, partner)(batch, rows).payload
    r = int(np.flatnonzero(partner >= 0)[0])
    invalid = (~valid[r]) & batch.present[r] & batch.present[partner[r]]
    assert torch.equal(hour[r][invalid], batch.payload[partner[r]][invalid])


def test_invariance_audit_masked_exact_unmasked_moves():
    cfg = config()
    spec = spec_for(cfg)
    split = make_split(4, n=120)
    batch = split.batch.to("cpu")
    for variant in ("masked", "block_local", "unmasked", "unmasked_novalidity"):
        model = T.build_model(cfg, META, variant, spec, seed=0)
        audit = X.invariance_audit(model, batch, split, spec, cfg)
        assert audit["used_equals_expected_mask"]
        fills = audit["fills"]
        if variant in ("masked", "block_local"):
            for fill, r in fills.items():
                assert r["bitwise_equal_rate"] == 1.0, (variant, fill)
                assert r["max_abs_change"] == 0.0 and r["nonfinite_rate"] == 0.0
                assert all(v == 0.0 for v in r["delta_nll_mean"].values())
        else:
            assert fills["random"]["bitwise_equal_rate"] == 0.0 and fills["random"]["max_abs_change"] > 1e-4
            assert fills["nan"]["nonfinite_rate"] == 1.0
            assert fills["hour_earlier"]["bitwise_equal_rate"] < 1.0


def test_compare_outputs():
    spec = spec_for()
    split = make_split(5, n=20)
    rng = np.random.default_rng(0)
    pmf = rng.dirichlet(np.ones(len(EDGES)), 20).astype(np.float32)
    base = {"qa": pmf, "qb": pmf, "qa_within60": rng.random(20).astype(np.float32),
            "qb_within60": rng.random(20).astype(np.float32), "level_bin": rng.dirichlet(np.ones(3), 20).astype(np.float32)}
    same = X.compare_outputs(base, {k: v.copy() for k, v in base.items()}, spec, split, config()["eval"])
    assert same["bitwise_equal_rate"] == 1.0 and same["max_abs_change"] == 0.0
    other = {k: v.copy() for k, v in base.items()}
    other["qa_within60"][3] += 0.25
    diff = X.compare_outputs(base, other, spec, split, config()["eval"])
    assert diff["bitwise_equal_rate"] == pytest.approx(19 / 20)
    assert diff["max_abs_change_noul"] == pytest.approx(0.25, abs=1e-6)


def random_survival_data(seed, n=300, k=len(EDGES)):
    rng = np.random.default_rng(seed)
    pmf = rng.dirichlet(np.ones(k) * 0.7, n)
    pmf[: n // 10] = pmf[0]
    time = rng.integers(1, 40, n).astype(np.float64) * 15
    event = rng.random(n) < 0.7
    return pmf, np.asarray(EDGES, np.float64), time, event


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_c_index_td_fast_equals_reference(seed):
    pmf, edges, time, event = random_survival_data(seed)
    for rule in ("linear", "pch", "upper"):
        assert X.c_index_td_fast(pmf, edges, time, event, rule) == pytest.approx(
            survival.c_index_td(pmf, edges, time, event, rule), abs=1e-12)


def test_brier_and_ibs_rows_equal_reference():
    pmf, edges, time, event = random_survival_data(3)
    censor = survival.censoring_km(time, event)
    horizons = [30.0, 60.0, 300.0]
    rows = X.brier_rows(pmf, edges, time, event, horizons, censor)
    np.testing.assert_allclose(rows.mean(0), survival.brier_ipcw(pmf, edges, time, event, horizons), atol=1e-12)
    grid = np.arange(5, 600, 5.0)
    assert X.ibs_rows(pmf, edges, time, event, grid, censor).mean() == pytest.approx(
        survival.integrated_brier(pmf, edges, time, event, grid), abs=1e-12)


def test_ece_ipcw_equals_calibration_at():
    pmf, edges, time, event = random_survival_data(4)
    censor = survival.censoring_km(time, event)
    for t in (60.0, 300.0):
        died, _, weight = X.ipcw_status(time, event, [t], censor)
        prob = np.clip(X.cdf_at_edge(pmf, edges, t), 0, 1)
        ref = survival.calibration_at(pmf, edges, time, event, t, n_bins=10)["ece"]
        assert X.ece_ipcw(prob, died[:, 0].astype(float), weight[:, 0], 10) == pytest.approx(ref, abs=1e-12)


def test_decision_blocks_stay_inside_segments():
    seg = np.repeat([3, 7], [1000, 500])
    tau = np.r_[T0 + 10.0 * np.arange(1000), T0 + 5e5 + 10.0 * np.arange(500)]
    c = X.decision_blocks(seg, tau, 3600)
    assert np.unique(c).size == 3 + 2
    for b in np.unique(c):
        assert np.unique(seg[c == b]).size == 1
        assert np.ptp(tau[c == b]) < 3600


def test_mean_ci_blank_below_min_clusters():
    rows = np.arange(100.0)
    boot = {"n_boot": 50, "alpha": 0.05, "seed": 0, "min_clusters": 10}
    few = X.mean_ci(rows, np.arange(100) // 20, boot)
    assert few["lo"] is None and few["point"] == pytest.approx(49.5)
    many = X.mean_ci(rows, np.arange(100) // 5, boot)
    assert many["lo"] < 49.5 < many["hi"]
    d = X.paired_mean_diff(rows + 1.0, rows, np.arange(100) // 5, boot)
    assert d["point"] == pytest.approx(1.0) and d["lo"] == pytest.approx(1.0)


def test_revalidation_stats_match_revalidate():
    split = make_split(6, n=80)
    b = split.batch
    stats = X.revalidation_stats(b, [1, 5, 15, 30])
    deps = b.dependency_set()
    for delta in (1, 5, 15, 30):
        lost = b.revalidate(deps, b.decision_time + delta)
        ref = np.mean([len(x) > 0 for x in lost])
        assert stats["deltas"][str(delta)]["frac_lose_any"] == pytest.approx(ref)
        assert stats["deltas"][str(delta)]["lost_per_decision"] == pytest.approx(np.mean([len(x) for x in lost]))
    assert stats["dependency_size_mean"] == pytest.approx(3.0)
    assert stats["unmasked_depends_on_invalid_rate"] == 1.0


def test_latency_benchmark_runs_on_cpu():
    cfg = config()
    spec = spec_for(cfg)
    models = {v: T.build_model(cfg, META, v, spec, seed=0) for v in ("masked", "block_local")}
    out = X.latency_benchmark(models, make_split(7, n=50).batch, spec, cfg, log=lambda m: None)
    r = out["devices"]["cpu"]["4"]
    assert set(r) == {"masked", "block_local"}
    assert r["masked"]["rescore"]["n"] == 2 and r["masked"]["rescore"]["p50"] > 0
    assert r["block_local"]["items_encoded_mean"]["rescore"] <= r["block_local"]["items_encoded_mean"]["fresh"]
    assert r["masked"]["items_encoded_mean"]["rescore"] == r["masked"]["items_encoded_mean"]["fresh"]
    lines = X.latency_table(out)
    assert len(lines) == 2 + 2 * 2 and "block_local" in lines[-1]


def test_find_runs_picks_latest_finished_matching_run(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_backbone.py"
    spec = importlib.util.spec_from_file_location("train_backbone_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    def write(name, variant, seed, sha, finished=True, version="23.05"):
        d = tmp_path / "model" / name
        d.mkdir(parents=True)
        rec = {"config": {"task": script.TASK, "version": version, "variant": variant, "seed": seed,
                          "shared_sha256": sha}}
        if finished:
            rec["finished_utc"] = "x"
        (d / "run.json").write_text(json.dumps(rec), encoding="utf-8")
        return d

    write("20260101-000000-a", "masked", 0, "h1")
    newer = write("20260102-000000-a", "masked", 0, "h1")
    write("20260103-000000-a", "masked", 0, "h1", finished=False)
    write("20260104-000000-a", "masked", 0, "other")
    write("20260105-000000-a", "masked", 0, "h1", version="22.04")
    km = write("20260101-000000-km", "km", None, "h1")
    found = script.find_runs("23.05", "h1", root=tmp_path)
    assert found == {("masked", 0): newer, ("km", None): km}


def test_deadline_outcomes_hand_values():
    time = np.array([10.0, 50.0, 30.0, 30.0])
    event = np.array([True, True, False, False])
    on, known = X.deadline_outcomes(time, event, np.array([5.0, 5.0, 5.0, 80.0]), 40.0)
    np.testing.assert_array_equal(on, [True, False, False, False])
    np.testing.assert_array_equal(known, [True, True, False, True])


def test_pipeline_components_match_their_samplers():
    dc = config()["deadline"]
    comps = X.pipeline_components(dc)
    draws = X.pipeline_draws(dc, 200000, seed=1)
    p = timecalc.prob_total_within(comps, np.array([10.0, 20.0, 50.0]), "linear", "inf", "mid", 2048)
    emp = np.array([(draws <= d).mean() for d in (10.0, 20.0, 50.0)])
    np.testing.assert_allclose(p, emp, atol=4e-3)


def test_deadline_gates_convolution_covers_quantile_sum_may_not():
    dc = config()["deadline"]
    edges = np.asarray(EDGES, np.float64)
    comps = X.pipeline_components(dc)
    rng = np.random.default_rng(0)
    n = 4000
    pmf = np.zeros((n, len(edges)))
    pmf[:, 1] = rng.uniform(0.9, 1.0, n)
    pmf[:, 2] = 1 - pmf[:, 1]
    deadlines, deltas = [35.0, 45.0], [0.05, 0.1]
    g = X.deadline_gates(pmf, edges, comps, deadlines, deltas, n_grid=1024)
    t_sim = X.sample_pmf(pmf, edges, np.random.default_rng(1))
    lat = X.pipeline_draws(dc, n, seed=2)
    n_conv = 0
    for j, d in enumerate(deadlines):
        on = t_sim + lat <= d
        assert np.all(g["lower"][:, j] <= g["mid"][:, j] + 1e-12)
        for delta in deltas:
            conv = g["gates"][f"{delta:g}"]["conv"][:, j]
            n_conv += int(conv.sum())
            if conv.any():
                assert on[conv].mean() >= 1 - delta - 3 * np.sqrt(delta * (1 - delta) / conv.sum())
    assert n_conv > 1000
    qs = g["gates"]["0.05"]["qsum"][:, 0]
    on35 = t_sim + lat <= 35.0
    assert qs.sum() > 1000 and on35[qs].mean() < 0.94
    assert not g["gates"]["0.05"]["conv"][:, 0].any()


def test_sample_pmf_follows_the_linear_rule():
    edges = np.array([0.0, 10.0, 30.0])
    pmf = np.tile([0.5, 0.3, 0.2], (200000, 1))
    t = X.sample_pmf(pmf, edges, np.random.default_rng(0))
    assert np.mean(np.isinf(t)) == pytest.approx(0.2, abs=0.005)
    assert np.mean(t <= 5.0) == pytest.approx(0.25, abs=0.005)
    assert np.mean(t <= 20.0) == pytest.approx(0.65, abs=0.005)


def test_evaluate_split_and_format_results():
    cfg = config()
    spec = spec_for(cfg)
    train, val = make_split(0, n=400, split="train"), make_split(1, n=400)
    km = T.KMPredictor(spec, 5, (30, 120), 10).fit(train)
    preds = {k: [km.predict(val, k)] for k in T.BASELINES}
    rng = np.random.default_rng(0)
    fake = []
    for s in range(2):
        p = km.predict(val, "km_state_age")
        p["noul"] = {k: np.clip(v + rng.normal(0, 0.02, v.size), 0, 1) for k, v in p["noul"].items()}
        fake.append(p)
    preds["masked"] = fake
    res = X.evaluate_split(val, preds, spec, cfg, log=lambda m: None)
    m = res["questions"]["qa"]
    assert set(m) == {"km_marginal", "km_state", "km_state_age", "masked"}
    assert m["masked"]["nll"]["point"] == pytest.approx(
        survival.nll_discrete(fake[0]["pmf"]["qa"], val.time[:, 0], val.event[:, 0], spec.edges))
    assert m["km_marginal"]["c_td"]["point"] == pytest.approx(0.5)
    assert m["masked"]["nll"]["sd"] == pytest.approx(0.0)
    assert res["paired"]["qa"]["masked - km_marginal"]["nll"]["point"] < 0
    coh = res["coherence"]["qa"]["masked"]
    assert 0 < coh["mean_abs_gap"]["point"] < 0.05
    assert set(res["deadline"]["qa"]) == {"km_state", "masked"}
    assert res["score"]["km_state"]["accuracy"]["point"] > res["score"]["km_marginal"]["accuracy"]["point"]
    audit = {"used_equals_expected_mask": True, "used_items_per_decision": 3.0, "invalid_items": 5,
             "arrived_invalid_items": 4, "fills": {"nan": {"bitwise_equal_rate": 1.0, "nonfinite_rate": 0.0,
                                                           "max_abs_change": 0.0, "max_abs_change_survival": 0.0,
                                                           "max_abs_change_noul": 0.0, "mean_abs_change": 0.0,
                                                           "frac_change_gt_0.01": 0.0,
                                                           "delta_nll_mean": {"qa": None, "qb": 0.0},
                                                           "delta_nll_abs_mean": {"qa": 0.0, "qb": 0.0}}}}
    agg = X.aggregate_audits([audit, audit])
    assert agg["fills"]["nan"]["bitwise_equal_rate_min"] == 1.0 and agg["fills"]["nan"]["delta_nll_mean"] == 0.0
    results = {"versions": {"syn": {"n_seeds": {"masked": 2}, "splits": {"val": res},
                                    "invariance": {"val": {"masked": agg}},
                                    "revalidation": {"val": X.revalidation_stats(val.batch, [1, 5])}}},
               "latency": None}
    md = X.format_results(results, cfg, note="unit test")
    for header in ("## Provenance", "### 1. Time-to-event quality: val", "### 1b.", "### 1c.", "### 2.", "### 3.",
                   "### 4a.", "### 5.", "unit test"):
        assert header in md
    assert "held-out HAI test split was never loaded" in md.replace("\n", " ")


@pytest.mark.data
def test_real_dev_test_loads_with_consistent_targets(processed_dir):
    try:
        split = T.load_split("23.05", "dev_test")
    except FileNotFoundError:
        pytest.skip("HAI tasks not prepared")
    assert split.batch.n == split.time.shape[0] and split.time.shape[1] == len(split.names)
    assert np.all(split.time > 0) and np.all(split.time <= 600)
    y = T.within_target(split.time, split.event, 60)
    assert np.all(y[split.event & (split.time <= 60)] == 1)
    c = X.decision_blocks(split.segment, split.decision_time, 3600)
    assert c.max() + 1 >= 10


@pytest.mark.data
def test_real_masked_audit_is_bitwise(processed_dir):
    try:
        split = T.load_split("23.05", "val", rows=np.arange(3000, 3600))
        meta = T.hai_tasks.load_meta("23.05")
    except FileNotFoundError:
        pytest.skip("HAI tasks not prepared")
    cfg = config()
    cfg["invariance"]["hour_s"] = 3600
    spec = T.QuestionSpec.from_config(cfg, split.names, int(meta["score"]["n_bins"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = split.batch.to(device)
    for variant in ("masked", "block_local"):
        model = T.build_model(cfg, meta, variant, spec, seed=0, device=device)
        audit = X.invariance_audit(model, batch, split, spec, cfg)
        assert audit["used_equals_expected_mask"]
        assert all(r["bitwise_equal_rate"] == 1.0 for r in audit["fills"].values())
    model = T.build_model(cfg, meta, "unmasked", spec, seed=0, device=device)
    audit = X.invariance_audit(model, batch, split, spec, cfg)
    assert audit["fills"]["hour_earlier"]["bitwise_equal_rate"] < 0.5


def test_p_values_below_bootstrap_resolution_are_bounds_not_zero():
    assert X._p(0.0, 1000) == "< 0.002"
    assert X._p(0.0015, 1000) == "< 0.002"
    assert X._p(0.2, 1000) == "0.2"
    assert X._p(None, 1000) == "-" and X._p(float("nan"), 1000) == "-"


def test_deadline_rates_pool_seeds_instead_of_hiding_them():
    old = {"n_feasible": {"per_seed": [315, 0, 436]},
           "on_time_rate": {"per_seed": [0.911, float("nan"), 0.892]},
           "sim_on_time_rate": {"per_seed": [0.95, float("nan"), 0.96]}}
    assert X._pooled(old, "on_time") == pytest.approx((0.911 * 315 + 0.892 * 436) / 751)
    assert np.isnan(X._pooled({"n_feasible": {"per_seed": [0, 0]}, "on_time_rate": {"per_seed": [np.nan, np.nan]}},
                              "on_time"))
    assert X._pooled({**old, "on_time_pooled": 0.5}, "on_time") == 0.5
