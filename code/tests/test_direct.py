import importlib.util

import numpy as np
import pytest
import torch

from cpsdefense.agents.flat_baseline import (FlatBaseline, candidate_targets, evidence_summary, feature_matrix,
                                             local_argmin)
from cpsdefense.model.direct import (TRIP_EDGES, DirectConfig, DirectDecisionModel, choose, derived_voi,
                                     gather_act_loss, losses, pad_candidates)
from cpsdefense.model.evidence import EvidenceBatch
from cpsdefense.paths import ROOT

T0 = 1.7e9
PAYLOAD_DIM = (5, 3)
F = 6
OUTPUTS = ("J_pred", "trip_logit", "resolve_logit", "log_duration", "hazard_logits")
MODES = ("masked", "block_local", "unmasked")


def load_script():
    spec = importlib.util.spec_from_file_location("train_direct", ROOT / "scripts" / "train_direct.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(seed, n=4, m=10):
    rng = np.random.default_rng(seed)
    tau = T0 + rng.uniform(0, 1000, n)
    age = rng.uniform(1, 600, (n, m))
    event_time = tau[:, None] - age
    arrival = event_time + rng.uniform(0, 1, (n, m)) * age
    valid_until = tau[:, None] + rng.uniform(-300, 900, (n, m))
    source = rng.integers(0, len(PAYLOAD_DIM), (n, m))
    payload = rng.normal(size=(n, m, max(PAYLOAD_DIM))).astype(np.float32)
    for s, dim in enumerate(PAYLOAD_DIM):
        payload[..., dim:][source == s] = 0
    return EvidenceBatch.from_arrays(tau, np.arange(n * m).reshape(n, m), source, event_time, arrival, valid_until,
                                     payload, segment=np.arange(n))


def make_model(mode="masked", seed=0, dropout=0.0, **config):
    backbone = dict(payload_dim=PAYLOAD_DIM, n_kinds=1, mode=mode, d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_pool=2, branch_layers=2, n_query_ids=0, descriptor_dim=F, dropout=dropout)
    torch.manual_seed(seed)
    return DirectDecisionModel(DirectConfig(backbone=backbone, head_hidden=16, **config)).eval()


def candidates(seed, n=4, k=5):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, k, F)).astype(np.float32)
    mask = np.ones((n, k), bool)
    mask[0, 3:] = False
    mask[2, 4] = False
    return feats, mask


def run(model, batch, feats, mask):
    with torch.no_grad():
        out = model(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))
    return {k: out[k] for k in OUTPUTS}


@pytest.mark.parametrize("mode", MODES)
def test_reordering_candidates_permutes_outputs_exactly(mode):
    model, batch = make_model(mode), make_batch(1)
    feats, mask = candidates(2)
    base = run(model, batch, feats, mask)
    perm = np.array([3, 0, 4, 1, 2])
    moved = run(model, batch, feats[:, perm], mask[:, perm])
    for key in OUTPUTS:
        real = torch.as_tensor(mask[:, perm])
        assert torch.equal(base[key][:, perm][real], moved[key][real]), key


@pytest.mark.parametrize("mode", MODES)
def test_other_candidates_and_padding_content_do_not_change_a_candidate(mode):
    model, batch = make_model(mode), make_batch(3)
    feats, mask = candidates(4)
    base = run(model, batch, feats, mask)
    rng = np.random.default_rng(5)
    for fill in (np.nan, np.inf, -1e30, "random"):
        junk = feats.copy()
        junk[~mask] = rng.normal(0, 1e3, junk[~mask].shape) if fill == "random" else fill
        junk[:, 1] = rng.normal(size=junk[:, 1].shape)
        other = run(model, batch, junk, mask)
        keep = torch.as_tensor(mask)
        keep[:, 1] = False
        for key in OUTPUTS:
            assert torch.equal(base[key][keep], other[key][keep]), (fill, key)
            assert torch.isfinite(other[key][torch.as_tensor(mask)]).all()


@pytest.mark.parametrize("mode", MODES)
def test_adding_and_removing_candidates_changes_nothing_within_tolerance(mode):
    model, batch = make_model(mode), make_batch(6)
    feats, mask = candidates(7)
    base = run(model, batch, feats, mask)
    rng = np.random.default_rng(8)
    extra = np.concatenate([feats, rng.normal(size=(4, 6, F)).astype(np.float32)], 1)
    extra_mask = np.concatenate([mask, rng.random((4, 6)) < 0.5], 1)
    more = run(model, batch, extra, extra_mask)
    fewer = run(model, batch, feats[:, [0, 2]], mask[:, [0, 2]])
    alone = [run(model, batch, feats[:, [j]], mask[:, [j]]) for j in range(feats.shape[1])]
    for key in OUTPUTS:
        real = torch.as_tensor(mask)
        torch.testing.assert_close(more[key][:, :5][real], base[key][real], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fewer[key][torch.as_tensor(mask[:, [0, 2]])], base[key][:, [0, 2]][
            torch.as_tensor(mask[:, [0, 2]])], atol=1e-5, rtol=1e-5)
        for j in range(feats.shape[1]):
            torch.testing.assert_close(alone[j][key][:, 0][real[:, j]], base[key][:, j][real[:, j]],
                                       atol=1e-5, rtol=1e-5)


def test_decision_rows_are_independent_of_each_other():
    model, batch = make_model(), make_batch(9)
    feats, mask = candidates(10)
    base = run(model, batch, feats, mask)
    for r in range(batch.n):
        one = run(model, batch.take([r]), feats[[r]], mask[[r]])
        for key in OUTPUTS:
            torch.testing.assert_close(one[key][0][torch.as_tensor(mask[r])], base[key][r][torch.as_tensor(mask[r])],
                                       atol=1e-5, rtol=1e-5)


def labels(mask, seed, R=3):
    rng = np.random.default_rng(seed)
    n, k = mask.shape
    J = rng.uniform(0, 30, (n, k))
    trip_time = np.where(rng.random((n, k, R)) < 0.3, rng.uniform(100, 5000, (n, k, R)), 10800.0)
    y = {"J": J, "p_trip": (trip_time < 10800).mean(-1), "trip_time": trip_time, "trip_event": trip_time < 10800,
         "p_resolved": rng.random((n, k)), "duration": rng.uniform(10, 900, (n, k)), "voi": rng.normal(size=(n, k)),
         "observe": rng.random((n, k)) < 0.4}
    y["observe"][:, 0], y["observe"][:, 1] = True, False
    for key, v in y.items():
        if v.dtype == bool:
            v[~mask] = False
        else:
            v[~mask] = np.nan
    return {k: torch.as_tensor(v, dtype=torch.bool if v.dtype == bool else torch.float32) for k, v in y.items()}


@pytest.mark.parametrize("mode", MODES)
def test_padded_rows_give_finite_loss_and_gradients(mode):
    model = make_model(mode, dropout=0.1, weights={"gva": 1.0}).train()
    batch = make_batch(11)
    feats, mask = candidates(12)
    feats[~mask] = np.nan
    feats[0, 4] = np.inf
    y = labels(mask, 13)
    out = model(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))
    loss, parts = losses(out, y, torch.as_tensor(mask), model.config)
    loss.backward()
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in parts.values())
    assert set(parts) == {"J", "J_centered", "rank", "trip", "hazard", "resolve", "duration", "gva"}
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert len(TRIP_EDGES) - 1 == out["hazard_logits"].shape[-1]


def test_zero_weights_drop_their_terms():
    model = make_model(weights={"rank": 0.0, "trip": 0.0, "hazard": 0.0, "resolve": 0.0, "duration": 0.0})
    batch = make_batch(14)
    feats, mask = candidates(15)
    y = labels(mask, 16)
    out = model(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))
    loss, parts = losses(out, y, torch.as_tensor(mask), model.config)
    assert float(loss.detach()) == pytest.approx(parts["J"], rel=1e-6)


def test_j_loss_options_and_the_centred_term():
    feats, mask = candidates(20)
    y = labels(mask, 21)
    m = torch.as_tensor(mask)
    rng = np.random.default_rng(22)
    J_out = torch.as_tensor(rng.normal(size=mask.shape), dtype=torch.float32)
    shift = torch.as_tensor(rng.normal(size=(mask.shape[0], 1)) * 5, dtype=torch.float32)
    out = {"J": J_out, "J_pred": J_out * 10.0, "trip_logit": torch.zeros(mask.shape),
           "resolve_logit": torch.zeros(mask.shape), "log_duration": torch.zeros(mask.shape),
           "hazard_logits": torch.zeros(mask.shape + (len(TRIP_EDGES) - 1,))}
    mse = DirectConfig(backbone={}, j_loss="mse")
    _, parts = losses(out, y, m, mse)
    keep = m & torch.isfinite(y["J"])
    expected = (0.5 * (J_out - torch.nan_to_num(y["J"]) / 10.0) ** 2)[keep].mean()
    assert parts["J"] == pytest.approx(float(expected), rel=1e-5)
    shifted = {**out, "J": J_out + shift, "J_pred": (J_out + shift) * 10.0}
    _, parts_shifted = losses(shifted, y, m, mse)
    assert parts_shifted["J_centered"] == pytest.approx(parts["J_centered"], rel=1e-4)
    assert parts_shifted["J"] != pytest.approx(parts["J"])
    with pytest.raises(ValueError):
        losses(out, y, m, DirectConfig(backbone={}, j_loss="l1"))


def test_choose_and_derived_voi_on_hand_made_inputs():
    J = torch.tensor([[5.0, 3.0, 4.0, -10.0], [2.0, 2.5, 7.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    observe = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    assert choose(J, mask).tolist() == [1, 0, 0]
    voi = derived_voi(J, observe, mask)
    expected = torch.tensor([[-2.0, np.nan, np.nan, np.nan], [np.nan, -0.5, -5.0, np.nan],
                             [np.nan, np.nan, np.nan, np.nan]])
    torch.testing.assert_close(voi, expected, equal_nan=True)


def test_argmin_gathers_exactly_when_some_derived_voi_is_positive():
    rng = np.random.default_rng(17)
    J = torch.as_tensor(rng.normal(size=(500, 6)))
    mask = torch.as_tensor(rng.random((500, 6)) < 0.8)
    mask[:, 0] = mask[:, 1] = True
    observe = torch.as_tensor(rng.random((500, 6)) < 0.4)
    observe[:, 0], observe[:, 1] = True, False
    chosen = choose(J, mask)
    gathers = observe[torch.arange(500), chosen]
    positive = (torch.nan_to_num(derived_voi(J, observe, mask), nan=-1.0) > 0).any(-1)
    assert torch.equal(gathers, positive)


def test_gather_act_loss_rewards_the_right_family_and_skips_near_ties():
    J = torch.tensor([[5.0, 8.0, 9.0], [5.0, 3.0, 9.0], [5.0, 5.1, 9.0]])
    observe = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.bool)
    keep = torch.ones(3, 3, dtype=torch.bool)
    config = DirectConfig(backbone={}, gva_margin=0.5)
    right = gather_act_loss(J, J, observe, keep, config)
    wrong = gather_act_loss(torch.tensor([[9.0, 8.0, 9.0], [1.0, 3.0, 9.0], [5.0, 5.1, 9.0]]), J, observe, keep, config)
    assert float(right) < float(wrong)
    tie_only = gather_act_loss(J[2:], J[2:], observe[2:], keep[2:], config)
    assert float(tie_only) == 0.0
    balanced = gather_act_loss(J, J, observe, keep, DirectConfig(backbone={}, gva_balance=True))
    assert torch.isfinite(balanced)
    J2 = torch.tensor([[5.0, 8.0, 9.0], [5.0, 5.8, 9.0]])
    wrong_big = torch.tensor([[9.0, 8.0, 9.0], [5.0, 6.0, 9.0]])
    wrong_small = torch.tensor([[5.0, 6.0, 9.0], [9.0, 8.0, 9.0]])
    plain, weighted = DirectConfig(backbone={}), DirectConfig(backbone={}, gva_regret_weight=True)
    args = (J2, observe[:2], keep[:2])
    assert float(gather_act_loss(wrong_big, *args, plain)) == pytest.approx(
        float(gather_act_loss(wrong_small, *args, plain)))
    assert float(gather_act_loss(wrong_big, *args, weighted)) > float(gather_act_loss(wrong_small, *args, weighted))
    pred = torch.tensor([[6.0, 9.0, 1.0], [2.0, 3.0, 0.5]], requires_grad=True)
    gather_act_loss(pred, J2, *args[1:], DirectConfig(backbone={}, gva_labelled_best=True)).backward()
    assert (pred.grad[:, 2] == 0).all() and (pred.grad[:, :2] != 0).all()
    pred.grad = None
    gather_act_loss(pred, J2, *args[1:], plain).backward()
    assert (pred.grad[:, 2] != 0).all() and (pred.grad[:, 1] == 0).all()


def test_pad_candidates_keeps_row_order_within_decisions():
    dec = np.array([1, 0, 1, 2, 0, 1])
    mask, out = pad_candidates(dec, 3, {"x": np.arange(6.0)})
    assert mask.tolist() == [[True, True, False], [True, True, True], [True, False, False]]
    np.testing.assert_array_equal(out["x"], [[1, 4, np.nan], [0, 2, 5], [3, np.nan, np.nan]])


def toy_vocab():
    measurements = ["m1", "m2", "m3"]
    links = ["sensor:a", "sensor:b", "actuator:a"]
    targets = {"link": links, "source": ["source:a", "source:b"], "sensor": measurements, "plant": ["plant"]}
    flat = [t for k in ("link", "source", "sensor", "plant") for t in targets[k]]
    layout = {
        "window:a": ["m1:last", "m1:frac_unchanged", "m2:last", "m2:frac_unchanged"],
        "window:b": ["m3:last", "m3:frac_unchanged"],
        "field_check:a": ["m1:field", "m2:field", "m1:published", "m2:published", "m1:published_minus_field",
                          "m2:published_minus_field"],
        "field_check:b": ["m3:field", "m3:published", "m3:published_minus_field"],
        "alert": ["stale", "deviation", "score", "subject=source:a", "subject=source:b"]
                 + [f"measurement={m}" for m in measurements],
        "link_health": ["loss", "backup"] + [f"subject={x}" for x in links],
        "trace_link": ["healthy", "backup"] + [f"subject={x}" for x in links],
        "operator_report": ["cause=none", "cause=link_fault"],
    }
    return {
        "groups": ["a", "b"], "measurements": measurements, "links": links,
        "group_measurements": {"a": ["m1", "m2"], "b": ["m3"]}, "sources": list(layout), "payload_layout": layout,
        "subject_blocks": {"alert": [{"key": "subject", "offset": 3, "ids": ["source:a", "source:b"]},
                                     {"key": "measurement", "offset": 5, "ids": measurements}],
                           "link_health": [{"key": "subject", "offset": 2, "ids": links}],
                           "trace_link": [{"key": "subject", "offset": 2, "ids": links}]},
        "targets": targets, "target_kinds": ["link", "source", "sensor", "plant"],
        "action_blocks": {"target": [0, len(flat)]}, "action_columns": [f"target:{t}" for t in flat],
    }


def toy_batch(vocab):
    src = {s: i for i, s in enumerate(vocab["sources"])}
    rows = []

    def alert(stale, group, meas, score, event, arrival, until, retracted=np.inf):
        p = np.zeros(8)
        p[0 if stale else 1], p[2], p[3 + group], p[5 + meas] = 1.0, score, 1.0, 1.0
        rows.append((src["alert"], p, event, arrival, until, retracted))

    alert(True, 0, 0, 2.0, 900, 910, 2000)
    alert(True, 0, 1, 5.0, 950, 960, 1500)
    alert(True, 1, 2, 9.0, 500, 510, 800)
    alert(False, 1, 2, 9.0, 700, 710, 2000, retracted=990)
    alert(False, 1, 2, 9.0, 980, 1200, 3000)
    rows.append((src["link_health"], np.array([1, 0, 0, 1, 0.0]), 800, 810, 2000, np.inf))
    rows.append((src["link_health"], np.array([0, 1, 0, 1, 0.0]), 950, 955, 2000, np.inf))
    rows.append((src["trace_link"], np.array([1, 0, 1, 0, 0.0]), 990, 995, 2000, np.inf))
    rows.append((src["field_check:a"], np.array([1, 1, 1.2, -2.0, 0.2, -3.0]), 970, 975, 2000, np.inf))
    rows.append((src["window:a"], np.array([0.3, 1.0, 0.1, 0.5]), 990, 992, 1100, np.inf))
    rows.append((src["operator_report"], np.array([0, 1.0]), 900, 990, 5000, np.inf))
    m = len(rows)
    P = max(len(v) for v in vocab["payload_layout"].values())
    payload = np.zeros((2, m, P), np.float32)
    fields = {k: np.zeros((2, m)) for k in ("source", "event", "arrival", "until", "retracted")}
    for j, (s, p, e, a, u, r) in enumerate(rows):
        payload[:, j, :len(p)] = p
        for k, v in zip(fields, (s, e, a, u, r)):
            fields[k][:, j] = v
    fields["until"][1] = 0.0
    return EvidenceBatch.from_arrays(T0 + np.array([1000.0, 1000.0]), np.arange(2 * m).reshape(2, m),
                                     fields["source"], T0 + fields["event"], T0 + fields["arrival"],
                                     T0 + fields["until"], payload, retracted_at=T0 + fields["retracted"])


def test_evidence_summary_reads_only_items_valid_at_tau():
    vocab = toy_vocab()
    s = evidence_summary(toy_batch(vocab), vocab)
    g, lk = s["group"], s["link"]
    assert s["n_valid"].tolist() == [8, 0]
    np.testing.assert_array_equal(g["stale"][0], [2, 0])
    np.testing.assert_array_equal(g["deviation"][0], [0, 0])
    np.testing.assert_array_equal(g["stale_share"][0], [1.0, 0.0])
    assert g["max_score"][0, 0] == 5.0 and np.isnan(g["max_score"][0, 1])
    assert s["since_first_alert"][0] == 100 and s["since_last_alert"][0] == 50
    np.testing.assert_array_equal(lk["reports"][0], [0, 2, 0])
    np.testing.assert_array_equal(lk["losses"][0], [0, 1, 0])
    assert lk["latest_loss"][0, 1] == 0.0 and lk["latest_backup"][0, 1] == 1.0
    assert lk["trace_healthy"][0, 0] == 1.0 and np.isnan(lk["trace_healthy"][0, 1])
    assert g["field_gap"][0, 0] == pytest.approx(3.0) and np.isnan(g["field_gap"][0, 1])
    assert g["frozen"][0, 0] == pytest.approx(0.75)
    np.testing.assert_array_equal(s["operator"], [[0, 1], [0, 0]])
    assert s["alerts"][1] == 0 and np.isnan(s["since_first_alert"][1]) and np.isnan(g["field_gap"][1]).all()


def test_candidate_features_read_the_summary_at_the_target():
    vocab = toy_vocab()
    s = evidence_summary(toy_batch(vocab), vocab)
    cols = vocab["action_columns"]
    names = ["sensor:b", "m1", "actuator:a", "plant"]
    feats = np.zeros((4, len(cols)), np.float32)
    for r, t in enumerate(names):
        feats[r, cols.index(f"target:{t}")] = 1.0
    np.testing.assert_array_equal(candidate_targets(feats, vocab), [[1, 1, -1], [0, 0, 0], [0, 2, -1], [-1, -1, -1]])
    X, fnames = feature_matrix(s, feats, np.zeros(4, np.int64), vocab)
    assert X.shape[1] == len(fnames)
    col = {n: X[:, fnames.index(n)] for n in ("target_link:reports", "target_group:stale",
                                               "target_measurement:stale", "target_link:trace_healthy")}
    np.testing.assert_array_equal(col["target_link:reports"][:3], [2, 0, 0])
    np.testing.assert_array_equal(col["target_group:stale"][:3], [0, 2, 2])
    assert col["target_measurement:stale"][1] == 1 and np.isnan(col["target_measurement:stale"][0])
    assert col["target_link:trace_healthy"][1] == 1.0
    aligned = [i for i, n in enumerate(fnames) if n.startswith("target_")]
    assert len(aligned) == 14 and np.isnan(X[3, aligned]).all()


def receipt_vocab():
    vocab = toy_vocab()
    tools, statuses = ["trace_link", "repair_link", "monitor"], ["accepted", "rejected", "running", "succeeded"]
    subjects = [t for k in vocab["target_kinds"] for t in vocab["targets"][k]]
    vocab["payload_layout"]["receipt"] = ([f"tool={t}" for t in tools] + [f"status={s}" for s in statuses]
                                          + [f"subject={s}" for s in subjects])
    vocab["sources"] = list(vocab["payload_layout"])
    vocab["subject_blocks"]["receipt"] = [{"key": "subject", "offset": len(tools) + len(statuses), "ids": subjects}]
    vocab.update(tools=tools, receipt_statuses=statuses)
    n_targets = len(vocab["action_columns"])
    vocab["action_blocks"]["tool"] = [n_targets, n_targets + len(tools)]
    vocab["action_columns"] = vocab["action_columns"] + [f"tool:{t}" for t in tools]
    return vocab


def test_receipt_features_count_valid_receipts_per_tool_and_at_the_target():
    vocab = receipt_vocab()
    src = {s: i for i, s in enumerate(vocab["sources"])}
    tools, statuses = vocab["tools"], vocab["receipt_statuses"]
    subjects = vocab["subject_blocks"]["receipt"][0]["ids"]
    P = len(vocab["payload_layout"]["receipt"])
    rows = [("repair_link", "running", "sensor:b", 900, 2000), ("repair_link", "succeeded", "sensor:b", 950, 2000),
            ("trace_link", "rejected", "sensor:a", 960, 2000), ("trace_link", "succeeded", "sensor:a", 500, 800)]
    payload = np.zeros((1, len(rows), P), np.float32)
    for j, (tool, status, subject, _, _) in enumerate(rows):
        payload[0, j, [tools.index(tool), len(tools) + statuses.index(status),
                       len(tools) + len(statuses) + subjects.index(subject)]] = 1.0
    ev = T0 + np.array([[r[3] for r in rows]], float)
    until = T0 + np.array([[r[4] for r in rows]], float)
    batch = EvidenceBatch.from_arrays(T0 + np.array([1000.0]), np.arange(len(rows))[None],
                                      np.full((1, len(rows)), src["receipt"]), ev, ev + 1, until, payload)
    s = evidence_summary(batch, vocab)
    r = s["receipt"]
    np.testing.assert_array_equal(r["count"][0], [1, 2, 0])
    assert r["latest_ok"][0, 1] == 1.0 and r["latest_ok"][0, 0] == 0.0 and np.isnan(r["latest_ok"][0, 2])
    cols = vocab["action_columns"]
    feats = np.zeros((3, len(cols)), np.float32)
    for k, (target, tool) in enumerate([("sensor:b", "repair_link"), ("sensor:b", "trace_link"), ("plant", "monitor")]):
        feats[k, cols.index(f"target:{target}")] = feats[k, cols.index(f"tool:{tool}")] = 1.0
    X, names = feature_matrix(s, feats, np.zeros(3, np.int64), vocab)
    col = {n: X[:, names.index(n)] for n in ("target_receipt:n", "target_receipt:running",
                                             "target_receipt:same_tool_status", "receipt_count:repair_link")}
    np.testing.assert_array_equal(col["target_receipt:n"], [2, 2, 0])
    np.testing.assert_array_equal(col["target_receipt:running"], [1, 1, 0])
    assert col["target_receipt:same_tool_status"][0] == statuses.index("succeeded")
    assert np.isnan(col["target_receipt:same_tool_status"][1:]).all()
    np.testing.assert_array_equal(col["receipt_count:repair_link"], [2, 2, 2])


@pytest.mark.parametrize("center", [False, True])
def test_flat_baseline_learns_a_toy_rule(center):
    rng = np.random.default_rng(18)

    def make(n):
        alert = rng.integers(0, 2, n).astype(float)
        X = np.concatenate([np.tile(np.eye(3), (n, 1)), np.repeat(alert, 3)[:, None]], 1)
        J = np.c_[5 + 10 * alert, np.full(n, 7.0), 6 + 5 * alert].reshape(-1) + rng.normal(0, 0.3, 3 * n)
        p = np.c_[0.75 * alert, np.zeros(n), np.zeros(n)].reshape(-1)
        return X, J, p, np.repeat(np.arange(n), 3), alert

    X, J, p, dec, _ = make(300)
    fb = FlatBaseline(max_iter=100, learning_rate=0.1, max_leaf_nodes=7, min_samples_leaf=5, center=center)
    fb.fit(X, J, p, dec)
    Xt, _, _, dect, alert = make(50)
    J_pred, trip = fb.predict(Xt)
    chosen = local_argmin(J_pred, dect, 50)
    np.testing.assert_array_equal(chosen, np.where(alert > 0, 1, 0))
    risky = (np.arange(150) % 3 == 0) & (np.repeat(alert, 3) > 0)
    assert trip[risky].min() > trip[~risky].max()


def test_local_argmin_uses_row_order_within_decisions():
    dec = np.array([1, 0, 1, 0, 1])
    pred = np.array([3.0, 2.0, 1.0, 2.0, 1.0])
    assert local_argmin(pred, dec, 2).tolist() == [0, 1]


def fake_split(n=40, episodes=10, seed=19):
    rng = np.random.default_rng(seed)
    tools = ["trace_link", "repair_link", "monitor"]
    vocab = {"action_blocks": {"tool": [0, 3]}, "tools": tools,
             "action_columns": ["tool:trace_link", "tool:repair_link", "tool:monitor", "arg:monitor.review_after_s"]}
    feats = np.zeros((n, 4, 4), np.float32)
    feats[:, 0, 0] = feats[:, 1, 1] = feats[:, 2, 2] = feats[:, 3, 2] = 1.0
    feats[:, 3, 3] = 1.0
    J = rng.uniform(0, 10, (n, 4))
    observe = np.tile([True, False, False, False], (n, 1))
    voi = np.where(observe, J[:, 1:].min(1, keepdims=True) - J, np.nan)
    split = {"n": n, "mask": np.ones((n, 4), bool),
             "pad": {"J": J, "p_trip": rng.choice([0.0, 0.25, 0.5], (n, 4)), "observe": observe, "features": feats,
                     "voi": voi},
             "dec": {"episode_seed": np.repeat(np.arange(episodes) * 7, n // episodes), "kind": np.zeros(n, int),
                     "baseline_act_first": np.ones(n, int), "baseline_investigate_first": np.zeros(n, int)}}
    return split, vocab


def test_bootstrap_scores_cluster_by_episode_and_average_seeds():
    td = load_script()
    split, vocab = fake_split()
    ctx = td.context(split, vocab, material_gap=0.5)
    J = split["pad"]["J"]
    methods = td.baseline_methods(ctx)
    assert (ctx["monitor"] == 2).all()
    oracle = td.score(ctx, methods["oracle"], n_boot=200)
    assert oracle["regret"]["point"] == 0 and oracle["regret"]["lo"] == oracle["regret"]["hi"] == 0
    assert oracle["regret"]["n_clusters"] == 10
    mon = td.score(ctx, methods["monitor"], n_boot=200)
    assert mon["regret"]["point"] == pytest.approx(np.mean(J[:, 2] - J.min(1)))
    assert mon["regret"]["lo"] <= mon["regret"]["point"] <= mon["regret"]["hi"]
    assert td.score(ctx, methods["random"], 50)["regret"]["point"] == pytest.approx(np.mean(J - J.min(1, keepdims=True)))
    tg = J[:, 0] < J[:, 1:].min(1)
    assert td.score(ctx, methods["always_act"], 50)["gva_accuracy"]["point"] == pytest.approx(np.mean(~tg))
    assert "regret" not in td.score(ctx, methods["always_act"], 50)
    p = split["pad"]["p_trip"]
    rows = td.rows_for_predictions(ctx, np.stack([J, -J]), np.stack([p, p]))
    sc = td.score(ctx, rows, n_boot=200)
    worst = np.mean(J.max(1) - J.min(1))
    assert sc["regret"]["per_seed"] == pytest.approx([0.0, worst])
    assert sc["regret"]["point"] == pytest.approx(worst / 2) and sc["regret"]["seed_std"] == pytest.approx(worst / 2)
    assert sc["voi_sign_accuracy"]["per_seed"][0] == 1.0 and sc["trip_brier"]["point"] == 0.0
    same = td.paired(ctx, rows, rows, n_boot=100)
    assert all(d["point"] == 0 and d["lo"] == d["hi"] == 0 for d in same.values())
    diff = td.paired(ctx, methods["oracle"], methods["monitor"], n_boot=200)
    assert diff["regret"]["point"] == pytest.approx(-mon["regret"]["point"]) and diff["regret"]["hi"] < 0
    assert set(diff) >= {"regret", "regret_high_stakes", "trip_rate_high_stakes", "gva_accuracy"}
    rep = td.report(ctx, {"model": (np.stack([J, -J]), np.stack([p, p]))}, n_boot=50)
    assert rep["reference"] == "model" and "playbook_act_first" in rep["paired"]
    assert rep["scores"]["playbook_act_first"]["regret"]["point"] == pytest.approx(np.mean(J[:, 1] - J.min(1)))
    assert "| model |" in td.markdown(rep, "test")


def belief_split(n=40, episodes=10, seed=19):
    split, vocab = fake_split(n, episodes, seed)
    rng = np.random.default_rng(seed + 1)
    pad = split["pad"]
    Jb = np.take_along_axis(pad["J"], rng.permuted(np.tile(np.arange(4), (n, 1)), axis=1), 1)
    pad.update(J_belief=Jb, p_trip_belief=pad["p_trip"][:, ::-1].copy(),
               voi_belief=np.where(pad["observe"], Jb[:, 1:].min(1, keepdims=True) - Jb, np.nan))
    split["dec"].update(family=np.arange(n) % 2 == 0, backup_available=np.arange(n) % 4 < 2)
    split["cand"] = {"J": pad["J"].ravel(), "p_trip": pad["p_trip"].ravel(), "J_belief": Jb.ravel(),
                     "p_trip_belief": pad["p_trip_belief"].ravel()}
    return split, vocab


def test_belief_view_scores_against_belief_labels_with_the_hindsight_oracle_as_reference():
    td = load_script()
    split, vocab = belief_split()
    J, Jb = split["pad"]["J"], split["pad"]["J_belief"]
    ctx = td.context(split, vocab, labels="belief")
    np.testing.assert_array_equal(ctx["J"], Jb)
    assert ctx["cross_oracle"][0] == "hindsight_oracle"
    methods = td.baseline_methods(ctx)
    r = np.arange(len(J))
    regret = td.score(ctx, methods["hindsight_oracle"], n_boot=50)["regret"]["point"]
    assert regret == pytest.approx(np.mean(Jb[r, J.argmin(1)] - Jb.min(1)))
    assert td.score(ctx, methods["oracle"], n_boot=50)["regret"]["point"] == 0
    hind = td.baseline_methods(td.context(split, vocab))
    assert "belief_oracle" in hind and "hindsight_oracle" not in hind
    fake, _ = fake_split()
    assert td.context(fake, vocab)["cross_oracle"] is None
    with pytest.raises(ValueError):
        td.context(split, vocab, labels="posterior")


def test_label_view_swaps_the_training_targets_and_subsets_slice_decisions():
    td = load_script()
    split, _ = belief_split()
    view = td.label_view(split, "belief")
    for k in ("J", "p_trip", "voi"):
        np.testing.assert_array_equal(view["pad"][k], split["pad"][f"{k}_belief"])
    np.testing.assert_array_equal(view["cand"]["J"], split["cand"]["J_belief"])
    np.testing.assert_array_equal(split["pad"]["J"], fake_split()[0]["pad"]["J"])
    assert td.label_view(split, "hindsight") is split
    with pytest.raises(ValueError):
        td.label_view(fake_split()[0], "belief")
    subs = dict(td.subsets(split))
    assert list(subs) == ["all", "family", "backup_available", "backup_unavailable", "family_backup_available",
                          "family_backup_unavailable"]
    part = td.take(split, subs["family_backup_available"])
    assert part["n"] == 10 and part["dec"]["family"].all() and part["dec"]["backup_available"].all()
    np.testing.assert_array_equal(part["pad"]["J"], split["pad"]["J"][subs["family_backup_available"]])
    split["dec"]["backup_available"][:] = True
    assert [name for name, _ in td.subsets(split)] == ["all", "family"]


def test_evaluate_writes_both_label_sets_per_subset(tmp_path):
    td = load_script()
    split, vocab = belief_split()
    J = split["pad"]["J_belief"]
    cfg = {"eval": {"material_gap": 0.5, "n_boot": 20}}
    res = td.evaluate({"val": split}, {"val": {"model": (J[None], None)}}, vocab, cfg, tmp_path, "t", lambda m: None)
    views = res["val"]
    assert list(views)[0] == "belief/all" and "hindsight/family_backup_unavailable" in views
    assert views["belief/all"]["scores"]["model"]["regret"]["point"] == 0
    assert views["hindsight/all"]["scores"]["model"]["regret"]["point"] >= 0
    assert (tmp_path / "results_val_belief_family.md").exists() and (tmp_path / "summary_val.md").exists()


@pytest.mark.parametrize("ema, stop_on", [(0.0, "total"), (0.9, "J_centered"), (0.9, {"J_centered": 1.0, "gva": 0.5})])
def test_training_loop_runs_end_to_end(ema, stop_on):
    td = load_script()
    batch = make_batch(23, n=12, m=10)
    feats, _ = candidates(24, n=12)
    mask = np.ones((12, 5), bool)
    mask[::3, 4] = False
    y = {k: v.numpy() for k, v in labels(mask, 25).items()}
    split = {"n": 12, "batch": batch, "mask": mask, "pad": {**y, "features": np.where(mask[..., None], feats, 0)}}
    dev = {"device": "cpu", "train": td.to_device(split, "cpu"), "val": td.to_device(split, "cpu")}
    backbone = dict(payload_dim=PAYLOAD_DIM, n_kinds=1, mode="masked", d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_pool=2, branch_layers=1, n_query_ids=0, descriptor_dim=F, dropout=0.1)
    cfg = {"model": {"backbone": backbone, "head_hidden": 16, "weights": {"J_centered": 1.0}},
           "train": {"lr": 1e-3, "weight_decay": 0.0, "batch_decisions": 4, "max_epochs": 3, "patience": 5,
                     "grad_clip": 1.0, "ema": ema, "stop_on": stop_on}}
    model, info = td.train_one(cfg, dev, 0, lambda m: None)
    assert info["epochs"] == 3 and 1 <= info["best_epoch"] <= 3 and len(info["val_regret_history"]) == 3
    assert info["best_val_loss"] == pytest.approx(min(info["history"]))
    J, trip = td.predict(model, dev["val"], chunk=5)
    assert J.shape == trip.shape == (12, 5) and np.isfinite(J[mask]).all() and ((trip >= 0) & (trip <= 1)).all()


def test_ablation_configs_change_one_thing():
    td = load_script()
    main = td.run_config("model/direct.yaml")

    def stop(gva):
        return {("train", "stop_on", "J_centered"): 1.0, ("train", "stop_on", "gva"): gva}

    expected = {"direct_block_local": {("model", "backbone", "mode"): "block_local"},
                "direct_unmasked": {("model", "backbone", "mode"): "unmasked"},
                "direct_rank": {("model", "weights", "rank"): 1.0},
                "direct_pilot_recipe": {("model", "weights", "rank"): 1.0, ("model", "weights", "J_centered"): 0.0,
                                        ("train", "stop_on"): "total", ("train", "ema"): 0.0,
                                        ("train", "patience"): 30},
                "direct_j_only": {("model", "weights", k): 0.0 for k in ("trip", "hazard", "resolve", "duration")},
                "direct_gva_balanced": {("model", "weights", "gva"): 1.0, ("model", "gva_margin"): 0.5,
                                        ("model", "gva_balance"): True, **stop(1.0)},
                "direct_gva_regret": {("model", "weights", "gva"): 1.0, ("model", "gva_regret_weight"): True,
                                      **stop(1.0)},
                "direct_gva_regret_light": {("model", "weights", "gva"): 0.1, ("model", "gva_regret_weight"): True,
                                            **stop(0.1)},
                "direct_gva_labelled": {("model", "weights", "gva"): 1.0, ("model", "gva_regret_weight"): True,
                                        ("model", "gva_labelled_best"): True, **stop(1.0)}}

    def leaves(d, prefix=()):
        for k, v in d.items():
            yield from leaves(v, prefix + (k,)) if isinstance(v, dict) else [(prefix + (k,), v)]

    base = dict(leaves(main))
    for name, change in expected.items():
        cfg = dict(leaves(td.run_config(f"model/{name}.yaml")))
        assert {k: v for k, v in cfg.items() if base.get(k) != v} == change, name
    v2 = dict(leaves(td.run_config("model/direct_v2.yaml")))
    v2_expected = {"direct_v2_summary": {("model", "summary_descriptor"): True},
                   "direct_v2_hindsight": {("labels",): "hindsight"},
                   "direct_v2_summary_aux": {("model", "summary_aux"): True, ("model", "weights", "summary_aux"): 0.3}}
    for name, change in v2_expected.items():
        cfg = dict(leaves(td.run_config(f"model/{name}.yaml")))
        assert {k: v for k, v in cfg.items() if v2.get(k) != v} == change, name
    summary = dict(leaves(td.run_config("model/direct_v2_summary.yaml")))
    for name, change in {"direct_v2_summary_noitems": {("model", "item_path"): False},
                         "direct_v2_summary_stopregret": {("train", "stop_on"): "regret", ("train", "stop_smooth"): 5,
                                                          ("train", "min_epochs"): 10}}.items():
        cfg = dict(leaves(td.run_config(f"model/{name}.yaml")))
        assert {k: v for k, v in cfg.items() if summary.get(k) != v} == change, name


def receipt_batch(vocab):
    src = {s: i for i, s in enumerate(vocab["sources"])}
    tools, statuses = vocab["tools"], vocab["receipt_statuses"]
    subjects = vocab["subject_blocks"]["receipt"][0]["ids"]
    P = max(len(v) for v in vocab["payload_layout"].values())
    rows = {0: [], 1: []}

    def receipt(d, tool, status, subject, event, until, retracted=np.inf):
        p = np.zeros(P)
        p[[tools.index(tool), len(tools) + statuses.index(status),
           len(tools) + len(statuses) + subjects.index(subject)]] = 1.0
        rows[d].append((src["receipt"], p, event, event + 1, until, retracted))

    def alert(d, event, arrival):
        p = np.zeros(P)
        p[0], p[2], p[3], p[5] = 1.0, 2.0, 1.0, 1.0
        rows[d].append((src["alert"], p, event, arrival, 5000, np.inf))

    receipt(0, "repair_link", "succeeded", "sensor:b", 900, 2000)
    receipt(0, "trace_link", "succeeded", "sensor:a", 500, 800)
    receipt(0, "trace_link", "rejected", "sensor:a", 950, 2000, retracted=990)
    alert(0, 900, 910)
    receipt(1, "trace_link", "running", "sensor:a", 960, np.inf)
    alert(1, 990, 1200)
    m = max(len(r) for r in rows.values())
    payload = np.zeros((2, m, P), np.float32)
    f = {k: np.full((2, m), np.nan) for k in ("source", "event", "arrival", "until", "retracted")}
    present = np.zeros((2, m), bool)
    for d, rs in rows.items():
        for j, (s, p, e, a, u, r) in enumerate(rs):
            payload[d, j], present[d, j] = p, True
            for k, v in zip(f, (s, e, a, u, r)):
                f[k][d, j] = v
    f["source"] = np.nan_to_num(f["source"]).astype(np.int64)
    item_id = np.where(present, np.arange(2 * m).reshape(2, m), -1)
    return EvidenceBatch.from_arrays(T0 + np.array([1000.0, 1000.0]), item_id, f["source"], T0 + f["event"],
                                     T0 + f["arrival"], T0 + f["until"], payload, present=present,
                                     retracted_at=T0 + f["retracted"])


def summary_inputs(vocab):
    batch = receipt_batch(vocab)
    cols = vocab["action_columns"]
    cands = [(1, "sensor:a", "trace_link"), (0, "sensor:b", "repair_link"), (0, "sensor:a", "trace_link"),
             (1, "plant", "monitor")]
    feats = np.zeros((len(cands), len(cols)), np.float32)
    for k, (_, target, tool) in enumerate(cands):
        feats[k, cols.index(f"target:{target}")] = feats[k, cols.index(f"tool:{tool}")] = 1.0
    return batch, feats, np.array([c[0] for c in cands], np.int64)


def test_summary_descriptor_reads_only_valid_items_and_each_row_only_its_own_candidate():
    from cpsdefense.model.summary import SummaryDescriptor
    vocab = receipt_vocab()
    batch, feats, dec = summary_inputs(vocab)
    desc = SummaryDescriptor(vocab)
    S, names = desc.raw(batch, feats, dec)
    col = {n: S[:, i] for i, n in enumerate(names)}
    assert S.shape == (4, len(names)) and not any(n.startswith("target:") for n in names)
    np.testing.assert_array_equal(col["n_valid"], [1, 2, 2, 1])
    np.testing.assert_array_equal(col["target_receipt:n"], [1, 1, 0, 0])
    statuses = vocab["receipt_statuses"]
    assert col["target_receipt:same_tool_status"][0] == statuses.index("running")
    assert col["target_receipt:same_tool_status"][1] == statuses.index("succeeded")
    assert np.isnan(col["target_receipt:same_tool_status"][2:]).all()
    np.testing.assert_array_equal(col["target_receipt:running"], [1, 0, 0, 0])
    np.testing.assert_array_equal(col["receipt_count:trace_link"], [1, 0, 0, 1])
    junk = batch.replace(payload=batch.payload.copy(), source=batch.source.copy(),
                         event_time=batch.event_time.copy())
    invalid = ~batch.validity()
    assert (invalid & batch.present).sum() == 3
    junk.payload[invalid] = np.nan
    junk.source[invalid] = 0
    junk.event_time[invalid] = T0 + 999.0
    S2, _ = desc.raw(junk, feats, dec)
    np.testing.assert_array_equal(S, S2)
    S3, _ = desc.raw(batch, feats[[3, 1]], dec[[3, 1]])
    np.testing.assert_array_equal(S3, S[[3, 1]])
    target_cols = [i for i, n in enumerate(names) if n.startswith("target_")]
    decision_cols = [i for i, n in enumerate(names) if not n.startswith("target_")]
    np.testing.assert_array_equal(S[1, decision_cols], S[2, decision_cols])
    assert not np.array_equal(S[1, target_cols], S[2, target_cols], equal_nan=True)


def test_summary_descriptor_standardises_with_train_statistics_and_round_trips(tmp_path):
    from cpsdefense.model.summary import SummaryDescriptor
    vocab = receipt_vocab()
    S = np.array([[1.0, np.nan, 5.0], [3.0, 2.0, 5.0], [5.0, np.nan, 5.0]])
    desc = SummaryDescriptor(vocab, clip=1.0).fit(S, ["a", "b", "c"])
    assert desc.width == 4 and desc.column_names()[-1] == "missing:b"
    Z = desc.transform(np.array([[3.0, np.nan, 9.0], [100.0, 2.0, 5.0], [-100.0, 2.0, 5.0]]))
    np.testing.assert_allclose(Z[0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(Z[1], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(Z[2], [-1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(desc.transform(S)[0], [-1.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        desc.transform(S, ["a", "c", "b"])
    desc.save(tmp_path / "d.npz")
    back = SummaryDescriptor.load(tmp_path / "d.npz", vocab)
    np.testing.assert_array_equal(back.transform(S), desc.transform(S))
    with pytest.raises(ValueError):
        SummaryDescriptor(vocab).transform(S)
    empty = SummaryDescriptor(vocab).fit(np.array([[1.0, np.nan], [3.0, np.nan]]), ["a", "never"])
    np.testing.assert_allclose(empty.transform(np.array([[1.0, 7.0], [1.0, np.nan]])),
                               [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])


def test_add_summary_descriptor_fits_on_train_only_and_appends_after_the_action_features():
    from cpsdefense.model.summary import SummaryDescriptor
    td = load_script()
    vocab = receipt_vocab()
    batch, feats, dec = summary_inputs(vocab)

    def split(b, f, d, n):
        mask, pad = pad_candidates(d, n, {"features": f})
        return {"n": n, "batch": b, "mask": mask, "pad": {"features": pad["features"].copy()},
                "cand": {"features": f, "decision": d}}

    train = split(batch, feats, dec, 2)
    val = split(toy_batch(vocab), feats[[1, 0, 3]], np.array([0, 0, 1]), 2)
    data = {"train": train, "val": val}
    desc = td.add_summary_descriptor(data, vocab, lambda m: None)
    ref = SummaryDescriptor(vocab).fit(*SummaryDescriptor(vocab).raw(batch, feats, dec))
    for key in ("mean", "std", "miss_cols"):
        np.testing.assert_array_equal(getattr(desc, key), getattr(ref, key))
    F_ = feats.shape[1]
    for s, d in data.items():
        full = d["pad"]["features"]
        assert full.shape[-1] == F_ + desc.width
        expected = desc.transform(*desc.raw(d["batch"], d["cand"]["features"], d["cand"]["decision"]))
        rows = d["mask"]
        np.testing.assert_array_equal(full[rows][:, :F_], d["cand"]["features"][np.argsort(d["cand"]["decision"],
                                                                                            kind="stable")])
        np.testing.assert_array_equal(full[rows][:, F_:], expected[np.argsort(d["cand"]["decision"], kind="stable")])
        assert (full[~rows] == 0).all()
    assert not data["val"]["mask"].all()
    np.testing.assert_array_equal(data["val"]["cand"]["features"], feats[[1, 0, 3]])


def test_summary_only_arm_ignores_every_item_and_reports_the_valid_ones():
    model = make_model(item_path=False)
    batch, (feats, mask) = make_batch(30), candidates(31)
    base = run(model, batch, feats, mask)
    rng = np.random.default_rng(32)
    other = batch.replace(payload=rng.normal(size=batch.payload.shape).astype(np.float32) * 100,
                          valid_until=batch.valid_until + 1e6)
    moved = run(model, other, feats, mask)
    for key in OUTPUTS:
        assert torch.equal(base[key], moved[key]), key
    with torch.no_grad():
        out = model(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))
        later = model(batch, torch.as_tensor(feats), tau=batch.decision_time + 400.0, mask=torch.as_tensor(mask))
    assert torch.equal(out["used"].cpu(), torch.as_tensor(batch.validity()))
    assert torch.equal(later["used"].cpu(), torch.as_tensor(batch.validity(batch.decision_time + 400.0)))
    other_feats = feats.copy()
    other_feats[:, 1] += 1.0
    changed = run(model, batch, other_feats, mask)
    real = torch.as_tensor(mask)
    assert not torch.equal(changed["J_pred"][:, 1][real[:, 1]], base["J_pred"][:, 1][real[:, 1]])
    torch.testing.assert_close(changed["J_pred"][:, 0], base["J_pred"][:, 0])


def test_summary_aux_head_is_a_training_target_only():
    batch, (feats, mask) = make_batch(43), candidates(44)
    plain, aux = make_model(seed=5), make_model(seed=5, summary_aux=True, summary_aux_dim=7,
                                                weights={"summary_aux": 0.3})
    a, b = run(plain, batch, feats, mask), run(aux, batch, feats, mask)
    for key in OUTPUTS:
        assert torch.equal(a[key], b[key]), key
    aux.train()
    out = aux(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))
    assert out["summary_aux"].shape == (4, 5, 7)
    y = labels(mask, 45)
    y["summary"] = torch.as_tensor(np.random.default_rng(46).normal(size=(4, 5, 7)), dtype=torch.float32)
    loss, parts = losses(out, y, torch.as_tensor(mask), aux.config)
    assert np.isfinite(parts["summary_aux"]) and parts["summary_aux"] > 0
    loss.backward()
    enc_grad = [p.grad for p in aux.backbone.encoder.parameters() if p.grad is not None]
    assert enc_grad and any(float(g.abs().sum()) > 0 for g in enc_grad)
    with torch.no_grad():
        plain_out = plain(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))
    _, plain_parts = losses(plain_out, y, torch.as_tensor(mask), plain.config)
    assert "summary_aux" not in plain_parts
    with pytest.raises(ValueError):
        make_model(summary_aux=True)


def test_summary_descriptor_model_refuses_another_scoring_time():
    model = make_model(summary_descriptor=True)
    batch, (feats, mask) = make_batch(33), candidates(34)
    with pytest.raises(ValueError):
        model(batch, torch.as_tensor(feats), tau=batch.decision_time + 60.0, mask=torch.as_tensor(mask))
    model(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))


def test_ensemble_averages_members_and_keeps_menu_invariance():
    from cpsdefense.model.direct import EnsembleDecisionModel
    members = [make_model(seed=0), make_model(mode="unmasked", seed=1)]
    ens = EnsembleDecisionModel(members).eval()
    batch, (feats, mask) = make_batch(35), candidates(36)
    with torch.no_grad():
        outs = [m(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask)) for m in members]
        e = ens(batch, torch.as_tensor(feats), mask=torch.as_tensor(mask))
        perm = np.array([4, 2, 0, 3, 1])
        ep = ens(batch, torch.as_tensor(feats[:, perm]), mask=torch.as_tensor(mask[:, perm]))
    torch.testing.assert_close(e["J_pred"], (outs[0]["J_pred"] + outs[1]["J_pred"]) / 2)
    p = (torch.sigmoid(outs[0]["trip_logit"]) + torch.sigmoid(outs[1]["trip_logit"])) / 2
    torch.testing.assert_close(torch.sigmoid(e["trip_logit"]), p, atol=1e-5, rtol=1e-5)
    real = torch.as_tensor(mask[:, perm])
    torch.testing.assert_close(e["J_pred"][:, perm][real], ep["J_pred"][real])
    assert torch.equal(e["used"], outs[0]["used"] | outs[1]["used"])
    assert (e["used"] & ~outs[0]["used"]).any()
    with pytest.raises(ValueError):
        EnsembleDecisionModel([])


def test_regret_splits_into_family_and_within_and_counterfactuals_use_the_own_ranking():
    td = load_script()
    split, vocab = fake_split()
    ctx = td.context(split, vocab)
    J = split["pad"]["J"]
    rng = np.random.default_rng(37)
    Jp = np.stack([J + rng.normal(0, 3, J.shape), J + rng.normal(0, 3, J.shape)])
    rows = td.rows_for_predictions(ctx, Jp)
    np.testing.assert_allclose(rows["regret_family"] + rows["regret_within"], rows["regret"])
    assert (rows["regret_family"] >= 0).all() and (rows["regret_within"] >= -1e-12).all()
    oracle = td.rows_for_choice(ctx, J.argmin(1))
    assert np.all(oracle["regret_family"] == 0) and np.all(oracle["regret_within"] == 0)
    exact = td.rows_for_predictions(ctx, J[None])
    assert np.nanmax(np.abs(exact["forced_act"])) == 0 and np.all(exact["cf_labelled_family"] == 0)
    always_act = J[:, 1:].min(1) - J.min(1)
    np.testing.assert_allclose(exact["cf_always_act"][:, 0], always_act)
    sc = td.score(ctx, rows, n_boot=50)
    assert {"regret_family", "regret_within", "forced_act_regret", "cf_labelled_family_regret",
            "cf_always_act_regret"} <= set(sc)
    assert sc["regret"]["seed_std_ddof1"] == pytest.approx(np.std(sc["regret"]["per_seed"], ddof=1))


def hand_split(J, observe, mask):
    n, k = J.shape
    vocab = {"action_blocks": {"tool": [0, 3]}, "tools": ["trace_link", "repair_link", "monitor"],
             "action_columns": ["tool:trace_link", "tool:repair_link", "tool:monitor", "arg:monitor.review_after_s"]}
    feats = np.zeros((n, k, 4), np.float32)
    feats[..., 0], feats[..., 1] = observe, ~observe
    J = np.where(mask, J, np.nan)
    voi = np.where(observe, np.nanmin(np.where(mask & ~observe, J, np.nan), 1, keepdims=True) - J, np.nan)
    split = {"n": n, "mask": mask, "pad": {"J": J, "p_trip": np.zeros((n, k)), "observe": observe & mask,
                                         "features": feats, "voi": voi},
             "dec": {"episode_seed": np.arange(n), "kind": np.zeros(n, int)}}
    return split, vocab


def test_counterfactual_family_choices_use_the_methods_own_ranking():
    td = load_script()
    inf = np.inf
    J = np.array([[5.0, 1.0, 3.0, 4.0],
                  [2.0, 6.0, 4.0, inf]])
    observe = np.array([[True, True, False, False], [False, False, False, False]])
    mask = np.array([[True] * 4, [True, True, True, False]])
    split, vocab = hand_split(J, observe, mask)
    ctx = td.context(split, vocab)
    Jp = np.array([[[0.0, 2.0, 10.0, 9.0], [3.0, 1.0, 2.0, -50.0]]])
    rows = td.rows_for_predictions(ctx, Jp)
    assert rows["forced_act"][0, 0] == pytest.approx(4.0 - 3.0)
    assert np.isnan(rows["forced_act"][1, 0])
    assert rows["cf_labelled_family"][0, 0] == pytest.approx(5.0 - 1.0)
    assert rows["cf_always_act"][0, 0] == pytest.approx(4.0 - 1.0)
    assert rows["cf_labelled_family"][1, 0] == rows["cf_always_act"][1, 0] == pytest.approx(6.0 - 2.0)
    assert rows["regret"][0, 0] == pytest.approx(4.0)
    assert rows["regret_family"][0, 0] == 0 and rows["regret_within"][0, 0] == pytest.approx(4.0)


def test_gva_diagnostics_tail_measures():
    td = load_script()
    rng = np.random.default_rng(39)
    n = 60
    J = rng.uniform(0, 5, (n, 4))
    J[:20, 0] += 15.0
    observe = np.zeros((n, 4), bool)
    observe[:, 0] = True
    split, vocab = hand_split(J, observe, np.ones((n, 4), bool))
    ctx = td.context(split, vocab)
    tail = ctx["has_both"] & (ctx["gap"] >= 10)
    assert tail.sum() == 20
    gather_always = np.where(observe, -100.0, J)[None]
    d = td.gva_diagnostics(ctx, gather_always)
    assert d["tail_decisions"] == 20 and d["tail_gather_share"] == 1.0
    assert d["tail_family_regret"] == pytest.approx((J[:20, 0] - J[:20, 1:].min(1)).sum() / n)
    oracle = td.gva_diagnostics(ctx, J[None])
    assert oracle["tail_family_regret"] == 0 and oracle["tail_gather_share"] == 0.0
    assert oracle["tail_auc_gap"] == pytest.approx(1.0)
    assert td._auc(np.array([0, 1, 1]), np.array([0.1, np.nan, 0.9])) == 1.0


def test_gva_diagnostics_bias_free_measures():
    td = load_script()
    split, vocab = fake_split(n=200, episodes=20, seed=38)
    split["pad"]["p_trip"][:, 1] = 0.5
    split["pad"]["p_trip"][:, 2] = 0.0
    ctx = td.context(split, vocab)
    J = split["pad"]["J"]
    perfect = td.gva_diagnostics(ctx, J[None])
    assert perfect["gap_spearman"] == pytest.approx(1.0) and perfect["gap_auc_true_gather"] == pytest.approx(1.0)
    assert perfect["regret_family"] == 0 and perfect["tail_family_regret"] == 0
    assert perfect["act_corr_high_stakes"] == pytest.approx(1.0) and perfect["centred_slope"] == pytest.approx(1.0)
    assert perfect["bias_gap_zero_predictor"] == pytest.approx(np.mean(-ctx["gap"][ctx["has_both"]]))
    shrunk = td.gva_diagnostics(ctx, (0.5 * J)[None])
    assert shrunk["centred_slope"] == pytest.approx(0.5) and shrunk["gap_spearman"] == pytest.approx(1.0)


def test_crossfit_epoch_selection_chooses_on_one_half_and_scores_on_the_other():
    td = load_script()
    halves = [[(3.0, 0.9, 10), (3.0, 0.9, 30)],
              [(1.0, 0.5, 10), (2.5, 0.4, 30)],
              [(2.0, 0.6, 10), (1.0, 0.8, 30)]]
    res = td.crossfit_epoch_selection(halves, "regret")
    assert res["epochs"] == [2, 3]
    assert res["regret"] == pytest.approx((2.5 * 30 + 2.0 * 10) / 40)
    assert td.crossfit_epoch_selection(halves, "J_centered")["epochs"] == [2, 2]
    assert td.crossfit_epoch_selection(halves, "regret", min_epochs=2)["epochs"] == [3, 3]
    np.testing.assert_allclose(td.smoothed([1.0, 3.0, 5.0], 2), [1.0, 2.0, 4.0])
    per_dec = {"regret": np.array([1.0, 3.0, 5.0]), "jc_se": np.array([1.0, 1.0, 4.0]), "jc_n": np.array([2, 2, 4])}
    assert td.half_scores(per_dec, np.array([0, 1, 1])) == [(1.0, 0.5, 1), (4.0, 5.0 / 6.0, 2)]
    spiky = [[(r, 0.5, 10), (2.0, 0.5, 10)] for r in (3.0, 1.0, 4.0, 1.2, 1.2)]
    assert td.crossfit_epoch_selection(spiky, "regret")["epochs"][0] == 2
    assert td.crossfit_epoch_selection(spiky, "regret", smooth=2)["epochs"][0] == 5
    assert td.crossfit_epoch_selection(halves, "regret", min_epochs=10)["epochs"] == [3, 3]
    empty_half = [[(1.0, 0.5, 5), (np.nan, 0.0, 0)]]
    res = td.crossfit_epoch_selection(empty_half, "regret")
    assert np.isnan(res["regret"]) and res["epochs"] == [None, None]
    halves_of = td.episode_halves(np.array([7, 7, 3, 3, 9, 9, 11]))
    assert halves_of.tolist() == [1, 1, 0, 0, 0, 0, 1]


def test_subsample_episodes_keeps_whole_episodes_and_renumbers_candidates():
    td = load_script()
    n = 8
    batch = make_batch(47, n=n, m=6)
    dec_of = np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 7])
    feats = np.arange(dec_of.size, dtype=np.float32)[:, None].repeat(2, 1)
    mask, pad = pad_candidates(dec_of, n, {"features": feats, "J": feats[:, 0].astype(np.float64)})
    split = {"n": n, "batch": batch, "mask": mask, "pad": pad,
             "dec": {"episode_seed": np.array([10, 10, 11, 11, 12, 12, 13, 13]), "tau": np.arange(n, dtype=float)},
             "cand": {"features": feats, "decision": dec_of, "J": feats[:, 0].astype(np.float64)}}
    half = td.subsample_episodes(split, 0.5, seed=0)
    kept = set(half["episodes_kept"].tolist())
    assert len(kept) == 2 and set(half["dec"]["episode_seed"].tolist()) == kept
    rows = np.flatnonzero(np.isin(split["dec"]["episode_seed"], list(kept)))
    assert half["n"] == rows.size
    np.testing.assert_array_equal(half["dec"]["tau"], rows)
    np.testing.assert_array_equal(half["batch"].decision_time, batch.decision_time[rows])
    for i, r in enumerate(rows):
        np.testing.assert_array_equal(half["cand"]["J"][half["cand"]["decision"] == i], feats[dec_of == r, 0])
    np.testing.assert_array_equal(half["pad"]["J"], pad["J"][rows])
    quarter = td.subsample_episodes(split, 0.25, seed=0)
    assert set(quarter["episodes_kept"].tolist()) <= kept
    with pytest.raises(ValueError):
        td.subsample_episodes(split, 0.0)


def test_training_with_regret_stopping_smoothing_and_min_epochs():
    td = load_script()
    batch = make_batch(40, n=12, m=10)
    feats, _ = candidates(41, n=12)
    mask = np.ones((12, 5), bool)
    y = {k: v.numpy() for k, v in labels(mask, 42).items()}
    split = {"n": 12, "batch": batch, "mask": mask, "pad": {**y, "features": feats},
             "dec": {"episode_seed": np.repeat(np.arange(4), 3)}}
    dev = {"device": "cpu", "train": td.to_device(split, "cpu"), "val": td.to_device(split, "cpu")}
    backbone = dict(payload_dim=PAYLOAD_DIM, n_kinds=1, mode="masked", d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_pool=2, branch_layers=1, n_query_ids=0, descriptor_dim=F, dropout=0.0)
    cfg = {"model": {"backbone": backbone, "head_hidden": 16},
           "train": {"lr": 1e-3, "weight_decay": 0.0, "batch_decisions": 4, "max_epochs": 5, "patience": 10,
                     "grad_clip": 1.0, "ema": 0.0, "stop_on": "regret", "stop_smooth": 2, "min_epochs": 2}}
    model, info = td.train_one(cfg, dev, 0, lambda m: None)
    smooth = td.smoothed(info["val_regret_history"], 2)
    best, expected = np.inf, None
    for e in range(2, 5):
        if smooth[e] < best - 1e-4:
            best, expected = smooth[e], e + 1
    assert info["best_epoch"] == expected
    np.testing.assert_allclose(info["history"], smooth)
    assert len(info["val_halves"]) == info["epochs"] == 5
    assert set(info["crossfit_selection"]) == {"J_centered", "regret"}
    cfg["train"]["min_epochs"] = 9
    _, info = td.train_one(cfg, dev, 0, lambda m: None)
    assert info["best_epoch"] == 5
