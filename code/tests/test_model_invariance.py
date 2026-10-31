import time

import numpy as np
import pytest
import torch

from cpsdefense.model.backbone import DecisionBackbone, eval_determinism
from cpsdefense.model.branches import Question
from cpsdefense.model.evidence import (FIELDS, SLOT_FIELDS, EvidenceBatch, concat, derived_validity,
                                       validity)

T0 = 1.7e9
PAYLOAD_DIM = (6, 4, 2)
N_KINDS = 3
CATS = ("valid", "arrival_edge", "expired", "expired_edge", "retracted", "retracted_edge", "future", "padding")
PROBS = (0.42, 0.05, 0.1, 0.05, 0.1, 0.05, 0.12, 0.11)
FILLS = ("random", "huge", "nan", "inf", "mixed")
MASKED_MODES = ("masked", "block_local")
EDGES = [0.0, 5.0, 30.0, 120.0, 600.0, np.inf]


def make_batch(seed, n=6, m=16, empty_rows=(1,), id_offset=0):
    rng = np.random.default_rng(seed)
    tau = T0 + rng.uniform(0, 5000, n)
    cat = rng.choice(len(CATS), size=(n, m), p=PROBS)
    for row in range(n):
        if row in empty_rows:
            cat[row] = rng.choice(np.arange(2, len(CATS)), size=m)
        else:
            cat[row, :2] = 0
    name = np.asarray(CATS)[cat]
    t = tau[:, None]
    age = rng.uniform(1, 600, (n, m))
    event_time = t - age
    arrival = event_time + rng.uniform(0, 1, (n, m)) * age
    valid_until = np.where(rng.random((n, m)) < 0.1, np.inf, t + rng.uniform(1, 900, (n, m)))
    retracted = np.where(rng.random((n, m)) < 0.15, t + rng.uniform(1, 300, (n, m)), np.inf)
    arrival = np.where(name == "arrival_edge", t, arrival)
    valid_until = np.where(name == "expired", t - rng.uniform(1e-3, 300, (n, m)), valid_until)
    valid_until = np.where(name == "expired_edge", t, valid_until)
    retracted = np.where(name == "retracted", t - rng.uniform(0, 300, (n, m)), retracted)
    retracted = np.where(name == "retracted_edge", t, retracted)
    future = name == "future"
    arrival = np.where(future, t + rng.uniform(1e-3, 300, (n, m)), arrival)
    event_time = np.where(future, arrival - rng.uniform(0, 30, (n, m)), event_time)
    valid_until = np.where(future, arrival + rng.uniform(60, 900, (n, m)), valid_until)
    event_start = event_time - rng.uniform(0, 60, (n, m))
    source = rng.integers(0, len(PAYLOAD_DIM), (n, m))
    payload = rng.normal(size=(n, m, max(PAYLOAD_DIM))).astype(np.float32)
    for s, dim in enumerate(PAYLOAD_DIM):
        payload[..., dim:][source == s] = 0
    item_id = id_offset + np.arange(n * m).reshape(n, m)
    item_id[name == "padding"] = -1
    batch = EvidenceBatch.from_arrays(tau, item_id, source, event_time, arrival, valid_until, payload,
                                      segment=np.arange(n), kind=rng.integers(0, N_KINDS, (n, m)),
                                      event_start=event_start, retracted_at=retracted)
    return batch, name


def make_model(mode, seed=0, **overrides):
    config = dict(payload_dim=PAYLOAD_DIM, n_kinds=N_KINDS, mode=mode, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                  n_pool=2, branch_layers=2, n_query_ids=8, descriptor_dim=3, n_option_ids=5,
                  option_descriptor_dim=2, max_score_levels=6, hazard_hidden=16)
    config.update(overrides)
    torch.manual_seed(seed)
    return DecisionBackbone(config).eval()


def questions():
    return [
        Question("harm", "noul", qid=0),
        Question("resolve", "noul", descriptor=[0.5, -1.0, 2.0]),
        Question("action", "choice", qid=1, options=[0, 1, 2, 3]),
        Question("which", "choice", qid=2, options=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])),
        Question("severity", "score", qid=3, n_levels=5),
        Question("completion", "hazard", qid=4, descriptor=[1.0, 0.0, 0.0], bin_edges=EDGES),
    ]


def flat(out):
    return {f"{q}.{k}": v for q, o in out["outputs"].items() for k, v in o.items()
            if torch.is_tensor(v) and v.dtype.is_floating_point and k != "edges"}


def assert_bitwise(a, b):
    fa, fb = flat(a), flat(b)
    assert fa.keys() == fb.keys()
    for key in fa:
        assert torch.equal(fa[key], fb[key]), f"{key} differs: max |d| = {(fa[key] - fb[key]).abs().max()}"


def assert_close(a, b, rows_a=None, rows_b=None, atol=1e-5, rtol=1e-5):
    fa, fb = flat(a), flat(b)
    assert fa.keys() == fb.keys()
    worst = 0.0
    for key in fa:
        x = fa[key] if rows_a is None else fa[key][rows_a]
        y = fb[key] if rows_b is None else fb[key][rows_b]
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol, msg=key)
        worst = max(worst, float((x - y).abs().max()) if x.numel() else 0.0)
    return worst


def max_diff(a, b):
    fa, fb = flat(a), flat(b)
    return max(float((fa[k] - fb[k]).abs().nan_to_num(nan=np.inf).max()) for k in fa)


def ids(deps):
    return [sorted(np.asarray(d.cpu() if torch.is_tensor(d) else d).tolist()) for d in deps]


def fill_values(rng, shape, fill):
    if fill == "random":
        return rng.normal(0, 1e3, shape)
    if fill == "huge":
        return rng.choice([1e30, -1e30], shape)
    if fill == "nan":
        return np.full(shape, np.nan)
    if fill == "inf":
        return rng.choice([np.inf, -np.inf], shape)
    return rng.choice([np.nan, np.inf, -np.inf, 1e30, -1e30, 0.0, 7.0, -3.0], shape)


def witness(rng, base, shape, fill, sign):
    if fill == "nan":
        return np.full(shape, np.nan)
    if fill == "inf":
        return np.full(shape, sign * np.inf)
    if fill == "huge":
        return np.full(shape, sign * 1e30)
    if fill == "random":
        return base + sign * rng.uniform(1e-3, 1e4, shape)
    edge = base if sign < 0 else base + 1e-3
    pool = np.stack([np.full(shape, np.nan), np.full(shape, sign * np.inf), np.full(shape, sign * 1e30), edge], -1)
    return pool[np.arange(shape), rng.integers(0, 4, shape)]


def corrupt_invalid(batch, seed, fill, content_only=False, in_vocab=False):
    rng = np.random.default_rng(seed)
    b = batch.numpy()
    tau = np.broadcast_to(b.decision_time[:, None], b.item_id.shape)
    valid = b.validity()
    bad = ~valid
    k = int(bad.sum())
    new = {name: getattr(b, name).copy() for name in SLOT_FIELDS}
    new["payload"][bad] = fill_values(rng, (k, b.p), fill).astype(np.float32)
    for name in ("event_start", "event_time"):
        new[name][bad] = fill_values(rng, k, fill)
    for name, size in (("source", len(PAYLOAD_DIM)), ("kind", N_KINDS)):
        new[name][bad] = rng.integers(0, size, k) if in_vocab else rng.integers(-10**6, 10**6, k)
    new["item_id"][bad] = 10**12 + rng.integers(0, 10**9, k)
    if not content_only:
        padding = ~b.present
        future = b.present & ~(b.arrival_time <= tau)
        expired = b.present & ~future & ~(tau < b.valid_until)
        retracted = bad & ~padding & ~future & ~expired
        new["arrival_time"][bad] = tau[bad] - rng.uniform(0, 1e4, k)
        new["valid_until"][bad] = np.inf
        new["retracted_at"][bad] = np.inf
        new["arrival_time"][future] = witness(rng, tau[future], int(future.sum()), fill, +1)
        new["valid_until"][expired] = witness(rng, tau[expired], int(expired.sum()), fill, -1)
        new["retracted_at"][retracted] = witness(rng, tau[retracted], int(retracted.sum()), fill, -1)
    out = b.replace(**new)
    np.testing.assert_array_equal(out.validity(), valid)
    return out


def reference_valid(b, row, slot, t):
    return (bool(b.present[row, slot]) and b.arrival_time[row, slot] <= t and t < b.valid_until[row, slot]
            and t < b.retracted_at[row, slot])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_validity_matches_contract(backend):
    batch, name = make_batch(1)
    shift = np.random.default_rng(2).uniform(-100, 400, batch.n)
    b = batch if backend == "numpy" else batch.to("cpu")
    for t in (None, batch.decision_time + shift, float(batch.decision_time[0])):
        got = b.validity(t)
        got = got.numpy() if backend == "torch" else got
        times = batch.decision_time if t is None else np.broadcast_to(t, batch.n)
        ref = np.array([[reference_valid(batch, r, s, times[r]) for s in range(batch.m)] for r in range(batch.n)])
        np.testing.assert_array_equal(got, ref)
    valid = batch.validity()
    assert valid[name == "arrival_edge"].all()
    for cat in ("expired", "expired_edge", "retracted", "retracted_edge", "future", "padding"):
        assert not valid[name == cat].any(), cat
    assert not valid[1].any() and valid.sum(1)[0] >= 2


def test_nan_metadata_is_invalid():
    batch, _ = make_batch(3)
    valid = batch.validity()
    for field in ("arrival_time", "valid_until", "retracted_at"):
        values = getattr(batch, field).copy()
        values[valid] = np.nan
        assert not batch.replace(**{field: values}).validity().any()
    assert not batch.validity(np.nan).any()


def test_derived_validity_takes_worst_input():
    arrival = np.array([[1.0, 5.0, 3.0]])
    until = np.array([[50.0, 20.0, np.inf]])
    retracted = np.array([[np.inf, np.inf, 30.0]])
    a, u, r = derived_validity(arrival, until, retracted)
    assert (a[0], u[0], r[0]) == (5.0, 20.0, 30.0)
    ta, tu, tr = derived_validity(*(torch.as_tensor(x) for x in (arrival, until, retracted)))
    assert (float(ta[0]), float(tu[0]), float(tr[0])) == (5.0, 20.0, 30.0)
    until[0, 1] = np.nan
    _, u, _ = derived_validity(arrival, until, retracted)
    _, tu, _ = derived_validity(*(torch.as_tensor(x) for x in (arrival, until, retracted)))
    assert np.isnan(u[0]) and torch.isnan(tu[0])
    assert not validity(True, a, u, r, 10.0).any()


def test_batch_helpers_round_trip():
    batch, _ = make_batch(4)
    t = batch.to("cpu")
    back = t.numpy()
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(back, name), getattr(batch, name))
    compacted = batch.compact(batch.validity())
    assert compacted.m == int(batch.validity().sum(1).max())
    assert ids(compacted.dependency_set()) == ids(batch.dependency_set())
    assert compacted.validity().sum() == batch.validity().sum()
    wide = concat([batch, make_batch(5, n=2, m=21, id_offset=10**6)[0]])
    assert (wide.n, wide.m) == (8, 21)
    assert ids(wide.take(slice(0, 6)).dependency_set()) == ids(batch.dependency_set())
    torch_wide = concat([batch.to("cpu"), batch.to("cpu").pad_to(20)])
    assert torch_wide.m == 20 and ids(torch_wide.dependency_set()[:6]) == ids(batch.dependency_set())
    with pytest.raises(ValueError):
        EvidenceBatch.from_arrays(batch.decision_time, np.zeros_like(batch.item_id), batch.source,
                                  batch.event_time, batch.arrival_time, batch.valid_until, batch.payload)
    batch.check(PAYLOAD_DIM)
    bad = batch.payload.copy()
    bad[batch.present & (batch.source == 2), 3] = 1.0
    with pytest.raises(ValueError):
        batch.replace(payload=bad).check(PAYLOAD_DIM)


@pytest.mark.parametrize("mode", ["masked", "block_local", "unmasked"])
def test_dependency_set_is_the_valid_set(mode):
    batch, _ = make_batch(6)
    out = make_model(mode)(batch, questions=questions())
    if mode == "unmasked":
        expected = [batch.item_id[r][batch.known()[r]] for r in range(batch.n)]
        assert sum(len(d) for d in expected) > sum(len(d) for d in batch.dependency_set())
    else:
        expected = batch.dependency_set()
        np.testing.assert_array_equal(out["used"].numpy(), batch.validity())
    assert ids(out["dependency_set"]) == ids(expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_revalidate_returns_exactly_the_items_that_became_invalid(backend):
    batch, _ = make_batch(7, n=8, m=24)
    rng = np.random.default_rng(8)
    t_exec = batch.decision_time + rng.uniform(0, 400, batch.n)
    deps = batch.dependency_set()
    deps[0] = np.r_[deps[0], 99_999_999]
    deps[2] = np.zeros(0, np.int64)
    b = batch if backend == "numpy" else batch.to("cpu")
    got = b.revalidate(deps, t_exec)
    n_invalid = 0
    for row in range(batch.n):
        slot_of = {int(i): s for s, i in enumerate(batch.item_id[row]) if batch.present[row, s]}
        expected = [int(i) for i in deps[row]
                    if int(i) not in slot_of or not reference_valid(batch, row, slot_of[int(i)], t_exec[row])]
        assert sorted(np.asarray(got[row]).tolist()) == sorted(expected)
        n_invalid += len(expected)
    assert n_invalid > batch.n
    assert all(len(r) == 0 for r in batch.revalidate(batch.dependency_set(), batch.decision_time))


@pytest.mark.parametrize("mode", MASKED_MODES)
@pytest.mark.parametrize("fill", FILLS)
def test_outputs_bitwise_invariant_to_invalid_items(mode, fill):
    batch, _ = make_batch(10)
    model = make_model(mode)
    corrupted = corrupt_invalid(batch, 11, fill)
    with torch.no_grad():
        a = model(batch, questions=questions())
        b = model(corrupted, questions=questions())
    assert_bitwise(a, b)
    assert all(torch.isfinite(v).all() for v in flat(b).values())
    assert ids(a["dependency_set"]) == ids(b["dependency_set"])


@pytest.mark.parametrize("mode", MASKED_MODES)
def test_outputs_invariant_when_invalid_items_are_deleted(mode):
    batch, _ = make_batch(12)
    model = make_model(mode)
    valid = batch.validity()
    partial = valid | (np.random.default_rng(13).random(valid.shape) < 0.5)
    with torch.no_grad():
        full = model(corrupt_invalid(batch, 14, "mixed"), questions=questions())
        only_valid = model(batch.compact(valid), questions=questions())
        some = model(batch.compact(partial), questions=questions())
    assert batch.compact(valid).m < batch.m
    assert_bitwise(full, only_valid)
    assert_bitwise(full, some)


@pytest.mark.parametrize("mode", MASKED_MODES)
def test_outputs_invariant_to_item_order_and_batch_composition(mode):
    batch, _ = make_batch(15)
    model = make_model(mode)
    rng = np.random.default_rng(16)
    order = np.stack([rng.permutation(batch.m) for _ in range(batch.n)])
    other, _ = make_batch(17, n=3, m=27, empty_rows=(), id_offset=10**6)
    with torch.no_grad():
        base = model(batch, questions=questions())
        shuffled = model(corrupt_invalid(batch.permute_slots(order), 18, "mixed"), questions=questions())
        mixed = model(concat([other, batch.permute_slots(order)]), questions=questions())
        worst = assert_close(base, shuffled)
        worst = max(worst, assert_close(base, mixed, rows_b=slice(3, None)))
        for row in range(batch.n):
            alone = model(batch.take([row]).compact(batch.validity()[[row]]), questions=questions())
            worst = max(worst, assert_close(base, alone, rows_a=[row]))
    print(f"{mode}: max |diff| under reordering / batch composition = {worst:.2e}")


@pytest.mark.parametrize("hide_validity", [False, True])
def test_unmasked_baseline_fails_the_invariance_test(hide_validity):
    batch, name = make_batch(10)
    model = make_model("unmasked", hide_validity=hide_validity)
    with torch.no_grad():
        a = model(batch, questions=questions())
        b = model(corrupt_invalid(batch, 11, "random", in_vocab=True), questions=questions())
        c = model(corrupt_invalid(batch, 11, "nan", in_vocab=True), questions=questions())
        stale = batch.known() & ~batch.validity()
        payload = batch.payload.copy()
        payload[stale] = np.random.default_rng(12).normal(0, 3, (int(stale.sum()), batch.p))
        for s, dim in enumerate(PAYLOAD_DIM):
            payload[..., dim:][batch.source == s] = 0
        d = model(batch.replace(payload=payload), questions=questions())
    assert max_diff(a, b) > 1e-3
    assert any(torch.isnan(v).any() for v in flat(c).values())
    assert max_diff(a, d) > 1e-3


def test_expose_mask_count_weakens_the_guarantee_to_content():
    batch, _ = make_batch(19)
    model = make_model("masked", expose_mask_count=True)
    stale = batch.known() & ~batch.validity()
    with torch.no_grad():
        a = model(batch, questions=questions())
        b = model(corrupt_invalid(batch, 20, "mixed", content_only=True), questions=questions())
        c = model(batch.compact(~stale), questions=questions())
    assert_bitwise(a, b)
    assert max_diff(a, c) > 1e-4


@pytest.mark.parametrize("mode", MASKED_MODES)
def test_valid_item_fields_the_model_must_not_read(mode):
    batch, _ = make_batch(21)
    valid = batch.validity()
    payload = batch.payload.copy()
    for s, dim in enumerate(PAYLOAD_DIM):
        payload[..., dim:][valid & (batch.source == s)] = np.nan
    item_id = np.where(valid, batch.item_id + 5 * 10**6, batch.item_id)
    t = batch.decision_time[:, None]
    retracted = np.where(valid, t + np.random.default_rng(22).uniform(1e-3, 50, valid.shape), batch.retracted_at)
    changed = batch.replace(payload=payload, item_id=item_id, retracted_at=retracted)
    model = make_model(mode)
    with torch.no_grad():
        assert_bitwise(model(batch, questions=questions()), model(changed, questions=questions()))


@pytest.mark.parametrize("mode", MASKED_MODES)
def test_no_gradient_reaches_invalid_items(mode):
    batch = corrupt_invalid(make_batch(23)[0], 24, "nan").to("cpu")
    payload = batch.payload.clone().requires_grad_(True)
    model = make_model(mode)
    out = model(batch.replace(payload=payload), questions=questions())["outputs"]
    loss = sum(o["logit"].sum() if o["type"] == "noul" else o["logits"].sum() for o in out.values()
               if o["type"] in ("noul", "choice", "hazard"))
    loss.backward()
    valid = batch.validity()
    assert torch.isfinite(loss)
    assert torch.equal(payload.grad[~valid], torch.zeros_like(payload.grad[~valid]))
    assert torch.isfinite(payload.grad).all() and payload.grad[valid].abs().sum() > 0


def test_invariance_holds_in_train_mode_with_dropout():
    batch, _ = make_batch(25)
    model = make_model("masked", dropout=0.2).train()
    outs = []
    for b in (batch, corrupt_invalid(batch, 26, "mixed")):
        torch.manual_seed(123)
        outs.append(model(b, questions=questions()))
    assert_bitwise(*outs)


def test_decision_without_valid_items_is_finite():
    batch, _ = make_batch(27)
    model = make_model("masked")
    with torch.no_grad():
        out = model(batch, questions=questions())
        padded = model(batch.take([1]).compact(np.zeros((1, batch.m), bool)), questions=questions())
    assert all(torch.isfinite(v).all() for v in flat(out).values())
    assert_close(out, padded, rows_a=[1])


def test_time_translation_invariance():
    batch, _ = make_batch(28)
    shift = 86400.0 * 400
    moved = batch.replace(decision_time=batch.decision_time + shift,
                          **{f: getattr(batch, f) + shift for f in ("event_start", "event_time", "arrival_time",
                                                                     "valid_until", "retracted_at")})
    model = make_model("masked")
    with torch.no_grad():
        assert_close(model(batch, questions=questions()), model(moved, questions=questions()), atol=1e-4)


def item_states(model, batch):
    enc = model.encoder(batch.to("cpu"))
    b = batch.to("cpu")
    states = {}
    for row in range(b.n):
        row_ids = b.item_id[row][enc.used[row]]
        for k, i in enumerate(row_ids.tolist()):
            states[(row, i)] = enc.items[row, k]
    return states, enc.pooled


def test_set_encoder_is_permutation_equivariant():
    batch, _ = make_batch(29)
    rng = np.random.default_rng(30)
    order = np.stack([rng.permutation(batch.m) for _ in range(batch.n)])
    model = make_model("masked")
    with torch.no_grad():
        s1, p1 = item_states(model, batch)
        s2, p2 = item_states(model, batch.permute_slots(order))
    assert s1.keys() == s2.keys() and len(s1) > 10
    for key in s1:
        torch.testing.assert_close(s1[key], s2[key], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p1, p2, atol=1e-5, rtol=1e-5)


def test_block_local_item_states_do_not_depend_on_other_items():
    batch, _ = make_batch(31)
    valid = batch.validity()
    drop = valid.copy()
    drop[0, np.flatnonzero(valid[0])[0]] = False
    for mode, independent in (("block_local", True), ("masked", False)):
        model = make_model(mode)
        with torch.no_grad():
            s1, _ = item_states(model, batch)
            s2, _ = item_states(model, batch.compact(drop))
        diffs = [float((s1[k] - s2[k]).abs().max()) for k in s2 if k[0] == 0]
        if independent:
            assert max(diffs) <= 1e-6
        else:
            assert max(diffs) > 1e-4


def test_score_at_block_local_equals_fresh_forward():
    batch, _ = make_batch(32, n=8, m=24)
    t_exec = batch.decision_time + 200.0
    model = make_model("block_local")
    with torch.no_grad():
        decided = model(batch, questions=questions())
        rescored = model.score_at(batch, t_exec, questions(), cache=decided["cache"])
        fresh = model(batch, t_exec, questions())
    newly_valid = batch.validity(t_exec) & ~batch.validity()
    assert newly_valid.sum() > 0 and (batch.validity() & ~batch.validity(t_exec)).sum() > 0
    assert rescored["stats"]["n_encoded"] == int(newly_valid.sum())
    assert fresh["stats"]["n_encoded"] == int(batch.validity(t_exec).sum())
    assert ids(rescored["dependency_set"]) == ids(batch.dependency_set(t_exec))
    worst = assert_close(rescored, fresh, atol=2e-6, rtol=1e-5)
    print(f"score_at vs fresh forward: max |diff| = {worst:.2e}")
    with pytest.raises(ValueError):
        model.score_at(make_batch(33, n=8, m=24)[0], t_exec, questions(), cache=decided["cache"])


def test_score_at_masked_recomputes_exactly():
    batch, _ = make_batch(34)
    t_exec = batch.decision_time + 150.0
    model = make_model("masked")
    with torch.no_grad():
        assert_bitwise(model.score_at(batch, t_exec, questions()), model(batch, t_exec, questions()))


def _median_time(fn, repeats=7):
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def test_score_at_block_local_is_faster_than_fresh_forward():
    batch, _ = make_batch(35, n=64, m=128, empty_rows=())
    t_exec = batch.decision_time + 30.0
    qs = questions()
    config = dict(d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = make_model("block_local", **config)
    masked = make_model("masked", **config)
    b = batch.to("cpu")
    with torch.no_grad():
        cache = model(b, questions=qs)["cache"]
        rescore = _median_time(lambda: model.score_at(b, t_exec, qs, cache=cache))
        fresh = _median_time(lambda: model(b, t_exec, qs, keep_cache=False))
        masked_fresh = _median_time(lambda: masked(b, t_exec, qs))
    print(f"re-score {rescore * 1e3:.1f} ms, block_local fresh {fresh * 1e3:.1f} ms, "
          f"masked fresh {masked_fresh * 1e3:.1f} ms ({int(b.validity(t_exec).sum())} valid items)")
    assert rescore < fresh


def pick(out, name):
    return {k: v for k, v in flat(out).items() if k.startswith(name + ".")}


@pytest.mark.parametrize("mode", MASKED_MODES)
def test_branch_outputs_do_not_depend_on_other_questions(mode):
    batch, _ = make_batch(36)
    model = make_model(mode)
    qs = questions()
    extra = Question("extra", "score", descriptor=[3.0, 1.0, -2.0], n_levels=3)
    substitute = Question("resolve", "hazard", qid=7, bin_edges=[0.0, 1.0, 2.0])
    with torch.no_grad():
        base = model(batch, questions=qs)
        reordered = model(batch, questions=qs[::-1])
        swapped = model(batch, questions=[qs[0], substitute, *qs[2:]])
        alone = {q.name: model(batch, questions=[q]) for q in qs}
        more = model(batch, questions=[extra, *qs, Question("x2", "noul", qid=6)])
    for q in qs:
        a, r = pick(base, q.name), pick(reordered, q.name)
        for key in a:
            assert torch.equal(a[key], r[key]), key
            torch.testing.assert_close(a[key], pick(alone[q.name], q.name)[key], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(a[key], pick(more, q.name)[key], atol=1e-6, rtol=1e-6)
    for q in qs:
        if q.name != "resolve":
            for key, value in pick(base, q.name).items():
                assert torch.equal(value, pick(swapped, q.name)[key]), key


def test_per_decision_descriptor_queries():
    batch, _ = make_batch(37)
    model = make_model("masked")
    desc = np.random.default_rng(38).normal(size=(batch.n, 3))
    q = Question("candidate", "hazard", descriptor=desc, bin_edges=EDGES)
    with torch.no_grad():
        out = model(batch, questions=[q])["outputs"]["candidate"]
        rows = [model(batch.take([r]), questions=[Question("candidate", "hazard", descriptor=desc[r],
                                                             bin_edges=EDGES)])["outputs"]["candidate"]
                for r in range(batch.n)]
    for r in range(batch.n):
        torch.testing.assert_close(out["survival"][r], rows[r]["survival"][0], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        model(batch, questions=[Question("bad", "noul", descriptor=desc[:2])])


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MASKED_MODES)
def test_invariance_bitwise_on_cuda(mode):
    batch, _ = make_batch(39, n=16, m=40)
    model = make_model(mode).to("cuda")
    with eval_determinism(model), torch.no_grad():
        outs = [model(batch.to("cuda"), questions=questions())]
        for fill in FILLS:
            outs.append(model(corrupt_invalid(batch, 40, fill).to("cuda"), questions=questions()))
        outs.append(model(batch.compact(batch.validity()).to("cuda"), questions=questions()))
    for other in outs[1:]:
        assert_bitwise(outs[0], other)
