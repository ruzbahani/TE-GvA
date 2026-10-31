import math

import numpy as np
import pytest
import torch

from cpsdefense.model.backbone import BackboneConfig, DecisionBackbone
from cpsdefense.model.branches import Question
from cpsdefense.model.evidence import EvidenceBatch
from cpsdefense.model.heads import (ChoiceHead, HazardHead, NoulHead, ScoreHead, ordinal_log_probs,
                                    survival_from_hazard, survival_from_logits)
from cpsdefense.model.losses import bce_loss, ce_loss, hazard_nll, ordinal_nll, total_loss


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def extreme_logits(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    parts = [torch.randn(shape, generator=g) * scale for scale in (1.0, 30.0, 1e4, 1e30)]
    z = torch.cat(parts)
    z[0, :3] = torch.tensor([float("inf"), float("-inf"), 0.0])
    z[1, ::2] = float("inf")
    z[2, 1::2] = float("-inf")
    return z


def test_survival_monotone_and_bounded_for_any_logits():
    z = extreme_logits((64, 12))
    s = survival_from_logits(z)
    assert torch.isfinite(s).all()
    assert (s >= 0).all() and (s <= 1).all()
    assert (s.diff(dim=-1) <= 0).all()
    s_h = survival_from_hazard(torch.sigmoid(z))
    assert (s_h >= 0).all() and (s_h <= 1).all() and (s_h.diff(dim=-1) <= 0).all()


def test_survival_matches_product_formula():
    z = torch.randn(20, 7, dtype=torch.float64) * 3
    h = 1 / (1 + np.exp(-z.numpy()))
    expected = np.cumprod(1 - h, axis=-1)
    np.testing.assert_allclose(survival_from_logits(z).numpy(), expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(survival_from_hazard(torch.as_tensor(h)).numpy(), expected, rtol=1e-12)


def test_hazard_head_outputs_are_coherent_for_any_state():
    torch.manual_seed(0)
    head = HazardHead(16, d_hidden=8)
    edges = [0.0, 2.0, 10.0, 60.0, 300.0, float("inf")]
    for scale in (1.0, 1e3, 1e6):
        state = torch.randn(3, 5, 16) * scale
        out = head(state, edges)
        assert out["logits"].shape == (3, 5, 5) and out["pmf"].shape == (3, 5, 6)
        s = out["survival"]
        assert (s >= 0).all() and (s <= 1).all() and (s.diff(dim=-1) <= 0).all()
        assert (out["pmf"] >= 0).all() and torch.equal(out["pmf"][..., -1], s[..., -1])
        total = out["pmf"].sum(-1)
        torch.testing.assert_close(total, torch.ones_like(total), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        head(torch.randn(2, 16), [0.0, 5.0, 5.0])
    with pytest.raises(ValueError):
        head(torch.randn(2, 16), [float("-inf"), 5.0])


def test_ordinal_cumulative_probabilities_are_ordered():
    torch.manual_seed(1)
    head = ScoreHead(16, max_levels=9)
    for scale in (1.0, 1e2, 1e4):
        state = torch.randn(256, 16) * scale
        for k in (2, 3, 9):
            out = head(state, k)
            assert out["thresholds"].shape == (256, k - 1)
            assert (out["thresholds"].diff(dim=-1) >= 0).all()
            assert (out["p_greater"].diff(dim=-1) <= 0).all()
            assert torch.isfinite(out["log_probs"]).all()
            assert (out["probs"] >= 0).all()
            torch.testing.assert_close(out["probs"].sum(-1), torch.ones(256), atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        head(state, 10)


def test_noul_complement_is_coherent_by_construction():
    torch.manual_seed(2)
    head = NoulHead(8)
    for scale in (1.0, 1e2, 1e5):
        out = head(torch.randn(1000, 8) * scale)
        assert set(out) == {"type", "logit", "p_yes", "p_no"}
        assert (out["p_yes"] + out["p_no"] - 1).abs().max() <= 2.5e-7
    z = torch.tensor([-1e4, -20.0, 0.0, 20.0, 1e4])
    torch.testing.assert_close(torch.sigmoid(z) + torch.sigmoid(-z), torch.ones(5), atol=2.5e-7, rtol=0)


def test_choice_logits_are_option_local_but_probabilities_are_set_dependent():
    torch.manual_seed(3)
    head = ChoiceHead(8, n_option_ids=6, option_descriptor_dim=3)
    state = torch.randn(10, 8)
    small, large = head(state, [0, 1, 2]), head(state, [0, 1, 2, 3, 4])
    torch.testing.assert_close(small["logits"], large["logits"][:, :3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(small["probs"].sum(-1), torch.ones(10))
    assert (large["probs"][:, :3] < small["probs"]).all()
    desc = torch.randn(10, 4, 3)
    assert head(state, desc)["probs"].shape == (10, 4)
    with pytest.raises(ValueError):
        ChoiceHead(8)(state, [0, 1])


def test_bce_and_ce_match_hand_computation():
    got = bce_loss(torch.tensor([0.0, 2.0, -1.0]), torch.tensor([1.0, 0.0, float("nan")]))
    expected = (-math.log(sig(0.0)) - math.log(1 - sig(2.0))) / 2
    assert abs(float(got) - expected) < 1e-6
    got = ce_loss(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), torch.tensor([2, -1]))
    expected = -(3.0 - math.log(math.exp(1) + math.exp(2) + math.exp(3)))
    assert abs(float(got) - expected) < 1e-6
    got = bce_loss(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 0.0]), mask=torch.tensor([True, False]))
    assert abs(float(got) + math.log(sig(0.0))) < 1e-6


def test_ordinal_nll_matches_hand_computation():
    eta, theta = 0.4, [-1.0, 0.5, 2.0]
    gt = [sig(eta - t) for t in theta]
    probs = [1 - gt[0], gt[0] - gt[1], gt[1] - gt[2], gt[2]]
    targets = [0, 1, 2, 3, -1]
    got = ordinal_nll(torch.full((5,), eta), torch.tensor([theta] * 5), torch.tensor(targets))
    expected = -np.mean([math.log(probs[k]) for k in targets[:4]])
    assert abs(float(got) - expected) < 1e-6
    lp = ordinal_log_probs(torch.tensor([eta], dtype=torch.float64), torch.tensor([theta], dtype=torch.float64))
    np.testing.assert_allclose(lp.exp().numpy()[0], probs, rtol=1e-12)

EDGES = [0.0, 10.0, 20.0, 40.0]
Z = [0.3, -0.5, 1.2]


def hazard_case(times, events, rule="start", edges=EDGES, z=Z):
    logits = torch.tensor([z] * len(times), dtype=torch.float64)
    return float(hazard_nll(logits, edges, torch.tensor(times, dtype=torch.float64), torch.tensor(events), rule))


def test_hazard_nll_matches_hand_computation():
    h = [sig(v) for v in Z]
    s = [math.log(1 - v) for v in h]
    cases = [
        (15.0, 1, "start", -(s[0] + math.log(h[1]))),
        (10.0, 1, "start", -math.log(h[0])),
        (0.0, 1, "start", -math.log(h[0])),
        (39.9, 1, "start", -(s[0] + s[1] + math.log(h[2]))),
        (40.0, 1, "start", -(s[0] + s[1] + math.log(h[2]))),
        (40.5, 1, "start", -(s[0] + s[1] + s[2])),
        (15.0, 0, "start", -s[0]),
        (15.0, 0, "half", -(s[0] + s[1])),
        (12.0, 0, "half", -s[0]),
        (10.0, 0, "start", -s[0]),
        (20.0, 0, "half", -(s[0] + s[1])),
        (5.0, 0, "start", 0.0),
        (100.0, 0, "half", -(s[0] + s[1] + s[2])),
    ]
    for t, e, rule, expected in cases:
        assert abs(hazard_case([t], [e], rule) - expected) < 1e-12, (t, e, rule)
    together = hazard_case([15.0, 15.0, float("nan"), 40.5], [1, 0, 1, 1])
    assert abs(together - np.mean([cases[0][3], cases[6][3], cases[5][3]])) < 1e-12
    open_z = [0.1, -0.2]
    got = hazard_case([1e6], [0], "half", edges=[0.0, 10.0, float("inf")], z=open_z)
    assert abs(got + math.log(1 - sig(open_z[0]))) < 1e-12


def test_hazard_nll_errors_and_missing_targets():
    logits = torch.zeros(3, 3, requires_grad=True)
    with pytest.raises(ValueError):
        hazard_nll(logits, EDGES, torch.tensor([-1.0, 1.0, 2.0]), torch.tensor([1, 1, 1]))
    with pytest.raises(ValueError):
        hazard_nll(logits, [0.0, 1.0], torch.tensor([1.0, 1.0, 2.0]), torch.tensor([1, 1, 1]))
    with pytest.raises(ValueError):
        hazard_nll(logits, EDGES, torch.tensor([1.0, 1.0, 2.0]), torch.tensor([1, 1, 1]), censor_rule="end")
    loss = hazard_nll(logits, EDGES, torch.full((3,), float("nan")), torch.tensor([1, 0, 1]))
    loss.backward()
    assert float(loss) == 0.0 and torch.equal(logits.grad, torch.zeros(3, 3))
    for fn, target in ((bce_loss, torch.full((3,), float("nan"))), (ce_loss, torch.full((3,), -1))):
        x = torch.zeros(3, 3, requires_grad=True) if fn is ce_loss else torch.zeros(3, requires_grad=True)
        value = fn(x, target)
        value.backward()
        assert float(value) == 0.0 and torch.isfinite(x.grad).all()


def test_losses_are_finite_for_extreme_logits():
    z = torch.tensor([[1e4, -1e4, 30.0], [-1e4, 1e4, -30.0]], requires_grad=True)
    loss = hazard_nll(z, EDGES, torch.tensor([15.0, 25.0]), torch.tensor([1, 0]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert torch.isfinite(bce_loss(torch.tensor([1e4, -1e4]), torch.tensor([0.0, 1.0])))
    eta = torch.tensor([1e4, -1e4])
    theta = torch.tensor([[0.0, 1e-3, 5.0], [0.0, 1e-3, 5.0]])
    assert torch.isfinite(ordinal_nll(eta, theta, torch.tensor([1, 3]), gaps=theta.diff(dim=-1)))


def synthetic_batch(seed, n=128, m=10, decoy=True):
    rng = np.random.default_rng(seed)
    tau = 1.7e9 + rng.uniform(0, 1000, n)
    t = tau[:, None]
    label = rng.integers(0, 2, n)
    sign = np.where(label == 1, 1.0, -1.0)[:, None]
    expired = rng.random((n, m)) < 0.4
    expired[:, 0] = False
    event_time = t - rng.uniform(1, 100, (n, m))
    arrival = event_time + 1.0
    valid_until = np.where(expired, t - rng.uniform(0, 50, (n, m)), t + rng.uniform(10, 300, (n, m)))
    payload = np.zeros((n, m, 3), np.float32)
    strength = np.where(expired, -3.0 if decoy else 0.0, 1.0)
    payload[..., 0] = sign * strength * rng.uniform(0.5, 1.5, (n, m)) + rng.normal(0, 0.3, (n, m))
    payload[..., 1:] = rng.normal(size=(n, m, 2))
    item_id = np.arange(n * m).reshape(n, m)
    batch = EvidenceBatch.from_arrays(tau, item_id, np.zeros((n, m)), event_time, arrival, valid_until, payload)
    return batch, label


def all_type_questions():
    return [Question("harm", "noul", qid=0), Question("act", "choice", qid=1, options=[0, 1, 2]),
            Question("which", "choice", qid=2, options=np.eye(2, 3)),
            Question("sev", "score", descriptor=[1.0, 0.0], n_levels=4),
            Question("eta", "hazard", qid=3, bin_edges=[0.0, 10.0, 60.0, float("inf")])]


@pytest.mark.parametrize("mode", ["masked", "block_local", "unmasked"])
def test_gradients_reach_every_parameter(mode):
    batch, label = synthetic_batch(0, n=32)
    torch.manual_seed(0)
    model = DecisionBackbone(dict(payload_dim=(3,), mode=mode, d_model=32, n_heads=4, d_ff=64, n_query_ids=4,
                                  descriptor_dim=2, n_option_ids=3, option_descriptor_dim=3, max_score_levels=5))
    out = model(batch, questions=all_type_questions(), keep_cache=False)["outputs"]
    rng = np.random.default_rng(1)
    targets = {"harm": torch.tensor(label, dtype=torch.float32), "act": torch.tensor(rng.integers(0, 3, 32)),
               "which": torch.tensor(rng.integers(-1, 2, 32)), "sev": torch.tensor(rng.integers(0, 4, 32)),
               "eta": {"time": torch.tensor(rng.uniform(0, 100, 32)), "event": torch.tensor(rng.integers(0, 2, 32))}}
    loss, parts = total_loss(out, targets)
    loss.backward()
    assert set(parts) == set(targets) and all(np.isfinite(v) for v in parts.values())
    silent = [name for name, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not silent, silent
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_backbone_learns_to_use_valid_evidence():
    torch.manual_seed(0)
    model = DecisionBackbone(dict(payload_dim=(3,), mode="masked", d_model=32, n_heads=4, d_ff=64, n_query_ids=2))
    q = [Question("harm", "noul", qid=0)]
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for step in range(120):
        batch, label = synthetic_batch(step)
        loss = bce_loss(model(batch, questions=q)["outputs"]["harm"]["logit"], torch.tensor(label, dtype=torch.float32))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
    model.eval()
    batch, label = synthetic_batch(10_000, n=512)
    with torch.no_grad():
        p = model(batch, questions=q)["outputs"]["harm"]["p_yes"].numpy()
    accuracy = float(np.mean((p > 0.5) == label))
    assert float(loss) < first and accuracy > 0.9, (first, float(loss), accuracy)


@pytest.mark.parametrize("mode", ["masked", "block_local", "unmasked"])
def test_float64_model_matches_float32(mode):
    batch, _ = synthetic_batch(3, n=16)
    torch.manual_seed(0)
    model = DecisionBackbone(dict(payload_dim=(3,), mode=mode, d_model=32, n_heads=4, d_ff=64, n_query_ids=4,
                                  descriptor_dim=2, n_option_ids=3, option_descriptor_dim=3,
                                  max_score_levels=5)).eval()
    with torch.no_grad():
        single = model(batch, questions=all_type_questions())["outputs"]
        double = model.double()(batch, questions=all_type_questions())["outputs"]
    for name, out in single.items():
        for key, value in out.items():
            if torch.is_tensor(value) and value.dtype.is_floating_point and key != "edges":
                assert double[name][key].dtype == torch.float64
                torch.testing.assert_close(double[name][key].float(), value, atol=1e-4, rtol=1e-4)


def test_question_and_config_validation():
    for kwargs in (dict(type="yesno", qid=0), dict(type="noul"), dict(type="score", qid=0, n_levels=1),
                   dict(type="choice", qid=0, options=[]), dict(type="hazard", qid=0, bin_edges=[0.0, 0.0]),
                   dict(type="hazard", qid=0, bin_edges=[5.0])):
        with pytest.raises(ValueError):
            Question("q", **kwargs)
    with pytest.raises(ValueError):
        BackboneConfig.from_dict({"payload_dim": (3,), "depth": 2})
    with pytest.raises(ValueError):
        DecisionBackbone(dict(payload_dim=(3,), mode="masked", hide_validity=True))
    with pytest.raises(ValueError):
        DecisionBackbone(dict(payload_dim=(3,), mode="unmasked", expose_mask_count=True))
    model = DecisionBackbone(dict(payload_dim=(3,), d_model=16, n_heads=2, d_ff=32))
    batch, _ = synthetic_batch(0, n=4)
    with pytest.raises(ValueError):
        model(batch, questions=[Question("a", "noul", qid=0), Question("a", "noul", qid=1)])
    with pytest.raises(ValueError):
        model(batch, questions=[Question("a", "noul", qid=99)])
    with pytest.raises(ValueError):
        model(batch, questions=[], keep_cache=True)
    assert BackboneConfig.from_dict(model.config.to_dict()) == model.config
