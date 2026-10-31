from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cpsdefense.model.backbone import BackboneConfig, DecisionBackbone
from cpsdefense.model.branches import Question
from cpsdefense.model.losses import bce_loss, hazard_nll, masked_mean

TRIP_EDGES = (0.0, 300.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0, 3600.0, 5400.0, 7200.0, 10800.0, float("inf"))


@dataclass
class DirectConfig:
    backbone: dict
    head_hidden: int = 128
    j_scale: float = 10.0
    j_loss: str = "huber"
    rank_temperature: float = 2.0
    gva_scale: float = 1.0
    gva_margin: float = 0.0
    gva_balance: bool = False
    gva_regret_weight: bool = False
    gva_labelled_best: bool = False
    weights: dict = None
    summary_descriptor: bool = False
    item_path: bool = True
    summary_aux: bool = False
    summary_aux_dim: int = 0

    def loss_weights(self):
        return {"J": 1.0, "J_centered": 0.0, "rank": 1.0, "trip": 1.0, "hazard": 0.5, "resolve": 0.5,
                "duration": 0.2, "gva": 0.0, "summary_aux": 0.0, **(self.weights or {})}


class CandidateHeads(nn.Module):
    def __init__(self, d_model, hidden, n_bins, aux_dim=0):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 4 + n_bins)
        self.aux = nn.Linear(hidden, aux_dim) if aux_dim else None

    def forward(self, state):
        h = self.body(state)
        z = self.out(h)
        out = {"J": z[..., 0], "trip_logit": z[..., 1], "resolve_logit": z[..., 2], "log_duration": z[..., 3],
               "hazard_logits": z[..., 4:]}
        if self.aux is not None:
            out["summary_aux"] = self.aux(h)
        return out


class DirectDecisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.summary_aux and config.summary_aux_dim <= 0:
            raise ValueError("summary_aux needs summary_aux_dim > 0 (set by the data pipeline)")
        self.backbone = DecisionBackbone(BackboneConfig.from_dict(config.backbone))
        self.heads = CandidateHeads(self.backbone.config.d_model, config.head_hidden, len(TRIP_EDGES) - 1,
                                    config.summary_aux_dim if config.summary_aux else 0)

    @property
    def device(self):
        return self.backbone.device

    def forward(self, batch, cand_features, tau=None, mask=None):
        if self.config.summary_descriptor and tau is not None:
            raise ValueError("the summary descriptor is computed at the decision time; recompute it to score at "
                             "another time")
        batch = batch.to(self.device)
        feats = torch.as_tensor(cand_features, dtype=torch.float32, device=self.device)
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            feats = torch.where(mask[..., None], feats, torch.zeros_like(feats))
        if self.config.item_path:
            enc = self.backbone.encoder(batch, tau)
            used = enc.used
        else:
            enc = self.backbone.encoder(batch.replace(present=torch.zeros_like(batch.present)).pad_absent(), tau)
            used = batch.validity(batch.time_column(tau))
        questions = [Question(f"c{k}", "noul", descriptor=feats[:, k]) for k in range(feats.shape[1])]
        state = self.backbone.branches(enc.items, enc.mask, enc.pooled, questions)
        out = self.heads(state)
        out["J_pred"] = out["J"] * self.config.j_scale
        out["used"] = used
        return out


class EnsembleDecisionModel(nn.Module):
    def __init__(self, members):
        super().__init__()
        members = list(members)
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = nn.ModuleList(members)
        self.config = members[0].config

    @property
    def device(self):
        return self.members[0].device

    def forward(self, batch, cand_features, tau=None, mask=None):
        outs = [m(batch, cand_features, tau=tau, mask=mask) for m in self.members]
        J = torch.stack([o["J_pred"] for o in outs]).mean(0)
        p = torch.stack([torch.sigmoid(o["trip_logit"]) for o in outs]).mean(0).clamp(1e-7, 1 - 1e-7)
        used = outs[0]["used"]
        for o in outs[1:]:
            used = used | o["used"]
        return {"J_pred": J, "trip_logit": torch.log(p) - torch.log1p(-p), "used": used}


def family_min(J, member):
    return torch.where(member, J, torch.full_like(J, float("inf"))).amin(-1)


def losses(out, y, mask, config):
    w = config.loss_weights()
    s = config.j_scale
    parts = {}
    J = y["J"]
    keep = mask & torch.isfinite(J)
    if config.j_loss not in ("huber", "mse"):
        raise ValueError(f"j_loss must be huber or mse, got {config.j_loss!r}")
    j_target = torch.nan_to_num(J / s)
    err = (F.huber_loss(out["J"], j_target, reduction="none", delta=1.0) if config.j_loss == "huber"
           else 0.5 * (out["J"] - j_target) ** 2)
    parts["J"] = masked_mean(err, keep)
    count = keep.sum(-1, keepdim=True).clamp(min=1)
    centred_pred = out["J"] - torch.where(keep, out["J"], 0.0).sum(-1, keepdim=True) / count
    centred_true = j_target - torch.where(keep, j_target, 0.0).sum(-1, keepdim=True) / count
    parts["J_centered"] = masked_mean(0.5 * (centred_pred - centred_true) ** 2, keep)
    t = config.rank_temperature
    big = torch.finfo(out["J"].dtype).max / 4
    logits_pred = torch.where(mask, -out["J_pred"] / t, torch.full_like(out["J"], -big))
    logits_true = torch.where(keep, -torch.nan_to_num(J) / t, torch.full_like(out["J"], -big))
    target = F.softmax(logits_true, -1)
    parts["rank"] = -(target * F.log_softmax(logits_pred, -1)).sum(-1).mean()
    parts["trip"] = bce_loss(out["trip_logit"], torch.where(mask, y["p_trip"], torch.nan), None)
    R = y["trip_time"].shape[-1]
    logits = out["hazard_logits"][:, :, None, :].expand(-1, -1, R, -1)
    hmask = mask[:, :, None].expand(-1, -1, R) & torch.isfinite(y["trip_time"])
    parts["hazard"] = hazard_nll(logits[hmask], TRIP_EDGES, y["trip_time"][hmask], y["trip_event"][hmask])
    parts["resolve"] = bce_loss(out["resolve_logit"], torch.where(mask, y["p_resolved"], torch.nan), None)
    dur = torch.log1p(y["duration"])
    dkeep = mask & torch.isfinite(dur)
    parts["duration"] = masked_mean((out["log_duration"] - torch.nan_to_num(dur)) ** 2, dkeep)
    parts["gva"] = gather_act_loss(out["J_pred"], J, y["observe"], keep, config)
    if "summary_aux" in out and "summary" in y:
        err = 0.5 * (out["summary_aux"] - y["summary"]) ** 2
        parts["summary_aux"] = masked_mean(err.mean(-1), mask)
    total = sum(w[k] * v for k, v in parts.items() if w[k] != 0)
    return total, {k: float(v.detach()) for k, v in parts.items()}


def gather_act_loss(J_pred, J, observe, keep, config):
    obs, act = keep & observe, keep & ~observe
    Jt = torch.nan_to_num(J)
    gap_true = family_min(Jt, act) - family_min(Jt, obs)
    if config.gva_labelled_best:
        inf = torch.full_like(Jt, float("inf"))
        best_act = torch.where(act, Jt, inf).argmin(-1, keepdim=True)
        best_obs = torch.where(obs, Jt, inf).argmin(-1, keepdim=True)
        gap_pred = (J_pred.gather(-1, best_act) - J_pred.gather(-1, best_obs)).squeeze(-1)
    else:
        gap_pred = family_min(J_pred, act) - family_min(J_pred, obs)
    use = obs.any(-1) & act.any(-1) & (gap_true.abs() >= config.gva_margin)
    target = (gap_true > 0).to(J_pred.dtype)
    logit = torch.where(use, gap_pred, torch.zeros_like(gap_pred)) / config.gva_scale
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if config.gva_balance:
        pos = (use & (target > 0)).sum().clamp(min=1)
        neg = (use & (target == 0)).sum().clamp(min=1)
        loss = loss * torch.where(target > 0, (pos + neg) / (2 * pos), (pos + neg) / (2 * neg))
    if config.gva_regret_weight:
        w = torch.where(use, gap_true.abs().clamp(max=config.j_scale), torch.zeros_like(gap_true))
        loss = loss * w / masked_mean(w, use).clamp(min=1e-6)
    return masked_mean(loss, use)


def choose(J_pred, mask):
    return torch.where(mask, J_pred, torch.full_like(J_pred, float("inf"))).argmin(-1)


def derived_voi(J_pred, observe, mask):
    best_act = family_min(J_pred, mask & ~observe)[..., None]
    ok = mask & observe & torch.isfinite(best_act)
    return torch.where(ok, best_act - J_pred, torch.full_like(J_pred, float("nan")))


def pad_candidates(cand_decision, n_decisions, arrays):
    cand_decision = np.asarray(cand_decision, np.int64)
    counts = np.bincount(cand_decision, minlength=n_decisions)
    K = int(counts.max()) if counts.size else 0
    order = np.argsort(cand_decision, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    slot = np.empty(len(cand_decision), np.int64)
    slot[order] = np.arange(len(cand_decision)) - np.repeat(starts, counts)
    mask = np.zeros((n_decisions, K), bool)
    mask[cand_decision, slot] = True
    out = {}
    for name, a in arrays.items():
        shape = (n_decisions, K) + a.shape[1:]
        fill = False if a.dtype == bool else np.nan if np.issubdtype(a.dtype, np.floating) else 0
        padded = np.full(shape, fill, dtype=a.dtype)
        padded[cand_decision, slot] = a
        out[name] = padded
    return mask, out
