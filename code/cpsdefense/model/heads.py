import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def survival_from_hazard(hazard):
    return torch.cumprod(1 - hazard, -1)


def survival_from_logits(logits):
    return torch.cumprod(torch.sigmoid(-logits), -1)


def ordinal_log_probs(eta, thresholds, gaps=None):
    a = eta[..., None] - thresholds
    log_gt, log_le = F.logsigmoid(a), F.logsigmoid(-a)
    gaps = thresholds[..., 1:] - thresholds[..., :-1] if gaps is None else gaps
    mid = log_gt[..., :-1] + log_le[..., 1:] + torch.log(-torch.expm1(-gaps))
    return torch.cat([log_le[..., :1], mid, log_gt[..., -1:]], -1)


class NoulHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, state):
        z = self.proj(state).squeeze(-1)
        return {"type": "noul", "logit": z, "p_yes": torch.sigmoid(z), "p_no": torch.sigmoid(-z)}


class ChoiceHead(nn.Module):
    def __init__(self, d_model, n_option_ids=0, option_descriptor_dim=0):
        super().__init__()
        self.d_model = int(d_model)
        self.option_emb = nn.Embedding(n_option_ids, d_model) if n_option_ids else None
        self.option_mlp = (nn.Sequential(nn.Linear(option_descriptor_dim, d_model), nn.GELU(),
                                         nn.Linear(d_model, d_model)) if option_descriptor_dim else None)
        self.query = nn.Linear(d_model, d_model)
        self.prior = nn.Linear(d_model, 1)

    def option_embeddings(self, options):
        device = self.query.weight.device
        if isinstance(options, torch.Tensor):
            options = options.to(device)
        is_ids = (options.dtype in (torch.int32, torch.int64) if isinstance(options, torch.Tensor)
                  else np.asarray(options).dtype.kind in "iu")
        if is_ids:
            if self.option_emb is None:
                raise ValueError("option ids given but n_option_ids is 0")
            return self.option_emb(torch.as_tensor(options, dtype=torch.int64, device=device))
        if self.option_mlp is None:
            raise ValueError("option descriptors given but option_descriptor_dim is 0")
        return self.option_mlp(torch.as_tensor(options, dtype=self.query.weight.dtype, device=device))

    def forward(self, state, options):
        opt = self.option_embeddings(options)
        q = self.query(state)[..., None, :]
        logits = (q * opt).sum(-1) / math.sqrt(self.d_model) + self.prior(opt).squeeze(-1)
        return {"type": "choice", "logits": logits, "probs": logits.softmax(-1)}


class ScoreHead(nn.Module):
    def __init__(self, d_model, max_levels=11, min_gap=1e-3):
        super().__init__()
        if max_levels < 2:
            raise ValueError("max_levels must be >= 2")
        self.max_levels, self.min_gap = int(max_levels), float(min_gap)
        self.eta = nn.Linear(d_model, 1)
        self.first = nn.Linear(d_model, 1)
        self.gap = nn.Linear(d_model, max_levels - 2) if max_levels > 2 else None

    def forward(self, state, n_levels):
        if not 2 <= n_levels <= self.max_levels:
            raise ValueError(f"n_levels must be in [2, {self.max_levels}], got {n_levels}")
        eta = self.eta(state).squeeze(-1)
        first = self.first(state)
        if n_levels > 2:
            gaps = F.softplus(self.gap(state)[..., :n_levels - 2]) + self.min_gap
            thresholds = torch.cat([first, first + gaps.cumsum(-1)], -1)
        else:
            gaps, thresholds = first[..., :0], first
        log_probs = ordinal_log_probs(eta, thresholds, gaps)
        return {"type": "score", "eta": eta, "thresholds": thresholds, "gaps": gaps,
                "p_greater": torch.sigmoid(eta[..., None] - thresholds), "log_probs": log_probs,
                "probs": log_probs.exp()}


def _signed_log(x, cap=1e7):
    x = x.clamp(-cap, cap)
    return torch.sign(x) * torch.log1p(x.abs())


class HazardHead(nn.Module):
    def __init__(self, d_model, d_hidden=64, max_dt=1e7):
        super().__init__()
        self.max_dt = float(max_dt)
        self.state_proj = nn.Linear(d_model, d_hidden)
        self.bin_proj = nn.Linear(4, d_hidden)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(d_hidden, 1))

    def bin_features(self, edges):
        lo, hi = edges[:-1], edges[1:]
        feats = [_signed_log(lo, self.max_dt), _signed_log(hi, self.max_dt), _signed_log(hi - lo, self.max_dt),
                 torch.isinf(hi).to(edges.dtype)]
        return torch.stack(feats, -1).to(self.bin_proj.weight.dtype)

    def logits(self, state, edges):
        edges = torch.as_tensor(edges, dtype=torch.float64, device=self.bin_proj.weight.device)
        if edges.ndim != 1 or edges.numel() < 2 or not torch.isfinite(edges[0]) or bool((edges.diff() <= 0).any()):
            raise ValueError("edges must be strictly increasing, >= 2 values, first finite")
        h = self.state_proj(state)[..., None, :] + self.bin_proj(self.bin_features(edges))
        return self.out(h).squeeze(-1)

    def forward(self, state, edges):
        edges = torch.as_tensor(edges, dtype=torch.float64, device=self.bin_proj.weight.device)
        z = self.logits(state, edges)
        hazard, survival = torch.sigmoid(z), survival_from_logits(z)
        before = torch.cat([torch.ones_like(survival[..., :1]), survival[..., :-1]], -1)
        pmf = torch.cat([hazard * before, survival[..., -1:]], -1)
        return {"type": "hazard", "logits": z, "hazard": hazard, "survival": survival, "pmf": pmf, "edges": edges}
