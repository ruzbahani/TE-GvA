import contextlib
import os
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn

from cpsdefense.model.branches import QuestionBranches, check_questions
from cpsdefense.model.encoder import FOURIER_PERIODS, LOG_SCALES, EvidenceEncoder
from cpsdefense.model.evidence import split_rows
from cpsdefense.model.heads import ChoiceHead, HazardHead, NoulHead, ScoreHead

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class BackboneConfig:
    payload_dim: tuple
    n_kinds: int = 1
    mode: str = "masked"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    n_pool: int = 4
    dropout: float = 0.0
    hide_validity: bool = False
    expose_mask_count: bool = False
    log_scales: tuple = LOG_SCALES
    fourier_periods: tuple = FOURIER_PERIODS
    max_dt: float = 1e7
    branch_layers: int = 2
    n_query_ids: int = 16
    descriptor_dim: int = 0
    n_option_ids: int = 0
    option_descriptor_dim: int = 0
    max_score_levels: int = 11
    min_threshold_gap: float = 1e-3
    hazard_hidden: int = 64

    @classmethod
    def from_dict(cls, config):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ValueError(f"unknown backbone config keys {unknown}")
        return cls(**config)

    def to_dict(self):
        return asdict(self)


class DecisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = config if isinstance(config, BackboneConfig) else BackboneConfig.from_dict(config)
        self.config = config
        c = config
        self.encoder = EvidenceEncoder(
            c.payload_dim, c.n_kinds, mode=c.mode, d_model=c.d_model, n_heads=c.n_heads, n_layers=c.n_layers,
            d_ff=c.d_ff, n_pool=c.n_pool, dropout=c.dropout, hide_validity=c.hide_validity,
            expose_mask_count=c.expose_mask_count, log_scales=c.log_scales, fourier_periods=c.fourier_periods,
            max_dt=c.max_dt)
        self.branches = QuestionBranches(c.d_model, c.n_heads, c.branch_layers, c.d_ff, c.n_query_ids,
                                         c.descriptor_dim, c.dropout)
        self.heads = nn.ModuleDict({
            "noul": NoulHead(c.d_model),
            "choice": ChoiceHead(c.d_model, c.n_option_ids, c.option_descriptor_dim),
            "score": ScoreHead(c.d_model, c.max_score_levels, c.min_threshold_gap),
            "hazard": HazardHead(c.d_model, c.hazard_hidden, c.max_dt),
        })

    @property
    def mode(self):
        return self.encoder.mode

    @property
    def device(self):
        return self.encoder.device

    def head(self, question, state):
        if question.type == "noul":
            return self.heads["noul"](state)
        if question.type == "choice":
            return self.heads["choice"](state, question.options)
        if question.type == "score":
            return self.heads["score"](state, question.n_levels)
        return self.heads["hazard"](state, question.bin_edges)

    def _run(self, batch, tau, questions, cache, build_cache):
        questions = check_questions(questions)
        batch = batch.to(self.device)
        enc = self.encoder(batch, tau, cache=cache, build_cache=build_cache)
        state = self.branches(enc.items, enc.mask, enc.pooled, questions)
        outputs = {q.name: self.head(q, state[:, i]) for i, q in enumerate(questions)}
        return {
            "outputs": outputs,
            "dependency_set": split_rows(batch.item_id, enc.used),
            "used": enc.used,
            "cache": enc.cache,
            "stats": {"n_items": int(enc.used.sum()), "n_encoded": enc.n_encoded},
        }

    def forward(self, batch, tau=None, questions=(), keep_cache=None):
        keep_cache = self.mode == "block_local" if keep_cache is None else keep_cache
        if keep_cache and self.mode != "block_local":
            raise ValueError("an item cache exists only in block_local mode")
        return self._run(batch, tau, questions, None, keep_cache)

    def score_at(self, batch, tau_exec, questions, cache=None):
        cache = cache if self.mode == "block_local" else None
        return self._run(batch, tau_exec, questions, cache, False)


@contextlib.contextmanager
def eval_determinism(model=None):
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    was_training = model.training if model is not None else None
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if model is not None:
        model.eval()
    try:
        yield model
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[1], previous[2]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[3], previous[4]
        if model is not None:
            model.train(was_training)
