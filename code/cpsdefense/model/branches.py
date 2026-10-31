from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cpsdefense.model.encoder import CrossBlock

QUESTION_TYPES = ("noul", "choice", "score", "hazard")


@dataclass(eq=False)
class Question:
    name: str
    type: str
    qid: int | None = None
    descriptor: object = None
    n_levels: int | None = None
    options: object = None
    bin_edges: object = None

    def __post_init__(self):
        if self.type not in QUESTION_TYPES:
            raise ValueError(f"{self.name}: type must be one of {QUESTION_TYPES}, got {self.type!r}")
        if self.qid is None and self.descriptor is None:
            raise ValueError(f"{self.name}: needs a qid or a descriptor")
        if self.type == "score" and (self.n_levels is None or self.n_levels < 2):
            raise ValueError(f"{self.name}: score questions need n_levels >= 2")
        if self.type == "choice" and (self.options is None or len(self.options) == 0):
            raise ValueError(f"{self.name}: choice questions need options")
        if self.type == "hazard":
            edges = np.asarray(self.bin_edges, dtype=np.float64) if self.bin_edges is not None else np.zeros(0)
            if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges[0]) or np.any(np.diff(edges) <= 0):
                raise ValueError(f"{self.name}: bin_edges must be strictly increasing, >= 2 values, first finite")


def check_questions(questions):
    questions = list(questions)
    names = [q.name for q in questions]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate question names in {names}")
    return questions


class QuestionBranches(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, d_ff=256, n_query_ids=16, descriptor_dim=0,
                 dropout=0.0):
        super().__init__()
        self.d_model, self.descriptor_dim = int(d_model), int(descriptor_dim)
        self.type_emb = nn.Embedding(len(QUESTION_TYPES), d_model)
        self.query_emb = nn.Embedding(n_query_ids, d_model) if n_query_ids else None
        self.descriptor = (nn.Sequential(nn.Linear(descriptor_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
                           if descriptor_dim else None)
        self.blocks = nn.ModuleList(CrossBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def query(self, question, n):
        device = self.type_emb.weight.device
        q = self.type_emb.weight[QUESTION_TYPES.index(question.type)]
        if question.qid is not None:
            if self.query_emb is None or not 0 <= question.qid < self.query_emb.num_embeddings:
                raise ValueError(f"{question.name}: qid {question.qid} outside the learned query table")
            q = q + self.query_emb.weight[question.qid]
        if question.descriptor is not None:
            if self.descriptor is None:
                raise ValueError(f"{question.name}: descriptor given but descriptor_dim is 0")
            desc = torch.as_tensor(question.descriptor, dtype=self.type_emb.weight.dtype, device=device)
            if desc.shape[-1] != self.descriptor_dim or desc.ndim not in (1, 2) or (desc.ndim == 2 and len(desc) != n):
                raise ValueError(f"{question.name}: descriptor must be [{self.descriptor_dim}] or "
                                 f"[{n}, {self.descriptor_dim}], got {tuple(desc.shape)}")
            q = q + self.descriptor(desc)
        return q.expand(n, self.d_model)

    def forward(self, items, mask, pooled, questions):
        n = items.shape[0]
        if not questions:
            return items.new_zeros(n, 0, self.d_model)
        kv = torch.cat([items, pooled], 1)
        kv_mask = torch.cat([mask, mask.new_ones(n, pooled.shape[1])], 1)
        h = torch.stack([self.query(q, n) for q in questions], 1)
        for block in self.blocks:
            h = block(h, kv, kv_mask)
        return self.norm(h)
