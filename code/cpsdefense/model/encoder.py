import math
from dataclasses import dataclass

import torch
from torch import nn

from cpsdefense.model.evidence import EvidenceBatch

MODES = ("masked", "block_local", "unmasked")
LOG_SCALES = (1.0, 10.0, 100.0, 1e3, 1e4)
FOURIER_PERIODS = (60.0, 3600.0, 86400.0)


def time_features(dt, log_scales=LOG_SCALES, fourier_periods=FOURIER_PERIODS, max_dt=1e7, dtype=torch.float32):
    x = dt.clamp(-max_dt, max_dt)
    feats = [torch.sign(x) * torch.log1p(x.abs() / s) / math.log1p(max_dt / s) for s in log_scales]
    for period in fourier_periods:
        angle = x * (2 * math.pi / period)
        feats += [torch.sin(angle), torch.cos(angle)]
    return torch.stack(feats, -1).to(dtype)


def n_time_features(log_scales=LOG_SCALES, fourier_periods=FOURIER_PERIODS):
    return len(log_scales) + 2 * len(fourier_periods)


class FeedForward(nn.Sequential):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh = n_heads, d_model // n_heads
        self.q, self.k, self.v, self.out = (nn.Linear(d_model, d_model) for _ in range(4))
        self.null_k = nn.Parameter(torch.randn(n_heads, 1, self.dh) * 0.02)
        self.null_v = nn.Parameter(torch.zeros(n_heads, 1, self.dh))
        self.drop = nn.Dropout(dropout)

    def _heads(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.h, self.dh).transpose(1, 2)

    def forward(self, x, kv, kv_mask):
        b = x.shape[0]
        q = self._heads(self.q(x))
        k = torch.cat([self.null_k.expand(b, -1, -1, -1), self._heads(self.k(kv))], 2)
        v = torch.cat([self.null_v.expand(b, -1, -1, -1), self._heads(self.v(kv))], 2)
        keep = torch.cat([kv_mask.new_ones(b, 1), kv_mask], 1)[:, None, None, :]
        scores = (q @ k.transpose(-1, -2)) * self.dh ** -0.5
        weights = self.drop(scores.masked_fill(~keep, float("-inf")).softmax(-1))
        return self.out((weights @ v).transpose(1, 2).reshape(b, x.shape[1], -1))


class SetBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm_attn, self.norm_ff = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, mask):
        x = self.norm_attn(h)
        h = h + self.drop(self.attn(x, x, mask))
        h = h + self.drop(self.ff(self.norm_ff(h)))
        return h.masked_fill(~mask[..., None], 0.0)


class ItemBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        return h + self.drop(self.ff(self.norm(h)))


class CrossBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ff = (nn.LayerNorm(d_model) for _ in range(3))
        self.attn = Attention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, kv, kv_mask):
        h = h + self.drop(self.attn(self.norm_q(h), self.norm_kv(kv), kv_mask))
        return h + self.drop(self.ff(self.norm_ff(h)))


class AttentionPool(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_pool=4, dropout=0.0):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(n_pool, d_model) * 0.02)
        self.block = CrossBlock(d_model, n_heads, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, items, mask):
        seeds = self.seeds.expand(items.shape[0], -1, -1)
        return self.norm(self.block(seeds, items, mask))


class PackIndex:
    def __init__(self, selected):
        self.n = selected.shape[0]
        self.rows, self.slots = selected.nonzero(as_tuple=True)
        self.pos = (selected.cumsum(1) - 1)[self.rows, self.slots]
        self.length = max(int(selected.sum(1).max()), 1) if self.n else 1
        self.mask = torch.zeros(self.n, self.length, dtype=torch.bool, device=selected.device)
        self.mask[self.rows, self.pos] = True

    def gather(self, x):
        return x[self.rows, self.slots]

    def pack(self, flat):
        out = flat.new_zeros((self.n, self.length, *flat.shape[1:]))
        return out.index_put((self.rows, self.pos), flat)


@dataclass
class ItemCache:
    item_id: torch.Tensor
    states: torch.Tensor
    encoded: torch.Tensor


@dataclass
class EncoderOutput:
    items: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor
    used: torch.Tensor
    cache: ItemCache | None
    n_encoded: int


class EvidenceEncoder(nn.Module):
    def __init__(self, payload_dim, n_kinds=1, mode="masked", d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 n_pool=4, dropout=0.0, hide_validity=False, expose_mask_count=False,
                 log_scales=LOG_SCALES, fourier_periods=FOURIER_PERIODS, max_dt=1e7):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if hide_validity and mode != "unmasked":
            raise ValueError("hide_validity applies to the unmasked mode only")
        if expose_mask_count and mode == "unmasked":
            raise ValueError("expose_mask_count applies to the masked modes only")
        self.mode, self.hide_validity, self.expose_mask_count = mode, bool(hide_validity), bool(expose_mask_count)
        self.payload_dim = [int(p) for p in payload_dim]
        self.n_sources, self.n_kinds, self.d_model = len(self.payload_dim), int(n_kinds), int(d_model)
        self.log_scales, self.fourier_periods, self.max_dt = tuple(log_scales), tuple(fourier_periods), float(max_dt)
        f = n_time_features(self.log_scales, self.fourier_periods)

        self.source_proj = nn.ModuleList(nn.Linear(p, d_model) for p in self.payload_dim)
        self.source_emb = nn.Embedding(self.n_sources, d_model)
        self.kind_emb = nn.Embedding(self.n_kinds, d_model)
        self.intrinsic = nn.Linear(2 * f, d_model)
        n_relative = 2 if self.hide_validity else 3
        self.relative = nn.Sequential(nn.Linear(n_relative * f, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.status = nn.Linear(2, d_model) if mode == "unmasked" and not self.hide_validity else None
        if mode == "block_local":
            self.item_blocks = nn.ModuleList(ItemBlock(d_model, d_ff, dropout) for _ in range(n_layers))
            self.time_block = ItemBlock(d_model, d_ff, dropout)
        else:
            self.blocks = nn.ModuleList(SetBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm_out = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model, n_heads, d_ff, n_pool, dropout)
        self.count_proj = nn.Linear(1, d_model) if self.expose_mask_count else None

    @property
    def device(self):
        return self.source_emb.weight.device

    @property
    def dtype(self):
        return self.source_emb.weight.dtype

    def _features(self, dt):
        return time_features(dt, self.log_scales, self.fourier_periods, self.max_dt, self.dtype)

    def _check_indices(self, source, kind):
        if source.numel() and (int(source.min()) < 0 or int(source.max()) >= self.n_sources):
            raise ValueError(f"source index outside [0, {self.n_sources}) on a used item")
        if kind.numel() and (int(kind.min()) < 0 or int(kind.max()) >= self.n_kinds):
            raise ValueError(f"kind index outside [0, {self.n_kinds}) on a used item")

    def _project(self, payload, source):
        out = payload.new_zeros(payload.shape[0], self.d_model)
        for s, proj in enumerate(self.source_proj):
            rows = (source == s).nonzero(as_tuple=True)[0]
            if rows.numel():
                out = out.index_put((rows,), proj(payload[rows, :self.payload_dim[s]]))
        return out

    def content(self, batch, rows, slots):
        source, kind = batch.source[rows, slots], batch.kind[rows, slots]
        self._check_indices(source, kind)
        event_time = batch.event_time[rows, slots]
        span = self._features(event_time - batch.event_start[rows, slots])
        delay = self._features(batch.arrival_time[rows, slots] - event_time)
        return (self._project(batch.payload[rows, slots].to(self.dtype), source) + self.source_emb(source)
                + self.kind_emb(kind) + self.intrinsic(torch.cat([span, delay], -1)))

    def relative_time(self, batch, idx, tau):
        tau = tau[idx.rows, 0]
        parts = [self._features(tau - idx.gather(batch.event_time)),
                 self._features(tau - idx.gather(batch.arrival_time))]
        if not self.hide_validity:
            parts.append(self._features(idx.gather(batch.valid_until) - tau))
        x = self.relative(torch.cat(parts, -1))
        if self.status is not None:
            expired = ~(tau < idx.gather(batch.valid_until))
            retracted = ~(tau < idx.gather(batch.retracted_at))
            x = x + self.status(torch.stack([expired, retracted], -1).to(x.dtype))
        return x

    def _content_states(self, batch, idx, cache):
        if cache is None:
            new = None
            states = None
        else:
            if tuple(cache.item_id.shape) != (batch.n, batch.m) or not torch.equal(cache.item_id, batch.item_id):
                raise ValueError("item cache was built for a different batch")
            reuse = idx.gather(cache.encoded)
            states = idx.gather(cache.states)
            new = (~reuse).nonzero(as_tuple=True)[0]
        rows, slots = (idx.rows, idx.slots) if new is None else (idx.rows[new], idx.slots[new])
        if rows.numel() == 0:
            return states, 0
        h = self.content(batch, rows, slots)
        if states is not None:
            states = states.to(h.dtype)
        for block in self.item_blocks:
            h = block(h)
        states = h if new is None else states.index_put((new,), h)
        return states, int(rows.numel())

    def _build_cache(self, batch, idx, states, cache):
        base = (cache.states.clone() if cache is not None
                else states.new_zeros(batch.n, batch.m, self.d_model))
        base[idx.rows, idx.slots] = states.detach()
        encoded = torch.zeros_like(batch.present)
        encoded[idx.rows, idx.slots] = True
        if cache is not None:
            encoded = encoded | cache.encoded
        return ItemCache(item_id=batch.item_id, states=base, encoded=encoded)

    def forward(self, batch, tau=None, cache=None, build_cache=False):
        if not isinstance(batch, EvidenceBatch):
            raise TypeError("batch must be an EvidenceBatch")
        batch = batch.to(self.device)
        t = batch.time_column(tau)
        used = batch.known(t) if self.mode == "unmasked" else batch.validity(t)
        idx = PackIndex(used)
        n_encoded, new_cache = idx.rows.numel(), None
        if self.mode == "block_local":
            states, n_encoded = self._content_states(batch, idx, cache)
            if states is None:
                states = torch.zeros(0, self.d_model, device=self.device, dtype=self.dtype)
            h = idx.pack(self.time_block(states + self.relative_time(batch, idx, t)))
            if build_cache:
                new_cache = self._build_cache(batch, idx, states, cache)
        else:
            h = idx.pack(self.content(batch, idx.rows, idx.slots) + self.relative_time(batch, idx, t))
            for block in self.blocks:
                h = block(h, idx.mask)
        items = self.norm_out(h).masked_fill(~idx.mask[..., None], 0.0)
        pooled = self.pool(items, idx.mask)
        if self.count_proj is not None:
            masked = (batch.known(t) & ~used).sum(1, keepdim=True)
            pooled = pooled + self.count_proj(torch.log1p(masked.to(pooled.dtype)))[:, None]
        return EncoderOutput(items=items, mask=idx.mask, pooled=pooled, used=used, cache=new_cache,
                             n_encoded=int(n_encoded))
