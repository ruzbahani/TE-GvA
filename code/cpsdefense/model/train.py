import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from cpsdefense.data import hai, hai_tasks
from cpsdefense.eval.survival import KaplanMeier, KMBaseline, km_pmf
from cpsdefense.model.backbone import BackboneConfig, DecisionBackbone, eval_determinism
from cpsdefense.model.branches import Question
from cpsdefense.model.evidence import FIELDS, EvidenceBatch
from cpsdefense.model.losses import question_loss

VARIANTS = {
    "unmasked": {"mode": "unmasked", "hide_validity": False},
    "unmasked_novalidity": {"mode": "unmasked", "hide_validity": True},
    "masked": {"mode": "masked", "hide_validity": False},
    "block_local": {"mode": "block_local", "hide_validity": False},
}
BASELINES = ("km_marginal", "km_state", "km_state_age")
SCORE_NAME = "level_bin"
TARGET_FIELDS = ("time_to_event", "event_observed", "state_at_tau", "time_since_event", "attack_at_tau",
                 "attack_in_horizon")
SCORE_FIELDS = ("score_bin", "score_bin_now")


@dataclass(eq=False)
class TaskSplit:
    version: str
    split: str
    names: list
    batch: EvidenceBatch
    time: np.ndarray
    event: np.ndarray
    state: np.ndarray
    since: np.ndarray
    score: np.ndarray | None
    score_now: np.ndarray | None
    attack_at_tau: np.ndarray
    attack_in_horizon: np.ndarray

    @property
    def n(self):
        return self.batch.n

    @property
    def segment(self):
        return self.batch.segment

    @property
    def decision_time(self):
        return self.batch.decision_time

    def take(self, rows):
        pick = lambda a: None if a is None else a[rows]
        return TaskSplit(self.version, self.split, self.names, self.batch.take(rows), self.time[rows],
                         self.event[rows], self.state[rows], self.since[rows], pick(self.score),
                         pick(self.score_now), self.attack_at_tau[rows], self.attack_in_horizon[rows])


def load_split(version, split, root=None, rows=None):
    hai_tasks.check_split(split)
    meta = hai_tasks.load_meta(version, root)
    fields = list(FIELDS) + list(TARGET_FIELDS) + (list(SCORE_FIELDS) if meta.get("score") else [])
    a = hai_tasks.load_tasks(version, split, root=root, fields=fields)
    if rows is not None:
        a = {k: v[rows] for k, v in a.items()}
    batch = EvidenceBatch(**{k: a[k] for k in FIELDS})
    return TaskSplit(
        version=hai.normalize_version(version), split=split, names=list(meta["question_names"]),
        batch=batch, time=a["time_to_event"].astype(np.float64), event=a["event_observed"].astype(bool),
        state=a["state_at_tau"].astype(np.int64), since=a["time_since_event"].astype(np.float64),
        score=a["score_bin"].astype(np.int64) if meta.get("score") else None,
        score_now=a["score_bin_now"].astype(np.int64) if meta.get("score") else None,
        attack_at_tau=a["attack_at_tau"], attack_in_horizon=a["attack_in_horizon"])


def within_target(time, event, horizon):
    time, event = np.asarray(time, np.float64), np.asarray(event, bool)
    return np.where(event & (time <= horizon), 1.0, np.where(time >= horizon, 0.0, np.nan))


def noul_name(name, horizon):
    return f"{name}_within{horizon:g}"


@dataclass
class QuestionSpec:
    names: list
    edges: np.ndarray
    within_s: float
    n_levels: int | None

    @classmethod
    def from_config(cls, config, names, n_levels=None):
        q = config["questions"]
        edges = np.asarray(q["hazard_edges"], np.float64)
        if edges[0] != 0 or not np.all(np.diff(edges) > 0) or not np.isfinite(edges).all():
            raise ValueError("hazard_edges must start at 0, increase strictly and be finite")
        return cls(list(names), edges, float(q["noul_within_s"]), n_levels if q.get("score", True) else None)

    def questions(self):
        qs = [Question(n, "hazard", qid=i, bin_edges=self.edges) for i, n in enumerate(self.names)]
        qs += [Question(noul_name(n, self.within_s), "noul", qid=len(self.names) + i)
               for i, n in enumerate(self.names)]
        if self.n_levels:
            qs.append(Question(SCORE_NAME, "score", qid=2 * len(self.names), n_levels=self.n_levels))
        return qs

    @property
    def n_query_ids(self):
        return 2 * len(self.names) + 1

    def to_dict(self):
        return {"names": self.names, "edges": self.edges.tolist(), "within_s": self.within_s,
                "n_levels": self.n_levels}

    @classmethod
    def from_dict(cls, d):
        return cls(list(d["names"]), np.asarray(d["edges"], np.float64), float(d["within_s"]), d["n_levels"])


class DeviceTargets:
    def __init__(self, split, spec, device):
        self.spec = spec
        as_t = lambda a, dtype: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        self.time = as_t(split.time, torch.float64)
        self.event = as_t(split.event, torch.bool)
        self.within = as_t(within_target(split.time, split.event, spec.within_s), torch.float32)
        self.score = None if split.score is None or not spec.n_levels else as_t(split.score, torch.int64)

    def __call__(self, idx):
        out = {}
        for i, name in enumerate(self.spec.names):
            out[name] = {"time": self.time[idx, i], "event": self.event[idx, i]}
            out[noul_name(name, self.spec.within_s)] = self.within[idx, i]
        if self.score is not None:
            out[SCORE_NAME] = self.score[idx]
        return out

    def kept(self, idx):
        n = int(len(idx))
        out = {}
        for i, name in enumerate(self.spec.names):
            out[name] = n
            out[noul_name(name, self.spec.within_s)] = int((~torch.isnan(self.within[idx, i])).sum())
        if self.score is not None:
            out[SCORE_NAME] = int((self.score[idx] >= 0).sum())
        return out


def build_model(config, meta, variant, spec, seed, device=None):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    cfg = dict(config["model"])
    cfg.update(VARIANTS[variant])
    cfg.update(payload_dim=tuple(meta["payload_dim"]), n_kinds=len(meta["kinds"]), n_query_ids=spec.n_query_ids,
               max_score_levels=max(2, spec.n_levels or 2))
    torch.manual_seed(seed)
    model = DecisionBackbone(BackboneConfig.from_dict(cfg))
    return model.to(device) if device is not None else model


def n_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def save_model(path, model, spec, extra=None):
    torch.save({"config": model.config.to_dict(), "spec": spec.to_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "extra": extra or {}}, path)


def load_model(path, device=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(ckpt["config"])
    config["payload_dim"] = tuple(config["payload_dim"])
    model = DecisionBackbone(BackboneConfig.from_dict(config))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device) if device is not None else model
    return model.eval(), QuestionSpec.from_dict(ckpt["spec"])


def batch_losses(outputs, targets, censor_rule):
    return {name: question_loss(outputs[name], target, censor_rule) for name, target in targets.items()}


def evaluate_loss(model, gpu_batch, targets, questions, censor_rule="start", chunk=4096):
    was_training = model.training
    model.eval()
    sums, counts = {}, {}
    with torch.no_grad():
        for start in range(0, gpu_batch.n, chunk):
            idx = torch.arange(start, min(start + chunk, gpu_batch.n), device=gpu_batch.device)
            out = model(gpu_batch.take(idx), questions=questions, keep_cache=False)["outputs"]
            kept = targets.kept(idx)
            for name, loss in batch_losses(out, targets(idx), censor_rule).items():
                sums[name] = sums.get(name, 0.0) + float(loss) * kept[name]
                counts[name] = counts.get(name, 0) + kept[name]
    model.train(was_training)
    parts = {name: sums[name] / max(counts[name], 1) for name in sums}
    parts["total"] = float(sum(parts.values()))
    return parts


def train_model(model, train, val, spec, config, seed, device, log=print):
    tc = config["train"]
    rule = config["questions"].get("censor_rule", "start")
    questions = spec.questions()
    gpu_train, gpu_val = train.batch.to(device), val.batch.to(device)
    y_train, y_val = DeviceTargets(train, spec, device), DeviceTargets(val, spec, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=float(tc["lr_factor"]),
                                                       patience=int(tc["lr_patience"]))
    gen = torch.Generator().manual_seed(int(seed))
    bs, n = int(tc["batch_size"]), gpu_train.n
    max_steps, eval_every, patience = int(tc["max_steps"]), int(tc["eval_every"]), int(tc["patience"])
    budget_s = 60.0 * float(tc["max_minutes"])
    best, best_state, bad, curve = math.inf, None, 0, []
    perm, pos, step, running, n_running = torch.randperm(n, generator=gen), 0, 0, 0.0, 0
    stop = "max_steps"
    t0 = time.perf_counter()
    model.train()
    while step < max_steps:
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos:pos + bs].to(device)
        pos += bs
        out = model(gpu_train.take(idx), questions=questions, keep_cache=False)["outputs"]
        loss = sum(batch_losses(out, y_train(idx), rule).values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
        opt.step()
        running += loss.detach()
        n_running += 1
        step += 1
        if step % eval_every == 0 or step == max_steps:
            val_parts = evaluate_loss(model, gpu_val, y_val, questions, rule, int(tc["eval_chunk"]))
            elapsed = time.perf_counter() - t0
            train_loss = float(running) / max(n_running, 1)
            running, n_running = 0.0, 0
            improved = val_parts["total"] < best - 1e-6
            curve.append({"step": step, "elapsed_s": elapsed, "train_loss": train_loss, "val": val_parts,
                          "lr": opt.param_groups[0]["lr"], "improved": bool(improved)})
            log(f"  step {step:6d} {elapsed:6.0f}s train {train_loss:.4f} val {val_parts['total']:.4f}"
                f"{' *' if improved else ''}")
            if improved:
                best, bad = val_parts["total"], 0
                best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
            else:
                bad += 1
            sched.step(val_parts["total"])
            if bad >= patience:
                stop = "early_stopping"
                break
            if elapsed > budget_s:
                stop = "time_budget"
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    best_row = min(curve, key=lambda r: r["val"]["total"]) if curve else None
    return {"curve": curve, "best_step": best_row["step"] if best_row else 0,
            "best_val": best_row["val"] if best_row else None, "steps": step, "stop": stop,
            "train_seconds": time.perf_counter() - t0, "n_parameters": n_parameters(model)}


def flat_outputs(outputs, questions):
    out = {}
    for q in questions:
        o = outputs[q.name]
        key = {"hazard": "pmf", "noul": "p_yes", "score": "probs", "choice": "probs"}[q.type]
        out[q.name] = o[key]
    return out


def predict(model, gpu_batch, questions, chunk=4096, transform=None, deterministic=True):
    parts = {q.name: [] for q in questions}
    ctx = eval_determinism(model) if deterministic else torch.no_grad()
    with ctx, torch.no_grad():
        model.eval()
        for start in range(0, gpu_batch.n, chunk):
            rows = torch.arange(start, min(start + chunk, gpu_batch.n), device=gpu_batch.device)
            sub = gpu_batch.take(rows)
            if transform is not None:
                sub = transform(sub, rows)
            flat = flat_outputs(model(sub, questions=questions, keep_cache=False)["outputs"], questions)
            for name, value in flat.items():
                parts[name].append(value.detach().cpu().numpy())
    return {name: np.concatenate(v) for name, v in parts.items()}


def to_predictions(raw, spec):
    out = {"pmf": {}, "noul": {}, "score": None}
    for name in spec.names:
        p = np.clip(raw[name].astype(np.float64), 0.0, None)
        out["pmf"][name] = p / p.sum(-1, keepdims=True)
        out["noul"][name] = raw[noul_name(name, spec.within_s)].astype(np.float64)
    if spec.n_levels and SCORE_NAME in raw:
        s = np.clip(raw[SCORE_NAME].astype(np.float64), 0.0, None)
        out["score"] = s / s.sum(-1, keepdims=True)
    return out


def save_predictions(path, preds):
    arrays = {f"pmf/{k}": v for k, v in preds["pmf"].items()}
    arrays.update({f"noul/{k}": v for k, v in preds["noul"].items()})
    if preds["score"] is not None:
        arrays["score"] = preds["score"]
    np.savez_compressed(path, **arrays)


def load_predictions(path):
    out = {"pmf": {}, "noul": {}, "score": None}
    with np.load(path) as z:
        for key in z.files:
            if key == "score":
                out["score"] = z[key]
            else:
                group, name = key.split("/", 1)
                out[group][name] = z[key]
    return out


def age_bin(since, edges):
    since = np.asarray(since, np.float64)
    b = np.searchsorted(np.asarray(edges, np.float64), np.nan_to_num(since, nan=0.0), side="right")
    return np.where(np.isnan(since), len(edges) + 1, b).astype(np.int64)


class KMPredictor:
    def __init__(self, spec, min_events=10, age_edges=(10, 30, 60, 120, 300, 600, 1800), shrink_events=10):
        self.spec = spec
        self.min_events = int(min_events)
        self.age_edges = tuple(float(a) for a in age_edges)
        self.shrink = float(shrink_events)

    def fit(self, train):
        e = self.spec.edges
        self.state, self.state_age, self.age_events = {}, {}, {}
        for i, name in enumerate(self.spec.names):
            t, d, s = train.time[:, i], train.event[:, i], train.state[:, i]
            self.state[name] = KMBaseline(e, self.min_events).fit(t, d, s)
            groups = s * 100 + age_bin(train.since[:, i], self.age_edges)
            rows, events = {}, {}
            for g in np.unique(groups):
                sel = groups == g
                if d[sel].sum() >= 1:
                    rows[g], events[g] = km_pmf(KaplanMeier(t[sel], d[sel]), e), float(d[sel].sum())
            self.state_age[name], self.age_events[name] = rows, events
        self.score_marginal = self.score_by_now = None
        if self.spec.n_levels and train.score is not None:
            ok = train.score >= 0
            k = self.spec.n_levels
            counts = np.bincount(train.score[ok], minlength=k).astype(np.float64)
            self.score_marginal = counts / counts.sum()
            self.score_by_now = {}
            for b in range(k):
                sel = ok & (train.score_now == b)
                c = np.bincount(train.score[sel], minlength=k).astype(np.float64)
                self.score_by_now[b] = c / c.sum() if c.sum() else self.score_marginal
        return self

    def _cdf_at(self, pmf, t):
        k = int(np.searchsorted(self.spec.edges, t))
        if self.spec.edges[k] != t:
            raise ValueError(f"within_s = {t} must be a hazard edge")
        return pmf[:, :k].sum(-1)

    def predict(self, split, kind):
        if kind not in BASELINES:
            raise ValueError(f"unknown baseline {kind!r}")
        out = {"pmf": {}, "noul": {}, "score": None}
        for i, name in enumerate(self.spec.names):
            km = self.state[name]
            s = split.state[:, i]
            if kind == "km_marginal":
                pmf = km.pmf(n=split.n)
            else:
                pmf = km.pmf(groups=s)
                if kind == "km_state_age":
                    groups = s * 100 + age_bin(split.since[:, i], self.age_edges)
                    rows, events = self.state_age[name], self.age_events[name]
                    for g in np.unique(groups):
                        if g in rows:
                            sel = groups == g
                            d = events[g]
                            pmf[sel] = (d * rows[g] + self.shrink * pmf[sel]) / (d + self.shrink)
            out["pmf"][name] = pmf
            out["noul"][name] = self._cdf_at(pmf, self.spec.within_s)
        if self.score_marginal is not None and split.score is not None:
            if kind == "km_marginal":
                out["score"] = np.tile(self.score_marginal, (split.n, 1))
            else:
                out["score"] = np.stack([self.score_by_now.get(int(b), self.score_marginal) for b in split.score_now])
        return out
