import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.agents.flat_baseline import FlatBaseline, evidence_summary, feature_matrix
from cpsdefense.eval.metrics import cluster_bootstrap_ci, paired_cluster_bootstrap_diff
from cpsdefense.experience.dataset import KINDS
from cpsdefense.experience.tensors import evidence_batch
from cpsdefense.model.direct import DirectConfig, DirectDecisionModel, derived_voi, losses, pad_candidates
from cpsdefense.model.summary import SummaryDescriptor
from cpsdefense.paths import ROOT
from cpsdefense.utils.config import load_config, merged, set_dotted
from cpsdefense.utils.runs import start_run
from cpsdefense.utils.seeding import set_global_seed

LABELS = ("J", "p_trip", "trip_time", "trip_event", "p_resolved", "duration", "voi", "observe")
BELIEF = {"J": "J_belief", "p_trip": "p_trip_belief", "voi": "voi_belief"}
LABEL_SETS = ("hindsight", "belief")
FLAT_GRID = [{"center": c, "learning_rate": lr, "max_leaf_nodes": leaves}
             for c in (False, True) for lr in (0.05, 0.1) for leaves in (15, 63)]
TABLE = ("regret", "regret_high_stakes", "trip_rate_high_stakes", "gva_accuracy", "gva_accuracy_material",
         "voi_sign_accuracy", "trip_brier")


def run_config(path, overrides=()):
    cfg = load_config(path)
    base = cfg.pop("base", None)
    if base is not None:
        cfg = merged(run_config(base), cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        set_dotted(cfg, key, yaml.safe_load(value))
    return cfg


def load_split(root, split):
    a = dict(np.load(Path(root) / f"{split}.npz", allow_pickle=False))
    n = len(a["dec_tau"])
    belief = [b for b in BELIEF.values() if f"y_{b}" in a]
    flat = {"features": a["cand_features"], **{k: a[f"y_{k}"] for k in (*LABELS, *belief)}}
    mask, pad = pad_candidates(a["cand_decision"], n, flat)
    pad["features"] = np.where(mask[..., None], pad["features"], 0.0).astype(np.float32)
    return {"name": split, "batch": evidence_batch(a), "mask": mask, "pad": pad, "n": n,
            "dec": {k[4:]: v for k, v in a.items() if k.startswith("dec_")},
            "cand": {"features": a["cand_features"], "decision": a["cand_decision"], "J": a["y_J"],
                     "p_trip": a["y_p_trip"], **{b: a[f"y_{b}"] for b in belief if b != BELIEF["voi"]}}}


def subsample_episodes(split, fraction, seed=0):
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    episodes = np.unique(split["dec"]["episode_seed"])
    order = np.random.default_rng(seed).permutation(episodes)
    keep_ep = order[:max(1, int(round(fraction * len(episodes))))]
    rows = np.flatnonzero(np.isin(split["dec"]["episode_seed"], keep_ep))
    new_index = np.full(split["n"], -1, np.int64)
    new_index[rows] = np.arange(rows.size)
    cand_rows = np.flatnonzero(new_index[split["cand"]["decision"]] >= 0)
    cand = {k: v[cand_rows] for k, v in split["cand"].items()}
    cand["decision"] = new_index[split["cand"]["decision"][cand_rows]]
    return {**split, "n": int(rows.size), "batch": split["batch"].take(rows), "mask": split["mask"][rows],
            "pad": {k: v[rows] for k, v in split["pad"].items()}, "dec": {k: v[rows] for k, v in split["dec"].items()},
            "cand": cand, "episodes_kept": keep_ep}


def has_belief(split):
    return all(b in split["pad"] for b in BELIEF.values())


def label_key(key, labels):
    if labels not in LABEL_SETS:
        raise ValueError(f"labels must be one of {LABEL_SETS}, got {labels!r}")
    return BELIEF.get(key, key) if labels == "belief" else key


def label_view(split, labels):
    if labels == "hindsight":
        return split
    if not has_belief(split):
        raise ValueError(f"{split.get('name')}: no belief labels in the data (label format 3 needed)")
    pad = {**split["pad"], **{k: split["pad"][label_key(k, labels)] for k in BELIEF}}
    cand = {**split["cand"], **{k: split["cand"][label_key(k, labels)] for k in ("J", "p_trip")}}
    return {**split, "pad": pad, "cand": cand}


def add_summary_descriptor(data, vocab, log=print, desc=None):
    fitted = desc is not None
    desc = desc if fitted else SummaryDescriptor(vocab)
    raw = {s: desc.raw(d["batch"], d["cand"]["features"], d["cand"]["decision"]) for s, d in data.items()}
    if not fitted:
        desc.fit(*raw["train"])
    for s, d in data.items():
        full = np.concatenate([np.asarray(d["cand"]["features"], np.float32), desc.transform(*raw[s])], 1)
        mask, pad = pad_candidates(d["cand"]["decision"], d["n"], {"features": full})
        if not np.array_equal(mask, d["mask"]):
            raise AssertionError(f"{s}: candidate layout changed while adding the summary")
        d["pad"]["features"] = np.where(mask[..., None], pad["features"], 0.0).astype(np.float32)
    log(f"summary descriptor: {len(desc.names)} summary columns + {int(desc.miss_cols.sum())} missing indicators "
        f"({'loaded, fitted on train at training time' if fitted else 'fitted on train'}) appended to "
        f"{len(vocab['action_columns'])} action features")
    return desc


def add_summary_targets(data, vocab, log=print):
    desc = SummaryDescriptor(vocab)
    raw = {s: desc.raw(d["batch"], d["cand"]["features"], d["cand"]["decision"]) for s, d in data.items()}
    desc.fit(*raw["train"])
    for s, d in data.items():
        mask, pad = pad_candidates(d["cand"]["decision"], d["n"], {"summary": desc.transform(*raw[s])})
        if not np.array_equal(mask, d["mask"]):
            raise AssertionError(f"{s}: candidate layout changed while adding the summary targets")
        d["pad"]["summary"] = np.where(mask[..., None], pad["summary"], 0.0).astype(np.float32)
    log(f"summary targets for the auxiliary head: width {desc.width} (fitted on train)")
    return desc


def take(split, sel):
    sel = np.asarray(sel, bool)
    return {"name": split.get("name"), "n": int(sel.sum()), "mask": split["mask"][sel],
            "pad": {k: v[sel] for k, v in split["pad"].items()}, "dec": {k: v[sel] for k, v in split["dec"].items()}}


def subsets(split):
    n, dec = split["n"], split["dec"]
    out = [("all", np.ones(n, bool))]
    if "family" in dec:
        out.append(("family", dec["family"].astype(bool)))
    if "backup_available" in dec and 0 < dec["backup_available"].sum() < n:
        backup = dec["backup_available"].astype(bool)
        out += [("backup_available", backup), ("backup_unavailable", ~backup)]
        if "family" in dec:
            fam = dec["family"].astype(bool)
            out += [("family_backup_available", fam & backup), ("family_backup_unavailable", fam & ~backup)]
    return [(name, sel) for name, sel in out if sel.any()]


def episode_halves(cluster):
    return np.unique(np.asarray(cluster), return_inverse=True)[1].ravel() % 2


def to_device(split, device):
    pad = split["pad"]
    y = {k: torch.as_tensor(pad[k]).to(device) for k in LABELS}
    for k in ("J", "p_trip", "trip_time", "p_resolved", "duration", "voi"):
        y[k] = y[k].to(torch.float32)
    y["trip_event"] = torch.nan_to_num(y["trip_event"].to(torch.float32)) > 0.5
    y["observe"] = y["observe"].to(torch.bool)
    if "summary" in pad:
        y["summary"] = torch.as_tensor(pad["summary"], dtype=torch.float32, device=device)
    cluster = split.get("dec", {}).get("episode_seed", np.arange(split["n"]))
    return {"n": split["n"], "batch": split["batch"].to(device), "y": y,
            "feats": torch.as_tensor(pad["features"], device=device),
            "mask": torch.as_tensor(split["mask"], device=device), "half": episode_halves(cluster)}


def rows_of(d, rows):
    return d["batch"].take(rows), d["feats"][rows], d["mask"][rows], {k: v[rows] for k, v in d["y"].items()}


def val_loss(model, d, chunk=64, per_decision=False):
    model.eval()
    total, n, parts, regret = 0.0, 0, {}, 0.0
    reg_dec, se_dec, n_dec = [], [], []
    with torch.no_grad():
        for s in range(0, d["n"], chunk):
            rows = torch.arange(s, min(d["n"], s + chunk), device=d["feats"].device)
            batch, feats, mask, y = rows_of(d, rows)
            out = model(batch, feats, mask=mask)
            loss, p = losses(out, y, mask, model.config)
            total += float(loss) * len(rows)
            n += len(rows)
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v * len(rows)
            keep = mask & torch.isfinite(y["J"])
            Jt = torch.where(keep, y["J"], torch.inf)
            chosen = torch.where(mask, out["J_pred"], torch.inf).argmin(-1, keepdim=True)
            r = Jt.gather(-1, chosen).squeeze(-1) - Jt.amin(-1)
            regret += float(r.sum())
            if per_decision:
                jt = torch.nan_to_num(y["J"] / model.config.j_scale)
                cnt = keep.sum(-1, keepdim=True).clamp(min=1)
                cp = out["J"] - torch.where(keep, out["J"], 0.0).sum(-1, keepdim=True) / cnt
                ct = jt - torch.where(keep, jt, 0.0).sum(-1, keepdim=True) / cnt
                reg_dec.append(r.float().cpu().numpy())
                se_dec.append(torch.where(keep, 0.5 * (cp - ct) ** 2, 0.0).sum(-1).float().cpu().numpy())
                n_dec.append(keep.sum(-1).cpu().numpy())
    out = (total / n, {k: v / n for k, v in parts.items()}, regret / n)
    if per_decision:
        out += ({"regret": np.concatenate(reg_dec), "jc_se": np.concatenate(se_dec), "jc_n": np.concatenate(n_dec)},)
    return out


def half_scores(per_dec, half):
    out = []
    for h in (0, 1):
        sel = half == h
        out.append((float(per_dec["regret"][sel].mean()) if sel.any() else np.nan,
                    float(per_dec["jc_se"][sel].sum() / max(per_dec["jc_n"][sel].sum(), 1)), int(sel.sum())))
    return out


def smoothed(values, k):
    values = np.asarray(values, np.float64)
    return np.array([values[max(0, i - k + 1):i + 1].mean() for i in range(len(values))])


def crossfit_epoch_selection(halves, rule, smooth=1, min_epochs=0):
    col = {"regret": 0, "J_centered": 1}[rule]
    h = np.asarray(halves, np.float64)
    if not len(h) or (h[:, :, 2] == 0).any():
        return {"regret": float("nan"), "epochs": [None, None]}
    first = min(int(min_epochs), len(h) - 1)
    chosen, total, count = [], 0.0, 0.0
    for a, b in ((0, 1), (1, 0)):
        crit = smoothed(h[:, a, col], smooth)
        e = first + int(np.nanargmin(crit[first:]))
        chosen.append(e + 1)
        total += h[e, b, 0] * h[e, b, 2]
        count += h[e, b, 2]
    return {"regret": float(total / count), "epochs": chosen}


def train_one(cfg, dev, seed, log):
    set_global_seed(seed)
    tc = cfg["train"]
    model = DirectDecisionModel(DirectConfig(**cfg["model"])).to(dev["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    decay = float(tc.get("ema", 0.0))
    ema = copy.deepcopy(model) if decay else model
    stop_on = tc.get("stop_on", "total")
    smooth, min_epochs = int(tc.get("stop_smooth", 1)), int(tc.get("min_epochs", 0))
    train, val = dev["train"], dev["val"]
    rng = np.random.default_rng(seed)
    best, best_state, best_parts, bad, history, regrets, halves = np.inf, None, None, 0, [], [], []
    raw = []
    t0 = time.perf_counter()
    for epoch in range(tc["max_epochs"]):
        model.train()
        order = torch.as_tensor(rng.permutation(train["n"]), device=train["feats"].device)
        for s in range(0, train["n"], tc["batch_decisions"]):
            batch, feats, mask, y = rows_of(train, order[s:s + tc["batch_decisions"]])
            loss, _ = losses(model(batch, feats, mask=mask), y, mask, model.config)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            if decay:
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(p, 1.0 - decay)
        total, parts, vregret, per_dec = val_loss(ema, val, per_decision=True)
        halves.append(half_scores(per_dec, val["half"]))
        raw.append(total if stop_on == "total" else vregret if stop_on == "regret"
                   else parts[stop_on] if isinstance(stop_on, str)
                   else sum(float(w) * parts[name] for name, w in stop_on.items()))
        vloss = float(smoothed(raw, smooth)[-1])
        history.append(vloss)
        regrets.append(vregret)
        if tc.get("log_epochs"):
            log(f"  seed {seed} epoch {epoch + 1}: val loss {total:.4f} (J {parts['J']:.3f}, J_centered "
                f"{parts['J_centered']:.4f}) val regret {vregret:.3f}")
        if epoch + 1 <= min_epochs:
            best_parts, best_state, k = parts, copy.deepcopy(ema.state_dict()), epoch
        elif vloss < best - 1e-4:
            best, best_parts, best_state, bad, k = vloss, parts, copy.deepcopy(ema.state_dict()), 0, epoch
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
        if time.perf_counter() - t0 > 60 * tc.get("max_minutes", np.inf):
            log(f"seed {seed}: time budget of {tc['max_minutes']} min reached")
            break
    model = ema
    model.load_state_dict(best_state)
    secs = time.perf_counter() - t0
    log(f"seed {seed}: {epoch + 1} epochs in {secs:.0f}s, best val {stop_on} {best:.4f} at epoch {k + 1} "
        + " ".join(f"{name}={v:.3f}" for name, v in best_parts.items())
        + f"; val regret there {regrets[k]:.3f} (lowest {min(regrets):.3f} at epoch {int(np.argmin(regrets)) + 1})")
    crossfit = {rule: crossfit_epoch_selection(halves, rule, smooth, min_epochs) for rule in ("J_centered", "regret")}
    return model, {"seed": seed, "epochs": epoch + 1, "best_epoch": k + 1, "stop_on": stop_on, "best_val_loss": best,
                   "best_val_parts": best_parts, "history": history, "val_regret_history": regrets,
                   "val_halves": halves, "crossfit_selection": crossfit, "seconds": secs}


def predict(model, d, chunk=64):
    model.eval()
    J, trip = [], []
    with torch.no_grad():
        for s in range(0, d["n"], chunk):
            rows = torch.arange(s, min(d["n"], s + chunk), device=d["feats"].device)
            batch, feats, mask, _ = rows_of(d, rows)
            o = model(batch, feats, mask=mask)
            J.append(o["J_pred"].float().cpu().numpy())
            trip.append(torch.sigmoid(o["trip_logit"]).float().cpu().numpy())
    return np.concatenate(J), np.concatenate(trip)


def flat_predictions(data, vocab, log, grid=FLAT_GRID):
    t0 = time.perf_counter()
    X = {s: feature_matrix(evidence_summary(d["batch"], vocab), d["cand"]["features"], d["cand"]["decision"],
                           vocab)[0] for s, d in data.items()}
    n_action = len(vocab["action_columns"])
    tr, val = data["train"], data["val"]
    ctx = context(val, vocab)
    table, best = [], None
    for params in grid:
        fb = FlatBaseline(**params).fit(X["train"], tr["cand"]["J"], tr["cand"]["p_trip"], tr["cand"]["decision"],
                                        trip=False)
        J_pred = pad_candidates(val["cand"]["decision"], val["n"], {"J": fb.predict(X["val"])[0]})[1]["J"]
        regret = float(np.mean(rows_for_choice(ctx, np.where(ctx["mask"], J_pred, np.inf).argmin(1))["regret"]))
        table.append({**params, "val_regret": regret})
        if best is None or regret < best[0]:
            best = (regret, params)
    fitted = {"flat_gbm": (FlatBaseline(**best[1]), slice(None)),
              "action_prior": (FlatBaseline(center=True, learning_rate=0.05, max_leaf_nodes=15), slice(0, n_action))}
    out = {}
    for name, (fb, cols) in fitted.items():
        fb.fit(X["train"][:, cols], tr["cand"]["J"], tr["cand"]["p_trip"], tr["cand"]["decision"])
        out[name] = {}
        for s, d in data.items():
            if s != "train":
                J_pred, trip = fb.predict(X[s][:, cols])
                pad = pad_candidates(d["cand"]["decision"], d["n"], {"J": J_pred, "trip": trip})[1]
                out[name][s] = (pad["J"][None], pad["trip"][None])
    log(f"flat baseline: {X['train'].shape[1]} features, chosen {best[1]} (val regret {best[0]:.3f}); "
        f"action prior on {n_action} action features; {time.perf_counter() - t0:.0f}s")
    return out, table, best[1]


def context(split, vocab, material_gap=0.5, labels="hindsight"):
    mask, pad, n = split["mask"], split["pad"], split["n"]
    J = np.where(mask, pad[label_key("J", labels)].astype(np.float64), np.inf)
    p = np.where(mask, pad[label_key("p_trip", labels)].astype(np.float64), 0.0)
    other = "belief" if labels == "hindsight" else "hindsight"
    cross = None
    if label_key("J", other) in pad:
        cross = (f"{other}_oracle", np.where(mask, pad[label_key("J", other)].astype(np.float64), np.inf).argmin(1))
    observe = mask & pad["observe"].astype(bool)
    act = mask & ~observe
    best = J.min(1)
    pmin, pmax = np.where(mask, p, np.inf).min(1), np.where(mask, p, -np.inf).max(1)
    obs_min, act_min = np.where(observe, J, np.inf).min(1), np.where(act, J, np.inf).min(1)
    has_both = observe.any(1) & act.any(1)
    with np.errstate(invalid="ignore"):
        gap = np.where(has_both, obs_min - act_min, np.nan)
    lo, hi = vocab["action_blocks"]["tool"]
    tool = pad["features"][..., lo:hi].argmax(-1)
    review = pad["features"][..., vocab["action_columns"].index("arg:monitor.review_after_s")]
    monitor = np.where(mask & (tool == vocab["tools"].index("monitor")), review, np.inf).argmin(1)
    voi = pad[label_key("voi", labels)].astype(np.float64)
    voi_valid = observe & np.isfinite(voi)
    return {
        "labels": labels, "cross_oracle": cross, "obs_min": obs_min, "act_min": act_min,
        "n": n, "mask": mask, "J": J, "p_trip": p, "observe": observe, "best": best, "pmin": pmin,
        "stakes": pmax > pmin, "has_both": has_both, "gap": gap, "true_gather": has_both & (gap < 0),
        "material": has_both & (np.abs(np.nan_to_num(gap)) >= material_gap), "material_gap": material_gap,
        "monitor": monitor, "voi": voi, "voi_valid": voi_valid, "voi_n": voi_valid.sum(1),
        "voi_pos_n": (voi_valid & (voi > 0)).sum(1), "voi_neg_n": (voi_valid & ~(voi > 0)).sum(1),
        "brier_n": mask.sum(1), "cluster": split["dec"]["episode_seed"], "kind": split["dec"]["kind"],
        "playbooks": {m: split["dec"].get(f"baseline_{m}") for m in ("act_first", "investigate_first")},
    }


def rows_for_choice(ctx, chosen):
    r = np.arange(ctx["n"]) if chosen.ndim == 1 else np.arange(ctx["n"])[:, None]
    col = (lambda x: x) if chosen.ndim == 1 else (lambda x: x[:, None])
    gather = ctx["observe"][r, chosen]
    J = ctx["J"][r, chosen]
    fam_best = np.where(gather, col(ctx["obs_min"]), col(ctx["act_min"]))
    return {"regret": J - col(ctx["best"]), "trip": ctx["p_trip"][r, chosen],
            "gather": gather.astype(float), "gva_correct": (gather == col(ctx["true_gather"])).astype(float),
            "regret_family": fam_best - col(ctx["best"]), "regret_within": J - fam_best}


def own_ranking_rows(ctx, Jp):
    mask, obs, J, hb, best = ctx["mask"], ctx["observe"], ctx["J"], ctx["has_both"], ctx["best"]
    act = mask & ~obs
    r = np.arange(ctx["n"])[None, :]
    pick = lambda allowed: np.where(allowed[None], Jp, np.inf).argmin(-1)
    fam = np.where(hb[:, None], np.where(ctx["true_gather"][:, None], obs, act), mask)
    fa = pick(act)
    return {"forced_act": np.where(hb[None], J[r, fa] - ctx["act_min"][None], np.nan).T,
            "cf_labelled_family": (J[r, pick(fam)] - best[None]).T,
            "cf_always_act": (J[r, pick(np.where(hb[:, None], act, mask))] - best[None]).T}


def rows_for_predictions(ctx, J_pred, trip=None):
    mask = ctx["mask"]
    Jp = np.where(mask[None], J_pred, np.inf)
    rows = rows_for_choice(ctx, Jp.argmin(-1).T)
    rows.update(own_ranking_rows(ctx, Jp))
    voi = derived_voi(torch.as_tensor(Jp), torch.as_tensor(ctx["observe"])[None], torch.as_tensor(mask)[None]).numpy()
    pos = ctx["voi"] > 0
    hit = ((voi > 0) == pos[None]) & ctx["voi_valid"][None]
    rows.update(voi_correct=hit.sum(-1).T, voi_pos_hit=(hit & pos[None]).sum(-1).T,
                voi_neg_hit=(hit & ~pos[None]).sum(-1).T)
    if trip is not None:
        rows["brier_se"] = np.where(mask[None], (trip - ctx["p_trip"][None]) ** 2, 0.0).sum(-1).T
    return rows


def baseline_methods(ctx, log=print):
    mask, J, p, obs, tg = ctx["mask"], ctx["J"], ctx["p_trip"], ctx["observe"], ctx["true_gather"]
    k = mask.sum(1)
    share = obs.sum(1) / k
    out = {"random": {"regret": np.where(mask, J - ctx["best"][:, None], 0).sum(1) / k, "trip": p.sum(1) / k,
                      "gather": share, "gva_correct": np.where(tg, share, 1 - share)},
           "monitor": rows_for_choice(ctx, ctx["monitor"])}
    for mode, idx in ctx["playbooks"].items():
        if idx is None or (idx < 0).any():
            log(f"playbook {mode}: coverage {0 if idx is None else (idx >= 0).mean():.3f} < 1, not scored")
            continue
        out[f"playbook_{mode}"] = rows_for_choice(ctx, idx)
    out["oracle"] = rows_for_choice(ctx, J.argmin(1))
    if ctx.get("cross_oracle") is not None:
        name, idx = ctx["cross_oracle"]
        out[name] = rows_for_choice(ctx, idx)
    out["always_act"] = {"gather": np.zeros(ctx["n"]), "gva_correct": (~tg).astype(float)}
    out["always_gather"] = {"gather": np.ones(ctx["n"]), "gva_correct": tg.astype(float)}
    return out


def m_mean(x):
    return float(np.mean(x)) if x.size else np.nan


def m_mean_where(x, where):
    x = x[where.astype(bool)]
    return float(np.mean(x)) if x.size else np.nan


def m_ratio(num, den):
    d = den.sum(0)
    return float(np.mean(num.sum(0) / d)) if np.all(d > 0) else np.nan


def m_balanced(pos_hit, pos_n, neg_hit, neg_n):
    return 0.5 * (m_ratio(pos_hit, pos_n) + m_ratio(neg_hit, neg_n))

METRICS = {
    "regret": (m_mean, ("regret",)),
    "regret_family": (m_mean, ("regret_family",)),
    "regret_within": (m_mean, ("regret_within",)),
    "forced_act_regret": (m_mean_where, ("forced_act", "has_both")),
    "cf_labelled_family_regret": (m_mean, ("cf_labelled_family",)),
    "cf_always_act_regret": (m_mean, ("cf_always_act",)),
    "regret_high_stakes": (m_mean_where, ("regret", "stakes")),
    "trip_rate": (m_mean, ("trip",)),
    "trip_rate_high_stakes": (m_mean_where, ("trip", "stakes")),
    "gather_rate": (m_mean_where, ("gather", "has_both")),
    "gva_accuracy": (m_mean_where, ("gva_correct", "has_both")),
    "gva_accuracy_material": (m_mean_where, ("gva_correct", "material")),
    "voi_sign_accuracy": (m_ratio, ("voi_correct", "voi_n")),
    "voi_balanced_accuracy": (m_balanced, ("voi_pos_hit", "voi_pos_n", "voi_neg_hit", "voi_neg_n")),
    "trip_brier": (m_ratio, ("brier_se", "brier_n")),
}


def _arrays(ctx, rows, keys):
    if keys[0] not in rows or any(k not in rows and k not in ctx for k in keys):
        return None
    return [rows[k] if k in rows else ctx[k] for k in keys]


def score(ctx, rows, n_boot, seed=0):
    out = {}
    for name, (fn, keys) in METRICS.items():
        arrays = _arrays(ctx, rows, keys)
        if arrays is None:
            continue
        ci = cluster_bootstrap_ci(fn, arrays, ctx["cluster"], n_boot=n_boot, seed=seed)
        if arrays[0].ndim == 2:
            per_seed = [fn(*(a[:, s] if a.ndim == 2 else a for a in arrays)) for s in range(arrays[0].shape[1])]
            ci.update(per_seed=per_seed, seed_std=float(np.std(per_seed)),
                      seed_std_ddof1=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan)
        out[name] = ci
    return out


def paired(ctx, rows_a, rows_b, n_boot, seed=0):
    out = {}
    for name, (fn, keys) in METRICS.items():
        a, b = _arrays(ctx, rows_a, keys), _arrays(ctx, rows_b, keys)
        if a is not None and b is not None:
            out[name] = paired_cluster_bootstrap_diff(fn, a, b, ctx["cluster"], n_boot=n_boot, seed=seed)
    return out


def gva_by_kind(ctx, methods):
    out = {}
    for k, kind in enumerate(KINDS):
        sel = ctx["has_both"] & (ctx["kind"] == k)
        mat = ctx["material"] & (ctx["kind"] == k)
        row = {"n": int(sel.sum()), "n_material": int(mat.sum()),
               "true_gather_share": m_mean_where(ctx["true_gather"].astype(float), sel)}
        for name, rows in methods.items():
            if "gva_correct" in rows:
                row[name] = m_mean_where(rows["gva_correct"], sel)
                row[f"{name}@material"] = m_mean_where(rows["gva_correct"], mat)
        out[kind] = row
    return out


def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() >= 3 else np.nan


def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores, np.float64)
    ok = np.isfinite(scores)
    labels, scores = np.asarray(labels, bool)[ok], scores[ok]
    return float(roc_auc_score(labels, scores)) if 0 < labels.sum() < labels.size else np.nan


def _corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.size >= 3 and np.std(x) > 0 and np.std(y) > 0 else np.nan


def act_rank_corr(ctx, Jp, sel):
    act = ctx["mask"] & ~ctx["observe"]
    keep = sel & (act.sum(1) >= 3)
    a = act[keep]
    Jl, Jq = np.where(a, ctx["J"][keep], np.nan), np.where(a, Jp[keep], np.nan)
    with np.errstate(invalid="ignore"):
        Jl, Jq = Jl - np.nanmean(Jl, 1, keepdims=True), Jq - np.nanmean(Jq, 1, keepdims=True)
    ok = np.isfinite(Jl) & np.isfinite(Jq)
    return _corr(Jl[ok], Jq[ok])


def centred_fit(ctx, Jp):
    mask = ctx["mask"]
    Jl, Jq = np.where(mask, ctx["J"], np.nan), np.where(mask, Jp, np.nan)
    Jl, Jq = Jl - np.nanmean(Jl, 1, keepdims=True), Jq - np.nanmean(Jq, 1, keepdims=True)
    ok = mask & np.isfinite(Jl) & np.isfinite(Jq)
    x, y = Jl[ok], Jq[ok]
    return (float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan), _corr(x, y)


def gva_diagnostics(ctx, J_pred, thresholds=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0), tail_gap=10.0):
    mask, obs, J, hb = ctx["mask"], ctx["observe"], ctx["J"], ctx["has_both"]
    act = mask & ~obs
    Jf = np.where(mask, J, np.nan)
    tg, gap = ctx["true_gather"], ctx["gap"]
    near = np.abs(np.nan_to_num(gap)) < ctx["material_gap"]
    r = np.arange(ctx["n"])
    seeds = []
    for Jp_raw in J_pred:
        Jp = np.where(mask, Jp_raw, np.inf)
        pmin_obs, pmin_act = np.where(obs, Jp, np.inf).min(1), np.where(act, Jp, np.inf).min(1)
        tmin_obs, tmin_act = np.where(obs, J, np.inf).min(1), np.where(act, J, np.inf).min(1)
        with np.errstate(invalid="ignore"):
            gp, bias_obs, bias_act = pmin_obs - pmin_act, pmin_obs - tmin_obs, pmin_act - tmin_act
        pg = hb & (gp < 0)
        err = hb & (pg != tg)
        chosen = Jp.argmin(1)
        fam_best = np.where(obs[r, chosen], tmin_obs, tmin_act)
        res = {
            "pred_gather_share": m_mean_where(pg.astype(float), hb),
            "true_gather_share": m_mean_where(tg.astype(float), hb),
            "false_gather": float((err & pg).sum()), "false_act": float((err & ~pg).sum()),
            "true_gather_hit": float((hb & pg & tg).sum()), "true_act_hit": float((hb & ~pg & ~tg).sum()),
            "errors_near_tie_share": m_mean_where(near.astype(float), err),
            "bias_observe": float(np.nanmean((Jp_raw - Jf)[obs])), "bias_act": float(np.nanmean((Jp_raw - Jf)[act])),
            "mae_observe": float(np.nanmean(np.abs(Jp_raw - Jf)[obs])), "mae_act": float(np.nanmean(np.abs(Jp_raw - Jf)[act])),
            "bias_best_observe": m_mean_where(bias_obs, hb), "bias_best_act": m_mean_where(bias_act, hb),
            "bias_gap": m_mean_where(gp - gap, hb),
            "gap_corr": float(np.corrcoef(gp[hb], gap[hb])[0, 1]),
            "gap_sign_corr_material": m_mean_where((np.sign(gp) == np.sign(gap)).astype(float), ctx["material"]),
            "regret_family": m_mean(fam_best - ctx["best"]),
            "regret_within_family": m_mean(J[r, chosen] - fam_best),
            "bias_gap_zero_predictor": m_mean_where(-np.nan_to_num(gap), hb),
            "gap_spearman": _spearman(gp[hb], gap[hb]),
            "gap_auc_true_gather": _auc(tg[hb], -gp[hb]),
        }
        tail = hb & (np.nan_to_num(gap) >= tail_gap)
        res["tail_decisions"] = float(tail.sum())
        res["tail_family_regret"] = float((fam_best - ctx["best"])[tail].sum() / ctx["n"])
        res["tail_gather_share"] = m_mean_where(obs[r, chosen].astype(float), tail)
        res["tail_auc_gap"] = _auc(tail[hb], gp[hb])
        res["act_corr_high_stakes"] = act_rank_corr(ctx, Jp, hb & ctx["stakes"])
        res["centred_slope"], res["centred_corr"] = centred_fit(ctx, Jp)
        regret_at, correct_at = [], []
        for d in thresholds:
            gather = hb & (pmin_act - pmin_obs > d)
            allowed = np.where(gather[:, None], obs, np.where(hb[:, None], act, mask))
            c = np.where(allowed, Jp, np.inf).argmin(1)
            regret_at.append(J[r, c] - ctx["best"])
            correct_at.append((gather == tg).astype(float))
            res[f"margin_{d:g}_regret"] = m_mean(regret_at[-1])
            res[f"margin_{d:g}_gva_accuracy"] = m_mean_where(correct_at[-1], hb)
        fold = np.unique(ctx["cluster"], return_inverse=True)[1].ravel() % 2
        regret_cf, correct_cf = np.empty(ctx["n"]), np.empty(ctx["n"])
        for f in (0, 1):
            pick = int(np.argmin([m.mean() for m in (x[fold == f] for x in regret_at)]))
            regret_cf[fold != f], correct_cf[fold != f] = regret_at[pick][fold != f], correct_at[pick][fold != f]
        res["margin_crossfit_regret"] = m_mean(regret_cf)
        res["margin_crossfit_gva_accuracy"] = m_mean_where(correct_cf, hb)
        seeds.append(res)
    return {k: float(np.mean([s[k] for s in seeds])) for k in seeds[0]}


def report(ctx, predictions, n_boot, log=print):
    methods = {}
    for name, (J_pred, trip) in predictions.items():
        methods[name] = rows_for_predictions(ctx, J_pred, trip)
    methods.update(baseline_methods(ctx, log))
    ref = next(iter(predictions))
    out = {"decisions": int(ctx["n"]), "episodes": int(np.unique(ctx["cluster"]).size),
           "labels": ctx.get("labels"), "high_stakes": int(ctx["stakes"].sum()),
           "high_stakes_episodes": int(np.unique(ctx["cluster"][ctx["stakes"]]).size),
           "trip_rate_min_high_stakes": m_mean_where(ctx["pmin"], ctx["stakes"]),
           "has_both": int(ctx["has_both"].sum()), "material": int(ctx["material"].sum()),
           "material_gap": ctx["material_gap"], "reference": ref,
           "scores": {name: score(ctx, rows, n_boot) for name, rows in methods.items()},
           "paired": {name: paired(ctx, methods[ref], rows, n_boot) for name, rows in methods.items() if name != ref},
           "gva_by_kind": gva_by_kind(ctx, methods),
           "gva_diagnostics": {name: gva_diagnostics(ctx, J_pred) for name, (J_pred, _) in predictions.items()}}
    return out


def fmt(ci, digits=3):
    if ci is None:
        return "-"
    return f"{ci['point']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def markdown(rep, title):
    labels = f" Labels: {rep['labels']}." if rep.get("labels") else ""
    lines = [f"# {title}", "",
             f"{rep['decisions']} decisions from {rep['episodes']} episodes;{labels} high-stakes (the choice changes the trip "
             f"probability) {rep['high_stakes']} from {rep['high_stakes_episodes']} episodes (lowest trip probability "
             f"among their candidates: {rep['trip_rate_min_high_stakes']:.3f} on average); gather-vs-act defined on "
             f"{rep['has_both']} decisions, {rep['material']} with |labelled gap| >= {rep['material_gap']}. "
             "95% cluster-bootstrap CIs over episodes; seeded models: seed mean, seed std in results.json. "
             "Labels are relative to the candidates tried and to the playbook as follow-up policy.", "",
             "| method | " + " | ".join(TABLE) + " |", "|---" * (len(TABLE) + 1) + "|"]
    for name, sc in rep["scores"].items():
        lines.append(f"| {name} | " + " | ".join(fmt(sc.get(m)) for m in TABLE) + " |")
    lines += ["", f"## Paired differences, {rep['reference']} minus method (bold: 95% CI excludes 0)", "",
              "| method | " + " | ".join(TABLE) + " |", "|---" * (len(TABLE) + 1) + "|"]
    for name, diffs in rep["paired"].items():
        cells = []
        for m in TABLE:
            d = diffs.get(m)
            cells.append("-" if d is None else (f"**{fmt(d)}**" if d["lo"] > 0 or d["hi"] < 0 else fmt(d)))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    kinds = rep["gva_by_kind"]
    names = [k for k in next(iter(kinds.values())) if k not in ("n", "n_material", "true_gather_share")
             and not k.endswith("@material")]
    lines += ["", "## Gather-vs-act accuracy by incident kind (all / material gap)", "",
              "| kind | n | n material | true gather share | " + " | ".join(names) + " |",
              "|---" * (len(names) + 4) + "|"]
    for kind, row in kinds.items():
        lines.append(f"| {kind} | {row['n']} | {row['n_material']} | {row['true_gather_share']:.3f} | "
                     + " | ".join(f"{row[k]:.3f} / {row[k + '@material']:.3f}" for k in names) + " |")
    lines += ["", "## Gather-vs-act diagnostics (seed mean)", ""]
    for name, diag in rep["gva_diagnostics"].items():
        lines.append(f"- {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in diag.items()))
    return "\n".join(lines) + "\n"

SUMMARY = ("regret", "regret_family", "regret_within", "cf_labelled_family_regret", "regret_high_stakes",
           "trip_rate_high_stakes", "gva_accuracy")
TRAIN_DIAG_KEYS = ("pred_gather_share", "true_gather_share", "regret_family", "regret_within_family", "gap_spearman",
                   "gap_auc_true_gather", "act_corr_high_stakes", "centred_slope")


def summary_markdown(views, title):
    lines = [f"# {title}", "", "Per view: score [95% cluster-bootstrap CI]; diff = reference minus method.", ""]
    for key, rep in views.items():
        lines += [f"## {key}: {rep['decisions']} decisions, {rep['episodes']} episodes, {rep['high_stakes']} high-stakes, "
                  f"{rep['has_both']} with both families ({rep['material']} material)", "",
                  "| method | " + " | ".join(SUMMARY) + " | diff regret | diff gva_accuracy |",
                  "|---" * (len(SUMMARY) + 3) + "|"]
        for name, sc in rep["scores"].items():
            d = rep["paired"].get(name, {})
            diff = [("-" if d.get(m) is None else (f"**{fmt(d[m])}**" if d[m]["lo"] > 0 or d[m]["hi"] < 0
                                                   else fmt(d[m]))) for m in ("regret", "gva_accuracy")]
            lines.append(f"| {name} | " + " | ".join(fmt(sc.get(m)) for m in SUMMARY) + " | " + " | ".join(diff) + " |")
        lines.append("")
    return "\n".join(lines)


def evaluate(data, predictions, vocab, cfg, run_dir, name, log):
    results = {}
    for s, preds in predictions.items():
        label_sets = ("belief", "hindsight") if has_belief(data[s]) else ("hindsight",)
        results[s] = {}
        for labels in label_sets:
            for sub, sel in subsets(data[s]):
                ctx = context(take(data[s], sel), vocab, cfg["eval"]["material_gap"], labels)
                view = {m: (J[:, sel], None if trip is None else trip[:, sel]) for m, (J, trip) in preds.items()}
                rep = report(ctx, view, cfg["eval"]["n_boot"], log)
                results[s][f"{labels}/{sub}"] = rep
                (run_dir / f"results_{s}_{labels}_{sub}.md").write_text(markdown(rep, f"{name}: {s}, {labels} labels, "
                                                                                      f"{sub}"), encoding="utf-8")
        md = summary_markdown(results[s], f"{name}: {s}")
        (run_dir / f"summary_{s}.md").write_text(md, encoding="utf-8")
        log(md)
    return results


def trip_trained(cfg):
    return DirectConfig(**cfg["model"]).loss_weights()["trip"] > 0


def load_predictions(run_dir, split):
    run_dir = Path(run_dir)
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    p = np.load(run_dir / f"predictions_{split}.npz")
    cfg = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["config"]
    return (p["J_pred"], p["trip"] if trip_trained(cfg) else None), p["episode_seed"], cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="model/direct.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--no-flat", action="store_true", help="skip the flat gradient-boosting baseline")
    parser.add_argument("--flat-labels", nargs="*", default=None, choices=LABEL_SETS,
                        help="label sets the flat baselines are fitted on (default: the training labels); with more "
                             "than one, each method name gets the label set as a suffix")
    parser.add_argument("--compare", nargs="*", default=None, help="run dirs whose saved predictions to compare")
    parser.add_argument("--labels", nargs="*", default=None, help="method names for --compare runs")
    parser.add_argument("--ensemble", action="store_true",
                        help="also score each seeded model's ensemble (mean prediction over its seeds) as "
                             "'<name>_ensemble'")
    parser.add_argument("--diagnose-train", action="store_true",
                        help="also predict the train split with each trained model and store in-sample gather-vs-act "
                             "diagnostics (results.json 'train_diagnostics') and predictions_train.npz")
    parser.add_argument("--reference", default=None,
                        help="method whose paired differences against every other method are reported (default: "
                             "the first model)")
    parser.add_argument("--name", default="direct")
    args = parser.parse_args()
    cfg = run_config(args.config, args.set)
    splits = ["train", "val"] + (["test"] if args.eval_test else [])
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.perf_counter()
    predictions = {s: {} for s in splits if s != "train"}
    extra = {}
    if args.compare:
        labels = args.labels or [Path(r).name.split("-", 2)[-1] for r in args.compare]
        cfgs = []
        for label, r in zip(labels, args.compare):
            for s in predictions:
                pred, episodes, rcfg = load_predictions(r, s)
                predictions[s][label] = pred
            cfgs.append(rcfg)
        cfg = {**cfg, "data": cfgs[0]["data"], "compare": dict(zip(labels, args.compare))}
        if any(c["data"] != cfg["data"] for c in cfgs):
            raise ValueError("compared runs use different data")
    meta = json.loads((Path(cfg["data"]) / "meta.json").read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    data = {s: load_split(cfg["data"], s) for s in splits}
    train_labels = cfg.setdefault("labels", "hindsight")
    fraction = float(cfg.get("train_fraction", 1.0))
    if fraction < 1.0:
        if args.compare:
            raise ValueError("train_fraction applies to training runs, not to --compare")
        data["train"] = subsample_episodes(data["train"], fraction, int(cfg.get("train_fraction_seed", 0)))
        log(f"train_fraction {fraction}: {len(data['train']['episodes_kept'])} train episodes, "
            f"{data['train']['n']} decisions")
    desc = aux_desc = None
    if not args.compare and cfg["model"].get("summary_descriptor"):
        desc = add_summary_descriptor(data, vocab, log)
    if not args.compare and cfg["model"].get("summary_aux"):
        aux_desc = add_summary_targets(data, vocab, log)
        cfg["model"]["summary_aux_dim"] = int(aux_desc.width)
    fit_data = {s: label_view(d, train_labels) for s, d in data.items()}
    run = start_run("model_direct", cfg, seed=None, name=args.name)
    if desc is not None:
        desc.save(run.dir / "summary_descriptor.npz")
    if aux_desc is not None:
        aux_desc.save(run.dir / "summary_targets.npz")
    log(f"data {cfg['data']}: " + ", ".join(f"{s} {data[s]['n']} decisions" for s in splits)
        + f"; training labels: {train_labels}")
    if args.compare:
        for r in args.compare:
            for s in predictions:
                if not np.array_equal(load_predictions(r, s)[1], data[s]["dec"]["episode_seed"]):
                    raise ValueError(f"saved {s} predictions of {r} do not match the data")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["model"]["backbone"] = {**cfg["model"]["backbone"],
                                    "payload_dim": tuple(int(x) for x in vocab["payload_dim"]),
                                    "n_kinds": len(vocab["kinds"]),
                                    "descriptor_dim": int(data["train"]["pad"]["features"].shape[-1])}
        run.save_json("model_config.json", cfg["model"])
        dev = {"device": device, **{s: to_device(d, device) for s, d in fit_data.items()}}
        per_seed = {s: [] for s in [*predictions, *(["train"] if args.diagnose_train else [])]}
        train_info = []
        for seed in cfg["seeds"]:
            model, info = train_one(cfg, dev, seed, log)
            train_info.append(info)
            for s in per_seed:
                per_seed[s].append(predict(model, dev[s]))
            torch.save(model.state_dict(), run.dir / f"model_s{seed}.pt")
        run.save_json("ensemble.json", {"members": [f"model_s{seed}.pt" for seed in cfg["seeds"]],
                                        "model_config": "model_config.json",
                                        "summary_descriptor": "summary_descriptor.npz" if desc is not None else None,
                                        "rule": "mean predicted J over members (EnsembleDecisionModel)"})
        for s in per_seed:
            J_pred, trip = (np.stack([p[i] for p in per_seed[s]]) for i in (0, 1))
            np.savez_compressed(run.dir / f"predictions_{s}.npz", J_pred=J_pred, trip=trip,
                                seeds=np.asarray(cfg["seeds"]), episode_seed=data[s]["dec"]["episode_seed"])
            if s in predictions:
                predictions[s][args.name] = (J_pred, trip if trip_trained(cfg) else None)
        if args.diagnose_train:
            J_train = np.stack([p[0] for p in per_seed["train"]])
            extra["train_diagnostics"] = {
                labels: gva_diagnostics(context(data["train"], vocab, cfg["eval"]["material_gap"], labels), J_train)
                for labels in (("belief", "hindsight") if has_belief(data["train"]) else ("hindsight",))}
            log("train (in-sample) diagnostics: " + json.dumps(
                {k: {m: round(v, 3) for m, v in d.items() if m in TRAIN_DIAG_KEYS}
                 for k, d in extra["train_diagnostics"].items()}))
        extra["train"] = train_info
        cf = {rule: float(np.mean([i["crossfit_selection"][rule]["regret"] for i in train_info]))
              for rule in ("J_centered", "regret")}
        log(f"cross-fitted epoch selection (seed mean of the out-of-half val regret; training labels): {cf}")
    if not args.no_flat:
        flat_labels = args.flat_labels or [train_labels]
        extra["flat"] = {}
        for labels in flat_labels:
            suffix = f"_{labels}" if len(flat_labels) > 1 else ""
            flat, table, params = flat_predictions({s: label_view(d, labels) for s, d in data.items()}, vocab, log)
            for name, per_split in flat.items():
                for s in predictions:
                    predictions[s][name + suffix] = per_split[s]
            extra["flat"][labels] = {"grid": table, "chosen": params}
    if args.ensemble:
        for s, preds in predictions.items():
            out = {}
            for name, (J_pred, trip) in preds.items():
                out[name] = (J_pred, trip)
                if J_pred.shape[0] > 1:
                    out[f"{name}_ensemble"] = (J_pred.mean(0, keepdims=True),
                                               None if trip is None else trip.mean(0, keepdims=True))
            predictions[s] = out
    if args.reference:
        for s, preds in predictions.items():
            if args.reference not in preds:
                raise ValueError(f"--reference {args.reference!r} is not one of {list(preds)}")
            predictions[s] = {args.reference: preds[args.reference],
                              **{k: v for k, v in preds.items() if k != args.reference}}
    results = evaluate(data, predictions, vocab, cfg, run.dir, args.name, log)
    run.save_json("results.json", {"results": results, **extra, "log": lines, "wall_s": time.perf_counter() - t0,
                                   "eval_test": args.eval_test})
    run.finish({s: {view: {name: {m: v["point"] for m, v in sc.items()} for name, sc in rep["scores"].items()}
                    for view, rep in views.items()} for s, views in results.items()})

if __name__ == "__main__":
    main()
