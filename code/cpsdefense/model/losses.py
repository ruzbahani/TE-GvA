import torch
import torch.nn.functional as F

from cpsdefense.model.heads import ordinal_log_probs

CENSOR_RULES = ("start", "half")


def masked_mean(values, keep):
    total = torch.where(keep, values, torch.zeros_like(values)).sum()
    return total / keep.sum().clamp(min=1).to(values.dtype)


def _keep(missing, mask):
    keep = ~missing
    return keep if mask is None else keep & torch.as_tensor(mask, dtype=torch.bool, device=keep.device)


def bce_loss(logits, target, mask=None):
    target = torch.as_tensor(target, dtype=logits.dtype, device=logits.device)
    keep = _keep(torch.isnan(target), mask)
    target = torch.where(keep, target, torch.zeros_like(target))
    return masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), keep)


def ce_loss(logits, target, mask=None):
    target = torch.as_tensor(target, dtype=torch.int64, device=logits.device)
    keep = _keep(target < 0, mask)
    target = torch.where(keep, target, torch.zeros_like(target))
    return masked_mean(F.cross_entropy(logits, target, reduction="none"), keep)


def ordinal_nll(eta, thresholds, target, gaps=None, mask=None):
    target = torch.as_tensor(target, dtype=torch.int64, device=eta.device)
    keep = _keep(target < 0, mask)
    target = torch.where(keep, target, torch.zeros_like(target))
    log_probs = ordinal_log_probs(eta, thresholds, gaps)
    return masked_mean(-log_probs.gather(-1, target[..., None]).squeeze(-1), keep)


def hazard_nll(logits, edges, time, event, censor_rule="start", mask=None):
    if censor_rule not in CENSOR_RULES:
        raise ValueError(f"censor_rule must be one of {CENSOR_RULES}")
    device = logits.device
    edges = torch.as_tensor(edges, dtype=torch.float64, device=device)
    n_bins = logits.shape[-1]
    if edges.shape != (n_bins + 1,):
        raise ValueError(f"edges must have {n_bins + 1} values for {n_bins} bins")
    time = torch.as_tensor(time, dtype=torch.float64, device=device)
    event = torch.as_tensor(event, device=device).to(torch.bool)
    keep = _keep(torch.isnan(time), mask)
    if bool((keep & (time < edges[0])).any()):
        raise ValueError("a time lies before the first bin edge")
    time = torch.where(keep, time, edges[0]).contiguous()
    k = (torch.searchsorted(edges, time, right=False) - 1).clamp(0, n_bins)
    inside = k < n_bins
    log_surv = torch.cat([torch.zeros_like(logits[..., :1]), F.logsigmoid(-logits).cumsum(-1)], -1)
    k_bin = k.clamp(max=n_bins - 1)
    log_event = (log_surv.gather(-1, k_bin[:, None]).squeeze(-1)
                 + F.logsigmoid(logits).gather(-1, k_bin[:, None]).squeeze(-1))
    credited = torch.searchsorted(edges[1:].contiguous(), time, right=True)
    if censor_rule == "half":
        lo, hi = edges[k_bin], edges[k_bin + 1]
        partial = inside & (credited == k) & torch.isfinite(hi) & (time >= (lo + hi) / 2)
        credited = credited + partial.to(credited.dtype)
    log_censored = log_surv.gather(-1, credited[:, None]).squeeze(-1)
    ll = torch.where(event & inside, log_event, torch.where(event, log_surv[:, -1], log_censored))
    return masked_mean(-ll, keep)


def question_loss(output, target, censor_rule="start"):
    kind = output["type"]
    if kind == "noul":
        return bce_loss(output["logit"], target)
    if kind == "choice":
        return ce_loss(output["logits"], target)
    if kind == "score":
        return ordinal_nll(output["eta"], output["thresholds"], target, gaps=output["gaps"])
    if kind == "hazard":
        return hazard_nll(output["logits"], output["edges"], target["time"], target["event"], censor_rule)
    raise ValueError(f"unknown output type {kind!r}")


def total_loss(outputs, targets, weights=None, censor_rule="start"):
    weights = weights or {}
    parts = {name: question_loss(outputs[name], target, censor_rule) for name, target in targets.items()}
    if not parts:
        raise ValueError("no targets")
    loss = sum(weights.get(name, 1.0) * value for name, value in parts.items())
    return loss, {name: float(value.detach()) for name, value in parts.items()}
