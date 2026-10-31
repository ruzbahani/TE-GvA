import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.defense.registry import arg_grid, load_registry
from cpsdefense.experience import belief
from cpsdefense.experience.dataset import (
    BELIEF_DECISION_KEYS, BELIEF_KEYS, CENSORED, KINDS, SPLITS, TOOL_RESULT_KINDS, backup_available, baseline_index,
    belief_labels, candidate_labels, in_family, later_action_ids, own_action_receipts, own_action_tool_results,
    split_of, submitted_action_ids)
from cpsdefense.paths import PROCESSED_DIR, ROOT, RUNS_DIR
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_file
from cpsdefense.utils.runs import start_run

EPISODES = PROCESSED_DIR / "tep_experience" / "20260927-003712-v2-natural"
ARRAYS = PROCESSED_DIR / "tep_decisions" / "v2-natural"
DEV_SPLITS = ("train", "val")
TOL = 1e-9
EPS = 0.01
COMPONENTS = ("attack", "service", "physical", "compute")
OUTCOME_KEYS = ("total", "shutdown", "time_to_shutdown", "resolved", "mitigated")
REFUSED = ("rejected", "timed_out")
KERNEL = ROOT / "third_party" / "tep-studio" / "temexd_mod" / "temexd_mod.c"
LICENCE = ROOT / "third_party" / "tep-studio" / "LICENSE"
PROVENANCE_FILES = (
    "configs/sim/tep.yaml", "configs/sim/incidents.yaml", "configs/sim/incidents_stress.yaml",
    "configs/sim/telemetry.yaml", "configs/sim/costs.yaml", "configs/sim/tep_env.yaml", "configs/defense/tools.yaml",
    "configs/defense/policy.yaml", "configs/experience/generate.yaml", "configs/experience/generate_stress.yaml",
    "configs/experience/tensors.yaml", "scripts/generate_experience.py", "scripts/build_dataset.py",
    "cpsdefense/experience/rollout.py", "cpsdefense/experience/dataset.py", "cpsdefense/experience/tensors.py",
    "cpsdefense/experience/belief.py",
    "cpsdefense/agents/playbook.py", "cpsdefense/sim/tep_env.py", "cpsdefense/sim/tep_plant.py",
    "cpsdefense/sim/tep_world.py", "cpsdefense/sim/tep_telemetry.py", "cpsdefense/sim/incidents.py",
    "cpsdefense/defense/registry.py", "cpsdefense/defense/executor.py", "cpsdefense/defense/policy.py",
    "cpsdefense/defense/evidence.py",
)
CHECKED_FIELDS = ("y_J", "y_J_std", "y_p_trip", "y_voi", "y_p_resolved", "y_p_success", "y_duration",
                  "y_first_censored", "y_duration_time", "y_duration_censored", "dec_tau", "dec_episode_seed",
                  "dec_kind", "dec_explored", "dec_n_candidates", "dec_baseline_act_first",
                  "dec_baseline_investigate_first", "dec_backup_available",
                  *(f"y_{k}" for k in BELIEF_KEYS), *(f"dec_{k}" for k in BELIEF_DECISION_KEYS))
CALIBRATION_BINS = 10


def summary(values, scale=1.0):
    x = np.array([np.nan if v is None else v for v in np.ravel(np.asarray(values, dtype=object))], np.float64)
    x = x[np.isfinite(x)] * scale
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4], "max": float(x.max())}


def counts(values):
    c = collections.Counter(str(v) for v in values)
    n = sum(c.values())
    return {"n": n, "counts": dict(c.most_common()), "shares": {k: v / n for k, v in c.most_common()}}


def share(mask):
    mask = np.asarray(mask, bool)
    return float(mask.mean()) if mask.size else None


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def by_group(groups, mask, values=None):
    groups, mask = np.asarray(groups), np.asarray(mask, bool)
    out = {}
    for g in sorted(set(groups.tolist())):
        sel = groups == g
        if values is None:
            out[g] = {"n": int(sel.sum()), "share": share(mask[sel])}
        else:
            v = np.asarray(values, np.float64)[sel & mask]
            out[g] = {"n": int((sel & mask).sum()), "mean": float(v.mean()) if v.size else None}
    return out


def credit(names_per_row):
    c = collections.Counter()
    for names in names_per_row:
        for n in names:
            c[n] += 1.0 / len(names)
    total = len(names_per_row)
    return {k: v / total for k, v in c.most_common()} if total else {}


def episode_files(root):
    return {int(p.stem[3:]): p for p in sorted(Path(root).glob("ep_*.pkl"))}


def load_episodes(root, splits=DEV_SPLITS, fractions=(0.6, 0.2, 0.2)):
    if "test" in splits:
        raise ValueError("the held-out test split is not read by the dataset statistics")
    out = {s: [] for s in splits}
    for seed, path in episode_files(root).items():
        s = split_of(seed, fractions)
        if s in out:
            with open(path, "rb") as handle:
                out[s].append(pickle.load(handle))
    return out


def load_arrays(root, splits=DEV_SPLITS):
    if "test" in splits:
        raise ValueError("the held-out test split is not read by the dataset statistics")
    return {s: dict(np.load(Path(root) / f"{s}.npz", allow_pickle=False)) for s in splits
            if (Path(root) / f"{s}.npz").exists()}


def load_item_ids(root, splits=DEV_SPLITS):
    if "test" in splits:
        raise ValueError("the held-out test split is not read by the dataset statistics")
    return {s: json.loads((Path(root) / f"{s}_item_ids.json").read_text(encoding="utf-8"))["keys"] for s in splits
            if (Path(root) / f"{s}_item_ids.json").exists()}


def result_matrix(decision, key):
    reps = sorted({r["replicate"] for r in decision["results"]})
    out = np.full((len(decision["candidates"]), len(reps)), np.nan)
    for r in decision["results"]:
        v = r[key]
        out[r["candidate"], reps.index(r["replicate"])] = np.nan if v is None else float(v)
    return out


def component_matrices(decision):
    reps = sorted({r["replicate"] for r in decision["results"]})
    out = {k: np.zeros((len(decision["candidates"]), len(reps))) for k in COMPONENTS}
    for r in decision["results"]:
        for k in COMPONENTS:
            out[k][r["candidate"], reps.index(r["replicate"])] = r["cost"][k]
    return out


def pairwise_noise(total):
    i, j = np.triu_indices(total.shape[0], 1)
    diff = total[i] - total[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = total.var(1, ddof=1)
        return {"sd_diff": diff.std(1, ddof=1), "sd_indep": np.sqrt(var[i] + var[j]),
                "identical": np.all(np.abs(diff) <= TOL, 1)}


def incident_class(hidden):
    if not hidden["onset_happened"]:
        return "pre_onset"
    parts = hidden["cause"].split(":")
    if parts[0] == "link_fault":
        return f"link_fault:{parts[1]}"
    if parts[0] == "sensor_fault":
        return f"sensor_fault:{parts[-1]}"
    return parts[0]


def wait_index(candidates, review_after_s=300.0):
    return next((i for i, c in enumerate(candidates) if c["tool_id"] == "monitor"
                 and float(c["args"].get("review_after_s", 0)) == review_after_s), -1)


def outcome_labels(decision):
    return np.stack([component_matrices(decision)[k] for k in COMPONENTS]
                    + [result_matrix(decision, k) for k in OUTCOME_KEYS], -1)


def monitor_pair(decision, labels, total):
    i, j = wait_index(decision["candidates"]), wait_index(decision["candidates"], 900.0)
    if i < 0 or j < 0:
        return None, None
    return (bool(np.all(np.abs(total[i] - total[j]) <= TOL)),
            bool(np.array_equal(np.round(labels[i], 9), np.round(labels[j], 9), equal_nan=True)))


def receipt_items(episode):
    return [i for i in episode["items"] if i["kind"] == "receipt"]


def tool_result_items(episode):
    return [i for i in episode["items"] if i["kind"] in TOOL_RESULT_KINDS]


def without_action_id(items):
    return sum(not isinstance(i["payload"].get("action_id"), str) for i in items)


def backup_group(episode):
    return "available" if backup_available(episode) else "unavailable"


def refusals_at_accept_time(episode):
    accepted = {(i["payload"]["tool_id"], i["subject_id"], i["arrival_time"]) for i in receipt_items(episode)
                if i["payload"]["status"] == "accepted"}
    return sum((i["payload"]["tool_id"], i["subject_id"], i["arrival_time"]) in accepted
               for i in receipt_items(episode) if i["payload"]["status"] in REFUSED)


def unknown_valid(episode, d):
    valid, tau = set(d["valid_evidence"]), d["tau"]
    kinds = [i["kind"] for i in episode["items"] if "arrival_time" in i and i["evidence_id"] not in valid
             and i["arrival_time"] <= tau < i["valid_until"] and tau < i.get("retracted_at", np.inf)]
    return kinds.count("receipt"), len(kinds) - kinds.count("receipt")


def belief_fields(episode, d, lab):
    family = in_family(episode)
    bl = belief_labels(d, family)
    J, Jb, observe = lab["J"], bl["J_belief"], lab["observe"]
    has_obs, has_act = bool(observe.any()), bool((~observe).any())
    best_obs = Jb[observe].min() if has_obs else np.nan
    best_act = Jb[~observe].min() if has_act else np.nan
    best_h, best_b = set(np.flatnonzero(J <= J.min() + TOL)), set(np.flatnonzero(Jb <= Jb.min() + TOL))
    incident = (episode.get("scenario") or {}).get("incident") or {}
    group = belief.family_group(incident) if family else None
    twin = d.get("twin") or {}
    is_link = family and incident["kind"] == belief.LINK
    return {
        "family": family, "family_group": group or "outside_family", "twin_record": bool(twin),
        "twin_ok": bl["twin_ok"], "belief_fallback": bl["belief_fallback"],
        "belief_used": family and not bl["belief_fallback"], "posterior": bl["posterior"],
        "p_link": (bl["posterior"] if is_link else 1.0 - bl["posterior"]) if family else np.nan,
        "is_link": bool(is_link),
        "gather_belief": has_obs and has_act and best_obs < best_act - TOL,
        "best_observe_J_belief": best_obs, "best_act_J_belief": best_act,
        "voi_max_belief": np.nanmax(bl["voi_belief"]) if has_obs and has_act else np.nan,
        "argmin_agree": bool(best_h & best_b),
        "hindsight_choice_belief_regret": float(min(Jb[i] for i in best_h) - Jb.min()),
        "belief_choice_hindsight_regret": float(min(J[i] for i in best_b) - J.min()),
        "deterministic_identical": twin["belief"]["deterministic_identical"] if twin else None,
        "pi_after_memory_differs": twin.get("pi_after_memory_differs") if twin else None,
        "candidates_unmatched": len(twin.get("candidates_unmatched") or ()) if twin else 0,
        "J_abs_diff": np.abs(Jb - J), "J_belief": Jb, "voi_belief": bl["voi_belief"],
    }


def decision_row(episode, d):
    lab = candidate_labels(d)
    cands, h, b = d["candidates"], d["hidden"], d["behaviour"]
    total = result_matrix(d, "total")
    J, p, observe = lab["J"], lab["p_trip"], lab["observe"]
    tools = [c["tool_id"] for c in cands]
    families = [c["family"] for c in cands]
    best = np.flatnonzero(J <= J.min() + TOL)
    order = np.argsort(J, kind="stable")
    rest = order[J[order] > J.min() + TOL]
    gap = se = sd_pair = np.nan
    if rest.size:
        diff = total[rest[0]] - total[best[0]]
        gap, sd_pair = J[rest[0]] - J[best[0]], diff.std(ddof=1)
        se = sd_pair / np.sqrt(diff.size)
    statuses = [[None] * total.shape[1] for _ in cands]
    reps = sorted({r["replicate"] for r in d["results"]})
    for r in d["results"]:
        statuses[r["candidate"]][reps.index(r["replicate"])] = r["first_status"]
    rejected = np.array([all(s == "rejected" for s in row) for row in statuses])
    censored = np.array([[s in CENSORED for s in row] for row in statuses])
    ids = [c["action_id"] for c in cands]
    wait = wait_index(cands)
    first = b["actions"][0] if b["actions"] else None
    b_idx = wait if first is None else ids.index(first) if first in ids else -1
    b_prob = b["probabilities"][0] if b["probabilities"] else b.get("empty_plan_probability", np.nan)
    stored = {i.get("evidence_id") for i in episode["items"]}
    valid = set(d["valid_evidence"])
    unknown_receipts, unknown_other = unknown_valid(episode, d)
    own_receipts, _ = own_action_receipts(d, episode["items"])
    own_results, _ = own_action_tool_results(d, episode["items"])
    same_J, same_labels = monitor_pair(d, outcome_labels(d), total)
    base = {m: baseline_index(d, m) for m in ("act_first", "investigate_first")}
    has_obs, has_act = bool(observe.any()), bool((~observe).any())
    best_obs = J[observe].min() if has_obs else np.nan
    best_act = J[~observe].min() if has_act else np.nan
    comps = component_matrices(d)
    means = {k: v.mean(1) for k, v in comps.items()}
    outcome = means["service"] + means["physical"] + means["attack"]
    gather_diff = None
    if has_obs and has_act:
        i_obs = np.flatnonzero(observe)[np.argmin(J[observe])]
        i_act = np.flatnonzero(~observe)[np.argmin(J[~observe])]
        gather_diff = {k: means[k][i_act] - means[k][i_obs] for k in COMPONENTS}
    oracle = h["oracle_time_to_shutdown"]
    bf = belief_fields(episode, d, lab)
    return {
        **{k: v for k, v in bf.items() if k not in ("J_abs_diff", "J_belief", "voi_belief")},
        "seed": episode["seed"], "cause": h["cause"], "kind": h["kind"] if h["onset_happened"] else "pre_onset",
        "incident_class": incident_class(h), "onset_happened": bool(h["onset_happened"]),
        "backup": backup_group(episode),
        "since_onset_s": d["tau"] - h["onset_time"], "tau": d["tau"],
        "oracle_tts_s": np.nan if oracle is None else float(oracle),
        "n_valid_evidence": len(d["valid_evidence"]), "n_cited_evidence": len(d["cited_evidence"]),
        "n_candidates": len(cands), "n_distinct": len({tuple(np.round(row, 9)) for row in total}),
        "n_replicates": total.shape[1], "horizon_s": min(r["horizon_s"] for r in d["results"]),
        "J_min": J.min(), "J_range": J.max() - J.min(), "p_min": p.min(), "p_max": p.max(),
        "best_tools": [tools[i] for i in best], "best_families": [families[i] for i in best],
        "n_best": len(best), "gap": gap, "gap_se": se, "gap_sd": sd_pair,
        "has_observe": has_obs, "has_act": has_act, "best_observe_J": best_obs, "best_act_J": best_act,
        "voi_max": np.nanmax(lab["voi"]) if has_obs and has_act else np.nan,
        "regret": {m: (J[i] - J.min()) if i >= 0 else np.nan for m, i in base.items()},
        "trip_excess": {m: (p[i] - p.min()) if i >= 0 else np.nan for m, i in base.items()},
        "baseline_index": base, "wait_index": wait, "wait_regret": (J[wait] - J.min()) if wait >= 0 else np.nan,
        "random_regret": J.mean() - J.min(),
        "mode": b["mode"], "epsilon": b["epsilon"], "explored": bool(b["explored"]), "plan_length": len(b["actions"]),
        "behaviour_index": b_idx, "behaviour_probability": b_prob,
        "empty_plan_probability": b.get("empty_plan_probability"),
        "behaviour_regret": (J[b_idx] - J.min()) if b_idx >= 0 else np.nan,
        "all_rejected": bool(rejected.all()), "n_rejected_candidates": int(rejected.sum()),
        "monitor_pair_identical": same_J, "monitor_pair_identical_labels": same_labels,
        "first_censored": censored, "valid_evidence_missing": sum(e not in stored for e in d["valid_evidence"]),
        "valid_receipt": any(i["evidence_id"] in valid for i in receipt_items(episode)),
        "unknown_valid_receipts": unknown_receipts, "unknown_valid_other": unknown_other,
        "own_action_receipts": len(own_receipts), "own_action_tool_results": len(own_results),
        "label_identical": int(len(cands) - len({tuple(np.round(row, 9)) for row in total})),
        "signature": (tuple(ids_key(c) for c in cands), tuple(np.round(total, 9).ravel())),
        "action_ids_unique": len(set(ids)) == len(ids),
        "cand": {"tool": tools, "family": families, "J": J, "J_sd": total.std(1, ddof=1) if total.shape[1] > 1
                 else np.full(len(J), np.nan), "p_trip": p, "p_resolved": lab["p_resolved"],
                 "p_success": lab["p_success"], "voi": lab["voi"],
                 "nominal_median": [c["expected_duration"] for c in cands],
                 "nominal_q90": [c["duration_q90"] for c in cands],
                 "J_belief": bf["J_belief"], "voi_belief": bf["voi_belief"], "J_abs_diff": bf["J_abs_diff"],
                 "belief_used": [bf["belief_used"]] * len(cands), "backup": [backup_group(episode)] * len(cands),
                 "family_group": [bf["family_group"]] * len(cands)},
        "results": d["results"], "pairs": pairwise_noise(total),
        "component_variance": {k: float(v.var(1, ddof=1).sum()) if v.shape[1] > 1 else 0.0 for k, v in comps.items()},
        "outcome_range": outcome.max() - outcome.min(), "gather_diff": gather_diff,
    }


def ids_key(c):
    return (c["tool_id"], c["target_id"], tuple(sorted(c["args"].items())))


def episode_row(e):
    sc, inc = e["scenario"], e["scenario"]["incident"]
    bg = tuple(sc["background"])
    return {"seed": e["seed"], "kind": inc["kind"], "target": inc["target"], "fault_kind": inc["fault_kind"],
            "cause": e["outcome"]["hidden"]["cause"], "background": "+".join(f"IDV{k}" for k in bg) or "none",
            "onset_s": inc["onset_s"], "mode": e["outcome"]["behaviour_mode"],
            "tripped": e["outcome"]["shutdown"] is not None, "cost": e["outcome"]["total"],
            "n_decisions": len(e["decisions"]), "n_items": len(e["items"]),
            "item_kinds": [i["kind"] for i in e["items"]], "seconds": e["seconds"],
            "receipt_items": len(receipt_items(e)), "refusals_at_accept_time": refusals_at_accept_time(e),
            "receipts_without_action_id": without_action_id(receipt_items(e)),
            "tool_results_without_action_id": without_action_id(tool_result_items(e)),
            "backup": backup_group(e), "incidents": e.get("incidents")}


def block(episodes):
    eps = [episode_row(e) for e in episodes]
    rows = [decision_row(e, d) for e in episodes for d in e["decisions"]]
    col = {k: np.array([r[k] for r in rows]) for k in (
        "kind", "incident_class", "onset_happened", "since_onset_s", "oracle_tts_s", "n_candidates", "n_distinct",
        "J_range", "p_min", "p_max", "n_best", "gap", "gap_se", "gap_sd", "has_observe", "has_act",
        "best_observe_J", "best_act_J", "voi_max", "wait_index", "wait_regret", "random_regret", "explored",
        "plan_length",
        "behaviour_index", "behaviour_probability", "behaviour_regret", "all_rejected", "horizon_s",
        "n_valid_evidence", "n_cited_evidence", "n_replicates", "label_identical", "outcome_range",
        "valid_evidence_missing", "valid_receipt", "unknown_valid_receipts", "unknown_valid_other", "backup",
        "own_action_receipts", "own_action_tool_results", "family", "family_group", "twin_record", "twin_ok",
        "belief_fallback", "belief_used", "posterior", "p_link", "is_link", "gather_belief", "best_observe_J_belief",
        "best_act_J_belief", "voi_max_belief", "argmin_agree", "hindsight_choice_belief_regret",
        "belief_choice_hindsight_regret", "candidates_unmatched")}
    cand = {k: np.concatenate([np.asarray(r["cand"][k]) for r in rows]) for k in rows[0]["cand"]} if rows else {}
    out = {"episodes": episode_stats(eps), "twins": twin_stats(episodes, rows)}
    if not rows:
        return out
    stakes = col["p_max"] > col["p_min"] + TOL
    unavoidable = col["p_min"] >= 1.0
    both = col["has_observe"] & col["has_act"]
    gather = both & (col["best_observe_J"] < col["best_act_J"] - TOL)
    out["decisions"] = decision_stats(rows, col)
    out["candidates"] = candidate_stats(rows, cand)
    out["labels"] = label_stats(rows, col, cand)
    out["stakes"] = {
        "high_stakes_share": share(stakes), "high_stakes_n": int(stakes.sum()),
        "decisive_share": share((col["p_min"] <= 0) & (col["p_max"] >= 1)),
        "trip_difference_ge_half_share": share(col["p_max"] - col["p_min"] >= 0.5),
        "unavoidable_trip_share": share(unavoidable), "unavoidable_trip_n": int(unavoidable.sum()),
        "trip_free_share": share(col["p_max"] <= 0),
        "trip_uniform_between_share": share((col["p_min"] > 0) & (col["p_max"] < 1) & ~stakes),
        "high_stakes_by_class": by_group(col["incident_class"], stakes),
        "high_stakes_by_backup": by_group(col["backup"], stakes),
        "unavoidable_by_class": by_group(col["incident_class"], unavoidable),
        "loose_definition_share": share((col["p_max"] > 0) & (col["p_min"] < 1)),
        "loose_definition_without_trip_difference": int(((col["p_max"] > 0) & (col["p_min"] < 1) & ~stakes).sum()),
        "J_range_high_stakes": summary(col["J_range"][stakes]), "J_range_other": summary(col["J_range"][~stakes]),
        "tool_cost_only_share": share(col["outcome_range"] <= EPS),
        "outcome_range_per_decision": summary(col["outcome_range"]),
    }
    out["best"] = best_stats(rows, col)
    out["voi"] = {
        "observe_candidates": summary(cand["voi"][cand["family"] == "observe"]),
        "positive_share": share(cand["voi"][cand["family"] == "observe"] > TOL),
        "zero_share": share(np.abs(cand["voi"][cand["family"] == "observe"]) <= TOL),
        "negative_share": share(cand["voi"][cand["family"] == "observe"] < -TOL),
        "by_tool": {t: {**summary(cand["voi"][cand["tool"] == t]),
                        "positive_share": share(cand["voi"][cand["tool"] == t] > TOL)}
                    for t in sorted(set(cand["tool"][cand["family"] == "observe"].tolist()))},
        "max_per_decision": summary(col["voi_max"]),
        "decisions_with_both": int(both.sum()),
        "gather_beats_act_share": share(gather[both]), "gather_beats_act_n": int(gather.sum()),
        "gather_beats_act_by_eps_share": share((col["best_observe_J"] < col["best_act_J"] - EPS)[both]),
        "positive_by_eps_share": share(cand["voi"][cand["family"] == "observe"] > EPS),
        "gather_ties_act_share": share(np.abs(col["best_observe_J"] - col["best_act_J"])[both] <= TOL),
        "gather_beats_act_high_stakes_share": share(gather[both & stakes]),
        "gather_beats_act_by_kind": by_group(col["kind"][both], gather[both]),
        "gather_beats_act_by_class": by_group(col["incident_class"][both], gather[both]),
        "gather_beats_act_by_backup": by_group(col["backup"][both], gather[both]),
        "gather_beats_act_high_stakes_by_backup": by_group(col["backup"][both & stakes], gather[both & stakes]),
        "positive_margin": summary(col["voi_max"][gather]),
        "positive_margin_high_stakes": summary(col["voi_max"][gather & stakes]),
        "gather_beats_act_by_1_share": share((col["best_observe_J"] < col["best_act_J"] - 1.0)[both]),
        "gather_beats_act_tool_cost_only_share": share([
            abs(r["gather_diff"]["service"]) <= EPS and abs(r["gather_diff"]["physical"]) <= EPS
            for r, g in zip(rows, gather) if g]),
        "gather_margin_by_component": {k: summary([r["gather_diff"][k] for r, g in zip(rows, gather) if g])
                                       for k in ("service", "physical", "compute")},
    }
    out["noise"] = noise_stats(rows, col, cand)
    out["behaviour"] = behaviour_stats(eps, rows, col)
    out["playbooks"] = playbook_stats(rows, col, stakes)
    out["quality"] = quality_stats(rows, col, cand)
    out["checks"] = check_stats(eps, rows, col, out)
    out["belief"] = belief_stats(rows, col, cand, gather)
    return out


def calibration_table(p_link, is_link, bins=CALIBRATION_BINS):
    p, y = np.asarray(p_link, np.float64), np.asarray(is_link, np.float64)
    keep = np.isfinite(p)
    p, y = p[keep], y[keep]
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        sel = idx == b
        row = {"bin": [float(edges[b]), float(edges[b + 1])], "n": int(sel.sum()),
               "mean_p_link": float(p[sel].mean()) if sel.any() else None,
               "link_frequency": float(y[sel].mean()) if sel.any() else None}
        if sel.any():
            ece += sel.sum() / p.size * abs(row["mean_p_link"] - row["link_frequency"])
        table.append(row)
    return {"n": int(p.size), "bins": table, "brier": float(np.mean((p - y) ** 2)) if p.size else None,
            "ece": float(ece) if p.size else None}


def belief_stats(rows, col, cand, gather):
    both = col["has_observe"] & col["has_act"]
    used = col["belief_used"]
    fam = col["family"]
    gb = col["gather_belief"].astype(bool)
    obs = cand["family"] == "observe"
    voi_b = cand["voi_belief"].astype(np.float64)
    post = col["onset_happened"]
    sel = fam & np.isfinite(col["p_link"].astype(np.float64))
    identical = np.array([r["deterministic_identical"] for r in rows if r["belief_used"]
                          and r["deterministic_identical"] is not None], bool)
    return {
        "family_decisions": int(fam.sum()), "family_share": share(fam), "belief_decisions": int(used.sum()),
        "twin_labelled_share_in_family": share(col["twin_ok"][fam]),
        "fallback_share_in_family": share(col["belief_fallback"][fam]),
        "family_without_twin_record": int((fam & ~col["twin_record"]).sum()),
        "posterior_undefined_in_family": int((fam & ~np.isfinite(col["posterior"].astype(np.float64))).sum()),
        "decisions_with_both": int(both.sum()),
        "gather_beats_act_share": share(gb[both]), "gather_beats_act_n": int((gb & both).sum()),
        "gather_beats_act_family_share": share(gb[both & used]), "gather_beats_act_family_n": int((gb & both & used).sum()),
        "gather_beats_act_by_backup": by_group(col["backup"][both], gb[both]),
        "gather_beats_act_family_by_backup": by_group(col["backup"][both & used], gb[both & used]),
        "gather_beats_act_by_group": by_group(col["family_group"][both], gb[both]),
        "gather_beats_act_by_class": by_group(col["incident_class"][both], gb[both]),
        "voi": {
            "observe_candidates": summary(voi_b[obs]), "positive_share": share(voi_b[obs] > TOL),
            "observe_candidates_family": summary(voi_b[obs & cand["belief_used"]]),
            "positive_share_family": share(voi_b[obs & cand["belief_used"]] > TOL),
            "by_backup": {g: {**summary(voi_b[obs & (cand["backup"] == g)]),
                              "positive_share": share(voi_b[obs & (cand["backup"] == g)] > TOL)}
                          for g in sorted(set(cand["backup"][obs].tolist()))},
            "by_group": {g: {**summary(voi_b[obs & (cand["family_group"] == g)]),
                             "positive_share": share(voi_b[obs & (cand["family_group"] == g)] > TOL)}
                         for g in sorted(set(cand["family_group"][obs].tolist()))},
            "max_per_decision": summary(col["voi_max_belief"]),
            "max_per_decision_family": summary(col["voi_max_belief"][used]),
        },
        "hindsight_vs_belief": {
            "n": int(used.sum()), "with_both": int((both & used).sum()),
            "gather_beats_act_hindsight_share": share(gather[both & used]),
            "gather_beats_act_belief_share": share(gb[both & used]),
            "gather_only_under_belief_share": share((gb & ~gather)[both & used]),
            "gather_only_under_hindsight_share": share((gather & ~gb)[both & used]),
            "argmin_agree_share": share(col["argmin_agree"][used]),
            "hindsight_choice_belief_regret": summary(col["hindsight_choice_belief_regret"][used]),
            "hindsight_choice_belief_regret_positive_share": share(col["hindsight_choice_belief_regret"][used] > TOL),
            "belief_choice_hindsight_regret": summary(col["belief_choice_hindsight_regret"][used]),
            "J_abs_diff": summary(cand["J_abs_diff"][cand["belief_used"]]),
            "voi_max_hindsight": summary(col["voi_max"][both & used]),
            "voi_max_belief": summary(col["voi_max_belief"][both & used]),
            "posterior_true_cause": summary(col["posterior"][used]),
            "posterior_true_cause_post_onset": summary(col["posterior"][used & post]),
            "twin_excluded_share": share(~identical),
        },
        "calibration": calibration_table(col["p_link"][sel], col["is_link"][sel]),
        "calibration_post_onset": calibration_table(col["p_link"][sel & post], col["is_link"][sel & post]),
    }


def twin_stats(episodes, rows):
    fam = [e for e in episodes if in_family(e)]
    twins = [e["twin"] for e in fam if e.get("twin")]
    div = [t["divergence"] for t in twins]
    firsts = [d["first_deterministic_difference"] for d in div if d["first_deterministic_difference"]]
    replayed = [r for t in twins for r in t["replayed"]]
    decisions = [r for r in rows if r["family"] and r["twin_record"]]
    identical = [r["deterministic_identical"] for r in decisions]
    memory = [r["pi_after_memory_differs"] for r in decisions if r["pi_after_memory_differs"] is not None]
    trips = [(e["outcome"]["shutdown"] is not None, e["twin"]["outcome"]["shutdown"] is not None)
             for e in fam if e.get("twin")]
    return {
        "family_episodes": len(fam), "twinned_episodes": len(twins),
        "receipt_status_differs_share": share([d["receipt_status_differs"] for d in div]),
        "receipt_differences": int(sum(len(d["receipt_differences"]) for d in div)),
        "replayed_submissions": len(replayed), "not_replayed_submissions": int(sum(len(t["not_replayed"])
                                                                                   for t in twins)),
        "replays_citation_changed": int(sum(r["citation_changed"] for r in replayed)),
        "replays_not_matched": int(sum(not r["matched"] for r in replayed)),
        "diverged_share": share([d["first_deterministic_difference"] is not None for d in div]),
        "first_difference_since_onset_min": summary([f["since_onset_s"] for f in firsts], 1 / 60),
        "first_difference_kind": counts([f["kind"] for f in firsts]),
        "primary_tripped_share": share([p for p, _ in trips]), "twin_tripped_share": share([t for _, t in trips]),
        "trip_outcome_differs_share": share([p != t for p, t in trips]),
        "decisions": {"n": len(decisions), "labelled_share": share([r["twin_ok"] for r in decisions]),
                      "deterministic_identical_share": share(identical),
                      "candidates_unmatched": int(sum(r["candidates_unmatched"] for r in decisions)),
                      "pi_after_memory_differs_share": share(memory)},
    }


def check_stats(eps, rows, col, out):
    censored = [(t, c) for r in rows for t, row in zip(r["cand"]["tool"], r["first_censored"]) for c in row]
    tools = sorted({t for t, _ in censored})
    with_decisions = [e for e in eps if e["n_decisions"]]
    return {
        "all_candidates_rejected_share": share(col["all_rejected"]),
        "monitor_pair_identical_share": out["quality"]["monitor_pair_identical_share"],
        "monitor_pair_identical_labels_share": share([r["monitor_pair_identical_labels"] for r in rows
                                                      if r["monitor_pair_identical_labels"] is not None]),
        "gather_beats_act_tool_cost_only_share": out["voi"]["gather_beats_act_tool_cost_only_share"],
        "first_action_censored_share": share([c for _, c in censored]),
        "first_action_censored_by_tool": {t: share([c for u, c in censored if u == t]) for t in tools},
        "receipt_items": int(sum(e["receipt_items"] for e in eps)),
        "episodes_with_receipt_items_share": share([e["receipt_items"] > 0 for e in with_decisions]),
        "decisions_with_valid_receipt_share": share(col["valid_receipt"]),
        "receipts_valid_by_time_not_known": int(col["unknown_valid_receipts"].sum()),
        "other_items_valid_by_time_not_known": int(col["unknown_valid_other"].sum()),
        "refusals_at_accept_time": int(sum(e["refusals_at_accept_time"] for e in eps)),
        "decisions_with_valid_evidence_missing_share": share(col["valid_evidence_missing"] > 0),
        "own_action_receipts_stored": int(col["own_action_receipts"].sum()),
        "receipt_items_without_action_id": int(sum(e["receipts_without_action_id"] for e in eps)),
        "own_action_tool_results_stored": int(col["own_action_tool_results"].sum()),
        "tool_results_without_action_id": int(sum(e["tool_results_without_action_id"] for e in eps)),
    }


def episode_stats(eps):
    kinds = [e["kind"] for e in eps]
    return {
        "n": len(eps), "with_decisions": sum(e["n_decisions"] > 0 for e in eps),
        "without_decisions": [{"seed": e["seed"], "cause": e["cause"], "background": e["background"]}
                              for e in eps if e["n_decisions"] == 0],
        "incidents_config": counts([e["incidents"] for e in eps]),
        "incident_kind": counts(kinds),
        "incident_target": {k: counts([e["target"] for e in eps if e["kind"] == k]) for k in sorted(set(kinds))},
        "sensor_fault_kind": counts([e["fault_kind"] for e in eps if e["kind"] == "sensor_fault"]),
        "background": counts([e["background"] for e in eps]),
        "backup": counts([e["backup"] for e in eps]),
        "onset_after_warmup_min": summary([e["onset_s"] for e in eps], 1 / 60),
        "behaviour_mode": counts([e["mode"] for e in eps]),
        "tripped_under_behaviour_share": share([e["tripped"] for e in eps]),
        "tripped_under_behaviour_by_kind": by_group(kinds, [e["tripped"] for e in eps]),
        "cost_under_behaviour": summary([e["cost"] for e in eps]),
        "decisions_per_episode": counts([e["n_decisions"] for e in eps]),
        "items_per_episode": summary([e["n_items"] for e in eps]),
        "item_kinds": counts([k for e in eps for k in e["item_kinds"]]),
        "generation_seconds": summary([e["seconds"] for e in eps]),
    }


def decision_stats(rows, col):
    post = col["onset_happened"]
    oracle = np.isfinite(col["oracle_tts_s"])
    truncated = col["horizon_s"] < col["horizon_s"].max() - 1e-6
    return {
        "n": len(rows), "kind": counts(col["kind"]), "incident_class": counts(col["incident_class"]),
        "backup": counts(col["backup"]), "pre_onset_share": share(~post),
        "since_onset_min": summary(col["since_onset_s"][post], 1 / 60),
        "before_onset_min": summary(-col["since_onset_s"][~post], 1 / 60),
        "oracle_deadline_share": share(oracle[post]), "oracle_deadline_min": summary(col["oracle_tts_s"][post], 1 / 60),
        "oracle_deadline_by_class": by_group(col["incident_class"][post], oracle[post]),
        "valid_evidence": summary(col["n_valid_evidence"]), "cited_evidence": summary(col["n_cited_evidence"]),
        "label_horizon_min": summary(col["horizon_s"], 1 / 60), "truncated_horizon_n": int(truncated.sum()),
        "truncated_horizon_min": summary(col["horizon_s"][truncated], 1 / 60),
    }


def candidate_stats(rows, cand):
    tools = sorted(set(cand["tool"].tolist()))
    present = {t: np.array([t in r["cand"]["tool"] for r in rows]) for t in tools}
    per = {t: np.array([r["cand"]["tool"].count(t) for r in rows]) for t in tools}
    return {
        "n": int(cand["J"].size), "per_decision": summary([r["n_candidates"] for r in rows]),
        "per_decision_counts": counts([r["n_candidates"] for r in rows]),
        "distinct_outcomes_per_decision": summary([r["n_distinct"] for r in rows]),
        "family": counts(cand["family"]),
        "per_tool": {t: {"n": int(per[t].sum()), "decisions_share": share(present[t]),
                         "per_decision_when_present": float(per[t][present[t]].mean())} for t in tools},
    }


def label_stats(rows, col, cand):
    p, res = cand["p_trip"], cand["p_resolved"]
    tools = sorted(set(cand["tool"].tolist()))
    trip_times, status, reasons = [], collections.defaultdict(list), []
    durations = collections.defaultdict(list)
    for r in rows:
        for x in r["results"]:
            status[x["tool_id"]].append(x["first_status"])
            if x["first_status"] != "succeeded":
                reasons.append(f"{x['tool_id']}:{x['first_status']}:{x['first_reason']}")
            if x["shutdown"]:
                trip_times.append(x["time_to_shutdown"])
            if x["first_duration"] is not None:
                durations[x["tool_id"]].append(x["first_duration"])
    defined = np.isfinite(res)
    return {
        "J": summary(cand["J"]),
        "J_by_family": {f: summary(cand["J"][cand["family"] == f]) for f in sorted(set(cand["family"].tolist()))},
        "J_by_tool": {t: summary(cand["J"][cand["tool"] == t]) for t in tools},
        "J_min_per_decision": summary([r["J_min"] for r in rows]),
        "J_range_per_decision": summary(col["J_range"]),
        "p_trip": {**summary(p), "zero_share": share(p <= 0), "one_share": share(p >= 1),
                   "between_share": share((p > 0) & (p < 1))},
        "p_trip_by_tool": {t: float(p[cand["tool"] == t].mean()) for t in tools},
        "trip_time_min": summary(trip_times, 1 / 60),
        "p_resolved": {**summary(res), "defined_share": share(defined), "one_share": share(res[defined] >= 1),
                       "zero_share": share(res[defined] <= 0)},
        "p_resolved_by_tool": {t: float(np.nanmean(res[cand["tool"] == t])) if np.isfinite(
            res[cand["tool"] == t]).any() else None for t in tools},
        "p_success_by_tool": {t: float(np.nanmean(cand["p_success"][cand["tool"] == t]))
                              if np.isfinite(cand["p_success"][cand["tool"] == t]).any() else None for t in tools},
        "first_status_by_tool": {t: counts(status[t]) for t in tools},
        "not_succeeded_reasons": counts(reasons),
        "first_duration_min_by_tool": {t: {**summary(durations[t], 1 / 60),
                                           "nominal_median_min": float(np.median(cand["nominal_median"][
                                               cand["tool"] == t])) / 60,
                                           "nominal_q90_min": float(np.median(cand["nominal_q90"][
                                               cand["tool"] == t])) / 60,
                                           "censored": sum(status[t].count(s) for s in CENSORED)} for t in tools},
    }


def best_stats(rows, col):
    have = np.isfinite(col["gap"])
    positive = have & (col["gap_se"] > 0)
    t95 = float(sps.t.ppf(0.975, max(int(col["n_replicates"].min()) - 1, 1)))
    classes = col["incident_class"]
    return {
        "tool_share": credit([r["best_tools"] for r in rows]),
        "family_share": credit([r["best_families"] for r in rows]),
        "tool_share_by_class": {c: credit([r["best_tools"] for r in rows if r["incident_class"] == c])
                                for c in sorted(set(classes.tolist()))},
        "argmin_size": counts(col["n_best"]), "tie_share": share(col["n_best"] > 1),
        "tie_across_tools_share": share([len(set(r["best_tools"])) > 1 for r in rows]),
        "wait_in_argmin_share": share(["monitor" in r["best_tools"] for r in rows]),
        "wait_only_argmin_share": share([set(r["best_tools"]) == {"monitor"} for r in rows]),
        "observe_in_argmin_share": share(["observe" in r["best_families"] for r in rows]),
        "respond_in_argmin_share": share(["respond" in r["best_families"] for r in rows]),
        "runner_up_defined_n": int(have.sum()), "gap_to_runner_up": summary(col["gap"]),
        "near_tie_share": share(col["gap"][have] <= EPS),
        "gap_over_se": summary(col["gap"][positive] / col["gap_se"][positive]),
        "runner_up_zero_variance_share": share(col["gap_se"][have] <= TOL),
        "t95_critical": t95,
        "unresolved_2se_share": share(col["gap"][have] <= 2 * col["gap_se"][have]),
        "unresolved_t95_share": share(col["gap"][have] <= t95 * col["gap_se"][have]),
        "unresolved_t95_material_share": share((col["gap"][have] > EPS)
                                               & (col["gap"][have] <= t95 * col["gap_se"][have])),
        "unresolved_t95_gap": summary(col["gap"][have & (col["gap"] <= t95 * col["gap_se"])]),
    }


def noise_stats(rows, col, cand):
    sd_diff = np.concatenate([r["pairs"]["sd_diff"] for r in rows])
    sd_indep = np.concatenate([r["pairs"]["sd_indep"] for r in rows])
    identical = np.concatenate([r["pairs"]["identical"] for r in rows])
    reps = np.concatenate([np.full(r["pairs"]["sd_diff"].size, r["n_replicates"]) for r in rows])
    keep = ~identical & np.isfinite(sd_diff) & np.isfinite(sd_indep)
    denom = keep & (sd_indep > TOL)
    ratio = sd_diff[denom] ** 2 / sd_indep[denom] ** 2
    p = cand["p_trip"]
    tools = sorted(set(cand["tool"].tolist()))
    comp = {k: sum(r["component_variance"][k] for r in rows) for k in COMPONENTS}
    comp_total = sum(comp.values())
    return {
        "replicates_per_candidate": counts(col["n_replicates"]),
        "J_sd": summary(cand["J_sd"]), "J_sd_zero_share": share(cand["J_sd"] <= TOL),
        "J_sd_below_eps_share": share(cand["J_sd"] <= EPS),
        "J_sd_one_share": share(np.abs(cand["J_sd"] - 1.0) <= 1e-6),
        "J_sd_ge_10_share": share(cand["J_sd"] >= 10.0),
        "variance_share_by_component": {k: v / comp_total for k, v in comp.items()} if comp_total > 0 else None,
        "J_sd_by_tool": {t: {"median": float(np.median(cand["J_sd"][cand["tool"] == t])),
                             "zero_share": share(cand["J_sd"][cand["tool"] == t] <= TOL)} for t in tools},
        "trip_disagreement_share": share((p > 0) & (p < 1)),
        "pairs": int(identical.size), "pairs_identical_share": share(identical),
        "pair_sd_diff": summary(sd_diff[keep]), "pair_sd_indep": summary(sd_indep[keep]),
        "pair_se_diff": summary(sd_diff[keep] / np.sqrt(reps[keep])),
        "pair_sd_diff_zero_share": share(sd_diff[keep] <= TOL),
        "pooled_variance_ratio": float(np.sum(sd_diff[denom] ** 2) / np.sum(sd_indep[denom] ** 2))
        if denom.any() else None,
        "pair_variance_ratio": summary(ratio), "pairs_reduced_share": share(ratio < 1.0),
        "pairs_increased_share": share(ratio > 1.0),
        "runner_up_sd_diff": summary(col["gap_sd"]),
    }


def behaviour_stats(eps, rows, col):
    modes = np.array([r["mode"] for r in rows])
    nonempty = col["plan_length"] > 0
    reg = col["behaviour_regret"]
    return {
        "epsilon": counts([r["epsilon"] for r in rows]),
        "mode_by_episode": counts([e["mode"] for e in eps if e["n_decisions"]]),
        "mode_by_decision": counts(modes),
        "explored_share": share(col["explored"]), "explored_by_mode": by_group(modes, col["explored"]),
        "plan_length": counts(col["plan_length"]), "empty_plan_share": share(~nonempty),
        "empty_plan_implied_probability": counts([1.0 - r["epsilon"] for r in rows if not r["plan_length"]]),
        "empty_plan_recorded_probability": counts([r["empty_plan_probability"] for r in rows if not r["plan_length"]]),
        "first_action_in_candidates_share": share(col["behaviour_index"][nonempty] >= 0),
        "first_action_probability": summary(col["behaviour_probability"][nonempty]),
        "taken_probability": summary(col["behaviour_probability"]),
        "min_candidate_probability": summary([r["epsilon"] / r["n_candidates"] for r in rows]),
        "regret": summary(reg), "zero_regret_share": share(reg[np.isfinite(reg)] <= TOL),
        "near_zero_regret_share": share(reg[np.isfinite(reg)] <= EPS),
        "regret_explored": summary(reg[col["explored"]]), "regret_playbook": summary(reg[~col["explored"]]),
    }


def playbook_stats(rows, col, stakes):
    out = {}
    for m in ("act_first", "investigate_first"):
        idx = np.array([r["baseline_index"][m] for r in rows])
        reg = np.array([r["regret"][m] for r in rows])
        exc = np.array([r["trip_excess"][m] for r in rows])
        ok = idx >= 0
        out[m] = {"coverage": share(ok), "regret": summary(reg[ok]), "zero_regret_share": share(reg[ok] <= TOL),
                  "near_zero_regret_share": share(reg[ok] <= EPS),
                  "regret_gt_1_share": share(reg[ok] > 1.0), "regret_ge_50_share": share(reg[ok] >= 50.0),
                  "trip_excess_mean": float(exc[ok].mean()), "trip_excess_positive_share": share(exc[ok] > TOL),
                  "regret_high_stakes": summary(reg[ok & stakes]),
                  "trip_excess_high_stakes_mean": float(exc[ok & stakes].mean()) if (ok & stakes).any() else None,
                  "regret_by_class": by_group(col["incident_class"], ok, reg),
                  "wait_share": share(idx[ok] == col["wait_index"][ok])}
    a = np.array([r["baseline_index"]["act_first"] for r in rows])
    i = np.array([r["baseline_index"]["investigate_first"] for r in rows])
    out["modes_agree_share"] = share(a == i)
    out["reference"] = {"random_candidate_regret": summary(col["random_regret"]),
                        "wait_regret": summary(col["wait_regret"]),
                        "wait_near_zero_regret_share": share(col["wait_regret"][np.isfinite(col["wait_regret"])] <= EPS),
                        "random_candidate_regret_high_stakes": summary(col["random_regret"][stakes]),
                        "wait_regret_high_stakes": summary(col["wait_regret"][stakes])}
    return out


def quality_stats(rows, col, cand):
    keys = collections.Counter((r["seed"], round(r["tau"], 6)) for r in rows)
    sigs = collections.Counter(r["signature"] for r in rows)
    mon = [r["monitor_pair_identical"] for r in rows if r["monitor_pair_identical"] is not None]
    statuses = collections.Counter(x["first_status"] for r in rows for x in r["results"])
    return {
        "duplicate_decisions_same_episode_time": sum(v - 1 for v in keys.values()),
        "duplicate_label_signatures": sum(v - 1 for v in sigs.values()),
        "decisions_with_duplicate_action_ids": sum(not r["action_ids_unique"] for r in rows),
        "label_identical_candidates": int(col["label_identical"].sum()),
        "label_identical_candidates_share": float(col["label_identical"].sum() / cand["J"].size),
        "monitor_pair_decisions": len(mon), "monitor_pair_identical_share": share(mon),
        "all_candidates_rejected_decisions": int(col["all_rejected"].sum()),
        "all_candidates_rejected_seeds": sorted({r["seed"] for r in rows if r["all_rejected"]}),
        "candidates_rejected_in_every_replicate": int(sum(r["n_rejected_candidates"] for r in rows)),
        "all_J_equal_decisions": int((col["J_range"] <= TOL).sum()),
        "results": int(sum(statuses.values())), "result_status": counts([x["first_status"] for r in rows
                                                                           for x in r["results"]]),
        "nan_J_candidates": int(np.isnan(cand["J"]).sum()),
    }


def array_stats(arrays, meta):
    out = {}
    sources = meta["vocab"]["sources"]
    for split, a in arrays.items():
        tau = a["decision_time"][:, None]
        valid = a["present"] & (a["arrival_time"] <= tau) & (tau < a["valid_until"]) & (tau < a["retracted_at"])
        m = a["present"].shape[1]
        out[split] = {
            "schema": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in a.items()},
            "decisions": int(a["dec_tau"].size), "candidates": int(a["cand_decision"].size),
            "slots": int(m), "payload_width": int(a["payload"].shape[-1]), "action_features": int(
                a["cand_features"].shape[-1]),
            "present_per_decision": summary(a["present"].sum(1)), "valid_per_decision": summary(valid.sum(1)),
            "at_slot_cap_share": share(a["present"].sum(1) >= m),
            "present_not_valid_share": float((a["present"] & ~valid).sum() / max(a["present"].sum(), 1)),
            "present_future_share": float((a["present"] & (a["arrival_time"] > tau)).sum() / max(a["present"].sum(), 1)),
            "decisions_with_future_items_share": share((a["present"] & (a["arrival_time"] > tau)).any(1)),
            "valid_by_source": counts([sources[s] for s in a["source"][valid]]),
            "label_nan_share": {k: share(np.isnan(v.astype(np.float64))) for k, v in a.items()
                                if k.startswith("y_") and v.dtype.kind == "f"},
            "baseline_coverage": {m_: share(a[f"dec_baseline_{m_}"] >= 0) for m_ in ("act_first", "investigate_first")},
        }
    return out


def consistency(episodes, a):
    eps = [e for e in sorted(episodes, key=lambda e: e["seed"]) if e["decisions"]]
    decs = [(e, d) for e in eps for d in e["decisions"]]
    labs = [candidate_labels(d) for _, d in decs]
    got = {f"y_{k}": np.concatenate([lab[k] for lab in labs]) for k in ("J", "J_std", "p_trip", "voi", "p_resolved",
                                                                         "p_success", "duration")}
    R = max(lab["first_censored"].shape[1] for lab in labs) if labs else 0
    for k in ("first_censored", "duration_time", "duration_censored"):
        got[f"y_{k}"] = np.concatenate([np.pad(lab[k].astype(np.float64), ((0, 0), (0, R - lab[k].shape[1])),
                                               constant_values=np.nan) for lab in labs])
    beliefs = [belief_labels(d, in_family(e)) for e, d in decs]
    got.update({f"y_{k}": np.concatenate([b[k] for b in beliefs]) if beliefs else np.zeros(0) for k in BELIEF_KEYS})
    got.update({f"dec_{k}": np.array([b[k] for b in beliefs], np.float64) for k in BELIEF_DECISION_KEYS})
    got.update({
        "dec_tau": np.array([d["tau"] for _, d in decs]), "dec_episode_seed": np.array([e["seed"] for e, _ in decs]),
        "dec_kind": np.array([KINDS.index(d["hidden"]["kind"]) if d["hidden"]["onset_happened"] else 0
                              for _, d in decs]),
        "dec_explored": np.array([d["behaviour"]["explored"] for _, d in decs]),
        "dec_n_candidates": np.array([len(d["candidates"]) for _, d in decs]),
        "dec_baseline_act_first": np.array([baseline_index(d, "act_first") for _, d in decs]),
        "dec_baseline_investigate_first": np.array([baseline_index(d, "investigate_first") for _, d in decs]),
        "dec_backup_available": np.array([backup_available(e) for e, _ in decs]),
    })
    out = {}
    for k in CHECKED_FIELDS:
        if k not in a:
            out[k] = {"match": None, "missing_in_arrays": True}
            continue
        x, y = np.asarray(got[k], np.float64), np.asarray(a[k], np.float64)
        if x.shape != y.shape:
            out[k] = {"match": False, "shape_raw": list(x.shape), "shape_arrays": list(y.shape)}
            continue
        same_nan = np.array_equal(np.isnan(x), np.isnan(y))
        fin = np.isfinite(x) & np.isfinite(y)
        diff = float(np.max(np.abs(x[fin] - y[fin]))) if fin.any() else 0.0
        out[k] = {"match": bool(same_nan and diff <= 1e-9), "max_abs_diff": diff}
    return out


def own_action_items_in_rows(episodes, a, keys):
    eps = [e for e in sorted(episodes, key=lambda e: e["seed"]) if e["decisions"]]
    decs = [(seg, d) for seg, e in enumerate(eps) for d in e["decisions"]]
    if len(decs) != a["item_id"].shape[0] or not np.array_equal([d["tau"] for _, d in decs], a["decision_time"]):
        raise ValueError("the arrays' rows do not follow the raw episodes' decisions")
    items = [{i["evidence_id"]: i for i in e["items"]} for e in eps]
    group = {"receipt": "receipt", **{k: "tool_result" for k in TOOL_RESULT_KINDS}}
    out = {"rows": len(decs)}
    for g in ("receipt", "tool_result"):
        out.update({f"own_action_{g}_slots": 0, f"rows_with_own_action_{g}s": 0, f"later_action_{g}_slots": 0,
                    f"{g}_slots_without_action_id": 0})
    for r, (seg, d) in enumerate(decs):
        own, later = set(submitted_action_ids(d)), later_action_ids(eps[seg], d["tau"])
        hits = {"receipt": 0, "tool_result": 0}
        for item_id in a["item_id"][r][a["present"][r]]:
            segment, evidence_id = keys[int(item_id)]
            item = items[seg].get(evidence_id) if segment == seg else None
            if item is None:
                raise ValueError(f"row {r}: item {evidence_id!r} of segment {segment} is not a stored item of "
                                 f"the row's episode")
            if item["kind"] not in group:
                continue
            g, action = group[item["kind"]], item["payload"].get("action_id")
            if not isinstance(action, str):
                out[f"{g}_slots_without_action_id"] += 1
            elif action in own:
                hits[g] += 1
            elif action in later:
                out[f"later_action_{g}_slots"] += 1
        for g, n in hits.items():
            out[f"own_action_{g}_slots"] += n
            out[f"rows_with_own_action_{g}s"] += n > 0
    return out


def registry_space(vocab):
    out = {}
    for spec in load_registry():
        n = sum(len(vocab["targets"].get(k, ())) for k in spec.target_kinds) * len(arg_grid(spec))
        out[spec.tool_id] = n
    return {"per_tool": out, "total": sum(out.values())}


def runs_referencing(path):
    path = Path(path)
    needles = (f"{path.parent.name}/{path.name}", f"{path.parent.name}\\\\{path.name}")
    out = []
    for rec in sorted(RUNS_DIR.rglob("run.json")):
        text = rec.read_text(encoding="utf-8", errors="replace")
        if any(n in text for n in needles):
            out.append({"run": rec.parent.relative_to(ROOT).as_posix(), "kind": json.loads(text).get("kind")})
    return out


def provenance(episode_root, array_root):
    episode_root, array_root = Path(episode_root), Path(array_root)
    tep = load_config("sim/tep.yaml")["kernel"]
    out = {"kernel": {"commit": tep["commit"], "temexd_mod_c_sha256_config": tep["temexd_mod_c_sha256"],
                      "temexd_mod_c_sha256_file": sha256_file(KERNEL) if KERNEL.exists() else None},
           "wrapper_licence": LICENCE.read_text(encoding="utf-8").splitlines()[0] if LICENCE.exists() else None}
    configured = str(out["kernel"]["temexd_mod_c_sha256_config"])
    out["kernel"]["config_hash_well_formed"] = len(configured) == 64 and all(c in "0123456789abcdef"
                                                                             for c in configured.lower())
    out["kernel"]["match"] = configured == out["kernel"]["temexd_mod_c_sha256_file"]
    try:
        import tep_studio
        out["tep_studio_version"] = getattr(tep_studio, "__version__", None)
    except ImportError:
        out["tep_studio_version"] = None
    record = RUNS_DIR / "experience" / episode_root.name / "run.json"
    if record.exists():
        rec = json.loads(record.read_text(encoding="utf-8"))
        files = rec["source"]["files"]
        now = {f: sha256_file(ROOT / f) if (ROOT / f).exists() else None for f in PROVENANCE_FILES}
        out["generation_run"] = {
            "record": record.parent.relative_to(ROOT).as_posix(), "argv": rec["argv"], "config": rec["config"],
            "config_sha256": rec["config_sha256"], "started_utc": rec["started_utc"],
            "finished_utc": rec.get("finished_utc"), "environment": rec["environment"],
            "wall_s": rec.get("metrics", {}).get("wall_s"), "source_combined": rec["source"]["combined"],
            "source_changed_during_run": rec.get("source_changed_during_run"),
            "files_at_generation": {f: files.get(f) for f in PROVENANCE_FILES},
            "changed_since_generation": [f for f in PROVENANCE_FILES if files.get(f) != now[f]],
            "all_changed_since_generation": sorted(f for f, h in files.items()
                                                   if (ROOT / f).exists() and sha256_file(ROOT / f) != h),
        }
        out["files_now"] = now
    raw = episode_files(episode_root)
    digest = hashlib.sha256()
    for p in raw.values():
        digest.update(f"{p.name}:{sha256_file(p)}".encode())
    out["raw_episodes"] = {"files": len(raw), "manifest_sha256": digest.hexdigest()}
    out["arrays"] = {p.name: sha256_file(p) for p in sorted(array_root.glob("*")) if p.is_file()}
    out["runs_referencing_arrays"] = runs_referencing(array_root)
    return out


def split_table(meta, episode_root):
    files = episode_files(episode_root)
    fractions = tuple(meta["fractions"])
    none = {"episodes": 0, "decisions": 0, "candidates": 0, "seeds": []}
    out = {}
    for s in SPLITS:
        v = meta["splits"].get(s, none)
        seeds_with = set(v["seeds"])
        seeds_all = sorted(k for k in files if split_of(k, fractions) == s)
        out[s] = {"episode_files": len(seeds_all), "episodes_with_decisions": v["episodes"],
                  "episodes_without_decisions": sorted(set(seeds_all) - seeds_with), "decisions": v["decisions"],
                  "candidates": v["candidates"], "seeds": sorted(seeds_with)}
    out["total"] = {k: sum(out[s][k] for s in SPLITS) for k in (
        "episode_files", "episodes_with_decisions", "decisions", "candidates")}
    return out


def headline(stats):
    dev = stats["development"]
    array_checks = stats.get("array_checks")
    return {
        "episodes": stats["splits"]["total"]["episode_files"], "decisions": stats["splits"]["total"]["decisions"],
        "decisions_by_split": {s: stats["splits"][s]["decisions"] for s in ("train", "val", "test")},
        "dev_decisions": dev["decisions"]["n"], "dev_candidates": dev["candidates"]["n"],
        "candidates_per_decision_mean": dev["candidates"]["per_decision"]["mean"],
        "distinct_outcomes_per_decision_mean": dev["candidates"]["distinct_outcomes_per_decision"]["mean"],
        "high_stakes_share": dev["stakes"]["high_stakes_share"],
        "unavoidable_trip_share": dev["stakes"]["unavoidable_trip_share"],
        "gather_beats_act_share": dev["voi"]["gather_beats_act_share"],
        "gather_beats_act_tool_cost_only_share": dev["voi"]["gather_beats_act_tool_cost_only_share"],
        "tool_cost_only_decisions_share": dev["stakes"]["tool_cost_only_share"],
        "wait_in_argmin_share": dev["best"]["wait_in_argmin_share"], "tie_share": dev["best"]["tie_share"],
        "monitor_pair_identical_share": dev["quality"]["monitor_pair_identical_share"],
        "all_candidates_rejected_decisions": dev["quality"]["all_candidates_rejected_decisions"],
        "pooled_variance_ratio_crn": dev["noise"]["pooled_variance_ratio"],
        "argmin_unresolved_t95_share": dev["best"]["unresolved_t95_share"],
        "regret_act_first_mean": dev["playbooks"]["act_first"]["regret"].get("mean"),
        "regret_investigate_first_mean": dev["playbooks"]["investigate_first"]["regret"].get("mean"),
        "regret_random_mean": dev["playbooks"]["reference"]["random_candidate_regret"].get("mean"),
        "regret_wait_mean": dev["playbooks"]["reference"]["wait_regret"].get("mean"),
        "label_identical_candidates_share": dev["quality"]["label_identical_candidates_share"],
        "loose_high_stakes_without_trip_difference": dev["stakes"]["loose_definition_without_trip_difference"],
        "high_stakes_by_backup": dev["stakes"]["high_stakes_by_backup"],
        "gather_beats_act_by_backup": dev["voi"]["gather_beats_act_by_backup"],
        "belief": {k: dev["belief"][k] for k in (
            "family_decisions", "belief_decisions", "twin_labelled_share_in_family", "fallback_share_in_family",
            "posterior_undefined_in_family", "gather_beats_act_share", "gather_beats_act_family_share",
            "gather_beats_act_by_backup", "gather_beats_act_family_by_backup")}
        | {"hindsight_gather_beats_act_family_share":
           dev["belief"]["hindsight_vs_belief"]["gather_beats_act_hindsight_share"],
           "voi_positive_share_family": dev["belief"]["voi"]["positive_share_family"],
           "calibration_ece": dev["belief"]["calibration"]["ece"],
           "calibration_brier": dev["belief"]["calibration"]["brier"],
           "twin_receipt_status_differs_share": dev["twins"]["receipt_status_differs_share"],
           "twin_decisions_identical_share": dev["twins"]["decisions"]["deterministic_identical_share"]},
        "checks": {k: v for k, v in dev["checks"].items() if not isinstance(v, dict)},
        **{name: sum(c[key] for c in array_checks.values()) if array_checks else None for name, key in (
            ("own_action_receipts_in_rows", "own_action_receipt_slots"),
            ("later_action_receipts_in_rows", "later_action_receipt_slots"),
            ("receipt_slots_without_action_id", "receipt_slots_without_action_id"),
            ("own_action_tool_results_in_rows", "own_action_tool_result_slots"),
            ("later_action_tool_results_in_rows", "later_action_tool_result_slots"),
            ("tool_result_slots_without_action_id", "tool_result_slots_without_action_id"))},
        "kernel_hash_matches_config": stats["provenance"]["kernel"]["match"],
        "arrays_reproduce_from_raw": all(v["match"] for s in stats.get("consistency", {}).values()
                                         for v in s.values() if v["match"] is not None),
        "array_fields_missing": sorted({k for s in stats.get("consistency", {}).values() for k, v in s.items()
                                        if v["match"] is None}),
    }


def compute(episodes_by_split, arrays=None, meta=None, item_ids=None):
    stats = {s: block(eps) for s, eps in episodes_by_split.items()}
    stats["development"] = block([e for s in DEV_SPLITS for e in episodes_by_split.get(s, [])])
    if arrays:
        stats["arrays"] = array_stats(arrays, meta) if meta else {}
        stats["consistency"] = {s: consistency(episodes_by_split[s], a) for s, a in arrays.items()
                                if s in episodes_by_split}
        if item_ids:
            stats["array_checks"] = {s: own_action_items_in_rows(episodes_by_split[s], a, item_ids[s])
                                     for s, a in arrays.items() if s in episodes_by_split and s in item_ids}
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default=str(EPISODES))
    parser.add_argument("--arrays", default=str(ARRAYS))
    parser.add_argument("--name", default="v2-natural-stats")
    args = parser.parse_args()
    episode_root, array_root = Path(args.episodes), Path(args.arrays)
    meta = json.loads((array_root / "meta.json").read_text(encoding="utf-8"))
    cfg = {"episodes": str(episode_root), "arrays": str(array_root), "splits": list(DEV_SPLITS), "tol": TOL}
    run = start_run("datasets", cfg, seed=None, name=args.name)
    episodes = load_episodes(episode_root, DEV_SPLITS, tuple(meta["fractions"]))
    stats = compute(episodes, load_arrays(array_root, DEV_SPLITS), meta, load_item_ids(array_root, DEV_SPLITS))
    stats["splits"] = split_table(meta, episode_root)
    stats["registry_space"] = registry_space(meta["vocab"])
    stats["vocab"] = {"sources": len(meta["vocab"]["sources"]), "P": meta["vocab"]["P"], "F": meta["vocab"]["F"],
                      "kinds": meta["vocab"]["kinds"], "tools": meta["vocab"]["tools"],
                      "targets": {k: len(v) for k, v in meta["vocab"]["targets"].items()}}
    stats["provenance"] = provenance(episode_root, array_root)
    stats["provenance"]["array_build"] = meta.get("provenance")
    stats["headline"] = headline(stats)
    stats = jsonable(stats)
    run.save_json("stats.json", stats)
    run.finish(stats["headline"])
    print(json.dumps(stats["headline"], indent=1))
    print(f"-> {run.dir / 'stats.json'}")

if __name__ == "__main__":
    main()
