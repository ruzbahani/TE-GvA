import json
import pickle
from pathlib import Path

import numpy as np

from cpsdefense.core.types import EvidenceItem
from cpsdefense.defense.registry import load_registry
from cpsdefense.experience import belief
from cpsdefense.experience import tensors as T
from cpsdefense.sim.incidents import Incident, Scenario
from cpsdefense.sim.tep_env import TEPEnv

SPLITS = ("train", "val", "test")
KINDS = ("none", "link_fault", "io_module_fault", "sensor_fault", "disturbance")
CENSORED = ("running", "accepted")
TOOL_RESULT_KINDS = ("trace_link", "historian", "field_check", "operator_report")
ACTION_ITEM_KINDS = ("receipt", *TOOL_RESULT_KINDS)
LABEL_FORMAT = "tep-decision-labels/3"
LABEL_KEYS = ("J", "J_std", "regret", "p_trip", "trip_event", "trip_time", "horizon", "p_resolved", "p_mitigated",
              "p_success", "first_censored", "duration", "duration_time", "duration_censored", "voi", "observe")
BELIEF_KEYS = ("J_belief", "p_trip_belief", "voi_belief")
BELIEF_DECISION_KEYS = ("posterior", "family", "twin_ok", "belief_fallback")


def split_of(seed, fractions=(0.6, 0.2, 0.2)):
    u = ((int(seed) * 2654435761) % 2**32) / 2**32
    return SPLITS[0] if u < fractions[0] else SPLITS[1] if u < fractions[0] + fractions[1] else SPLITS[2]


def load_episodes(root):
    return [pickle.load(open(p, "rb")) for p in sorted(Path(root).glob("ep_*.pkl"))]


def to_item(d):
    return EvidenceItem(evidence_id=d["evidence_id"], source=d["source"], kind=d["kind"], subject_id=d["subject_id"],
                        event_time=d["event_time"], arrival_time=d["arrival_time"], valid_until=d["valid_until"],
                        payload=d["payload"], provenance=d["provenance"], quality=d.get("quality", 1.0),
                        supersedes=tuple(d.get("supersedes", ())))


def backup_available(episode):
    if "backup_available" in episode:
        return bool(episode["backup_available"])
    return bool(episode["scenario"]["backup_available"])


def submitted_action_ids(decision):
    ids = decision.get("submitted_action_ids")
    return list(decision["behaviour"]["actions"] if ids is None else ids)


def later_action_ids(episode, tau):
    return {s["action_id"] for s in episode.get("submissions", ()) if s["time"] > tau}


def items_about(action_ids, items, kinds=ACTION_ITEM_KINDS):
    matching, unnamed = [], []
    for i in items:
        if i["kind"] not in kinds:
            continue
        action = i["payload"].get("action_id")
        if not isinstance(action, str):
            unnamed.append(i["evidence_id"])
        elif action in action_ids:
            matching.append(i["evidence_id"])
    return matching, unnamed


def receipts_about(action_ids, items):
    return items_about(action_ids, items, ("receipt",))


def tool_results_about(action_ids, items):
    return items_about(action_ids, items, TOOL_RESULT_KINDS)


def own_action_receipts(decision, items):
    return receipts_about(set(submitted_action_ids(decision)), items)


def own_action_tool_results(decision, items):
    return tool_results_about(set(submitted_action_ids(decision)), items)


def candidate_labels(decision):
    cands = decision["candidates"]
    reps = sorted({r["replicate"] for r in decision["results"]})
    C, R = len(cands), len(reps)
    total = np.full((C, R), np.nan)
    trip = np.zeros((C, R), bool)
    trip_time = np.full((C, R), np.nan)
    horizon = np.full((C, R), np.nan)
    resolved = np.full((C, R), np.nan)
    mitigated = np.full((C, R), np.nan)
    success = np.full((C, R), np.nan)
    censored = np.zeros((C, R), bool)
    duration = np.full((C, R), np.nan)
    duration_time = np.full((C, R), np.nan)
    for r in decision["results"]:
        i, k = r["candidate"], reps.index(r["replicate"])
        total[i, k] = r["total"]
        trip[i, k] = r["shutdown"]
        horizon[i, k] = r["horizon_s"]
        trip_time[i, k] = r["time_to_shutdown"] if r["shutdown"] else r["horizon_s"]
        resolved[i, k] = np.nan if r["resolved"] is None else float(r["resolved"])
        mitigated[i, k] = np.nan if r["mitigated"] is None else float(r["mitigated"])
        if r["first_status"] in CENSORED:
            censored[i, k] = True
            started = r.get("first_started") or 0.0
            duration_time[i, k] = 0.0 if r["first_status"] == "accepted" else max(trip_time[i, k] - started, 0.0)
        else:
            success[i, k] = float(r["first_status"] == "succeeded")
            if r["first_duration"] is not None:
                duration[i, k] = duration_time[i, k] = r["first_duration"]
    J = total.mean(1)
    observe = np.array([c["family"] == "observe" for c in cands])
    voi = voi_of(J, observe)
    return {
        "J": J, "J_std": total.std(1), "regret": J - J.min(), "p_trip": trip.mean(1),
        "trip_event": trip, "trip_time": trip_time, "horizon": horizon,
        "p_resolved": _row_nanmean(resolved), "p_mitigated": _row_nanmean(mitigated),
        "p_success": _row_nanmean(success), "first_censored": censored,
        "duration": _row_nanmean(duration), "duration_time": duration_time, "duration_censored": censored.copy(),
        "voi": voi, "observe": observe,
    }


def _row_nanmean(x):
    finite = np.isfinite(x)
    return np.where(finite.any(1), np.where(finite, x, 0.0).sum(1) / np.maximum(finite.sum(1), 1), np.nan)


def in_family(episode):
    return belief.family_group((episode.get("scenario") or {}).get("incident")) is not None


def voi_of(J, observe):
    act = J[~observe]
    return np.where(observe, (act.min() if act.size else np.nan) - J, np.nan)


def belief_labels(decision, family):
    lab = candidate_labels(decision)
    out = {"J_belief": lab["J"].copy(), "p_trip_belief": lab["p_trip"].copy(), "voi_belief": lab["voi"].copy(),
           "posterior": 1.0 if not family else np.nan, "family": bool(family), "twin_ok": False,
           "belief_fallback": True, "twin_weight": 0.0 if not family else np.nan}
    twin = decision.get("twin")
    if not family or twin is None:
        return out
    w = belief.posterior(twin["belief"])
    out.update(posterior=float(w[0]), twin_weight=float(w[1]), twin_ok=twin.get("results") is not None)
    if not np.isfinite(w).all():
        return out
    if w[1] == 0.0:
        out["belief_fallback"] = False
        return out
    if not out["twin_ok"]:
        return out
    other = candidate_labels({"candidates": decision["candidates"], "results": twin["results"]})
    J = w[0] * lab["J"] + w[1] * other["J"]
    out.update(J_belief=J, p_trip_belief=w[0] * lab["p_trip"] + w[1] * other["p_trip"],
               voi_belief=voi_of(J, lab["observe"]), belief_fallback=False)
    return out


def baseline_index(decision, mode):
    plans = decision.get("baselines")
    if plans is None:
        return -1
    ids = [c["action_id"] for c in decision["candidates"]]
    plan = plans.get(mode, [])
    if plan:
        return ids.index(plan[0]) if plan[0] in ids else -1
    for i, c in enumerate(decision["candidates"]):
        if c["tool_id"] == "monitor" and float(c["args"].get("review_after_s", 0)) == 300.0:
            return i
    return -1


def _vocab_env():
    return TEPEnv(Scenario(seed=0, incident=Incident("link_fault", "sensor:reactor", 600.0)))


def build(root, out_dir, fractions=(0.6, 0.2, 0.2), tensor_config=None, provenance=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(root)
    registry = load_registry()
    vocab = T.build_vocab(_vocab_env(), tensor_config)
    by_split = {s: [e for e in episodes if split_of(e["seed"], fractions) == s and e["decisions"]] for s in SPLITS}
    train_items = [to_item(d) for e in by_split["train"] for d in e["items"]]
    scaler = T.fit_payload_scaler(train_items, vocab, tensor_config)
    meta = {"root": str(root), "fractions": list(fractions), "vocab": vocab, "scaler": scaler, "splits": {},
            "label_format": LABEL_FORMAT, "label_keys": list(LABEL_KEYS), "belief_label_keys": list(BELIEF_KEYS),
            "belief_decision_keys": list(BELIEF_DECISION_KEYS), "provenance": provenance}
    for split, eps in by_split.items():
        ids = T.ItemIds()
        ev_parts, feats, cand_dec, labels = [], [], [], {}
        dec_rows = {"tau": [], "episode_seed": [], "segment": [], "kind": [], "oracle_tts": [], "onset": [],
                    "explored": [], "n_candidates": [], "baseline_act_first": [], "baseline_investigate_first": [],
                    "backup_available": [], **{k: [] for k in BELIEF_DECISION_KEYS}}
        n_dec = n_excluded = n_results_excluded = n_later_excluded = 0
        for seg, ep in enumerate(eps):
            items = [to_item(d) for d in ep["items"]]
            taus = [d["tau"] for d in ep["decisions"]]
            known = [d["valid_evidence"] for d in ep["decisions"]]
            exclude = []
            for d in ep["decisions"]:
                own, unnamed = own_action_receipts(d, ep["items"])
                own_results, unnamed_results = own_action_tool_results(d, ep["items"])
                later_ids = later_action_ids(ep, d["tau"])
                if (unnamed or unnamed_results) and (submitted_action_ids(d) or later_ids):
                    raise ValueError(f"episode {ep['seed']}: items {(unnamed + unnamed_results)[:3]} name no "
                                     f"action_id, so the receipts and tool results of the behaviour's actions at "
                                     f"and after the decision at {d['tau']} cannot be left out; regenerate the "
                                     f"episodes with the current generator")
                later = items_about(later_ids, ep["items"])[0]
                exclude.append(own + own_results + later)
                n_excluded += len(own)
                n_results_excluded += len(own_results)
                n_later_excluded += len(later)
            ev_parts.append(T.evidence_tensors(items, taus, seg, vocab, tensor_config, scaler=scaler, ids=ids,
                                               known=known, exclude=exclude))
            for d in ep["decisions"]:
                actions = [registry.make_action(c["tool_id"], c["target_id"], c["args"], d["tau"], c["action_id"])
                           for c in d["candidates"]]
                f, columns = T.action_features(actions, registry, vocab)
                feats.append(f)
                cand_dec.append(np.full(len(actions), n_dec, np.int64))
                lab = candidate_labels(d)
                for k, v in lab.items():
                    labels.setdefault(k, []).append(v)
                bl = belief_labels(d, in_family(ep))
                for k in BELIEF_KEYS:
                    labels.setdefault(k, []).append(bl[k])
                for k in BELIEF_DECISION_KEYS:
                    dec_rows[k].append(bl[k])
                h = d["hidden"]
                dec_rows["tau"].append(d["tau"])
                dec_rows["episode_seed"].append(ep["seed"])
                dec_rows["segment"].append(seg)
                dec_rows["kind"].append(KINDS.index(h["kind"]) if h["onset_happened"] else 0)
                dec_rows["oracle_tts"].append(np.nan if h["oracle_time_to_shutdown"] is None
                                              else h["oracle_time_to_shutdown"])
                dec_rows["onset"].append(bool(h["onset_happened"]))
                dec_rows["explored"].append(bool(d["behaviour"]["explored"]))
                dec_rows["n_candidates"].append(len(actions))
                for mode in ("act_first", "investigate_first"):
                    dec_rows[f"baseline_{mode}"].append(baseline_index(d, mode))
                dec_rows["backup_available"].append(backup_available(ep))
                n_dec += 1
        if not ev_parts:
            continue
        arrays = {k: np.concatenate([p[k] for p in ev_parts]) for k in ev_parts[0] if k != "payload_dim"}
        arrays["payload_dim"] = ev_parts[0]["payload_dim"]
        arrays["cand_features"] = np.concatenate(feats).astype(np.float32)
        arrays["cand_decision"] = np.concatenate(cand_dec)
        R = max(v.shape[1] for v in labels["trip_event"])
        for k, parts in labels.items():
            if parts[0].ndim == 2:
                parts = [np.pad(p.astype(np.float64), ((0, 0), (0, R - p.shape[1])), constant_values=np.nan)
                         for p in parts]
            arrays[f"y_{k}"] = np.concatenate(parts)
        for k, v in dec_rows.items():
            arrays[f"dec_{k}"] = np.asarray(v)
        np.savez_compressed(out_dir / f"{split}.npz", **arrays)
        (out_dir / f"{split}_item_ids.json").write_text(json.dumps(ids.to_dict()), encoding="utf-8")
        meta["splits"][split] = {"episodes": len(eps), "decisions": n_dec, "candidates": int(arrays["cand_decision"].size),
                                 "seeds": [e["seed"] for e in eps], "action_columns": columns,
                                 "own_action_receipts_excluded": n_excluded,
                                 "own_action_tool_results_excluded": n_results_excluded,
                                 "later_action_items_excluded": n_later_excluded,
                                 "family_decisions": int(arrays["dec_family"].sum()),
                                 "twin_labelled_decisions": int(arrays["dec_twin_ok"].sum()),
                                 "belief_fallback_decisions": int(arrays["dec_belief_fallback"].sum())}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    return meta
