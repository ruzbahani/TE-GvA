import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_stats as DS
from cpsdefense.agents.playbook import Playbook
from cpsdefense.eval.metrics import paired_cluster_bootstrap_diff
from cpsdefense.experience import belief
from cpsdefense.experience.dataset import candidate_labels, in_family, split_of, to_item
from cpsdefense.paths import ROOT
from cpsdefense.utils.config import load_config
from cpsdefense.utils.runs import start_run

SETS = {"natural": ("data/processed/tep_experience/20260927-003712-v2-natural", "sim/incidents.yaml"),
        "stress": ("data/processed/tep_experience/20260927-085004-v2-stress", "sim/incidents_stress.yaml")}
DEV = ("train", "val")
FRACTIONS = (0.6, 0.2, 0.2)
TOL, EPS = DS.TOL, DS.EPS
LABELS = {"belief": "Jb", "hindsight": "J"}
ACTION_CLASSES = ("trace_link_pair", "verify_sensor_pair", "escalate", "switch_to_backup_pair", "restore_primary",
                  "repair_pair", "recalibrate_sensor", "set_production", "monitor", "other_observe", "other")
INFO_CLASS = {"trace_link_pair": "diagnostic", "verify_sensor_pair": "diagnostic", "escalate": "diagnostic",
              "switch_to_backup_pair": "backup_probe", "repair_pair": "cause_revealing_repair",
              "other_observe": "observe_no_pair_info", "monitor": "wait", "restore_primary": "response_no_pair_info",
              "recalibrate_sensor": "response_no_pair_info", "set_production": "response_no_pair_info",
              "other": "response_no_pair_info"}
INFO_CLASSES = ("diagnostic", "backup_probe", "cause_revealing_repair", "observe_no_pair_info",
                "response_no_pair_info", "wait")
DIAGNOSTIC = ("trace_link_pair", "verify_sensor_pair", "escalate")
DEFINITIONS = {"D0_observe_family": ("trace_link_pair", "verify_sensor_pair", "other_observe"),
               "D1_diagnostic": DIAGNOSTIC,
               "D2_diagnostic_or_backup_probe": (*DIAGNOSTIC, "switch_to_backup_pair"),
               "D3_with_cause_revealing_repairs": (*DIAGNOSTIC, "switch_to_backup_pair", "repair_pair")}
ONSET_SHIFTS = (-1, 1)
ONSET_GRID = tuple(range(-5, 6))
RANGE_SUBGROUPS = ("backup_", "index_", "since_onset_")

DEFINITION_TEXT = {
    "development_pool": "train + val raw episodes, split by seed (dataset.split_of, fractions 0.6/0.2/0.2) before a "
                        "file is opened (dataset_stats.load_episodes); any other split is refused",
    "ambiguous_decisions": "decisions of episodes whose incident is a link fault on sensor:<g> or an I/O-module "
                           "fault on <g> (dataset.in_family = dec_family); g = belief.family_group(incident)",
    "labels": "hindsight: J of the true cause (dataset.candidate_labels); belief: J_belief = P(c|E) J(a|c) + "
              "P(c'|E) J(a|c',h) (dataset.belief_labels); twin world: J(a|c',h), the twin's branch labels",
    "co_optimal": "J <= min J + TOL, TOL = 1e-9 (dataset_stats.TOL)",
    "class_co_optimal_share": "share of decisions in which the class holds a co-optimal candidate",
    "class_unique_argmin_share": "share of decisions whose argmin is a single candidate of the class",
    "tie_share": "share of decisions with more than one co-optimal candidate; cross_class_tie_share: whose "
                 "co-optimal candidates fall in more than one class",
    "action_classes": {
        "trace_link_pair": "trace_link on sensor:<g>", "verify_sensor_pair": "verify_sensor on source:<g>",
        "escalate": "escalate (target incident)", "switch_to_backup_pair": "switch_to_backup on sensor:<g>",
        "restore_primary": "restore_primary (any link)",
        "repair_pair": "repair_link on sensor:<g> or replace_io_module on source:<g>",
        "recalibrate_sensor": "recalibrate_sensor (any instrument)", "set_production": "set_production",
        "monitor": "monitor (300 s and 900 s)",
        "other_observe": "every other observe-family candidate: query_historian, trace_link / verify_sensor on "
                         "other targets",
        "other": "every other candidate: switch_to_backup, repair_link, replace_io_module on other targets"},
    "info_classes": {"diagnostic": "no physical effect, own result discriminates the pair (trace_link_pair, "
                                   "verify_sensor_pair, escalate)",
                     "backup_probe": "switch_to_backup_pair", "cause_revealing_repair": "repair_pair",
                     "observe_no_pair_info": "other_observe",
                     "response_no_pair_info": "restore_primary, recalibrate_sensor, set_production, other",
                     "wait": "monitor"},
    "strictly_best": "Eq. gap with class D in place of the observe family: min J over D < min J over the other "
                     "candidates - TOL; denominator (with_both): decisions with candidates in D and outside D. "
                     "D0 is the datasheet's gather-beats-act (dataset_stats: gather_belief, voi.gather_beats_act)",
    "definitions_D": {k: list(v) for k, v in DEFINITIONS.items()},
    "co_optimal_D": "D holds a co-optimal candidate; denominator: all ambiguous decisions of the subset",
    "belief_minus_hindsight": "strictly-best share under belief minus hindsight labels on the with_both decisions; "
                              "95% paired cluster bootstrap over episodes (eval.metrics.paired_cluster_bootstrap_diff)",
    "belief_regret": "J_belief(candidate) - min J_belief",
    "minus_historian": "paired label difference J(tool) - J(query_historian on source:<g>) in the true world "
                       "(hindsight labels) and the twin world (twin labels), common random numbers",
    "effort_difference": "(resource_cost(tool) - resource_cost(query_historian)) * compute.per_resource_unit * "
                         "weights.compute (configs/defense/tools.yaml, configs/sim/costs.yaml)",
    "within_effort": "|J(tool) - J(historian) - effort difference| <= EPS = 0.01 (dataset_stats.EPS)",
    "trace_share_of_observe": "decisions in which trace_link_pair is co-optimal / decisions in which an "
                              "observe-family candidate is co-optimal (belief labels)",
    "groupwide_staleness": "stale alerts valid at tau (decision valid_evidence) on at least max(2, stale_fraction "
                           "* group size) distinct channels of a sensor group (agents/playbook.Playbook.plan, "
                           "stale_fraction 0.5)",
    "settled": "the twin's deterministic evidence differed by tau (deterministic_identical false): the posterior "
               "is exactly 0 or 1",
    "onset_shift": "both worlds' assumed onsets moved by delta; link-world state records the true onset alone "
                   "explains are set to the state the assumed onset implies; evidence rebuilt with "
                   "belief.belief_evidence from the primary's stored items and the stored state timelines",
    "onset_uniform": "likelihood of each cause averaged uniformly over the shifts, then the posterior",
    "argmin_set_changed": "the set of co-optimal candidates under J_belief with the shifted posterior differs",
    "gather_flipped": "D0 strictly best (gather-or-act answer) differs between the stored and shifted posterior",
    "calibration": "P(link | E) of ambiguous decisions with a defined posterior, 10 equal bins "
                   "(dataset_stats.calibration_table): ECE (count-weighted |mean p - frequency|) and Brier",
    "calibration_null": "ECE with outcomes drawn as independent Bernoulli(p) from the same posteriors, n_null draws, "
                        "rng seed fixed (the same for every subgroup)",
    "since_onset_terciles": "post-onset decisions split at the 1/3 and 2/3 quantiles of tau - onset",
    "subgroup_ece_range": "over backup availability, decision index and since-onset tercile subgroups",
    "prior": "pair_prior = (pi_L w_L(g)/W_L) / (pi_L w_L(g)/W_L + pi_M w_M(g)/W_M): pi type shares, w the group's "
             "sensor-link and module weights, W_L all link weights (sensor and actuator), W_M all module weights",
}


def load_dev(root, splits=DEV):
    if not splits or not set(splits) <= set(DEV):
        raise ValueError(f"only the development splits {DEV} are read, got {tuple(splits)}")
    episodes = DS.load_episodes(root, tuple(splits), FRACTIONS)
    for s, eps in episodes.items():
        assert s in DEV and all(split_of(e["seed"], FRACTIONS) == s for e in eps)
    return episodes


def action_class(c, g):
    tool, target = c["tool_id"], c["target_id"]
    link, source = f"sensor:{g}", f"source:{g}"
    if (tool, target) == ("trace_link", link):
        return "trace_link_pair"
    if (tool, target) == ("verify_sensor", source):
        return "verify_sensor_pair"
    if c["family"] == "observe":
        return "other_observe"
    if (tool, target) == ("switch_to_backup", link):
        return "switch_to_backup_pair"
    if (tool, target) in (("repair_link", link), ("replace_io_module", source)):
        return "repair_pair"
    if tool in ("escalate", "restore_primary", "recalibrate_sensor", "set_production", "monitor"):
        return tool
    return "other"


def argmin_set(J):
    return frozenset(np.flatnonzero(J <= J.min() + TOL).tolist())


def strictly_best(J, member):
    if not member.any() or member.all():
        return None
    return bool(J[member].min() < J[~member].min() - TOL)


def decision_rows(episodes):
    rows = []
    for e in episodes:
        if not in_family(e):
            continue
        g = belief.family_group(e["scenario"]["incident"])
        for k, d in enumerate(e["decisions"]):
            lab = candidate_labels(d)
            bf = DS.belief_fields(e, d, lab)
            action = np.array([action_class(c, g) for c in d["candidates"]])
            observe = np.isin(action, DEFINITIONS["D0_observe_family"])
            if not np.array_equal(observe, lab["observe"]):
                raise AssertionError(f"episode {e['seed']}: the classes do not partition the observe family")
            twin = d.get("twin") or {}
            Jt = (candidate_labels({"candidates": d["candidates"], "results": twin["results"]})["J"]
                  if twin.get("results") is not None else None)
            row = {"episode": e, "decision": d, "seed": e["seed"], "index": k, "group": g,
                   "backup": DS.backup_group(e), "action": action, "info": np.array([INFO_CLASS[c] for c in action]),
                   "tools": [(c["tool_id"], c["target_id"]) for c in d["candidates"]],
                   "J": lab["J"], "Jb": bf["J_belief"], "Jt": Jt, "belief_used": bf["belief_used"],
                   "posterior": bf["posterior"], "p_link": bf["p_link"], "is_link": bf["is_link"],
                   "settled": bf["deterministic_identical"] is False,
                   "onset_happened": bool(d["hidden"]["onset_happened"]),
                   "since_onset_s": d["tau"] - d["hidden"]["onset_time"]}
            if bf["belief_used"] and bool(strictly_best(row["Jb"], observe)) != bool(bf["gather_belief"]):
                raise AssertionError(f"episode {e['seed']}: D0 does not reproduce dataset_stats gather_belief")
            rows.append(row)
    return rows


def row_subsets(rows):
    out = [("all", rows)]
    groups = {r["backup"] for r in rows}
    if len(groups) > 1:
        out += [(f"backup_{b}", [r for r in rows if r["backup"] == b]) for b in sorted(groups)]
    return out


def prior_components(incidents, groups):
    types, links = incidents["types"], incidents["link_fault"]["links"]
    modules = incidents["io_module_fault"]["groups"]
    pi_L = types[belief.LINK] / sum(types.values())
    pi_M = types[belief.MODULE] / sum(types.values())
    W_L, W_M = float(sum(links.values())), float(sum(modules.values()))
    per_group = {}
    for g in groups:
        w_L, w_M = float(links.get(f"sensor:{g}", 0.0)), float(modules.get(g, 0.0))
        prior = belief.pair_prior(incidents, g)[belief.LINK]
        formula = pi_L * w_L / W_L / (pi_L * w_L / W_L + pi_M * w_M / W_M)
        if not math.isclose(prior, formula, rel_tol=1e-12):
            raise AssertionError(f"group {g}: pair_prior {prior} differs from the formula {formula}")
        per_group[g] = {"w_L": w_L, "w_M": w_M, "prior_link": prior}
    return {"pi_L": pi_L, "pi_M": pi_M, "W_L": W_L, "W_M": W_M,
            "W_L_sensor": float(sum(v for k, v in links.items() if k.startswith("sensor:"))),
            "W_L_actuator": float(sum(v for k, v in links.items() if k.startswith("actuator:"))),
            "groups": per_group}


def prior_stats(incidents, groups, episodes, rows):
    out = prior_components(incidents, groups)
    priors = {round(v["prior_link"], 12) for v in out["groups"].values()}
    fam = [e for e in episodes if in_family(e)]
    n_link = sum(e["scenario"]["incident"]["kind"] == belief.LINK for e in fam)
    p0 = priors.pop() if len(priors) == 1 else None
    out["same_for_every_group"] = p0 is not None
    out["prior_link"] = p0
    out["empirical_link_share"] = {"link_episodes": n_link, "ambiguous_episodes": len(fam),
                                   "share": n_link / len(fam) if fam else None,
                                   "binomial_se_at_prior": math.sqrt(p0 * (1 - p0) / len(fam)) if p0 and fam else None,
                                   "decision_share": float(np.mean([r["is_link"] for r in rows])) if rows else None}
    stored = [(r["decision"].get("twin") or {}).get("belief") for r in rows]
    out["stored_prior_matches"] = all(
        ev is None or np.allclose(ev["prior"], [belief.pair_prior(incidents, ev["group"])[c] for c in ev["causes"]],
                                  rtol=0, atol=1e-12) for ev in stored)
    return out


def class_table(rows, key, labels, classes):
    n = len(rows)
    co, unique, avail = collections.Counter(), collections.Counter(), collections.Counter()
    ties = spans = 0
    for r in rows:
        best = argmin_set(r[LABELS[labels]])
        held = {r[key][i] for i in best}
        co.update(held)
        avail.update(set(r[key].tolist()))
        if len(best) == 1:
            unique[r[key][next(iter(best))]] += 1
        ties += len(best) > 1
        spans += len(held) > 1
    return {"n": n, "tie_share": ties / n, "cross_class_tie_share": spans / n,
            "classes": {c: {"co_optimal": co[c] / n, "unique_argmin": unique[c] / n, "candidate": avail[c] / n}
                        for c in classes}}


def definition_table(rows, n_boot, seed):
    out = {}
    for name, members in DEFINITIONS.items():
        strict = {k: [] for k in LABELS}
        co = collections.Counter()
        clusters = []
        for r in rows:
            member = np.isin(r["action"], members)
            for labels, key in LABELS.items():
                co[labels] += bool(member.any() and r[key][member].min() <= r[key].min() + TOL)
                s = strictly_best(r[key], member)
                if s is not None:
                    strict[labels].append(float(s))
            if member.any() and not member.all():
                clusters.append(r["seed"])
        entry = {"members": list(members), "with_both": len(clusters), "decisions": len(rows)}
        for labels in LABELS:
            x = np.array(strict[labels])
            entry[labels] = {"strictly_best": {"count": int(x.sum()), "share": float(x.mean()) if x.size else None},
                             "co_optimal": {"count": co[labels], "share": co[labels] / len(rows)}}
        if clusters:
            entry["belief_minus_hindsight"] = paired_cluster_bootstrap_diff(
                np.mean, [np.array(strict["belief"])], [np.array(strict["hindsight"])], np.array(clusters),
                n_boot=n_boot, seed=seed)
        out[name] = entry
    return out


def effort_differences():
    tools, costs = load_config("defense/tools.yaml")["tools"], load_config("sim/costs.yaml")
    unit = float(costs["compute"]["per_resource_unit"]) * float(costs["weights"]["compute"])
    return {t: (float(tools[t]["resource_cost"]) - float(tools["query_historian"]["resource_cost"])) * unit
            for t in ("escalate", "verify_sensor")}


def information_tools(rows, effort):
    tools = {"escalate": "escalate", "verify_sensor_pair": "verify_sensor", "trace_link_pair": "trace_link"}
    out = {t: {"candidate_decisions": 0, "co_optimal": {k: 0 for k in LABELS}, "regret": []} for t in tools}
    diff = {t: {"true_world": [], "twin_world": [], "both_within": []} for t in tools}
    no_trace = trace_co = observe_co = 0
    for r in rows:
        idx = {t: np.flatnonzero(r["action"] == t) for t in tools}
        hist = [i for i, tt in enumerate(r["tools"]) if tt == ("query_historian", f"source:{r['group']}")]
        no_trace += idx["trace_link_pair"].size == 0
        best_b = argmin_set(r["Jb"])
        observe_co += any(r["action"][i] in DEFINITIONS["D0_observe_family"] for i in best_b)
        trace_co += any(r["action"][i] == "trace_link_pair" for i in best_b)
        for t, ix in idx.items():
            if not ix.size:
                continue
            out[t]["candidate_decisions"] += 1
            for labels, key in LABELS.items():
                J = r[key]
                out[t]["co_optimal"][labels] += bool(J[ix].min() <= J.min() + TOL)
            out[t]["regret"].append(r["Jb"][ix].min() - r["Jb"].min())
            if hist:
                i, h = ix[0], hist[0]
                diff[t]["true_world"].append(r["J"][i] - r["J"][h])
                if r["Jt"] is not None:
                    diff[t]["twin_world"].append(r["Jt"][i] - r["Jt"][h])
                    if tools[t] in effort:
                        diff[t]["both_within"].append(abs(r["J"][i] - r["J"][h] - effort[tools[t]]) <= EPS
                                                      and abs(r["Jt"][i] - r["Jt"][h] - effort[tools[t]]) <= EPS)
    for t, tool in tools.items():
        out[t]["belief_regret"] = DS.summary(out[t].pop("regret"))
        mh = {w: DS.summary(diff[t][w]) for w in ("true_world", "twin_world")}
        if tool in effort:
            mh["effort_difference"] = effort[tool]
            for w in ("true_world", "twin_world"):
                x = np.asarray(diff[t][w], float)
                mh[w]["within_effort_share"] = float(np.mean(np.abs(x - effort[tool]) <= EPS)) if x.size else None
            mh["within_effort_both_worlds_share"] = DS.share(diff[t]["both_within"])
        out[t]["minus_historian"] = mh
    n = len(rows)
    out["no_trace_candidate"] = {"count": int(no_trace), "decisions": n, "share": no_trace / n}
    out["trace_share_of_observe_co_optimal"] = {"trace_co_optimal": int(trace_co), "observe_co_optimal":
                                                int(observe_co), "share": trace_co / observe_co if observe_co else None}
    return out


def action_stats(rows, effort, n_boot, seed):
    return {"decisions": len(rows), "episodes": len({r["seed"] for r in rows}),
            "classes": {k: class_table(rows, "action", k, ACTION_CLASSES) for k in LABELS},
            "info_classes": {k: class_table(rows, "info", k, INFO_CLASSES) for k in LABELS},
            "definitions": definition_table(rows, n_boot, seed),
            "information_tools": information_tools(rows, effort)}


def groupwide_stale(items_by_id, valid_ids, sizes, fraction):
    stale = collections.defaultdict(set)
    for eid in valid_ids:
        item = items_by_id.get(eid)
        if item is not None and item["kind"] == "alert" and item["payload"].get("category") == "stale":
            stale[item["subject_id"].split(":", 1)[1]].add(item["payload"]["measurement"])
    return {g for g, ms in stale.items() if len(ms) >= max(2, fraction * sizes[g])}


def pair_conditioning(episodes, sizes, fraction):
    n = with_signal = in_pair = fam = fam_signal = pre = 0
    outside = collections.Counter()
    minutes = []
    for e in episodes:
        items = {i["evidence_id"]: i for i in e["items"]}
        g = belief.family_group(e["scenario"]["incident"])
        for d in e["decisions"]:
            held = groupwide_stale(items, d["valid_evidence"], sizes, fraction)
            n += 1
            if held:
                with_signal += 1
                if g is not None and held == {g}:
                    in_pair += 1
                else:
                    outside[f"{d['hidden']['cause']} | groups {sorted(held)}"] += 1
            if g is None:
                continue
            fam += 1
            if g in held:
                fam_signal += 1
            elif d["hidden"]["onset_happened"]:
                minutes.append((d["tau"] - d["hidden"]["onset_time"]) / 60.0)
            else:
                pre += 1
    return {"decisions": n, "with_groupwide_signal": with_signal, "signal_only_on_own_pair_group": in_pair,
            "signal_outside_pair": sum(outside.values()), "signal_outside_pair_causes": dict(outside.most_common()),
            "ambiguous": {"decisions": fam, "with_signal_on_g": fam_signal, "share": fam_signal / fam if fam else None,
                          "without_signal": fam - fam_signal, "without_signal_pre_onset": pre,
                          "without_signal_post_onset_minutes": DS.summary(minutes)}}


def scenario_onset(scenario):
    return float(scenario["warmup_s"]) + float(scenario["incident"]["onset_s"])


def shifted_world(cause, onset, dt, states, delta):
    o2 = onset + delta
    recs = []
    for t, mode, faulted in states:
        if cause == belief.LINK and delta > 0 and faulted and t - dt < o2:
            mode, faulted = ("up" if mode == "down" else mode), False
        elif cause == belief.LINK and delta < 0 and not faulted and o2 <= t - dt < onset:
            mode, faulted = ("down" if mode == "up" else mode), True
        recs.append((t, mode, faulted))
    return belief.world(cause, o2, dt, recs)


def onset_items(episode, tau, group):
    link = f"sensor:{group}"
    seen = [i for i in episode["items"] if i["arrival_time"] <= tau]
    ref = max((i for i in seen if i["kind"] in ("window_summary", "link_health")),
              key=lambda i: belief.window_index(i["evidence_id"]))
    keep = [i for i in seen if i is not ref and ((i["kind"] in ("link_health", "trace_link") and i["subject_id"] == link)
                                                 or i["kind"] == "operator_report")]
    return [to_item(i) for i in (ref, *keep)]


def onset_evidence(episode, stored, items, dt, delta=0.0):
    tau = stored["tau"]
    twin = episode["twin"]
    onsets = (scenario_onset(episode["scenario"]), scenario_onset(twin["scenario"]))
    timelines = (twin["primary_states"], twin["states"])
    worlds = [shifted_world(c, o, dt, [s for s in st if s[0] <= tau], delta)
              for c, o, st in zip(stored["causes"], onsets, timelines)]
    first = None if stored["deterministic_identical"] else tuple(stored["first_difference"])
    return belief.belief_evidence(items, tau, stored["group"], worlds, stored["params"], stored["prior"], first)


def mean_loglik(lls):
    L = np.asarray(lls, np.float64)
    finite = np.isfinite(L)
    m = np.where(finite.any(0), np.where(finite, L, -np.inf).max(0), 0.0)
    with np.errstate(divide="ignore"):
        return m + np.log(np.mean(np.exp(L - m), 0))


def posterior_from_loglik(prior, ll):
    with np.errstate(divide="ignore"):
        logs = np.log(np.asarray(prior, np.float64)) + np.asarray(ll, np.float64)
    if not np.isfinite(logs).any():
        return np.array([np.nan, np.nan])
    w = np.exp(logs - logs[np.isfinite(logs)].max())
    return w / w.sum()


def onset_grids(window_s, dt):
    out = {f"shift_{k:+d}_window": [k * window_s] for k in ONSET_SHIFTS}
    out["uniform_pm5_windows"] = [k * window_s for k in ONSET_GRID]
    n = int(round(max(ONSET_GRID) * window_s / dt))
    out["uniform_pm5_windows_control_steps"] = [k * dt for k in range(-n, n + 1)]
    return out


def shift_summary(dpi, changed, flipped):
    dpi = np.asarray(dpi, np.float64)
    n = dpi.size
    return {"decisions": n, "mean_abs_dpi": float(dpi.mean()), "q95_abs_dpi": float(np.percentile(dpi, 95)),
            "max_abs_dpi": float(dpi.max()), "argmin_set_changed": {"count": int(sum(changed)), "share": sum(changed) / n},
            "gather_flipped": {"count": int(sum(flipped)), "share": sum(flipped) / n}}


def onset_sensitivity(rows, dt):
    sel = [r for r in rows if r["belief_used"] and not r["settled"] and r["Jt"] is not None]
    if not sel:
        return {"decisions": 0}
    window_s = sel[0]["decision"]["twin"]["belief"]["params"]["window_s"]
    grids = onset_grids(window_s, dt)
    shifts = sorted({x for v in grids.values() for x in v})
    res = {name: ([], [], []) for name in grids}
    mismatch, post_diff, offsets, n_windows = 0, 0.0, [], []
    for r in sel:
        e, d = r["episode"], r["decision"]
        stored = d["twin"]["belief"]
        if stored["params"]["window_s"] != window_s:
            raise ValueError(f"episode {e['seed']}: telemetry window {stored['params']['window_s']} != {window_s}")
        items = onset_items(e, d["tau"], stored["group"])
        ev0 = onset_evidence(e, stored, items, dt)
        mismatch += any(ev0[k] != stored[k] for k in ("windows", "traces", "operator"))
        p0 = belief.posterior(stored)
        post_diff = max(post_diff, float(abs(belief.posterior(ev0)[0] - p0[0])), abs(float(p0[0]) - r["posterior"]))
        if stored["windows"]:
            offsets.append(stored["windows"][0]["end"] - scenario_onset(e["scenario"]))
        n_windows.append(len(stored["windows"]))
        observe = np.isin(r["action"], DEFINITIONS["D0_observe_family"])
        best0, gather0 = argmin_set(r["Jb"]), strictly_best(r["Jb"], observe)
        ll = {x: belief.log_likelihoods(onset_evidence(e, stored, items, dt, x)) for x in shifts}
        for name, grid in grids.items():
            p = posterior_from_loglik(stored["prior"], mean_loglik([ll[x] for x in grid]))
            Jb = p[0] * r["J"] + p[1] * r["Jt"]
            res[name][0].append(abs(p[0] - p0[0]))
            res[name][1].append(argmin_set(Jb) != best0)
            res[name][2].append(strictly_best(Jb, observe) != gather0)
    out = {"decisions": len(sel), "window_s": window_s, "control_step_s": dt,
           "reproduction": {"evidence_mismatches": int(mismatch), "posterior_max_abs_diff": post_diff},
           "first_window_end_minus_onset_s": {"min": min(offsets), "max": max(offsets)} if offsets else None,
           "windows_in_likelihood": DS.summary(n_windows)}
    out.update({name: shift_summary(*v) for name, v in res.items()})
    return out


def null_ece(p, n_draws, seed):
    rng = np.random.default_rng(seed)
    p = np.asarray(p, np.float64)
    vals = np.array([DS.calibration_table(p, rng.random(p.size) < p)["ece"] for _ in range(n_draws)])
    return float(vals.mean()), float(np.quantile(vals, 0.95))


def calibration_subgroups(rows):
    since = np.array([r["since_onset_s"] for r in rows], np.float64)
    post = np.array([r["onset_happened"] for r in rows], bool)
    settled = np.array([r["settled"] for r in rows], bool)
    backup = np.array([r["backup"] for r in rows])
    index = np.array([r["index"] for r in rows])
    out = [("all", np.ones(len(rows), bool)), ("not_settled", ~settled), ("settled", settled)]
    if len(set(backup.tolist())) > 1:
        out += [(f"backup_{b}", backup == b) for b in sorted(set(backup.tolist()))]
    out += [(f"index_{k}", index == k) for k in sorted(set(index.tolist()))]
    lo, hi = np.quantile(since[post], [1 / 3, 2 / 3]) if post.any() else (np.nan, np.nan)
    out += [("since_onset_t1", post & (since <= lo)), ("since_onset_t2", post & (since > lo) & (since <= hi)),
            ("since_onset_t3", post & (since > hi)), ("pre_onset", ~post)]
    return [(name, m) for name, m in out if m.any()], [float(lo / 60), float(hi / 60)]


def ece_range(subgroups):
    ranged = {k: v["ece"] for k, v in subgroups.items() if k.split(":")[-1].startswith(RANGE_SUBGROUPS)}
    if not ranged:
        return None
    lo, hi = min(ranged, key=ranged.get), max(ranged, key=ranged.get)
    return {"min": ranged[lo], "min_subgroup": lo, "max": ranged[hi], "max_subgroup": hi, "subgroups": sorted(ranged),
            "all_below_null_q95": all(subgroups[k]["below_null_q95"] for k in ranged)}


def calibration(rows, prior_link, n_null, seed):
    rows = [r for r in rows if np.isfinite(r["p_link"])]
    p = np.array([r["p_link"] for r in rows], np.float64)
    y = np.array([r["is_link"] for r in rows], np.float64)
    seeds = np.array([r["seed"] for r in rows])
    groups, edges = calibration_subgroups(rows)
    out = {"since_onset_tercile_edges_min": edges,
           "prior_only_brier": float(np.mean((prior_link - y) ** 2)) if prior_link is not None else None,
           "subgroups": {}}
    for name, m in groups:
        t = DS.calibration_table(p[m], y[m])
        mean, q95 = null_ece(p[m], n_null, seed)
        out["subgroups"][name] = {"decisions": t["n"], "episodes": int(np.unique(seeds[m]).size), "ece": t["ece"],
                                  "brier": t["brier"], "null_ece_mean": mean, "null_ece_q95": q95,
                                  "below_null_q95": t["ece"] < q95}
        if name in ("all", "not_settled"):
            out["subgroups"][name]["bins"] = t["bins"]
    out["subgroup_ece_range"] = ece_range(out["subgroups"])
    return out


def analyse(name, splits, args, log):
    t0 = time.perf_counter()
    root, incidents_path = SETS[name]
    episodes = load_dev(ROOT / root, splits)
    dev = [e for s in splits for e in episodes[s]]
    tep = load_config("sim/tep.yaml")
    sizes = {g: len(v) for g, v in tep["sensor_links"].items()}
    incidents = load_config(incidents_path)
    rows = decision_rows(dev)
    used = [r for r in rows if r["belief_used"]]
    n_dec = sum(len(e["decisions"]) for e in dev)
    log(f"{name}: {len(dev)} episodes, {n_dec} decisions, {len(rows)} ambiguous ({len(used)} with belief labels); "
        f"{time.perf_counter() - t0:.0f}s")
    out = {"root": root, "incidents": incidents_path,
           "episodes": {s: len(v) for s, v in episodes.items()},
           "decisions": {s: sum(len(e["decisions"]) for e in v) for s, v in episodes.items()},
           "ambiguous": {"episodes": len({r["seed"] for r in rows}), "decisions": len(rows),
                         "share_of_decisions": len(rows) / n_dec, "belief_decisions": len(used),
                         "belief_fallback": len(rows) - len(used)}}
    out["prior"] = prior_stats(incidents, list(sizes), dev, rows)
    effort = effort_differences()
    out["actions"] = {sub: action_stats(rr, effort, args.n_boot, args.seed) for sub, rr in row_subsets(used)}
    log(f"{name}: actions done; {time.perf_counter() - t0:.0f}s")
    out["pair_conditioning"] = pair_conditioning(dev, sizes, Playbook().stale_fraction)
    out["onset"] = onset_sensitivity(rows, float(tep["control_interval_s"]))
    log(f"{name}: onset done; {time.perf_counter() - t0:.0f}s")
    out["calibration"] = calibration(rows, out["prior"]["prior_link"], args.n_null, args.seed)
    out["seconds"] = time.perf_counter() - t0
    return out


def rounded(obj, digits=4):
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return round(x, digits) if isinstance(x, float) else x
    return walk(DS.jsonable(obj))


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def num(summary, key, digits=2):
    return f"{summary[key]:.{digits}f}" if summary.get(key) is not None else "-"


def md_classes(actions):
    subs = list(actions)
    head = " | ".join(f"{s} (n={actions[s]['decisions']}) {k}" for s in subs for k in LABELS)
    lines = []
    for title, key, classes in (("First-action class", "classes", ACTION_CLASSES),
                                ("Informational class", "info_classes", INFO_CLASSES)):
        lines += [f"### {title}: % co-optimal / % unique argmin", "", f"| class | {head} |",
                  "|---" * (2 * len(subs) + 1) + "|"]
        for c in classes:
            cells = [f"{pct(v['co_optimal'])} / {pct(v['unique_argmin'])}"
                     for s in subs for v in (actions[s][key][k]["classes"][c] for k in LABELS)]
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines += ["| ties / cross-class ties | " + " | ".join(
            f"{pct(actions[s][key][k]['tie_share'])} / {pct(actions[s][key][k]['cross_class_tie_share'])}"
            for s in subs for k in LABELS) + " |", ""]
    return lines


def md_definitions(actions):
    lines = ["### Strictly best: Eq. gap with D in place of the observe family (%)", "",
             "| subset | D | with both | belief | hindsight | belief - hindsight [95% CI] | co-optimal belief | "
             "co-optimal hindsight |", "|---" * 8 + "|"]
    for s, a in actions.items():
        for name, v in a["definitions"].items():
            ci = v["belief_minus_hindsight"]
            lines.append(f"| {s} | {name} | {v['with_both']} | {pct(v['belief']['strictly_best']['share'])} | "
                         f"{pct(v['hindsight']['strictly_best']['share'])} | {100 * ci['point']:+.1f} "
                         f"[{100 * ci['lo']:+.1f}, {100 * ci['hi']:+.1f}] | {pct(v['belief']['co_optimal']['share'])} "
                         f"| {pct(v['hindsight']['co_optimal']['share'])} |")
    return lines + [""]


def md_tools(actions):
    lines = ["### Escalate and the field check on the pair", ""]
    for s, a in actions.items():
        it = a["information_tools"]
        lines.append(f"- {s}: no trace_link_pair candidate in {it['no_trace_candidate']['count']} decisions "
                     f"({pct(it['no_trace_candidate']['share'])}%); trace_link_pair is co-optimal in "
                     f"{pct(it['trace_share_of_observe_co_optimal']['share'])}% of the decisions with a co-optimal "
                     "observe candidate (belief).")
        for t in ("escalate", "verify_sensor_pair"):
            v = it[t]
            mh = v["minus_historian"]
            worlds = " / ".join(f"{num(mh[w], 'median')} ({pct(mh[w].get('within_effort_share'))}% within {EPS} of "
                                f"{mh['effort_difference']:.2f})" for w in ("true_world", "twin_world"))
            lines.append(f"  - {t}: co-optimal in {v['co_optimal']['belief']} (belief) / {v['co_optimal']['hindsight']}"
                         f" (hindsight) of {v['candidate_decisions']} decisions; belief regret min "
                         f"{num(v['belief_regret'], 'min')}, median {num(v['belief_regret'], 'median')}; "
                         f"J - J(historian) median, true / twin world: {worlds}")
    return lines + [""]


def md_pair(pc):
    pa = pc["ambiguous"]
    mins = pa["without_signal_post_onset_minutes"]
    return ["### Pair conditioning", "",
            f"- {pc['with_groupwide_signal']} of {pc['decisions']} decisions have a group-wide staleness signal at tau; "
            f"{pc['signal_outside_pair']} of them lie outside the pair on that group.",
            f"- Ambiguous decisions with the signal on g: {pa['with_signal_on_g']} of {pa['decisions']} "
            f"({pct(pa['share'])}%); without: {pa['without_signal']} ({pa['without_signal_pre_onset']} pre-onset; "
            f"post-onset median {num(mins, 'median')} min, max {num(mins, 'max')} min after the onset).", ""]


def md_onset(on):
    if not on["decisions"]:
        return []
    rep = on["reproduction"]
    lines = [f"### Onset sensitivity: {on['decisions']} ambiguous decisions not settled (rebuilt evidence: "
             f"{rep['evidence_mismatches']} mismatches, max |posterior difference| {rep['posterior_max_abs_diff']:.1e})",
             "", "| assumed onset | mean abs dpi | q95 | max | % argmin set changed | % gather flipped |",
             "|---" * 6 + "|"]
    for k, v in on.items():
        if isinstance(v, dict) and "mean_abs_dpi" in v:
            lines.append(f"| {k} | {v['mean_abs_dpi']:.4f} | {v['q95_abs_dpi']:.4f} | {v['max_abs_dpi']:.3f} | "
                         f"{pct(v['argmin_set_changed']['share'])} | {pct(v['gather_flipped']['share'])} |")
    return lines + [""]


def md_calibration(cal):
    brier0 = "-" if cal["prior_only_brier"] is None else f"{cal['prior_only_brier']:.3f}"
    edges = cal["since_onset_tercile_edges_min"]
    lines = [f"### Calibration of P(link | E) (prior-only Brier {brier0}; since-onset tercile edges {edges[0]:.1f} / "
             f"{edges[1]:.1f} min)", "", "| subgroup | n | episodes | ECE | Brier | null ECE mean | null ECE q95 |",
             "|---" * 7 + "|"]
    for k, v in cal["subgroups"].items():
        lines.append(f"| {k} | {v['decisions']} | {v['episodes']} | {v['ece']:.4f} | {v['brier']:.4f} | "
                     f"{v['null_ece_mean']:.4f} | {v['null_ece_q95']:.4f} |")
    rng = cal.get("subgroup_ece_range")
    if rng:
        lines += ["", f"Subgroup ECE from {rng['min']:.4f} ({rng['min_subgroup']}) to {rng['max']:.4f} "
                      f"({rng['max_subgroup']}); all below their null q95: {rng['all_below_null_q95']}."]
    return lines + [""]


def summary_markdown(out, splits):
    lines = [f"# Label diagnostics, development pool ({' + '.join(splits)})", "",
             "Definitions in results.json (`definitions`).", ""]
    for name, r in out["sets"].items():
        amb, pr = r["ambiguous"], r["prior"]
        emp = pr["empirical_link_share"]
        per_split = ", ".join(f"{s} {n}" for s, n in r["decisions"].items())
        priors = ", ".join(sorted({f"{v['prior_link']:.4f}" for v in pr["groups"].values()}))
        lines += [f"## {name}: {sum(r['episodes'].values())} episodes, {sum(r['decisions'].values())} decisions "
                  f"({per_split}); {amb['decisions']} ambiguous ({pct(amb['share_of_decisions'])}%) from "
                  f"{amb['episodes']} episodes", "",
                  f"Prior P(link | pair) {priors} (pi_L {pr['pi_L']:.3f}, pi_M {pr['pi_M']:.3f}, W_L {pr['W_L']:g} = "
                  f"{pr['W_L_sensor']:g} sensor + {pr['W_L_actuator']:g} actuator, W_M {pr['W_M']:g}); link faults in "
                  f"{emp['link_episodes']} of {emp['ambiguous_episodes']} ambiguous episodes ({emp['share']:.3f}).", ""]
        lines += md_classes(r["actions"]) + md_definitions(r["actions"]) + md_tools(r["actions"])
        lines += md_pair(r["pair_conditioning"]) + md_onset(r["onset"]) + md_calibration(r["calibration"])
    rng = out.get("subgroup_ece_range_across_sets")
    if rng:
        lines += [f"Across sets, subgroup ECE from {rng['min']:.4f} ({rng['min_subgroup']}) to {rng['max']:.4f} "
                  f"({rng['max_subgroup']}).", ""]
    return "\n".join(lines)


def headline(r):
    d0 = r["actions"]["all"]["definitions"]["D0_observe_family"]
    cal = r["calibration"]["subgroups"]
    return {"gather_strictly_best_belief": d0["belief"]["strictly_best"]["share"],
            "gather_strictly_best_hindsight": d0["hindsight"]["strictly_best"]["share"],
            "ece": cal["all"]["ece"], "brier": cal["all"]["brier"], "ece_not_settled": cal["not_settled"]["ece"],
            "signal_share": r["pair_conditioning"]["ambiguous"]["share"],
            "onset_shift_+1_mean_abs_dpi": r["onset"].get("shift_+1_window", {}).get("mean_abs_dpi")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    parser.add_argument("--splits", nargs="+", choices=DEV, default=list(DEV))
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="dev")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = {"sets": {s: {"episodes": SETS[s][0], "incidents": SETS[s][1]} for s in args.sets},
           "splits": args.splits, "fractions": list(FRACTIONS), "tol": TOL, "eps": EPS, "n_null": args.n_null,
           "n_boot": args.n_boot, "onset_shifts_windows": list(ONSET_SHIFTS), "onset_grid_windows": list(ONSET_GRID)}
    run = start_run("label_diagnostics", cfg, seed=args.seed, name=args.name)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    results = {s: analyse(s, args.splits, args, log) for s in args.sets}
    across = ece_range({f"{s}:{k}": v for s, r in results.items() for k, v in r["calibration"]["subgroups"].items()})
    out = DS.jsonable({"definitions": DEFINITION_TEXT, "sets": results, "subgroup_ece_range_across_sets": across,
                       "log": lines, "wall_s": time.perf_counter() - t0})
    run.save_json("results.json", rounded(out))
    md = summary_markdown(out, args.splits)
    (run.dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    run.finish(rounded({s: headline(r) for s, r in out["sets"].items()}))
    print(f"-> {run.dir}")

if __name__ == "__main__":
    main()
