import argparse
import copy
import dataclasses
import json
import math
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.agents.playbook import MODES, Playbook, citable
from cpsdefense.experience import belief
from cpsdefense.experience.dataset import split_of
from cpsdefense.experience.rollout import as_record, branch_labels, relevant_candidates, relevant_evidence
from cpsdefense.paths import CONFIG_DIR, PROCESSED_DIR, ROOT, ensure_dir
from cpsdefense.sim.incidents import sample_scenario
from cpsdefense.sim.tep_env import TEPEnv
from cpsdefense.sim.tep_plant import TEPPlant
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_file
from cpsdefense.utils.runs import start_run

SEALED = "sealed"
SHARED_KINDS = ("alert", "window_summary")


def action_record(a):
    return {"action_id": a.action_id, "tool_id": a.tool_id, "tool_version": a.tool_version,
            "family": a.family.value, "target_id": a.target_id, "args": dict(a.bounded_arguments),
            "expected_duration": a.expected_duration, "duration_q90": a.duration_q90,
            "resource_limit": a.resource_limit, "side_effect_class": a.side_effect_class.value,
            "expiry": a.expiry, "compensation": a.compensation_or_recovery_id}


def item_record(i):
    payload = {k: (np.asarray(v) if k == "values" else v) for k, v in i.payload.items()}
    return {"evidence_id": i.evidence_id, "source": i.source, "kind": i.kind, "subject_id": i.subject_id,
            "event_time": i.event_time, "arrival_time": i.arrival_time, "valid_until": i.valid_until,
            "payload": payload, "provenance": i.provenance, "quality": i.quality, "supersedes": list(i.supersedes)}


def stored_items(log, taus, window):
    if not taus:
        return []
    lo, hi = min(taus) - window[0], max(taus) + window[1]
    return [{**item_record(log.get(eid)), "retracted_at": log.retracted_at(eid)} for eid in log
            if log.get(eid).arrival_time <= hi
            and (lo <= log.get(eid).event_time or any(log.valid(eid, t) for t in taus))]


def config_path(path):
    path = Path(path)
    return path if path.is_absolute() or path.exists() else CONFIG_DIR / path


def behaviour_step(behaviour, env, cands, cited, epsilon, rng):
    playbook_plan = copy.deepcopy(behaviour).plan(env)
    explore = bool(rng.random() < epsilon)
    if explore:
        pick = cands[int(rng.integers(len(cands)))]
        plan = [(pick, list(cited))]
        behaviour.note_submitted(pick)
    else:
        plan = behaviour.plan(env)
    in_playbook = {a.action_id for a, _ in playbook_plan}
    cand_ids = {a.action_id for a in cands}
    probs = [(epsilon / len(cands) if a.action_id in cand_ids else 0.0)
             + ((1.0 - epsilon) if a.action_id in in_playbook else 0.0) for a, _ in plan]
    return plan, {"mode": behaviour.mode, "epsilon": epsilon, "explored": explore,
                  "actions": [a.action_id for a, _ in plan], "probabilities": probs,
                  "empty_plan_probability": 0.0 if playbook_plan else 1.0 - epsilon}


def submit_plan(env, plan, decision_id):
    submitted = []
    for a, ev in plan:
        env.submit(env.make_bundle(a, ev, decision_id))
        submitted.append(a.action_id)
    return submitted


def world_state(env, group):
    h = env.hidden_state()
    link = f"sensor:{group}"
    return float(env.now), h["link_mode"][link], link in h["faulted_links"]


def playbook_memory(playbook):
    return {"traces": {link: bool(ok) for link, (ok, _) in playbook.traces.items()},
            "link_loss": sorted(link for link in playbook.link_loss if link.startswith("actuator:"))}


def observed_items(log, tau, size=None):
    ids = list(log)[:size] if size is not None else list(log)
    return [log.get(e) for e in ids if log.get(e).arrival_time <= tau]


def twin_action(env, record, candidates=None):
    by_id = {a.action_id: a for a in (env.candidates() if candidates is None else candidates)}
    if record["action_id"] in by_id:
        return by_id[record["action_id"]], True
    return env.registry.make_action(record["tool_id"], record["target_id"], record["args"], env.now,
                                    record["action_id"]), False


def target_group(env, target_id):
    if target_id.startswith(("sensor:", "actuator:", "source:")):
        group = target_id.split(":", 1)[1]
    else:
        group = env.world.group_of.get(target_id)
    return group if group in env.plant.sensor_links else None


def shared_evidence(env, targets):
    items = [env.log.get(e) for e in citable(env, env.valid_evidence())]
    items = sorted((i for i in items if i.kind in SHARED_KINDS), key=lambda i: i.arrival_time, reverse=True)
    groups = dict.fromkeys(g for g in (target_group(env, t) for t in targets) if g is not None)
    latest = [next((i.evidence_id for i in items if i.subject_id == f"source:{g}"), None) for g in groups]
    return [e for e in latest if e is not None] or [i.evidence_id for i in items[:1]]


def replay_citation(env, cited, targets):
    kept = citable(env, cited)
    return kept if kept or not cited else shared_evidence(env, targets)


def advance_twin(env, playbook, t, states, group):
    half = env.plant.dt_s / 2
    while not env.done and env.now < t - half:
        obs = env.advance_to_decision(max_s=t - env.now - half)
        playbook.update(obs, env)
        states.append(world_state(env, group))
    return not env.done and abs(env.now - t) < half


def replay_submissions(env, playbook, subs):
    out = []
    for s in subs:
        action, matched = twin_action(env, s)
        cite = replay_citation(env, s["evidence_ids"], [s["target_id"]])
        env.submit(env.make_bundle(action, cite, s["decision_id"]))
        playbook.note_submitted(action)
        out.append({"time": float(env.now), "decision_id": s["decision_id"], "action_id": s["action_id"],
                    "matched": matched, "evidence_ids": cite, "citation_changed": list(cite) != list(s["evidence_ids"])})
    if subs:
        playbook.escalated = bool(subs[-1]["escalated_after"])
    return out


def receipt_statuses(executor, keys, until=math.inf):
    out = {k: [] for k in keys}
    for r in executor.receipts:
        stamp = next(x for x in (r.finished_at, r.started_at, r.submitted_at) if x is not None)
        if r.idempotency_key in out and stamp <= until:
            out[r.idempotency_key].append((r.status.value, r.reason))
    return out


def label_twin(tenv, pb, follow, d, cfg):
    candidates = tenv.candidates()
    pairs = [twin_action(tenv, c, candidates) for c in d["candidates"]]
    evidence = replay_citation(tenv, d["cited_evidence"], [c["target_id"] for c in d["candidates"]])
    _, results = branch_labels(tenv, follow, cfg["horizon_s"], cfg["replicates"], tag=tuple(d["reseed_tag"]),
                               candidates=[a for a, _ in pairs], evidence_ids=evidence)
    oracle = tenv.oracle_time_to_shutdown(cfg["horizon_s"])
    return {"results": [as_record(r) for r in results], "cited_evidence": evidence,
            "candidates_unmatched": [c["action_id"] for c, (_, ok) in zip(d["candidates"], pairs) if not ok],
            "hidden": {**tenv.hidden_state(), "oracle_time_to_shutdown": oracle[0], "oracle_message": oracle[1]},
            "pi_after_memory_differs": playbook_memory(pb) != playbook_memory(follow)}


def run_twin(env, scenario, decisions, submissions, primary, mode, cfg, incidents):
    group = belief.family_group(scenario.incident)
    dt = env.plant.dt_s
    twin_sc = belief.twin_scenario(scenario, dt)
    tenv = TEPEnv(twin_sc)
    tenv.reset()
    pb = Playbook(mode)
    states = [world_state(tenv, group)]
    params = belief.params_from(env)
    pair = belief.pair_prior(incidents, group)
    causes = (scenario.incident.kind, twin_sc.incident.kind)
    onsets = (scenario.warmup_s + scenario.incident.onset_s, twin_sc.warmup_s + twin_sc.incident.onset_s)
    by_time = {}
    for s in submissions:
        by_time.setdefault(s["time"], []).append(s)
    labelled = {d["tau"]: i for i, d in enumerate(decisions)}
    replayed, not_replayed = [], []
    for t in sorted(set(by_time) | set(labelled)):
        at = advance_twin(tenv, pb, t, states, group)
        if t in labelled:
            i = labelled[t]
            d = decisions[i]
            seen = observed_items(env.log, t, primary["log_sizes"][i])
            first = belief.first_difference(seen, observed_items(tenv.log, t))
            worlds = [belief.world(causes[0], onsets[0], dt, [s for s in primary["states"] if s[0] <= t]),
                      belief.world(causes[1], onsets[1], dt, [s for s in states if s[0] <= t])]
            record = {"cause": twin_sc.incident.cause, "labelled": at,
                      "belief": belief.belief_evidence(seen, t, group, worlds, params, [pair[c] for c in causes],
                                                       first),
                      "results": None, "cited_evidence": [], "candidates_unmatched": [], "hidden": None,
                      "pi_after_memory_differs": None}
            if at:
                record.update(label_twin(tenv, pb, primary["follow"][i], d, cfg))
            d["twin"] = record
        if at:
            replayed += replay_submissions(tenv, pb, by_time.get(t, []))
        else:
            not_replayed += [{k: s[k] for k in ("time", "decision_id", "action_id")} for s in by_time.get(t, ())]
    advance_twin(tenv, pb, env.now, states, group)
    keys = [f"{s['decision_id']}:{s['action_id']}" for s in replayed]
    until = min(env.now, tenv.now)
    ours, theirs = receipt_statuses(env.executor, keys, until), receipt_statuses(tenv.executor, keys, until)
    first = belief.first_difference(observed_items(env.log, until), observed_items(tenv.log, until))
    first_item = None if first is None else (env.log.get(first[1]) or tenv.log.get(first[1]))
    c = tenv.cost()
    return {
        "cause": twin_sc.incident.cause, "group": group, "scenario": dataclasses.asdict(twin_sc),
        "onset_shift_s": twin_sc.extra["twin_onset_shift_s"], "prior": dict(pair), "params": params,
        "behaviour_mode": mode, "primary_states": primary["states"], "states": states, "replayed": replayed,
        "not_replayed": not_replayed,
        "divergence": {
            "receipt_status_differs": any(ours[k] != theirs[k] for k in keys),
            "receipt_differences": [{"key": k, "primary": ours[k], "twin": theirs[k]} for k in keys
                                    if ours[k] != theirs[k]],
            "first_deterministic_difference": None if first is None else {
                "time": float(first[0]), "evidence_id": first[1], "kind": first_item.kind,
                "subject_id": first_item.subject_id, "since_onset_s": float(first[0] - onsets[0])},
            "compared_until": float(until)},
        "outcome": {"cost": c.as_dict(), "total": c.total(), "shutdown": tenv.plant.shutdown,
                    "hidden": tenv.hidden_state()}}


def twin_summary(twin, decisions):
    if twin is None:
        return None
    first = twin["divergence"]["first_deterministic_difference"]
    return {"labelled": sum(bool(d.get("twin", {}).get("labelled")) for d in decisions),
            "receipt_status_differs": twin["divergence"]["receipt_status_differs"],
            "first_difference_since_onset_s": None if first is None else first["since_onset_s"]}


def run_episode(args):
    seed, cfg, nominal, incidents, out_dir = args
    rng = np.random.default_rng([seed, 31337])
    scenario = sample_scenario(seed, nominal, incidents)
    env = TEPEnv(scenario)
    env.reset()
    behaviour = Playbook(MODES[int(rng.integers(len(MODES)))])
    follow_mode = cfg["follow_mode"]
    group = belief.family_group(scenario.incident) if cfg.get("twins", False) else None
    primary = {"states": [world_state(env, group)] if group is not None else [], "log_sizes": [], "follow": []}
    decisions, taus, submissions, n = [], [], [], 0
    started = time.perf_counter()
    while not env.done:
        obs = env.advance_to_decision()
        if env.done:
            break
        if group is not None:
            primary["states"].append(world_state(env, group))
        behaviour.update(obs, env)
        after_onset = env.now >= env.world.onset_time
        pre_onset_labels = sum(not d["hidden"]["onset_happened"] for d in decisions)
        wants_label = len(decisions) < cfg["max_labels_per_episode"] and (
            after_onset or pre_onset_labels < cfg["max_labels_pre_onset"]) and (
            obs.alerts or (cfg["label_ticks_after_onset"] and after_onset and relevant_evidence(env)))
        cands = relevant_candidates(env) if wants_label else []
        record = None
        if cands:
            follow = dataclasses.replace(behaviour, mode=follow_mode) if follow_mode != behaviour.mode else behaviour
            if group is not None:
                primary["log_sizes"].append(len(env.log))
                primary["follow"].append(copy.deepcopy(follow))
            evidence = relevant_evidence(env)
            _, results = branch_labels(env, follow, cfg["horizon_s"], cfg["replicates"], tag=(n,),
                                       candidates=cands, evidence_ids=evidence)
            oracle = env.oracle_time_to_shutdown(cfg["horizon_s"])
            record = {"tau": env.now, "decision_id": f"ep{seed}d{n}", "reseed_tag": [n], "obs_version": obs.version,
                      "valid_evidence": sorted(env.valid_evidence()),
                      "cited_evidence": evidence, "candidates": [action_record(a) for a in cands],
                      "results": [as_record(r) for r in results],
                      "hidden": {**env.hidden_state(), "oracle_time_to_shutdown": oracle[0],
                                 "oracle_message": oracle[1]}}
            taus.append(env.now)
            record["baselines"] = {m: [a.action_id for a, _ in dataclasses.replace(copy.deepcopy(behaviour),
                                                                                   mode=m).plan(env)]
                                   for m in MODES}
        if record is None:
            plan = behaviour.plan(env)
        else:
            plan, record["behaviour"] = behaviour_step(behaviour, env, cands, record["cited_evidence"],
                                                       cfg["epsilon"], rng)
            decisions.append(record)
        submitted = submit_plan(env, plan, f"ep{seed}d{n}")
        submissions += [{"time": env.now, "decision_id": f"ep{seed}d{n}", "action_id": aid, "tool_id": a.tool_id,
                         "target_id": a.target_id, "args": dict(a.bounded_arguments), "evidence_ids": list(ev),
                         "escalated_after": bool(behaviour.escalated)} for (a, ev), aid in zip(plan, submitted)]
        if record is not None:
            record["submitted_action_ids"] = submitted
        n += 1
    c = env.cost()
    items = stored_items(env.log, taus, cfg["evidence_window_s"])
    seconds = time.perf_counter() - started
    twin = run_twin(env, scenario, decisions, submissions, primary, behaviour.mode, cfg, incidents) \
        if group is not None else None
    episode = {"seed": seed, "incidents": cfg["incidents"], "scenario": dataclasses.asdict(scenario),
               "backup_available": bool(scenario.backup_available), "decisions": decisions, "items": items,
               "submissions": submissions, "twin": twin,
               "outcome": {"cost": c.as_dict(), "total": c.total(), "shutdown": env.plant.shutdown,
                           "hidden": env.hidden_state(), "behaviour_mode": behaviour.mode},
               "seconds": time.perf_counter() - started, "twin_seconds": time.perf_counter() - started - seconds}
    path = Path(out_dir) / f"ep_{seed:06d}.pkl"
    with open(path, "wb") as handle:
        pickle.dump(episode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return (seed, len(decisions), episode["seconds"], scenario.incident.cause, env.plant.shutdown is not None,
            episode["backup_available"], twin_summary(twin, decisions))


def episode_row(seed, n_dec, secs, cause, tripped, backup, twin=None):
    sealed = split_of(seed) == "test"
    return {"seed": seed, "decisions": n_dec, "seconds": SEALED if sealed else secs,
            "cause": SEALED if sealed else cause, "tripped": SEALED if sealed else tripped, "backup_available": backup,
            "twin": SEALED if sealed else twin}


def report(results, total):
    rows = []
    for result in results:
        row = episode_row(*result)
        rows.append(row)
        if split_of(row["seed"]) == "test":
            print(f"[{len(rows)}/{total}] held-out test episode: {SEALED}", flush=True)
            continue
        twin = row["twin"] if row["twin"] is None else (
            f"{row['twin']['labelled']} labelled, receipts differ={row['twin']['receipt_status_differs']}, "
            f"first difference={row['twin']['first_difference_since_onset_s']}")
        print(f"[{len(rows)}/{total}] seed {row['seed']}: {row['decisions']} labelled decisions, "
              f"{row['seconds']:.0f}s, {row['cause']}, backup={row['backup_available']}, twin={twin}, "
              f"trip={row['tripped']}", flush=True)
    return sorted(rows, key=lambda r: r["seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="experience/generate.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--name", default="pilot")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    incidents_file = config_path(cfg["incidents"])
    incidents = load_config(incidents_file)
    run = start_run("experience", cfg, seed=None, name=args.name)
    out_dir = ensure_dir(PROCESSED_DIR / "tep_experience" / run.dir.name)
    (out_dir / "generation.json").write_text(json.dumps({
        "run": run.dir.relative_to(ROOT).as_posix(), "config_file": args.config, "config": cfg,
        "config_sha256": run.record["config_sha256"], "config_file_sha256": sha256_file(config_path(args.config)),
        "incidents_file": cfg["incidents"], "incidents_sha256": sha256_file(incidents_file),
        "incidents_config": incidents, "source_combined": run.record["source"]["combined"]}, indent=1),
        encoding="utf-8")
    plant = TEPPlant()
    plant.reset(seed=0)
    nominal = dict(zip(plant.measurement_names, plant.measured.tolist()))
    jobs = [(s, cfg, nominal, incidents, str(out_dir)) for s in range(*cfg["seeds"])]
    t0 = time.perf_counter()
    with Pool(int(cfg["workers"])) as pool:
        rows = report(pool.imap_unordered(run_episode, jobs), len(jobs))
    run.finish({"episodes": len(rows), "labelled_decisions": sum(r["decisions"] for r in rows),
                "wall_s": time.perf_counter() - t0, "out_dir": str(out_dir), "per_episode": rows})
    print(f"done: {len(rows)} episodes, {sum(r['decisions'] for r in rows)} labelled decisions -> {out_dir}")

if __name__ == "__main__":
    main()
