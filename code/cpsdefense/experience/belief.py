import math
from dataclasses import replace

import numpy as np

from cpsdefense.utils.config import load_config

LINK, MODULE = "link_fault", "io_module_fault"
PAIR = (LINK, MODULE)
MODES = ("up", "down", "backup")
REPORT_PATTERN = {"up": (1.0, 0.0), "down": (1.0, 0.0), "backup": (0.0, 1.0)}


def _incident_fields(incident):
    if incident is None:
        return None
    if isinstance(incident, dict):
        return incident.get("kind"), incident.get("target"), incident.get("onset_s")
    return incident.kind, incident.target, incident.onset_s


def family_group(incident):
    fields = _incident_fields(incident)
    if fields is None:
        return None
    kind, target, _ = fields
    if kind == LINK and isinstance(target, str) and target.startswith("sensor:"):
        return target[len("sensor:"):]
    if kind == MODULE and isinstance(target, str):
        return target
    return None


def twin_incident(incident, dt_s):
    group = family_group(incident)
    if group is None:
        raise ValueError(f"incident {incident} is not a link or I/O-module fault of a sensor group")
    if incident.kind == LINK:
        return replace(incident, kind=MODULE, target=group, onset_s=incident.onset_s - float(dt_s))
    return replace(incident, kind=LINK, target=f"sensor:{group}", onset_s=incident.onset_s + float(dt_s))


def twin_scenario(scenario, dt_s):
    twin = twin_incident(scenario.incident, dt_s)
    shift = twin.onset_s - scenario.incident.onset_s
    return replace(scenario, incident=twin, horizon_s=scenario.horizon_s - shift,
                   extra={**scenario.extra, "twin_of": scenario.incident.cause, "twin_onset_shift_s": shift})


def pair_prior(incidents, group):
    types = incidents["types"]
    total = float(sum(types.values()))
    links = incidents.get("link_fault", {}).get("links", {})
    groups = incidents.get("io_module_fault", {}).get("groups", {})
    p_link = float(types.get(LINK, 0.0)) / total * (float(links.get(f"sensor:{group}", 0.0))
                                                   / max(float(sum(links.values())), 1e-300))
    p_module = float(types.get(MODULE, 0.0)) / total * (float(groups.get(group, 0.0))
                                                       / max(float(sum(groups.values())), 1e-300))
    if p_link + p_module <= 0:
        raise ValueError(f"the incidents config gives neither a link nor a module fault on group {group!r}")
    return {LINK: p_link / (p_link + p_module), MODULE: p_module / (p_link + p_module)}


def _params(link_health, tools, causes, window_s, evidence_scale):
    return {"p_detect_down": float(link_health["p_detect_down"]), "p_false": float(link_health["p_false"]),
            "report_backup": bool(link_health["report_backup"]), "delay_s": float(link_health["delay_s"]),
            "delay_sigma": float(link_health["sigma"]), "evidence_scale": float(evidence_scale),
            "window_s": float(window_s), "trace_accuracy": float(tools["trace_accuracy"]),
            "operator_accuracy": float(tools["operator_accuracy"]), "causes": [str(c) for c in causes]}


def params_from(env):
    return _params(env.telemetry.config["link_health"], env.config["tools"], env.config["causes"],
                   env.telemetry.window_steps * env.plant.dt_s, env.scenario.evidence_scale)


def load_params(evidence_scale=1.0):
    telemetry, env = load_config("sim/telemetry.yaml"), load_config("sim/tep_env.yaml")
    dt = float(load_config("sim/tep.yaml")["control_interval_s"])
    return _params(telemetry["link_health"], env["tools"], env["causes"], int(telemetry["window_steps"]) * dt,
                   evidence_scale)


def delay_cdf(x, params):
    median = params["delay_s"] * params["evidence_scale"]
    if x <= 0:
        return 0.0
    if params["delay_sigma"] == 0:
        return 1.0 if x >= median else 0.0
    return 0.5 * math.erfc(-math.log(x / median) / (params["delay_sigma"] * math.sqrt(2.0)))


def world(cause, onset_time, dt_s, states):
    return {"cause": cause, "onset_time": float(onset_time), "dt": float(dt_s),
            "states": [(float(t), str(m), bool(f)) for t, m, f in states]}


def state_at(w, t):
    dt, onset = w["dt"], w["onset_time"]
    last = (-math.inf, "up", False)
    for s in w["states"]:
        if s[0] < t - dt / 2:
            last = s
        else:
            break
    t_r, mode, faulted = last
    if w["cause"] == LINK and t - dt >= onset and not t_r - dt >= onset:
        mode, faulted = ("down" if mode == "up" else mode), True
    return mode, faulted


def true_kind(w, t):
    return w["cause"] if t >= w["onset_time"] else "none"


def _canonical(value):
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in (value.tolist() if isinstance(value, np.ndarray) else value))
    if isinstance(value, (np.floating, float)):
        return ("nan",) if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return tuple(sorted((k, _canonical(v)) for k, v in value.items()))
    return value


def deterministic_view(item):
    if item.kind == "link_health":
        return None
    payload = dict(item.payload)
    if item.kind == "trace_link":
        payload["values"] = np.asarray(payload["values"])[1:]
    elif item.kind == "operator_report":
        payload.pop("values", None)
        payload.pop("cause", None)
    return (item.kind, item.source, item.subject_id, float(item.event_time), float(item.arrival_time),
            float(item.valid_until), item.provenance, tuple(item.supersedes), _canonical(payload))


def first_difference(primary, twin, until=math.inf):
    views = []
    for items in (primary, twin):
        seen = {}
        for i in items:
            view = deterministic_view(i) if i.arrival_time <= until else None
            if view is not None:
                seen[i.evidence_id] = (i.arrival_time, view)
        views.append(seen)
    a, b = views
    differing = [(min(x[0] for x in (a.get(e), b.get(e)) if x is not None), e) for e in set(a) | set(b)
                 if a.get(e) is None or b.get(e) is None or a[e][1] != b[e][1]]
    return min(differing) if differing else None


def window_index(evidence_id):
    return int(evidence_id.rsplit(":", 1)[1])


def belief_evidence(items, tau, group, worlds, params, prior, first_diff=None):
    link = f"sensor:{group}"
    items = [i for i in items if i.arrival_time <= tau]
    window_items = [i for i in items if i.kind in ("window_summary", "link_health")]
    if not window_items:
        raise ValueError("no telemetry window item observed: the window schedule is unknown")
    ref = max(window_items, key=lambda i: window_index(i.evidence_id))
    w_ref, e_ref = window_index(ref.evidence_id), float(ref.event_time)
    reports = {window_index(i.evidence_id): [float(v) for v in np.asarray(i.payload["values"])[:2]]
               for i in items if i.kind == "link_health" and i.subject_id == link}
    step = params["window_s"]
    first_onset = min(w["onset_time"] for w in worlds)
    windows = []
    w_lo = w_ref + int(math.floor((first_onset - e_ref) / step))
    w_hi = w_ref + int(math.floor((tau - e_ref) / step + 1e-9))
    for w in range(w_lo, w_hi + 1):
        end = e_ref + (w - w_ref) * step
        if end <= first_onset or end > tau + 1e-9:
            continue
        modes = [state_at(x, end)[0] for x in worlds]
        if modes[0] != modes[1]:
            windows.append({"w": w, "end": end, "modes": modes, "report": reports.get(w), "age": float(tau - end)})
    traces = [{"time": float(i.event_time), "reported_healthy": bool(np.asarray(i.payload["values"])[0] > 0.5),
               "healthy": [not state_at(x, i.event_time)[1] for x in worlds]}
              for i in items if i.kind == "trace_link" and i.subject_id == link]
    operator = [{"time": float(i.event_time), "named": str(i.payload["cause"]),
                 "true": [true_kind(x, i.event_time) for x in worlds]}
                for i in items if i.kind == "operator_report"]
    return {"group": group, "causes": [w["cause"] for w in worlds], "prior": [float(p) for p in prior],
            "tau": float(tau), "params": dict(params), "deterministic_identical": first_diff is None,
            "first_difference": None if first_diff is None else [float(first_diff[0]), str(first_diff[1])],
            "windows": windows, "traces": traces, "operator": operator}


def report_probability(mode, params):
    if mode == "down":
        return params["p_detect_down"]
    if mode == "up":
        return params["p_false"]
    if mode == "backup":
        return 1.0 if params["report_backup"] else 0.0
    raise ValueError(f"unknown link mode {mode!r}")


def window_probability(mode, report, age, params):
    p = report_probability(mode, params)
    if report is None:
        return 1.0 - p * delay_cdf(age, params)
    return p if tuple(float(v) for v in report) == REPORT_PATTERN[mode] else 0.0


def _log(p):
    return math.log(p) if p > 0 else -math.inf


def log_likelihoods(evidence):
    p = evidence["params"]
    ll = [0.0, 0.0]
    acc, op_acc = p["trace_accuracy"], p["operator_accuracy"]
    n_causes = len(p["causes"])
    for c in (0, 1):
        for w in evidence["windows"]:
            ll[c] += _log(window_probability(w["modes"][c], w["report"], w["age"], p))
        for t in evidence["traces"]:
            ll[c] += _log(acc if t["reported_healthy"] == t["healthy"][c] else 1.0 - acc)
        for o in evidence["operator"]:
            ll[c] += _log(op_acc if o["named"] == o["true"][c] else (1.0 - op_acc) / (n_causes - 1))
    if not evidence["deterministic_identical"]:
        ll[1] = -math.inf
    return ll


def posterior(evidence):
    logs = np.array([_log(p) + ll for p, ll in zip(evidence["prior"], log_likelihoods(evidence))])
    if not np.isfinite(logs).any():
        return np.array([np.nan, np.nan])
    w = np.exp(logs - logs[np.isfinite(logs)].max())
    return w / w.sum()
