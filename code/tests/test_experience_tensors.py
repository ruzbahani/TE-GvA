import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("tep_studio")

import torch

from cpsdefense.core.types import EvidenceItem, ExecStatus
from cpsdefense.defense.evidence import EvidenceLog
from cpsdefense.defense.registry import TARGET_KINDS
from cpsdefense.experience.dataset import TOOL_RESULT_KINDS
from cpsdefense.experience.tensors import (
    CONFIG, KINDS, RECEIPT_STATUSES, ItemIds, action_features, build_vocab, evidence_batch, evidence_tensors,
    fit_payload_scaler, item_payload, item_source, load_vocab, save_vocab)
from cpsdefense.model.backbone import DecisionBackbone
from cpsdefense.model.branches import Question
from cpsdefense.model.evidence import FIELDS, NUMPY_DTYPES, PAD_VALUES, ROW_FIELDS, EvidenceBatch
from cpsdefense.paths import CONFIG_DIR
from cpsdefense.sim.incidents import Incident, Scenario
from cpsdefense.sim.tep_env import RECEIPT_LIFETIME_S, TEPEnv
from cpsdefense.utils.config import load_config

ACTIONS = (("trace_link", "sensor:stripper"), ("query_historian", "source:stripper"),
           ("verify_sensor", "source:stripper"), ("escalate", "incident:1"), ("switch_to_backup", "sensor:stripper"))
TOOL_KINDS = {"trace_link", "historian", "field_check", "operator_report"}
RESULT_KIND = {"trace_link": "trace_link", "query_historian": "historian", "verify_sensor": "field_check",
               "escalate": "operator_report"}
SEGMENT = 7


def config(**selection):
    cfg = copy.deepcopy(load_config(CONFIG))
    cfg["selection"].update(selection)
    return cfg

WIDE = dict(max_items=4096, fixed_slots=False)


def pick(env, tool, target):
    return next(a for a in env.candidates() if a.tool_id == tool and a.target_id == target)


def receipt_time(r):
    return r.finished_at if r.finished_at is not None else r.started_at if r.started_at is not None else r.submitted_at


@pytest.fixture(scope="module")
def run():
    sc = Scenario(seed=3, incident=Incident("link_fault", "sensor:stripper", 600.0), background=(), horizon_s=3 * 3600)
    env = TEPEnv(sc)
    vocab = build_vocab(env)
    env.reset()
    while not env.done:
        obs = env.advance_to_decision()
        if obs.alerts:
            break
    alert = obs.alerts[0].evidence_ids[0]
    actions = {}
    for i, (tool, target) in enumerate(ACTIONS):
        a = pick(env, tool, target)
        actions[a.action_id] = a
        env.submit(env.make_bundle(a, [alert], f"d{i}"))
    taus, left = [obs.time], None
    while not env.done and len(taus) < 60 and left != 0:
        taus.append(env.advance_to_decision().time)
        if left is None and TOOL_KINDS <= {env.log.get(e).kind for e in env.log}:
            left = 3
        elif left is not None:
            left -= 1
    assert left == 0, "tool results missing"
    taus += [t + 17.3 for t in taus[:3]]
    return {"env": env, "vocab": vocab, "taus": np.array(taus), "actions": actions}


def slots(arrays):
    return list(zip(*np.nonzero(arrays["present"])))


def test_vocab_layout_and_json_round_trip(run, tmp_path):
    env, vocab = run["env"], run["vocab"]
    assert json.loads(json.dumps(vocab)) == vocab
    assert load_vocab(save_vocab(vocab, tmp_path / "vocab.json")) == vocab
    size = {g: len(idx) for g, idx in env.plant.sensor_links.items()}
    n_targets = len(env.plant.links) + len(size) + len(env.plant.measurement_names) + 2
    expected = {**{f"window:{g}": 5 * n for g, n in size.items()}, **{f"historian:{g}": n for g, n in size.items()},
                **{f"field_check:{g}": 3 * n for g, n in size.items()},
                "alert": 3 + len(size) + len(env.plant.measurement_names), "link_health": 2 + len(env.plant.links),
                "trace_link": 2 + len(env.plant.links), "operator_report": len(env.config["causes"]),
                "receipt": len(env.registry.specs) + 8 + n_targets}
    assert dict(zip(vocab["sources"], vocab["payload_dim"])) == expected
    assert vocab["P"] == max(expected.values()) == 95
    assert vocab["kinds"] == list(KINDS)
    assert vocab["receipt_statuses"] == list(RECEIPT_STATUSES) == [s.value for s in ExecStatus]
    assert vocab["payload_layout"]["receipt"][:len(vocab["tools"])] == [f"tool={t}" for t in vocab["tools"]]
    assert not any(vocab["payload_scaled"]["receipt"])
    block = vocab["subject_blocks"]["receipt"][0]
    assert block["ids"] == [t for k in TARGET_KINDS for t in vocab["targets"][k]]
    assert len(set(block["ids"])) == n_targets
    for s, name in enumerate(vocab["sources"]):
        assert len(vocab["payload_layout"][name]) == len(vocab["payload_scaled"][name]) == vocab["payload_dim"][s]
    assert vocab["F"] == len(vocab["action_columns"]) == len(set(vocab["action_columns"]))
    assert vocab == build_vocab(env)


def test_tensor_fields_follow_the_contract(run):
    vocab, taus = run["vocab"], run["taus"]
    arrays = evidence_tensors(run["env"].log, taus, SEGMENT, vocab, config(max_items=1024))
    n, m, p = len(taus), 1024, vocab["P"]
    for name in FIELDS:
        shape = (n,) if name in ROW_FIELDS else (n, m, p) if name == "payload" else (n, m)
        assert arrays[name].dtype == NUMPY_DTYPES[name] and arrays[name].shape == shape, name
    assert arrays["payload_dim"].dtype == np.int64 and arrays["payload_dim"].tolist() == vocab["payload_dim"]
    assert np.all(arrays["segment"] == SEGMENT) and np.array_equal(arrays["decision_time"], taus)
    batch = evidence_batch(arrays).check(payload_dim=vocab["payload_dim"])
    rebuilt = EvidenceBatch.from_arrays(**{k: arrays[k] for k in FIELDS})
    for name in FIELDS:
        assert np.array_equal(getattr(rebuilt, name), arrays[name], equal_nan=arrays[name].dtype.kind == "f"), name
    absent = ~arrays["present"]
    assert absent.any()
    for name, value in PAD_VALUES.items():
        got = arrays[name][absent]
        assert np.all(np.isnan(got)) if isinstance(value, float) and math.isnan(value) else np.all(got == value), name
    assert batch.to().n == n


def test_validity_matches_the_evidence_log_at_tau_and_later(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    ids = ItemIds()
    arrays = evidence_tensors(env.log, taus, SEGMENT, vocab, ids=ids)
    batch = evidence_batch(arrays)
    for dt in (0.0, 5.0, 60.0, 119.0, 600.0, 2000.0):
        valid = batch.validity(taus + dt)
        for r, c in slots(arrays):
            eid = ids.key(arrays["item_id"][r, c])[1]
            assert valid[r, c] == env.log.valid(eid, taus[r] + dt), (eid, dt)


def test_valid_items_hold_no_future_information_and_lookahead_items_are_present(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    cfg = config(**WIDE)
    lookahead, context = cfg["selection"]["lookahead_s"], cfg["selection"]["context_s"]
    ids = ItemIds()
    arrays = evidence_tensors(env.log, taus, SEGMENT, vocab, cfg, ids=ids)
    batch = evidence_batch(arrays)
    valid = batch.validity()
    tau = np.broadcast_to(taus[:, None], valid.shape)
    for name in ("event_start", "event_time", "arrival_time"):
        assert np.all(arrays[name][valid] <= tau[valid]), name
    ahead_total = ahead_future_event = 0
    items = [env.log.get(e) for e in env.log]
    for r, t in enumerate(taus):
        slot = {ids.key(i)[1]: c for c, i in enumerate(arrays["item_id"][r]) if i >= 0}
        expected = {it.evidence_id for it in items if it.arrival_time <= t + lookahead
                    and (it.event_time >= t - context or env.log.valid(it.evidence_id, t))}
        assert set(slot) == expected
        ahead = [it for it in items if it.evidence_id in expected and it.arrival_time > t]
        assert not any(valid[r, slot[it.evidence_id]] for it in ahead)
        ahead_total += len(ahead)
        ahead_future_event += sum(it.event_time > t for it in ahead)
        assert set(ids.evidence_ids(batch.dependency_set()[r])) == set(env.log.valid_ids(t))
    assert ahead_total > 0 and ahead_future_event > 0


def test_max_items_keeps_every_source_and_its_most_recent_items(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    ids = ItemIds()
    wide = evidence_tensors(env.log, taus, SEGMENT, vocab, config(**WIDE), ids=ids)
    small = evidence_tensors(env.log, taus, SEGMENT, vocab, config(max_items=24), ids=ids)
    assert small["item_id"].shape[1] == 24
    n_truncated = 0
    for r in range(len(taus)):
        eligible = {i: (s, a) for i, s, a, ok in zip(wide["item_id"][r], wide["source"][r], wide["arrival_time"][r],
                                                     wide["present"][r]) if ok}
        kept = set(small["item_id"][r][small["present"][r]].tolist())
        assert kept <= set(eligible) and len(kept) == min(24, len(eligible))
        count, total = {}, {}
        for i, (s, _) in eligible.items():
            total[s] = total.get(s, 0) + 1
            count[s] = count.get(s, 0) + (i in kept)
        assert len(total) <= 24 and min(count.values()) >= 1
        truncated = [s for s in total if count[s] < total[s]]
        if truncated:
            assert min(count[s] for s in truncated) >= max(count.values()) - 1
        for s in truncated:
            arrivals_kept = [a for i, (src, a) in eligible.items() if src == s and i in kept]
            arrivals_dropped = [a for i, (src, a) in eligible.items() if src == s and i not in kept]
            assert min(arrivals_kept) >= max(arrivals_dropped)
        n_truncated += len(truncated)
    assert n_truncated > 0


def test_payload_source_kind_and_subject_one_hots(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    ids = ItemIds()
    arrays = evidence_tensors(env.log, taus, SEGMENT, vocab, config(**WIDE), ids=ids)
    kinds = set()
    for r, c in slots(arrays):
        item = env.log.get(ids.key(arrays["item_id"][r, c])[1])
        s = arrays["source"][r, c]
        name, dim, row = vocab["sources"][s], vocab["payload_dim"][s], arrays["payload"][r, c]
        assert name == item_source(item, vocab)
        assert vocab["kinds"][arrays["kind"][r, c]] == vocab["source_kind"][s] == item.kind
        assert np.all(row[dim:] == 0)
        values = np.asarray(item.payload["values"], np.float32)
        assert np.array_equal(row[:values.size], values)
        if item.kind == "field_check":
            half = values.size // 2
            assert np.allclose(row[values.size:values.size + half], values[half:] - values[:half], atol=1e-3)
        for block in vocab["subject_blocks"].get(name, ()):
            part = row[block["offset"]:block["offset"] + len(block["ids"])]
            key = item.subject_id if block["key"] == "subject" else item.payload[block["key"]]
            assert part.sum() == 1 and block["ids"][int(part.argmax())] == key
        kinds.add(item.kind)
    assert kinds == set(KINDS)


def test_receipt_items_follow_the_shared_format(run):
    env, vocab, actions = run["env"], run["vocab"], run["actions"]
    items = [env.log.get(e) for e in env.log if env.log.get(e).kind == "receipt"]
    receipts = [r for r in env.executor.receipts if r.action_id in actions]
    assert len(items) == len(receipts) > len(actions)
    numbers = [int(it.evidence_id.split(":")[1]) for it in items]
    assert [it.evidence_id for it in items] == [f"receipt:{n}" for n in numbers]
    assert numbers == list(range(numbers[0], numbers[0] + len(items)))
    tools = vocab["tools"]
    running, superseded = {}, 0
    for it, r in zip(items, receipts):
        a, t = actions[r.action_id], receipt_time(r)
        assert (it.source, it.provenance, it.subject_id) == ("tool", "executor", a.target_id)
        assert it.event_time == it.arrival_time == it.payload["event_start"] == t
        assert (it.payload["tool_id"], it.payload["status"]) == (a.tool_id, r.status.value)
        assert (it.payload["action_id"], it.payload["idempotency_key"]) == (r.action_id, r.idempotency_key)
        assert isinstance(it.payload["action_id"], str) and isinstance(it.payload["idempotency_key"], str)
        if r.status == ExecStatus.RUNNING:
            assert it.valid_until == math.inf and it.supersedes == ()
            running[r.idempotency_key] = it.evidence_id
        else:
            assert it.valid_until == t + 3600.0
            ended = r.started_at is not None and r.idempotency_key in running
            assert it.supersedes == ((running.pop(r.idempotency_key),) if ended else ())
            if ended:
                assert env.log.retracted_at(it.supersedes[0]) == it.arrival_time
                superseded += 1
        values = np.asarray(it.payload["values"])
        assert values.dtype == np.float32 and values.shape == (len(tools) + len(RECEIPT_STATUSES),)
        assert values.sum() == 2 and values[tools.index(a.tool_id)] == 1
        assert values[len(tools) + RECEIPT_STATUSES.index(r.status.value)] == 1
    assert {"accepted", "running", "succeeded"} <= {it.payload["status"] for it in items} and superseded > 0
    for eid in running.values():
        assert env.log.valid(eid, env.now) and env.log.retracted_at(eid) == math.inf


def test_tool_result_items_name_the_action_that_produced_them(run):
    env, actions = run["env"], run["actions"]
    keys = {aid: f"d{i}:{aid}" for i, aid in enumerate(actions)}
    items = [env.log.get(e) for e in env.log if env.log.get(e).kind in TOOL_RESULT_KINDS]
    assert set(TOOL_RESULT_KINDS) == TOOL_KINDS == {it.kind for it in items}
    for it in items:
        assert isinstance(it.payload["action_id"], str) and isinstance(it.payload["idempotency_key"], str)
        a = actions[it.payload["action_id"]]
        assert (it.kind, it.subject_id) == (RESULT_KIND[a.tool_id], a.target_id)
        assert it.payload["idempotency_key"] == keys[a.action_id]


def test_receipt_payload_is_tool_status_and_target_one_hots_and_stays_unscaled(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    scaler = fit_payload_scaler(env.log, vocab)
    entry = scaler["sources"]["receipt"]
    assert entry["count"] > 0 and set(entry["origin"]) == {"fixed"}
    ids = ItemIds()
    arrays = evidence_tensors(env.log, taus, SEGMENT, vocab, config(**WIDE), scaler=scaler, ids=ids)
    s = vocab["sources"].index("receipt")
    targets = vocab["subject_blocks"]["receipt"][0]["ids"]
    n_tools, n_status = len(vocab["tools"]), len(RECEIPT_STATUSES)
    valid = evidence_batch(arrays).validity()
    seen = 0
    for r, c in zip(*np.nonzero(arrays["present"] & (arrays["source"] == s))):
        item = env.log.get(ids.key(arrays["item_id"][r, c])[1])
        row = arrays["payload"][r, c]
        assert vocab["kinds"][arrays["kind"][r, c]] == "receipt"
        assert vocab["tools"][int(row[:n_tools].argmax())] == item.payload["tool_id"] and row[:n_tools].sum() == 1
        assert RECEIPT_STATUSES[int(row[n_tools:n_tools + n_status].argmax())] == item.payload["status"]
        block = row[n_tools + n_status:n_tools + n_status + len(targets)]
        assert block.sum() == 1 and targets[int(block.argmax())] == item.subject_id
        assert np.all(row[n_tools + n_status + len(targets):] == 0)
        assert valid[r, c] == env.log.valid(item.evidence_id, taus[r])
        seen += 1
    assert seen > 0 and valid[arrays["source"] == s].any()


def test_excluded_items_leave_the_row_entirely_lookahead_included(run):
    env, vocab, taus, actions = run["env"], run["vocab"], run["taus"], run["actions"]
    cfg = config(**WIDE)
    ids = ItemIds()
    receipts = {e for e in env.log if env.log.get(e).kind == "receipt"
                and env.log.get(e).payload["action_id"] in actions}
    known = [env.log.valid_ids(t) for t in taus]
    full = evidence_tensors(env.log, taus, SEGMENT, vocab, cfg, ids=ids, known=known)
    exclude = [receipts] + [()] * (len(taus) - 1)
    cut = evidence_tensors(env.log, taus, SEGMENT, vocab, cfg, ids=ids, known=known, exclude=exclude)

    def row(a, r):
        return {ids.key(i)[1]: c for c, i in enumerate(a["item_id"][r]) if i >= 0}

    first = row(full, 0)
    shown = receipts & set(first)
    assert shown and any(full["arrival_time"][0, first[e]] > taus[0] for e in shown)
    assert set(row(cut, 0)) == set(first) - receipts
    for name in ("arrival_time", "valid_until", "retracted_at"):
        kept = row(cut, 0)
        assert all(cut[name][0, c] == full[name][0, first[e]] for e, c in kept.items()), name
    for r in range(1, len(taus)):
        assert np.array_equal(cut["item_id"][r], full["item_id"][r])
        assert np.array_equal(cut["payload"][r], full["payload"][r])
    with pytest.raises(ValueError):
        evidence_tensors(env.log, taus, SEGMENT, vocab, cfg, exclude=exclude[:-1])


def test_excluding_an_item_excludes_everything_derived_from_it(run):
    vocab = run["vocab"]
    items = synthetic_items(vocab["links"][0])
    cfg = config(lookahead_s=1000.0, context_s=1000.0, max_items=16)
    ids = ItemIds()
    arrays = evidence_tensors(items, [45.0, 45.0], 0, vocab, cfg, ids=ids, exclude=[{"b"}, set()])
    shown = [sorted(ids.key(i)[1] for i in arrays["item_id"][r] if i >= 0) for r in range(2)]
    assert shown == [["a", "c"], ["a", "b", "c", "d", "e", "f"]]


def test_a_receipt_whose_values_disagree_with_its_tool_or_status_is_refused(run):
    env, vocab = run["env"], run["vocab"]
    good = next(env.log.get(e) for e in env.log if env.log.get(e).kind == "receipt"
                and env.log.get(e).payload["status"] == "running")
    assert item_source(good, vocab) == "receipt"
    assert item_payload(good, vocab).size == vocab["payload_dim"][vocab["sources"].index("receipt")]
    other = next(t for t in vocab["tools"] if t != good.payload["tool_id"])
    for bad in (replace(good, payload={**good.payload, "status": "succeeded"}),
                replace(good, payload={**good.payload, "status": "paused"}),
                replace(good, payload={**good.payload, "tool_id": other}),
                replace(good, payload={**good.payload, "tool_id": "forged_tool"}),
                replace(good, payload={**good.payload, "values": good.payload["values"][:-1]}),
                replace(good, subject_id="no-such-target")):
        with pytest.raises(ValueError):
            item_payload(bad, vocab)
    n_tools = len(vocab["tools"])
    forged = replace(good, payload={**good.payload, "tool_id": "forged_tool",
                                    "values": np.r_[np.zeros(n_tools), good.payload["values"][n_tools:]]})
    assert item_payload(forged, vocab)[:n_tools].sum() == 0


def test_item_ids_round_trip(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    ids = ItemIds()
    a = evidence_tensors(env.log, taus, SEGMENT, vocab, config(**WIDE), ids=ids)
    seen = {}
    for r, c in slots(a):
        item_id = int(a["item_id"][r, c])
        segment, eid = ids.key(item_id)
        assert segment == SEGMENT and eid in env.log and ids.find(SEGMENT, eid) == item_id
        assert seen.setdefault(eid, item_id) == item_id
    restored = ItemIds.from_dict(json.loads(json.dumps(ids.to_dict())))
    assert len(restored) == len(ids)
    b = evidence_tensors(env.log, taus, SEGMENT, vocab, config(**WIDE), ids=restored)
    assert np.array_equal(a["item_id"], b["item_id"])
    other = evidence_tensors(env.log, taus, SEGMENT + 1, vocab, config(**WIDE), ids=ids)
    assert not set(other["item_id"][other["present"]]) & set(a["item_id"][a["present"]])
    deps = evidence_batch(a).to().dependency_set()
    assert set(ids.evidence_ids(deps[0])) == set(env.log.valid_ids(taus[0]))
    assert ids.find(SEGMENT, "no-such-id") == -1


def synthetic_items(link):
    def item(eid, arrival, lifetime, supersedes=(), parents=()):
        payload = {"values": np.array([1.0, 0.0], np.float32), "event_start": arrival - 2.0}
        if parents:
            payload["derived_from"] = list(parents)
        return EvidenceItem(eid, "tool", "trace_link", link, arrival - 1.0, arrival, arrival + lifetime, payload,
                            "test", supersedes=tuple(supersedes))
    return [item("a", 10.0, 100.0), item("b", 20.0, 100.0), item("c", 30.0, 100.0, supersedes=("a",)),
            item("d", 40.0, 30.0, parents=("b", "c")), item("e", 50.0, 100.0, parents=("d",)),
            item("f", 15.0, 100.0, parents=("b",))]


def test_retraction_supersession_and_derivation_follow_the_log(run):
    vocab = run["vocab"]
    items = synthetic_items(vocab["links"][0])
    cfg = config(lookahead_s=1000.0, context_s=1000.0, max_items=16)
    taus = np.array([12.0, 35.0, 45.0, 65.0, 150.0])
    retracted = EvidenceLog(items)
    retracted.retract("b", 60.0)
    for source, log in ((retracted, retracted), (items, EvidenceLog(items))):
        ids = ItemIds()
        arrays = evidence_tensors(source, taus, 0, vocab, cfg, ids=ids)
        batch = evidence_batch(arrays)
        assert batch.present.sum(1).tolist() == [len(items)] * len(taus)
        for t in np.arange(0.0, 250.0, 0.5):
            valid = batch.validity(t)
            for r, c in slots(arrays):
                eid = ids.key(arrays["item_id"][r, c])[1]
                assert valid[r, c] == log.valid(eid, t), (eid, t)
    row = {ids.key(i)[1]: c for c, i in enumerate(arrays["item_id"][0]) if i >= 0}
    assert arrays["retracted_at"][0, row["a"]] == 30.0 and arrays["arrival_time"][0, row["f"]] == 20.0
    assert arrays["valid_until"][0, row["e"]] == 70.0
    with pytest.raises(ValueError):
        evidence_tensors(items[3:], taus, 0, vocab, cfg)


def test_scaler_is_json_able_and_leaves_flags_and_one_hots_alone(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    scaler = fit_payload_scaler(env.log, vocab)
    assert json.loads(json.dumps(scaler)) == scaler
    window = scaler["sources"]["window:stripper"]
    for label, scaled, origin, scale in zip(vocab["payload_layout"]["window:stripper"],
                                            vocab["payload_scaled"]["window:stripper"], window["origin"],
                                            window["scale"]):
        assert origin == ("fitted" if scaled else "fixed") and scale > 0
        assert scaled == (not label.endswith(":frac_unchanged"))
    historian = scaler["sources"]["historian:stripper"]
    assert historian["count"] == 1 and historian["origin"][0] == "window:stripper/stripper_level:last"
    k = vocab["payload_layout"]["window:stripper"].index("stripper_level:last")
    assert historian["center"][0] == window["center"][k] and historian["scale"][0] == window["scale"][k]
    field = scaler["sources"]["field_check:stripper"]
    diff = vocab["payload_layout"]["field_check:stripper"].index("stripper_level:published_minus_field")
    assert field["center"][diff] == 0.0
    assert scaler == fit_payload_scaler([env.log.get(e) for e in env.log], vocab)

    ids = ItemIds()
    raw = evidence_tensors(env.log, taus, SEGMENT, vocab, ids=ids)
    z = evidence_tensors(env.log, taus, SEGMENT, vocab, scaler=scaler, ids=ids)
    assert np.array_equal(raw["item_id"], z["item_id"])
    clip = scaler["clip"]
    for r, c in slots(raw):
        name = vocab["sources"][raw["source"][r, c]]
        dim, entry = vocab["payload_dim"][raw["source"][r, c]], scaler["sources"][name]
        scaled = np.array(vocab["payload_scaled"][name])
        x = item_payload(env.log.get(ids.key(raw["item_id"][r, c])[1]), vocab)
        y = z["payload"][r, c, :dim]
        assert np.array_equal(y[~scaled], x[~scaled].astype(np.float32)) and np.all(z["payload"][r, c, dim:] == 0)
        expect = np.clip((x - np.array(entry["center"])) / np.array(entry["scale"]), -clip, clip)
        assert np.allclose(y[scaled], expect[scaled], rtol=1e-6, atol=1e-5) and np.all(np.abs(y[scaled]) <= clip)


def test_action_features_are_deterministic_one_hot_blocks(run):
    env, vocab = run["env"], run["vocab"]
    actions = env.candidates()
    x, columns = action_features(actions, env.registry, vocab)
    assert x.dtype == np.float32 and x.shape == (len(actions), vocab["F"]) and columns == vocab["action_columns"]
    again, columns2 = action_features(actions, env.registry, json.loads(json.dumps(vocab)))
    assert np.array_equal(x, again) and columns2 == columns
    assert np.array_equal(action_features(actions[::-1], env.registry, vocab)[0], x[::-1])
    for name, (lo, hi) in vocab["action_blocks"].items():
        assert np.all(x[:, lo:hi].sum(1) == 1), name
    col = {c: i for i, c in enumerate(columns)}
    tools, seen_args = set(), set()
    for row, a in zip(x, actions):
        assert row[col[f"tool:{a.tool_id}"]] == 1 and row[col[f"target:{a.target_id}"]] == 1
        assert row[col[f"family:{a.family.value}"]] == 1
        assert row[col[f"side_effect:{a.side_effect_class.value}"]] == 1
        assert row[col["log1p_duration_median_s"]] == np.float32(math.log1p(a.expected_duration))
        for arg in vocab["action_args"]:
            value = row[arg["column"]]
            assert 0.0 <= value <= 1.0
            if arg["tool"] == a.tool_id:
                v = a.bounded_arguments[arg["name"]]
                assert value == np.float32((v - arg["low"]) / (arg["high"] - arg["low"]))
                seen_args.add(arg["column"])
            else:
                assert value == 0.0
        tools.add(a.tool_id)
    assert "escalate" in tools and "recalibrate_sensor" in tools and seen_args


def test_backbone_forward_pass_on_the_batch(run):
    env, vocab, taus = run["env"], run["vocab"], run["taus"]
    scaler = fit_payload_scaler(env.log, vocab)
    batch = evidence_batch(evidence_tensors(env.log, taus, SEGMENT, vocab, scaler=scaler))
    x, _ = action_features(env.candidates(), env.registry, vocab)
    torch.manual_seed(0)
    model = DecisionBackbone(dict(payload_dim=tuple(vocab["payload_dim"]), n_kinds=len(vocab["kinds"]), d_model=32,
                                  n_heads=4, n_layers=1, d_ff=64, n_query_ids=0, descriptor_dim=vocab["F"])).eval()
    question = Question("trip", "hazard", descriptor=np.repeat(x[:1], batch.n, 0),
                        bin_edges=[0.0, 600.0, 1800.0, 3600.0, math.inf])
    with torch.no_grad():
        out = model(batch, questions=[question])
    pmf = out["outputs"]["trip"]["pmf"]
    assert pmf.shape == (batch.n, 5) and torch.isfinite(pmf).all()
    assert torch.allclose(pmf.sum(-1), torch.ones(batch.n), atol=1e-5)
    for got, expected in zip(out["dependency_set"], batch.dependency_set()):
        assert np.array_equal(got.numpy(), expected)
    assert out["stats"]["n_items"] == int(batch.validity().sum()) > 0
