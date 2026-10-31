import json

import numpy as np
import pandas as pd
import pytest

from cpsdefense.data import hai
from cpsdefense.data.splits import window_starts
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_file

T0 = pd.Timestamp("2021-01-01 00:00:00")
FMT = "%Y-%m-%d %H:%M:%S"
DEV_SPLITS = ("train", "val", "dev_test")


def _frame(start, n, seed, attack=None, with_label=True, gap_at=None, gap_s=30):
    rng = np.random.default_rng(seed)
    offsets = np.arange(n, dtype=np.int64)
    if gap_at is not None:
        offsets[gap_at:] += gap_s
    ts = start + pd.to_timedelta(offsets, unit="s")
    df = pd.DataFrame({"timestamp": ts.strftime(FMT)})
    df["f1"] = rng.normal(size=n).round(5)
    df["f2"] = rng.uniform(0, 100, size=n).round(5)
    df["C1"] = 5.0
    df["C2"] = 1.0
    if with_label:
        df["Attack"] = 0 if attack is None else attack
    return df


def _labels(n, runs):
    y = np.zeros(n, int)
    for a, b in runs:
        y[a:b] = 1
    return y


def _write_manifest(raw_root):
    files = {p.relative_to(raw_root).as_posix(): {"sha256": sha256_file(p)}
             for p in sorted(raw_root.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    (raw_root / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def _config(version, release, **overrides):
    return {"val_fraction": 0.2, "purge_s": 20, "purge_adjacent_files": True, "max_gap_s": 1,
            "verify_checksums": True, "releases": {version: release}, **overrides}


@pytest.fixture
def release_a(tmp_path):
    raw_root = tmp_path / "raw"
    src = raw_root / "hai" / "hai-90.01"
    src.mkdir(parents=True)
    test_a = _frame(T0, 400, 1, attack=_labels(400, [(100, 120), (300, 350)]))
    test_a.loc[300:349, "C1"] = 7.0
    test_a.to_csv(src / "testA.csv", index=False)
    _frame(T0 + pd.Timedelta(days=1), 1000, 2, gap_at=500).to_csv(src / "trainA.csv", index=False)
    test_b_start = T0 + pd.Timedelta(days=2)
    _frame(test_b_start, 300, 3, attack=_labels(300, [(50, 60)])).to_csv(src / "testB.csv", index=False)
    _frame(test_b_start + pd.Timedelta(seconds=300), 800, 4).to_csv(src / "trainB.csv", index=False)
    _write_manifest(raw_root)
    release = {"label_column": "Attack", "train": ["trainA", "trainB"], "dev_test": ["testA"], "test": ["testB"]}
    return raw_root, _config("90.01", release)


def _hms(s):
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _summary_text(start, end, stated_count, events, total_s=None, percent=None, rows=None):
    spans = [int((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds()) for a, b in events]
    total_s = sum(spans) if total_s is None else total_s
    percent = 100 * total_s / rows if percent is None else percent
    hms = _hms
    lines = [
        "=" * 81,
        "File Name\t: split_label.csv ",
        f"StartTime\t\t: {start}",
        f" EndTime\t\t: {end}",
        f"  Attacks\t\t: {stated_count} times",
        "-------------------------------|  Attacks |--------------------------------------",
        f" Duration\t\t: min {hms(min(spans))} / max {hms(max(spans))} / avg {hms(int(np.mean(spans)))}",
        f" Total\t\t: {hms(total_s)} ({percent:.2f} percent)",
        "-" * 81,
        "index\tStart Time\t\tEnd Time\t\t\tDuration",
    ]
    lines += [f"[{i + 1:3d}]\t{a}\t{b}\t{hms(s)}" for i, ((a, b), s) in enumerate(zip(events, spans))]
    return "\n".join(lines + ["=" * 81]) + "\n"


def _minute_label(ts):
    return f"{ts:%Y-%m-%d} {ts.hour}:{ts:%M}"


@pytest.fixture
def release_b(tmp_path):
    raw_root = tmp_path / "raw"
    src = raw_root / "hai" / "hai-91.01"
    src.mkdir(parents=True)
    _frame(T0, 500, 5, with_label=False).to_csv(src / "tr1.csv", index=False)
    te1 = _frame(T0 + pd.Timedelta(days=1), 200, 6, with_label=False)
    te1.to_csv(src / "te1.csv", index=False)
    pd.DataFrame({"timestamp": te1["timestamp"], "label": _labels(200, [(20, 30)])}).to_csv(src / "lab1.csv", index=False)
    start2 = T0 + pd.Timedelta(days=2, seconds=1)
    te2 = _frame(start2, 600, 7, with_label=False)
    te2.to_csv(src / "te2.csv", index=False)
    ts2 = pd.to_datetime(te2["timestamp"])
    y2 = _labels(600, [(100, 110), (400, 420)])
    pd.DataFrame({"timestamp": [_minute_label(t.floor("min")) for t in ts2], "label": y2}).to_csv(src / "lab2.csv", index=False)
    events = [(str(ts2[100]), str(ts2[109])), (str(ts2[400]), str(ts2[419]))]
    (src / "summary2.txt").write_text(_summary_text(ts2.iloc[0], ts2.iloc[-1], 3, events, rows=600), encoding="utf-8")
    _write_manifest(raw_root)
    release = {
        "label_column": None, "train": ["tr1"], "dev_test": ["te1"], "test": ["te2"],
        "label_files": {"te1": {"file": "lab1.csv", "timestamp_resolution": "s"},
                        "te2": {"file": "lab2.csv", "timestamp_resolution": "min"}},
        "summaries": {"te2": "summary2.txt"},
    }
    return raw_root, _config("91.01", release)


def _prepare(release, tmp_path, version):
    raw_root, config = release
    out_root = tmp_path / "processed"
    return hai.prepare_release(version, config=config, raw_root=raw_root, out_root=out_root, verbose=False), out_root


def test_prepare_release_splits_purges_and_constants(release_a, tmp_path):
    meta, out_root = _prepare(release_a, tmp_path, "90.01")
    files = meta["files"]
    assert meta["train_labels_all_zero"]
    assert meta["file_order"] == ["testA", "trainA", "testB", "trainB"]
    assert meta["constant_columns"] == {"C1": 5.0, "C2": 1.0}
    assert meta["constant_columns_vary_outside_train"] == {"C1": ["dev_test"], "C2": []}
    assert meta["feature_columns"] == ["f1", "f2"]
    assert meta["columns"] == ["f1", "f2", "C1", "C2"]

    ranges = [(r["split"], r["start_row"], r["stop_row"]) for r in files["trainA"]["split_ranges"]]
    assert ranges == [("train", 0, 781), ("purge", 781, 800), ("val", 800, 1000)]
    ranges = [(r["split"], r["start_row"], r["stop_row"]) for r in files["trainB"]["split_ranges"]]
    assert ranges == [("purge", 0, 19), ("train", 19, 621), ("purge", 621, 640), ("val", 640, 800)]
    assert meta["adjacent_files"] == [{"before": "testB", "after": "trainB", "gap_s": 1.0,
                                       "splits": ["test", "train"], "purged": {"file": "trainB", "rows": 19}}]
    assert meta["purged_rows"] == {"testA": 0, "trainA": 19, "testB": 0, "trainB": 38}

    assert files["trainA"]["cadence"]["n_segments"] == 2
    assert files["trainA"]["cadence"]["gaps"][0]["row"] == 500
    assert files["trainA"]["cadence"]["gaps"][0]["gap_s"] == 31.0
    assert not files["trainA"]["cadence"]["regular_1s"]
    assert files["testA"]["cadence"]["regular_1s"]

    events = files["testA"]["events"]
    assert [(e["start_row"], e["stop_row"], e["duration_s"]) for e in events] == [(100, 120, 20.0), (300, 350, 50.0)]
    assert events[0]["start"] == "2021-01-01 00:01:40" and events[0]["end"] == "2021-01-01 00:01:59"
    assert files["testA"]["label_fraction"] == pytest.approx(70 / 400)
    assert files["trainA"]["label_source"] == "column Attack"
    assert all(f["source"]["sha256_verified"] for f in files.values())

    splits = meta["splits"]
    assert splits["train"]["rows"] == 781 + 602 and splits["val"]["rows"] == 200 + 160
    assert splits["train"]["n_segments"] == 3
    assert splits["dev_test"] == {"files": ["testA"], "rows": 400, "attack_rows": 70,
                                  "label_fraction": 70 / 400, "n_events": 2, "n_segments": 1}
    assert splits["test"]["n_events"] == 1

    saved = json.loads((out_root / "hai" / "90.01" / "meta.json").read_text(encoding="utf-8"))
    assert saved["splits"] == splits
    parquet = pd.read_parquet(out_root / "hai" / "90.01" / "trainA.parquet")
    assert list(parquet.columns) == ["timestamp", "segment", "attack", "f1", "f2", "C1", "C2"]
    assert parquet["f1"].dtype == np.float32 and parquet["attack"].dtype == np.int8
    assert parquet["segment"].tolist() == [0] * 500 + [1] * 500


def test_load_split_contract(release_a, tmp_path):
    _, out_root = _prepare(release_a, tmp_path, "90.01")
    train = hai.load_split("90.01", "train", root=out_root)
    assert train["X"].dtype == np.float32 and train["X"].shape == (1383, 2)
    assert train["y"].dtype == np.int8 and not train["y"].any()
    assert train["columns"] == ["f1", "f2"]
    assert train["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert list(dict.fromkeys(train["file"])) == ["trainA", "trainB"]
    for seg in np.unique(train["segment"]):
        assert np.unique(train["file"][train["segment"] == seg]).size == 1
    assert np.unique(train["segment"]).tolist() == [0, 1, 2]
    for s in window_starts(train["segment"], 50, 7):
        assert np.unique(train["file"][s:s + 50]).size == 1
        assert np.all(np.diff(train["timestamp"][s:s + 50]) == np.timedelta64(1, "s"))

    dev = hai.load_split("90.01", "dev_test", root=out_root)
    assert dev["y"].sum() == 70 and dev["X"].shape == (400, 2)
    wide = hai.load_split("90.01", "dev_test", root=out_root, columns=["f1", "C1"])
    assert wide["X"].shape == (400, 2) and wide["columns"] == ["f1", "C1"]
    with pytest.raises(ValueError):
        hai.load_split("90.01", "holdout", root=out_root)
    with pytest.raises(ValueError):
        hai.load_split("90.01", "val", root=out_root, columns=["nope"])
    with pytest.raises(FileNotFoundError):
        hai.load_split("90.02", "val", root=out_root)


def _min_distance_s(a, b):
    a = np.sort(a.astype("datetime64[s]").astype(np.int64))
    b = np.sort(b.astype("datetime64[s]").astype(np.int64))
    idx = np.clip(np.searchsorted(b, a), 1, b.size - 1)
    return int(np.min(np.minimum(np.abs(a - b[idx - 1]), np.abs(a - b[idx]))))


def test_fit_rows_are_purge_separated_from_other_splits(release_a, tmp_path):
    _, out_root = _prepare(release_a, tmp_path, "90.01")
    parts = {s: hai.load_split("90.01", s, root=out_root)["timestamp"] for s in hai.SPLITS}
    assert _min_distance_s(parts["train"], parts["val"]) >= 20
    for other in ("dev_test", "test"):
        assert _min_distance_s(parts["train"], parts[other]) >= 20
        assert _min_distance_s(parts["val"], parts[other]) >= 20


def test_adjacent_purge_can_be_disabled(release_a, tmp_path):
    raw_root, config = release_a
    config = {**config, "purge_adjacent_files": False}
    meta, _ = _prepare((raw_root, config), tmp_path, "90.01")
    assert meta["purged_rows"]["trainB"] == 19
    assert meta["adjacent_files"][0]["purged"] is None


def test_dev_test_next_to_test_loses_purge_rows():
    dev, te = np.arange(10000.0, 12000.0), np.arange(12001.0, 14000.0)
    codes, order, contacts = hai.assign_splits({"dev": "dev_test", "te": "test"}, {"dev": dev, "te": te}, 0.2, 300)
    assert order == ["dev", "te"]
    assert contacts == [{"before": "dev", "after": "te", "gap_s": 2.0, "splits": ["dev_test", "test"],
                         "purged": {"file": "dev", "rows": 298}}]
    assert np.all(codes["te"] == hai.SPLIT_CODE["test"])
    kept = dev[codes["dev"] == hai.SPLIT_CODE["dev_test"]]
    assert np.all(codes["dev"][kept.size:] == hai.SPLIT_CODE["purge"]) and te[0] - kept[-1] == 300

    codes, _, contacts = hai.assign_splits({"te": "test", "dev": "dev_test"}, {"te": dev, "dev": te}, 0.2, 300)
    assert contacts[0]["splits"] == ["test", "dev_test"] and contacts[0]["purged"] == {"file": "dev", "rows": 298}
    assert np.all(codes["te"] == hai.SPLIT_CODE["test"])
    assert te[codes["dev"] == hai.SPLIT_CODE["dev_test"]][0] - dev[-1] == 300

    codes, _, contacts = hai.assign_splits({"dev": "dev_test", "te": "test"}, {"dev": dev, "te": te}, 0.2, 300,
                                           purge_adjacent=False)
    assert contacts[0]["purged"] is None and not np.any(codes["dev"] == hai.SPLIT_CODE["purge"])


def test_same_split_contacts_are_only_recorded():
    stamps = {"a": np.arange(0.0, 100.0), "b": np.arange(101.0, 200.0), "c": np.arange(1000.0, 1100.0),
              "d": np.arange(1101.0, 1200.0)}
    roles = {"a": "dev_test", "b": "dev_test", "c": "test", "d": "test"}
    codes, _, contacts = hai.assign_splits(roles, stamps, 0.2, 50)
    assert [(c["before"], c["after"], c["purged"]) for c in contacts] == [("a", "b", None), ("c", "d", None)]
    assert not any(np.any(c == hai.SPLIT_CODE["purge"]) for c in codes.values())


def test_dev_test_file_purged_away_fails():
    stamps = {"dev": np.arange(11800.0, 12000.0), "te": np.arange(12001.0, 14000.0)}
    with pytest.raises(ValueError, match="dev: no dev_test rows left"):
        hai.assign_splits({"dev": "dev_test", "te": "test"}, stamps, 0.2, 300)


def test_prepare_purges_dev_test_before_adjacent_test(tmp_path):
    raw_root = tmp_path / "raw"
    src = raw_root / "hai" / "hai-92.01"
    src.mkdir(parents=True)
    _frame(T0, 500, 8).to_csv(src / "trainA.csv", index=False)
    dev_start = T0 + pd.Timedelta(days=1)
    _frame(dev_start, 400, 9, attack=_labels(400, [(100, 120)])).to_csv(src / "devA.csv", index=False)
    test_start = dev_start + pd.Timedelta(seconds=401)
    _frame(test_start, 300, 10, attack=_labels(300, [(50, 60)])).to_csv(src / "testA.csv", index=False)
    _write_manifest(raw_root)
    release = {"label_column": "Attack", "train": ["trainA"], "dev_test": ["devA"], "test": ["testA"]}
    meta, out_root = _prepare((raw_root, _config("92.01", release)), tmp_path, "92.01")
    assert meta["adjacent_files"] == [{"before": "devA", "after": "testA", "gap_s": 2.0,
                                       "splits": ["dev_test", "test"], "purged": {"file": "devA", "rows": 18}}]
    assert meta["purged_rows"] == {"trainA": 19, "devA": 18, "testA": 0}
    ranges = [(r["split"], r["start_row"], r["stop_row"]) for r in meta["files"]["devA"]["split_ranges"]]
    assert ranges == [("dev_test", 0, 382), ("purge", 382, 400)]
    assert [r["split"] for r in meta["files"]["testA"]["split_ranges"]] == ["test"]
    assert meta["splits"]["dev_test"]["rows"] == 382 and meta["splits"]["dev_test"]["n_events"] == 1
    assert meta["splits"]["test"]["rows"] == 300
    dev = hai.load_split("92.01", "dev_test", root=out_root)["timestamp"]
    test = hai.load_split("92.01", "test", root=out_root)["timestamp"]
    assert _min_distance_s(dev, test) == 20


def test_label_files_minute_resolution_and_summary(release_b, tmp_path):
    meta, out_root = _prepare(release_b, tmp_path, "91.01")
    files = meta["files"]
    assert files["tr1"]["label_source"].startswith("none")
    assert files["te1"]["label_file"]["timestamp_resolution"] == "s"
    assert files["te2"]["label_file"]["timestamp_resolution"] == "min"
    assert files["te1"]["label_file"]["require_summary_match"] is False
    assert files["te2"]["label_file"]["require_summary_match"] is True
    assert [(e["start_row"], e["stop_row"]) for e in files["te2"]["events"]] == [(100, 110), (400, 420)]
    check = files["te2"]["summary_check"]
    assert check["listed_match_derived"] and check["listed_count"] == 2 and check["stated_count"] == 3
    assert check["discrepancies"] == [
        "header states 3 attacks but lists 2",
        "header states 3 attacks, labels contain 2 events",
    ]
    assert meta["discrepancies"] == [f"te2: {d}" for d in check["discrepancies"]]
    test = hai.load_split("91.01", "test", root=out_root)
    assert test["y"].sum() == 30
    assert hai.load_split("91.01", "dev_test", root=out_root)["y"].sum() == 10


def _roll_minute_labels(release, shift=7):
    raw_root, config = release
    path = raw_root / "hai" / "hai-91.01" / "lab2.csv"
    frame = pd.read_csv(path, dtype={"timestamp": str})
    frame["label"] = np.roll(frame["label"].to_numpy(), shift)
    frame.to_csv(path, index=False)
    config["verify_checksums"] = False
    return raw_root, config


def test_minute_labels_must_match_summary_events(release_b, tmp_path):
    release = _roll_minute_labels(release_b)
    with pytest.raises(ValueError, match="te2: events derived from lab2.csv do not match summary2.txt: "
                                         "2 listed not in labels, 2 label events not listed"):
        _prepare(release, tmp_path, "91.01")
    assert not (tmp_path / "processed" / "hai" / "91.01" / "meta.json").exists()


def test_minute_labels_need_a_summary(release_b, tmp_path):
    raw_root, config = release_b
    release = config["releases"]["91.01"]
    del release["summaries"]["te2"]
    with pytest.raises(ValueError, match="te2: lab2.csv has 'min' timestamps.*summaries"):
        _prepare((raw_root, config), tmp_path, "91.01")
    release["summaries"]["te2"] = "summary2.txt"
    release["label_files"]["te1"]["require_summary_match"] = True
    with pytest.raises(ValueError, match="te1: lab1.csv has 's' timestamps.*summaries"):
        _prepare((raw_root, config), tmp_path, "91.01")


def test_summary_match_opt_out_only_records(release_b, tmp_path):
    raw_root, config = _roll_minute_labels(release_b)
    release = config["releases"]["91.01"]
    release["label_files"]["te2"]["require_summary_match"] = False
    meta, _ = _prepare((raw_root, config), tmp_path, "91.01")
    assert meta["files"]["te2"]["label_file"]["require_summary_match"] is False
    assert not meta["files"]["te2"]["summary_check"]["listed_match_derived"]
    assert "te2: listed events differ from label events: 2 listed not in labels, 2 label events not listed" \
        in meta["discrepancies"]
    del release["summaries"]["te2"]
    meta, _ = _prepare((raw_root, config), tmp_path, "91.01")
    assert "summary_check" not in meta["files"]["te2"] and meta["discrepancies"] == []


def test_summary_parse_and_total_mismatch(tmp_path):
    events = [("2022-08-17 01:27:06", "2022-08-17 01:29:18"), ("2022-08-17 03:37:02", "2022-08-17 03:39:13")]
    path = tmp_path / "s.txt"
    path.write_text(_summary_text("2022-08-17 00:00:01", "2022-08-17 04:00:00", 2, events, total_s=500, rows=14400), encoding="utf-8")
    summary = hai.parse_summary(path)
    assert summary["stated_count"] == 2 and summary["stated_total_s"] == 500
    assert summary["events"][1] == {"index": 2, "start": events[1][0], "end": events[1][1], "duration_s": 131}
    derived = [{"start": a, "end": b} for a, b in events]
    check = hai.check_summary(summary, derived, "2022-08-17 00:00:01", "2022-08-17 04:00:00", 14400)
    assert check["listed_match_derived"]
    assert any("stated total 500 s != 263 s" in d for d in check["discrepancies"])
    shifted = [{"start": "2022-08-17 01:27:07", "end": events[0][1]}, derived[1]]
    check = hai.check_summary(summary, shifted, "2022-08-17 00:00:01", "2022-08-17 04:00:00", 14400)
    assert not check["listed_match_derived"]
    assert any("1 listed not in labels, 1 label events not listed" in d for d in check["discrepancies"])


def test_align_labels_rules():
    data = pd.date_range("2022-08-17 00:00:01", periods=180, freq="s").to_numpy()
    y = np.zeros(180, int)
    assert hai.align_labels(data, data, y).dtype == np.int8
    minutes = pd.DatetimeIndex(data).floor("min").to_numpy()
    assert hai.align_labels(data, minutes, y, "min").sum() == 0
    with pytest.raises(ValueError, match="label rows"):
        hai.align_labels(data, data[:-1], y[:-1])
    with pytest.raises(ValueError, match="misaligned"):
        hai.align_labels(data, np.roll(data, 1), y)
    with pytest.raises(ValueError, match="misaligned"):
        hai.align_labels(data, minutes, y, "s")
    with pytest.raises(ValueError, match="misaligned"):
        hai.align_labels(data, np.r_[minutes[1:], minutes[-1:]], y, "min")
    with pytest.raises(ValueError, match="0/1"):
        hai.align_labels(data, data, np.full(180, 2))


def test_minute_labels_rejected_when_declared_exact(release_b, tmp_path):
    raw_root, config = release_b
    config["releases"]["91.01"]["label_files"]["te2"]["timestamp_resolution"] = "s"
    with pytest.raises(ValueError, match="do not match"):
        _prepare((raw_root, config), tmp_path, "91.01")


def test_label_file_row_count_mismatch_fails(release_b, tmp_path):
    raw_root, config = release_b
    path = raw_root / "hai" / "hai-91.01" / "lab1.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    config["verify_checksums"] = False
    with pytest.raises(ValueError, match="199 label rows vs 200 data rows"):
        _prepare((raw_root, config), tmp_path, "91.01")


def test_train_file_with_attack_fails(release_a, tmp_path):
    raw_root, config = release_a
    path = raw_root / "hai" / "hai-90.01" / "trainA.csv"
    df = pd.read_csv(path)
    df.loc[10, "Attack"] = 1
    df.to_csv(path, index=False)
    config["verify_checksums"] = False
    with pytest.raises(ValueError, match="train file has 1 attack rows"):
        _prepare((raw_root, config), tmp_path, "90.01")


def test_checksum_mismatch_fails(release_a, tmp_path):
    raw_root, config = release_a
    path = raw_root / "hai" / "hai-90.01" / "testA.csv"
    df = pd.read_csv(path)
    df.loc[0, "f1"] += 1.0
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="sha256"):
        _prepare(release_a, tmp_path, "90.01")


def test_bad_raw_files_fail(tmp_path):
    good = _frame(T0, 20, 0)
    dup = good.copy()
    dup.loc[5, "timestamp"] = dup.loc[4, "timestamp"]
    dup.to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        hai.load_raw_file(tmp_path / "dup.csv", label_column="Attack")
    nan = good.copy()
    nan.loc[3, "f2"] = np.nan
    nan.to_csv(tmp_path / "nan.csv", index=False)
    with pytest.raises(ValueError, match="non-finite values in \\['f2'\\]"):
        hai.load_raw_file(tmp_path / "nan.csv", label_column="Attack")
    good.to_csv(tmp_path / "good.csv", index=False)
    with pytest.raises(ValueError, match="label-like"):
        hai.load_raw_file(tmp_path / "good.csv")
    frame = hai.load_raw_file(tmp_path / "good.csv", label_column="Attack", max_gap_s=1)
    assert list(frame.columns) == ["timestamp", "segment", "attack", "f1", "f2", "C1", "C2"]
    assert frame["timestamp"].dtype == np.dtype("datetime64[ns]")


def test_role_assignment_errors(release_a, tmp_path):
    raw_root, config = release_a
    release = config["releases"]["90.01"]
    src = raw_root / "hai" / "hai-90.01"
    with pytest.raises(ValueError, match="not assigned"):
        hai.assign_roles({**release, "test": []}, src)
    with pytest.raises(ValueError, match="both"):
        hai.assign_roles({**release, "dev_test": ["testA", "trainA"]}, src)
    with pytest.raises(FileNotFoundError):
        hai.assign_roles({**release, "test": ["testB", "testC"]}, src)


def test_column_mismatch_between_files_fails(release_a, tmp_path):
    raw_root, config = release_a
    path = raw_root / "hai" / "hai-90.01" / "testB.csv"
    pd.read_csv(path).drop(columns=["f2"]).to_csv(path, index=False)
    config["verify_checksums"] = False
    with pytest.raises(ValueError, match="columns differ"):
        _prepare((raw_root, config), tmp_path, "90.01")


def test_repo_config_matches_protocol():
    config = load_config(hai.CONFIG)
    r22, r23 = config["releases"]["22.04"], config["releases"]["23.05"]
    assert r22["dev_test"] == ["test1", "test2"] and r22["test"] == ["test3", "test4"]
    assert r23["dev_test"] == ["hai-test1"] and r23["test"] == ["hai-test2"]
    assert config["val_fraction"] == 0.2 and config["purge_s"] >= 300
    assert config["purge_adjacent_files"] is True
    assert r23["label_files"]["hai-test2"]["timestamp_resolution"] == "min"
    assert r23["label_files"]["hai-test2"].get("require_summary_match", True) is True
    assert r23["summaries"]["hai-test2"] == "summary_label2.txt"


def test_heldout_guard_covers_repo_test_files(heldout_names):
    assert heldout_names == {"test3", "test4", "hai-test2", "label-test2", "summary_label2"}


@pytest.mark.data
def test_raw_23_05_label_alignment(raw_dir):
    src = raw_dir / "hai" / "hai-23.05"
    if not src.exists():
        pytest.skip("HAI 23.05 not downloaded")
    frame = hai.load_raw_file(src / "hai-test1.csv", label_file=src / "label-test1.csv", label_resolution="s")
    assert int(frame["attack"].sum()) == 2981
    events = hai.event_records(frame["timestamp"].to_numpy(), frame["attack"].to_numpy(), frame["segment"].to_numpy())
    assert len(events) == 14
    with pytest.raises(ValueError, match="do not match"):
        hai.read_label_file(src / "label-test1.csv", "min")
    summary = hai.parse_summary(src / "summary_label1.txt")
    assert summary["stated_count"] == 14 and len(summary["events"]) == 14


@pytest.mark.data
def test_raw_22_04_train_labels_are_zero(raw_dir):
    src = raw_dir / "hai" / "hai-22.04"
    if not src.exists():
        pytest.skip("HAI 22.04 not downloaded")
    train = hai.load_raw_file(src / "train1.csv", label_column="Attack")
    assert len(train) == 93601 and not train["attack"].any()
    assert train.drop(columns=["timestamp", "segment", "attack"]).dtypes.eq(np.float32).all()
    test = hai.load_raw_file(src / "test1.csv", label_column="Attack")
    assert int(test["attack"].sum()) == 885


def _processed_meta(processed_dir, version):
    if not (processed_dir / "hai" / version / "meta.json").exists():
        pytest.skip(f"HAI {version} not prepared")
    return hai.load_meta(version, root=processed_dir)


def _min_distance_to_ranges_s(timestamps, ranges):
    t = timestamps.astype("datetime64[s]").astype(np.int64)
    best = np.inf
    for r in ranges:
        lo, hi = (pd.Timestamp(r[k]).value // 10**9 for k in ("start", "end"))
        best = min(best, int(np.min(np.maximum(np.maximum(lo - t, t - hi), 0))))
    return best


@pytest.mark.data
@pytest.mark.parametrize("version", ["22.04", "23.05"])
def test_processed_release_is_leakage_free(processed_dir, version):
    meta = _processed_meta(processed_dir, version)
    assert meta["train_labels_all_zero"]
    assert not set(meta["feature_columns"]) & set(meta["constant_columns"])
    purge = meta["config"]["purge_s"]
    for stem, info in meta["files"].items():
        covered = sum(r["rows"] for r in info["split_ranges"])
        assert covered == info["rows"]
        assert info["source"]["sha256"] is not None
    parts = {}
    for split in DEV_SPLITS:
        data = hai.load_split(version, split, root=processed_dir)
        assert data["X"].shape == (meta["splits"][split]["rows"], len(meta["feature_columns"]))
        assert np.isfinite(data["X"]).all()
        assert int(data["y"].sum()) == meta["splits"][split]["attack_rows"]
        assert np.unique(data["segment"]).size == meta["splits"][split]["n_segments"]
        seg_file = pd.DataFrame({"s": data["segment"], "f": data["file"]}).drop_duplicates()
        assert seg_file["s"].is_unique
        parts[split] = data["timestamp"]
    assert _min_distance_s(parts["train"], parts["val"]) >= purge
    for fit in ("train", "val"):
        assert _min_distance_s(parts[fit], parts["dev_test"]) >= purge

    test_files = meta["config"]["release"]["test"]
    assert meta["splits"]["test"]["files"] == [s for s in meta["file_order"] if s in test_files]
    ranges = []
    for stem in test_files:
        info = meta["files"][stem]
        assert info["role"] == "test" and meta["purged_rows"][stem] == 0
        (r,) = info["split_ranges"]
        assert (r["split"], r["start_row"], r["stop_row"], r["rows"]) == ("test", 0, info["rows"], info["rows"])
        assert (r["start"], r["end"]) == (info["start"], info["end"])
        edges = [0, *(x for e in info["events"] for x in (e["start_row"], e["stop_row"])), info["rows"]]
        assert np.all(np.diff(edges) >= 0) and all(e["start_row"] < e["stop_row"] for e in info["events"])
        assert sum(e["stop_row"] - e["start_row"] for e in info["events"]) == info["attack_rows"]
        ranges.append(r)
    test = meta["splits"]["test"]
    assert test["rows"] == sum(meta["files"][s]["rows"] for s in test_files)
    assert test["attack_rows"] == sum(meta["files"][s]["attack_rows"] for s in test_files)
    assert test["n_events"] == sum(len(meta["files"][s]["events"]) for s in test_files)
    for split in DEV_SPLITS:
        assert _min_distance_to_ranges_s(parts[split], ranges) >= purge


@pytest.mark.data
def test_processed_23_05_events_match_summaries(processed_dir):
    meta = _processed_meta(processed_dir, "23.05")
    for stem, n in (("hai-test1", 14), ("hai-test2", 38)):
        check = meta["files"][stem]["summary_check"]
        assert check["listed_match_derived"] and check["derived_count"] == n
    assert meta["files"]["hai-test1"]["summary_check"]["discrepancies"] == []
    heldout = meta["files"]["hai-test2"]
    assert heldout["label_file"]["timestamp_resolution"] == "min"
    assert heldout["label_file"]["require_summary_match"] is True
    assert heldout["attack_rows"] == 8403
    assert heldout["summary_check"]["stated_count"] == 40 and heldout["summary_check"]["listed_count"] == 38
    assert "header states 40 attacks but lists 38" in heldout["summary_check"]["discrepancies"]
    assert meta["splits"]["test"]["n_events"] == 38 and meta["splits"]["dev_test"]["n_events"] == 14
