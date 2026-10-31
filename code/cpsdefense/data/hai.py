import datetime as dt
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from cpsdefense.data.splits import events_from_labels, run_bounds, segment_by_gaps, time_split_with_purge
from cpsdefense.paths import PROCESSED_DIR, RAW_DIR, ensure_dir
from cpsdefense.utils.config import load_config
from cpsdefense.utils.hashing import sha256_file

CONFIG = "data/hai.yaml"
SPLITS = ("train", "val", "dev_test", "test")
ROLES = ("train", "dev_test", "test")
SPLIT_CODE = {"purge": -1, "train": 0, "val": 1, "dev_test": 2, "test": 3}
CODE_SPLIT = {code: name for name, code in SPLIT_CODE.items()}
TIME_FORMAT = {"s": "%Y-%m-%d %H:%M:%S", "min": "%Y-%m-%d %H:%M"}
META_COLUMNS = ["timestamp", "segment", "attack"]


def normalize_version(version):
    version = str(version)
    return version[4:] if version.startswith("hai-") else version


def raw_release_dir(version, raw_root=None):
    return Path(raw_root or RAW_DIR) / "hai" / f"hai-{normalize_version(version)}"


def processed_release_dir(version, root=None):
    return Path(root or PROCESSED_DIR) / "hai" / normalize_version(version)


def to_seconds(timestamps):
    return np.asarray(timestamps, dtype="datetime64[ns]").astype(np.int64) / 1e9


def _ts(value):
    return str(pd.Timestamp(value))


def parse_timestamps(values, resolution, source):
    fmt = TIME_FORMAT[resolution]
    try:
        parsed = pd.to_datetime(pd.Series(values), format=fmt)
    except ValueError as error:
        raise ValueError(f"{source}: timestamps do not match {fmt!r}: {error}") from None
    if parsed.isna().any():
        raise ValueError(f"{source}: {int(parsed.isna().sum())} missing timestamps")
    return parsed.to_numpy(dtype="datetime64[ns]")


def check_binary(values, source):
    values = np.asarray(values, dtype=np.float64)
    ok = (values == 0) | (values == 1)
    if not ok.all():
        raise ValueError(f"{source}: labels must be 0/1, found {np.unique(values[~ok])[:5].tolist()}")
    return values.astype(np.int8)


def read_csv_frame(path):
    with open(path, encoding="utf-8-sig") as handle:
        header = handle.readline().strip().split(",")
    if "timestamp" not in header:
        raise ValueError(f"{path}: no 'timestamp' column")
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: duplicate column names")
    types = {name: pa.string() if name == "timestamp" else pa.float64() for name in header}
    table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(column_types=types))
    return table.to_pandas()


def read_label_file(path, resolution):
    frame = pd.read_csv(path, dtype={"timestamp": str})
    if list(frame.columns) != ["timestamp", "label"]:
        raise ValueError(f"{path}: expected columns timestamp,label, got {list(frame.columns)}")
    return parse_timestamps(frame["timestamp"], resolution, path), frame["label"].to_numpy()


def align_labels(data_timestamps, label_timestamps, labels, resolution="s", source="labels"):
    if resolution not in TIME_FORMAT:
        raise ValueError(f"unknown timestamp resolution {resolution!r}")
    data_ts = pd.DatetimeIndex(data_timestamps)
    label_ts = pd.DatetimeIndex(label_timestamps)
    if len(label_ts) != len(data_ts) or len(labels) != len(label_ts):
        raise ValueError(f"{source}: {len(label_ts)} label rows vs {len(data_ts)} data rows")
    expected = data_ts if resolution == "s" else data_ts.floor("min")
    mismatch = np.flatnonzero(expected.values != label_ts.values)
    if mismatch.size:
        i = mismatch[0]
        raise ValueError(
            f"{source}: {mismatch.size} rows misaligned, first at row {i}: "
            f"label {label_ts[i]} vs data {data_ts[i]} (resolution {resolution})"
        )
    return check_binary(labels, source)


def load_raw_file(path, label_column=None, label_file=None, label_resolution="s", max_gap_s=1.0):
    if label_column is not None and label_file is not None:
        raise ValueError(f"{path}: give label_column or label_file, not both")
    raw = read_csv_frame(path)
    if raw.empty:
        raise ValueError(f"{path}: no rows")
    timestamp = parse_timestamps(raw.pop("timestamp"), "s", path)
    t = to_seconds(timestamp)
    bad_step = np.flatnonzero(np.diff(t) <= 0)
    if bad_step.size:
        i = bad_step[0]
        raise ValueError(
            f"{path}: timestamps not strictly increasing at row {i + 1} ({timestamp[i]} -> {timestamp[i + 1]})"
        )
    if label_column is not None:
        if label_column not in raw:
            raise ValueError(f"{path}: label column {label_column!r} missing")
        attack = check_binary(raw.pop(label_column), f"{path}:{label_column}")
    elif label_file is not None:
        label_ts, labels = read_label_file(label_file, label_resolution)
        attack = align_labels(timestamp, label_ts, labels, label_resolution, source=str(label_file))
    else:
        attack = np.zeros(len(raw), np.int8)
    label_like = [c for c in raw.columns if c.lower() in ("attack", "label") or c.lower().startswith("attack_")]
    if label_like:
        raise ValueError(f"{path}: label-like columns {label_like} would leak into the features")
    values = raw.to_numpy(np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(f"{path}: non-finite values in {raw.columns[~finite.all(axis=0)].tolist()}")
    head = pd.DataFrame({
        "timestamp": timestamp,
        "segment": segment_by_gaps(t, max_gap_s).astype(np.int32),
        "attack": attack,
    })
    return pd.concat([head, pd.DataFrame(values.astype(np.float32), columns=raw.columns)], axis=1)


def cadence_report(timestamps, segments):
    t = to_seconds(timestamps)
    steps, counts = np.unique(np.diff(t), return_counts=True)
    starts, _ = run_bounds(segments)
    gaps = [
        {"row": int(b), "after": _ts(timestamps[b - 1]), "before": _ts(timestamps[b]),
         "gap_s": float(t[b] - t[b - 1])}
        for b in starts[1:]
    ]
    return {
        "step_counts": {f"{s:g}": int(c) for s, c in zip(steps, counts)},
        "regular_1s": bool(np.all(steps == 1)),
        "n_segments": int(starts.size),
        "gaps": gaps,
    }


def event_records(timestamps, labels, segments):
    t = to_seconds(timestamps)
    return [
        {"start": _ts(timestamps[a]), "end": _ts(timestamps[b - 1]), "duration_s": float(t[b - 1] - t[a] + 1),
         "start_row": a, "stop_row": b}
        for a, b in events_from_labels(labels, segments)
    ]

_TS = r"(\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2})"
_HMS = r"(\d+):(\d{2}):(\d{2})"


def _hms(match, first):
    h, m, s = (int(match.group(first + k)) for k in range(3))
    return h * 3600 + m * 60 + s


def _span_s(start, end):
    return int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds())


def parse_summary(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {"file": Path(path).name}
    m = re.search(r"Attacks\s*:\s*(\d+)\s*times", text)
    out["stated_count"] = int(m.group(1)) if m else None
    for key, label in (("stated_start", "StartTime"), ("stated_end", "EndTime")):
        m = re.search(label + r"\s*:\s*" + _TS, text)
        out[key] = _ts(m.group(1)) if m else None
    m = re.search(r"Total\s*:\s*" + _HMS + r"\s*\(\s*([\d.]+)\s*percent\)", text)
    out["stated_total_s"] = _hms(m, 1) if m else None
    out["stated_percent"] = float(m.group(4)) if m else None
    m = re.search(r"min\s*" + _HMS + r"\s*/\s*max\s*" + _HMS + r"\s*/\s*avg\s*" + _HMS, text)
    out["stated_min_s"], out["stated_max_s"], out["stated_avg_s"] = (
        (_hms(m, 1), _hms(m, 4), _hms(m, 7)) if m else (None, None, None)
    )
    out["events"] = [
        {"index": int(m.group(1)), "start": _ts(m.group(2)), "end": _ts(m.group(3)), "duration_s": _hms(m, 4)}
        for m in re.finditer(r"\[\s*(\d+)\]\s+" + _TS + r"\s+" + _TS + r"\s+" + _HMS, text)
    ]
    return out


def _event_spans(events):
    return [(e["start"], e["end"]) for e in events]


def _event_difference(listed, derived):
    return (f"{len(set(listed) - set(derived))} listed not in labels, "
            f"{len(set(derived) - set(listed))} label events not listed")


def check_summary(summary, events, first, last, rows):
    listed = _event_spans(summary["events"])
    derived = _event_spans(events)
    listed_spans = [_span_s(a, b) for a, b in listed]
    derived_total = sum(_span_s(a, b) for a, b in derived)
    listed_total = sum(e["duration_s"] for e in summary["events"])
    issues = []
    if summary["stated_count"] != len(listed):
        issues.append(f"header states {summary['stated_count']} attacks but lists {len(listed)}")
    if summary["stated_count"] != len(derived):
        issues.append(f"header states {summary['stated_count']} attacks, labels contain {len(derived)} events")
    if listed != derived:
        issues.append(f"listed events differ from label events: {_event_difference(listed, derived)}")
    wrong = [e["index"] for e, s in zip(summary["events"], listed_spans) if e["duration_s"] != s]
    if wrong:
        issues.append(f"listed durations differ from end - start for events {wrong}")
    if summary["stated_total_s"] != listed_total:
        issues.append(f"stated total {summary['stated_total_s']} s != {listed_total} s summed over listed events")
    if summary["stated_percent"] is not None and abs(summary["stated_percent"] - 100 * listed_total / rows) > 0.006:
        issues.append(f"stated {summary['stated_percent']}% != {100 * listed_total / rows:.2f}% from listed events")
    if listed_spans:
        if summary["stated_min_s"] != min(listed_spans) or summary["stated_max_s"] != max(listed_spans):
            issues.append("stated min/max duration differ from listed events")
        if summary["stated_avg_s"] is not None and abs(summary["stated_avg_s"] - np.mean(listed_spans)) >= 1:
            issues.append(f"stated avg {summary['stated_avg_s']} s != {np.mean(listed_spans):.1f} s over listed events")
    if summary["stated_start"] != first or summary["stated_end"] != last:
        issues.append(f"stated range {summary['stated_start']} .. {summary['stated_end']} != file {first} .. {last}")
    return {
        "file": summary["file"],
        "stated_count": summary["stated_count"],
        "listed_count": len(listed),
        "derived_count": len(derived),
        "listed_match_derived": listed == derived,
        "stated_total_s": summary["stated_total_s"],
        "listed_total_s": listed_total,
        "derived_total_s": derived_total,
        "stated_percent": summary["stated_percent"],
        "discrepancies": issues,
    }


def load_manifest(raw_root):
    path = Path(raw_root) / "MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8")).get("files", {}) if path.exists() else {}


def source_record(path, raw_root, manifest, verify):
    key = Path(path).relative_to(raw_root).as_posix()
    entry = manifest.get(key) or {}
    record = {"path": key, "sha256": entry.get("sha256"), "in_manifest": bool(entry)}
    if verify:
        actual = sha256_file(path)
        if record["sha256"] and actual != record["sha256"]:
            raise ValueError(f"{key}: sha256 {actual} != MANIFEST.json {record['sha256']}")
        record["sha256_verified"] = bool(record["sha256"])
        if not record["sha256"]:
            record["sha256_computed"] = actual
    return record


def assign_roles(release, src_dir):
    roles = {}
    for role in ROLES:
        for stem in release.get(role) or []:
            if stem in roles:
                raise ValueError(f"{stem} assigned to both {roles[stem]} and {role}")
            roles[stem] = role
    if "train" not in roles.values():
        raise ValueError("release has no train files")
    label_names = {spec["file"] for spec in (release.get("label_files") or {}).values()}
    present = {p.stem for p in Path(src_dir).glob("*.csv") if p.name not in label_names}
    missing, extra = sorted(set(roles) - present), sorted(present - set(roles))
    if missing:
        raise FileNotFoundError(f"{src_dir}: missing data files {missing}")
    if extra:
        raise ValueError(f"{src_dir}: data files not assigned to any split: {extra}")
    return roles


def assign_splits(roles, stamps, val_fraction, purge_s, purge_adjacent=True):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    codes = {}
    for stem, role in roles.items():
        if role == "train":
            parts = time_split_with_purge(stamps[stem], (1 - val_fraction, val_fraction), purge_s)
            codes[stem] = parts.astype(np.int8)
        else:
            codes[stem] = np.full(stamps[stem].size, SPLIT_CODE[role], np.int8)
    order = sorted(roles, key=lambda s: stamps[s][0])
    base = {stem: c.copy() for stem, c in codes.items()}
    contacts = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            gap = stamps[b][0] - stamps[a][-1]
            if gap <= 0:
                raise ValueError(f"files {a} and {b} overlap in time")
            if gap >= purge_s:
                continue
            ca, cb = int(base[a][-1]), int(base[b][0])
            record = {"before": a, "after": b, "gap_s": float(gap),
                      "splits": [CODE_SPLIT[ca], CODE_SPLIT[cb]], "purged": None}
            if purge_adjacent and ca != cb:
                if cb < ca:
                    target, mask = b, stamps[b] < stamps[a][-1] + purge_s
                else:
                    target, mask = a, stamps[a] > stamps[b][0] - purge_s
                codes[target][mask] = SPLIT_CODE["purge"]
                record["purged"] = {"file": target, "rows": int(mask.sum())}
            contacts.append(record)
    for stem, role in roles.items():
        for split in ("train", "val") if role == "train" else (role,):
            if not np.any(codes[stem] == SPLIT_CODE[split]):
                raise ValueError(f"{stem}: no {split} rows left after purging")
    return codes, order, contacts


def split_ranges(codes, timestamps):
    starts, stops = run_bounds(codes)
    return [
        {"split": CODE_SPLIT[int(codes[a])], "start_row": int(a), "stop_row": int(b), "rows": int(b - a),
         "start": _ts(timestamps[a]), "end": _ts(timestamps[b - 1])}
        for a, b in zip(starts, stops)
    ]


def constant_columns(parquet_paths, columns, codes):
    lo = {s: np.full(len(columns), np.inf) for s in SPLITS}
    hi = {s: np.full(len(columns), -np.inf) for s in SPLITS}
    for stem, path in parquet_paths.items():
        values = pd.read_parquet(path, columns=columns).to_numpy()
        for split in SPLITS:
            mask = codes[stem] == SPLIT_CODE[split]
            if mask.any():
                lo[split] = np.minimum(lo[split], values[mask].min(axis=0))
                hi[split] = np.maximum(hi[split], values[mask].max(axis=0))
    constant = {c: float(lo["train"][i]) for i, c in enumerate(columns) if lo["train"][i] == hi["train"][i]}
    vary = {}
    for i, c in enumerate(columns):
        if c in constant:
            present = [s for s in SPLITS[1:] if np.isfinite(lo[s][i])]
            vary[c] = [s for s in present if lo[s][i] != constant[c] or hi[s][i] != constant[c]]
    return constant, vary


def split_summary(order, files, labels, segments):
    splits = {}
    for split in SPLITS:
        members, rows, attack_rows, n_events, n_segments = [], 0, 0, 0, 0
        for stem in order:
            for r in files[stem]["split_ranges"]:
                if r["split"] != split:
                    continue
                a, b = r["start_row"], r["stop_row"]
                if stem not in members:
                    members.append(stem)
                rows += b - a
                attack_rows += int(labels[stem][a:b].sum())
                n_events += len(events_from_labels(labels[stem][a:b], segments[stem][a:b]))
                n_segments += run_bounds(segments[stem][a:b])[0].size
        splits[split] = {"files": members, "rows": rows, "attack_rows": attack_rows,
                         "label_fraction": attack_rows / rows if rows else None,
                         "n_events": n_events, "n_segments": n_segments}
    return splits


def prepare_release(version, config=None, raw_root=None, out_root=None, verbose=True):
    version = normalize_version(version)
    config = load_config(CONFIG) if config is None else config
    releases = config.get("releases") or {}
    if version not in releases:
        raise KeyError(f"HAI release {version!r} not in config (have {sorted(releases)})")
    release = releases[version]
    raw_root = Path(raw_root or RAW_DIR)
    src_dir = raw_release_dir(version, raw_root)
    out_dir = ensure_dir(processed_release_dir(version, out_root))
    meta_path = out_dir / "meta.json"
    meta_path.unlink(missing_ok=True)

    roles = assign_roles(release, src_dir)
    label_files = release.get("label_files") or {}
    summaries = release.get("summaries") or {}
    for stem in [*label_files, *summaries]:
        if stem not in roles:
            raise ValueError(f"label/summary entry for unknown file {stem}")
    manifest = load_manifest(raw_root)
    verify = bool(config.get("verify_checksums", True))
    max_gap = float(config.get("max_gap_s", 1))
    purge_s = float(config.get("purge_s", 300))
    label_column = release.get("label_column")

    files, times, labels, segments = {}, {}, {}, {}
    columns = None
    for stem, role in roles.items():
        path = src_dir / f"{stem}.csv"
        spec = label_files.get(stem)
        resolution = spec.get("timestamp_resolution", "s") if spec else "s"
        if role != "train" and not (label_column or spec):
            raise ValueError(f"{stem}: {role} file has no labels")
        require_match = spec.get("require_summary_match", resolution == "min") if spec else False
        if not isinstance(require_match, bool):
            raise ValueError(f"{stem}: require_summary_match must be true or false, got {require_match!r}")
        if require_match and stem not in summaries:
            raise ValueError(
                f"{stem}: {spec['file']} has {resolution!r} timestamps and needs its events checked against a "
                f"summary file, but none is listed under 'summaries' (opt out with require_summary_match: false)"
            )
        frame = load_raw_file(
            path,
            label_column=None if spec else label_column,
            label_file=src_dir / spec["file"] if spec else None,
            label_resolution=resolution,
            max_gap_s=max_gap,
        )
        sensors = [c for c in frame.columns if c not in META_COLUMNS]
        if columns is None:
            columns = sensors
        elif sensors != columns:
            raise ValueError(f"{stem}: columns differ from the release's first file")
        ts = frame["timestamp"].to_numpy()
        attack = frame["attack"].to_numpy()
        seg = frame["segment"].to_numpy()
        if role == "train" and attack.any():
            raise ValueError(f"{stem}: train file has {int(attack.sum())} attack rows")
        info = {
            "role": role,
            "source": source_record(path, raw_root, manifest, verify),
            "rows": len(frame),
            "start": _ts(ts[0]),
            "end": _ts(ts[-1]),
            "cadence": cadence_report(ts, seg),
            "attack_rows": int(attack.sum()),
            "label_fraction": float(attack.mean()),
            "events": event_records(ts, attack, seg),
        }
        if spec:
            info["label_source"] = f"file {spec['file']}"
            info["label_file"] = {
                "source": source_record(src_dir / spec["file"], raw_root, manifest, verify),
                "timestamp_resolution": resolution,
                "alignment": "identical timestamps" if resolution == "s"
                else "same row count; label timestamp == data timestamp truncated to the minute, row by row",
                "require_summary_match": require_match,
            }
        elif label_column:
            info["label_source"] = f"column {label_column}"
        else:
            info["label_source"] = "none (normal-operation file, all rows 0)"
        if stem in summaries:
            summary_path = src_dir / summaries[stem]
            summary = parse_summary(summary_path)
            check = check_summary(summary, info["events"], info["start"], info["end"], len(frame))
            if require_match and not check["listed_match_derived"]:
                difference = _event_difference(_event_spans(summary["events"]), _event_spans(info["events"]))
                raise ValueError(
                    f"{stem}: events derived from {spec['file']} do not match {summaries[stem]}: {difference} "
                    f"(label rows may be misaligned; opt out with require_summary_match: false)"
                )
            check["source"] = source_record(summary_path, raw_root, manifest, verify)
            info["summary_check"] = check
        frame.to_parquet(out_dir / f"{stem}.parquet", index=False)
        files[stem], times[stem], labels[stem], segments[stem] = info, ts, attack, seg
        if verbose:
            print(f"[hai-{version}] {stem}: {info['rows']} rows {info['start']} .. {info['end']}, "
                  f"{len(info['events'])} events, {len(info['cadence']['gaps'])} gaps", flush=True)

    stamps = {stem: to_seconds(ts) for stem, ts in times.items()}
    val_fraction = float(config.get("val_fraction", 0.2))
    purge_adjacent = bool(config.get("purge_adjacent_files", True))
    codes, order, contacts = assign_splits(roles, stamps, val_fraction, purge_s, purge_adjacent)
    for stem in roles:
        files[stem]["split_ranges"] = split_ranges(codes[stem], times[stem])
    frames = {stem: out_dir / f"{stem}.parquet" for stem in order}
    constant, vary = constant_columns(frames, columns, codes)
    purged = {stem: int(np.sum(codes[stem] == SPLIT_CODE["purge"])) for stem in order}
    discrepancies = [
        f"{stem}: {issue}"
        for stem in order
        for issue in files[stem].get("summary_check", {}).get("discrepancies", [])
    ]
    meta = {
        "dataset": "HAI",
        "version": version,
        "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": {**{k: v for k, v in config.items() if k != "releases"}, "release": release},
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ("hai.py", "splits.py")},
        "columns": columns,
        "feature_columns": [c for c in columns if c not in constant],
        "constant_columns": constant,
        "constant_columns_vary_outside_train": vary,
        "train_labels_all_zero": all(files[s]["attack_rows"] == 0 for s, r in roles.items() if r == "train"),
        "file_order": order,
        "files": files,
        "adjacent_files": contacts,
        "purged_rows": purged,
        "splits": split_summary(order, files, labels, segments),
        "discrepancies": discrepancies,
    }
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)
    return meta


def summarize(meta):
    return {
        "version": meta["version"],
        "n_features": len(meta["feature_columns"]),
        "constant_columns": sorted(meta["constant_columns"]),
        "constant_but_vary_outside_train": {c: s for c, s in meta["constant_columns_vary_outside_train"].items() if s},
        "splits": {
            s: {"files": len(v["files"]), **{k: v[k] for k in ("rows", "label_fraction", "n_events", "n_segments")}}
            for s, v in meta["splits"].items()
        },
        "gaps": {stem: len(f["cadence"]["gaps"]) for stem, f in meta["files"].items()},
        "irregular_cadence": [stem for stem, f in meta["files"].items() if not f["cadence"]["regular_1s"]],
        "purged_rows": meta["purged_rows"],
        "adjacent_files": meta["adjacent_files"],
        "discrepancies": meta["discrepancies"],
    }


def load_meta(version, root=None):
    path = processed_release_dir(version, root) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_hai.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(version, split, root=None, columns=None):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    meta = load_meta(version, root)
    columns = list(meta["feature_columns"] if columns is None else columns)
    unknown = sorted(set(columns) - set(meta["columns"]))
    if unknown:
        raise ValueError(f"unknown columns {unknown}")
    base = processed_release_dir(version, root)
    X, y, timestamp, segment, file = [], [], [], [], []
    offset = 0
    for stem in meta["file_order"]:
        ranges = [r for r in meta["files"][stem]["split_ranges"] if r["split"] == split]
        if not ranges:
            continue
        frame = pd.read_parquet(base / f"{stem}.parquet", columns=[*META_COLUMNS, *columns])
        expected = meta["files"][stem]["rows"]
        if len(frame) != expected:
            raise ValueError(f"{stem}.parquet has {len(frame)} rows, meta says {expected}; re-run preparation")
        for r in ranges:
            part = frame.iloc[r["start_row"]:r["stop_row"]]
            seg = part["segment"].to_numpy()
            local = np.cumsum(np.r_[True, seg[1:] != seg[:-1]]) - 1
            X.append(part[columns].to_numpy(np.float32))
            y.append(part["attack"].to_numpy(np.int8))
            timestamp.append(part["timestamp"].to_numpy(dtype="datetime64[ns]"))
            segment.append(local + offset)
            file.append(np.full(len(part), stem))
            offset += int(local[-1]) + 1
    if not X:
        raise ValueError(f"split {split!r} of HAI {meta['version']} is empty")
    return {
        "X": np.concatenate(X),
        "y": np.concatenate(y),
        "timestamp": np.concatenate(timestamp),
        "segment": np.concatenate(segment).astype(np.int64),
        "file": np.concatenate(file),
        "columns": columns,
    }
