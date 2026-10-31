import datetime as dt
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from cpsdefense.paths import PROCESSED_DIR, RAW_DIR, ensure_dir
from cpsdefense.utils.seeding import spawn_generators

NETFLOW_RAW = RAW_DIR / "netflow"
NETFLOW_PROCESSED = PROCESSED_DIR / "netflow"
DATASETS = ("NF-UNSW-NB15-v3", "NF-ToN-IoT-v3")

_STR = pa.dictionary(pa.int32(), pa.string())

SCHEMA = pa.schema([
    ("FLOW_START_MILLISECONDS", pa.int64()),
    ("FLOW_END_MILLISECONDS", pa.int64()),
    ("IPV4_SRC_ADDR", _STR),
    ("L4_SRC_PORT", pa.uint16()),
    ("IPV4_DST_ADDR", _STR),
    ("L4_DST_PORT", pa.uint16()),
    ("PROTOCOL", pa.uint8()),
    ("L7_PROTO", pa.float32()),
    ("IN_BYTES", pa.int64()),
    ("IN_PKTS", pa.int32()),
    ("OUT_BYTES", pa.int64()),
    ("OUT_PKTS", pa.int32()),
    ("TCP_FLAGS", pa.uint8()),
    ("CLIENT_TCP_FLAGS", pa.uint8()),
    ("SERVER_TCP_FLAGS", pa.uint8()),
    ("FLOW_DURATION_MILLISECONDS", pa.int32()),
    ("DURATION_IN", pa.int32()),
    ("DURATION_OUT", pa.int32()),
    ("MIN_TTL", pa.uint8()),
    ("MAX_TTL", pa.uint8()),
    ("LONGEST_FLOW_PKT", pa.uint16()),
    ("SHORTEST_FLOW_PKT", pa.uint16()),
    ("MIN_IP_PKT_LEN", pa.uint16()),
    ("MAX_IP_PKT_LEN", pa.uint16()),
    ("SRC_TO_DST_SECOND_BYTES", pa.float32()),
    ("DST_TO_SRC_SECOND_BYTES", pa.float32()),
    ("RETRANSMITTED_IN_BYTES", pa.int64()),
    ("RETRANSMITTED_IN_PKTS", pa.int32()),
    ("RETRANSMITTED_OUT_BYTES", pa.int64()),
    ("RETRANSMITTED_OUT_PKTS", pa.int32()),
    ("SRC_TO_DST_AVG_THROUGHPUT", pa.int64()),
    ("DST_TO_SRC_AVG_THROUGHPUT", pa.int64()),
    ("NUM_PKTS_UP_TO_128_BYTES", pa.int32()),
    ("NUM_PKTS_128_TO_256_BYTES", pa.int32()),
    ("NUM_PKTS_256_TO_512_BYTES", pa.int32()),
    ("NUM_PKTS_512_TO_1024_BYTES", pa.int32()),
    ("NUM_PKTS_1024_TO_1514_BYTES", pa.int32()),
    ("TCP_WIN_MAX_IN", pa.uint16()),
    ("TCP_WIN_MAX_OUT", pa.uint16()),
    ("ICMP_TYPE", pa.uint16()),
    ("ICMP_IPV4_TYPE", pa.uint8()),
    ("DNS_QUERY_ID", pa.uint16()),
    ("DNS_QUERY_TYPE", pa.uint16()),
    ("DNS_TTL_ANSWER", pa.uint32()),
    ("FTP_COMMAND_RET_CODE", pa.uint16()),
    ("SRC_TO_DST_IAT_MIN", pa.int32()),
    ("SRC_TO_DST_IAT_MAX", pa.int32()),
    ("SRC_TO_DST_IAT_AVG", pa.int32()),
    ("SRC_TO_DST_IAT_STDDEV", pa.int32()),
    ("DST_TO_SRC_IAT_MIN", pa.int32()),
    ("DST_TO_SRC_IAT_MAX", pa.int32()),
    ("DST_TO_SRC_IAT_AVG", pa.int32()),
    ("DST_TO_SRC_IAT_STDDEV", pa.int32()),
    ("Label", pa.int8()),
    ("Attack", _STR),
])

CSV_COLUMNS = tuple(SCHEMA.names)
TIME_COLUMNS = ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS")
ENDPOINT_COLUMNS = ("IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT")
LABEL_COLUMNS = ("Label", "Attack")
FLOW_FEATURES = tuple(c for c in CSV_COLUMNS if c not in TIME_COLUMNS + ENDPOINT_COLUMNS + LABEL_COLUMNS)

SPLITS = ("train", "val", "test")
PURGED = 3
SPLIT_NAMES = SPLITS + ("purged",)
SPLIT_KINDS = ("temporal", "random")
_SPLIT_TYPE = pa.dictionary(pa.int8(), pa.string())
STORED_SCHEMA = pa.schema([
    pa.field("row_id", pa.int64()),
    *SCHEMA,
    pa.field("split_temporal", _SPLIT_TYPE),
    pa.field("split_random", _SPLIT_TYPE),
])

_READ_TYPES = {f.name: (pa.float64() if f.type == pa.float32() else f.type) for f in SCHEMA}
_READ_TYPES["L7_PROTO"] = _STR
_REQUIRED = TIME_COLUMNS + LABEL_COLUMNS

_PANDAS_INT = {
    pa.int8(): pd.Int8Dtype(), pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(), pa.int64(): pd.Int64Dtype(),
    pa.uint8(): pd.UInt8Dtype(), pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(), pa.uint64(): pd.UInt64Dtype(),
}


class ChecksumError(Exception):
    pass


class _Sha1Reader:
    def __init__(self, raw):
        self.raw = raw
        self.sha1 = hashlib.sha1()
        self.nbytes = 0
        self.closed = False

    def read(self, size=-1):
        data = self.raw.read(size)
        self.sha1.update(data)
        self.nbytes += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return False

    def close(self):
        self.raw.close()
        self.closed = True


class Bag:
    def __init__(self, zip_path, name=None):
        self.path = Path(zip_path)
        self.name = name or self.path.stem
        self.csv_rel = f"data/{self.name}.csv"
        self.archive = zipfile.ZipFile(self.path)
        try:
            names = self.archive.namelist()
            members = [m for m in names if m == self.csv_rel or m.endswith("/" + self.csv_rel)]
            if len(members) != 1:
                raise FileNotFoundError(f"{self.path.name}: expected one */{self.csv_rel}, found {members}")
            self.member = members[0]
            self.root = self.member[: -len(self.csv_rel)]
            manifest = self.root + "manifest-sha1.txt"
            if manifest not in names:
                raise ChecksumError(f"{self.path.name}: no {manifest}")
            self.manifest = {}
            for line in self.archive.read(manifest).decode("utf-8").splitlines():
                if line.strip():
                    digest, rel = line.split(maxsplit=1)
                    self.manifest[rel.strip()] = digest.lower()
            if self.csv_rel not in self.manifest:
                raise ChecksumError(f"{self.path.name}: {self.csv_rel} missing from manifest-sha1.txt")
            self.csv_size = self.archive.getinfo(self.member).file_size
        except BaseException:
            self.archive.close()
            raise

    def open_csv_member(self):
        return self.archive.open(self.member)

    def verify_payload(self, skip=()):
        verified = {}
        for rel, expected in sorted(self.manifest.items()):
            if rel in skip:
                continue
            if self.root + rel not in self.archive.namelist():
                raise ChecksumError(f"{rel} is listed in the manifest but missing from the bag")
            digest = hashlib.sha1()
            with self.archive.open(self.root + rel) as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            _check_sha1(rel, expected, digest.hexdigest())
            verified[rel] = expected
        return verified

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _check_sha1(rel, expected, actual):
    if actual != expected:
        raise ChecksumError(f"{rel}: sha1 {actual} != manifest {expected}")


def check_header(header):
    missing = [c for c in CSV_COLUMNS if c not in header]
    extra = [c for c in header if c not in CSV_COLUMNS]
    if missing or extra or len(header) != len(set(header)):
        raise ValueError(f"unexpected NetFlow v3 header: missing={missing} extra={extra}")


def _utc(ms):
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return (epoch + dt.timedelta(milliseconds=int(ms))).isoformat(timespec="milliseconds")


def _global_codes(array, lookup, names):
    lut = np.empty(len(array.dictionary), dtype=np.int16)
    for i, value in enumerate(array.dictionary.to_pylist()):
        if value not in lookup:
            lookup[value] = len(names)
            names.append(value)
        lut[i] = lookup[value]
    return lut[array.indices.to_numpy()]


def _check_float32(name, wide, narrow, offset):
    created = np.isinf(narrow.to_numpy(zero_copy_only=False)) & ~np.isinf(wide.to_numpy(zero_copy_only=False))
    if created.any():
        raise ValueError(f"{name}: {int(created.sum())} finite value(s) beyond the float32 range, "
                         f"first at row {offset + int(np.argmax(created))}")


def _narrow(batch, l7_raw, offset):
    arrays = []
    for field in SCHEMA:
        col = batch.column(field.name)
        if field.name == "L7_PROTO":
            l7_raw.update(v for v in col.dictionary.to_pylist() if v is not None)
            wide = pc.take(pc.cast(col.dictionary, pa.float64()), col.indices)
            col = pc.take(pc.cast(col.dictionary, pa.float32()), col.indices)
            _check_float32(field.name, wide, col, offset)
        elif field.type == pa.float32():
            wide, col = col, pc.cast(col, pa.float32(), safe=False)
            _check_float32(field.name, wide, col, offset)
        arrays.append(col)
    return pa.RecordBatch.from_arrays(arrays, schema=SCHEMA)


def _hash_frame(batch):
    return batch.to_pandas(types_mapper=_PANDAS_INT.get)


def _row_hash(frame, columns):
    return pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()


def _stream(bag, stage, block_size, rows_per_part, log):
    source = _Sha1Reader(bag.open_csv_member())
    try:
        reader = pcsv.open_csv(
            source,
            read_options=pcsv.ReadOptions(block_size=block_size),
            convert_options=pcsv.ConvertOptions(column_types=_READ_TYPES, strings_can_be_null=True),
        )
        header = reader.schema.names
        check_header(header)
        cols = {k: [] for k in ("start", "end", "attack", "label", "hash_all", "hash_features")}
        lookup, class_names, l7_raw = {}, [], set()
        nulls = dict.fromkeys(CSV_COLUMNS, 0)
        nonfinite = {c: 0 for c in CSV_COLUMNS if _READ_TYPES[c] == pa.float64()}
        parts, buffer, buffered, offset = [], [], 0, 0
        next_report = time.time() + 30

        def flush():
            name = f"part-{len(parts):05d}.parquet"
            pq.write_table(pa.Table.from_batches(buffer), stage / name, compression="lz4")
            parts.append(name)
            buffer.clear()

        for batch in reader:
            n = batch.num_rows
            for c in CSV_COLUMNS:
                nulls[c] += batch.column(c).null_count
            for c in nonfinite:
                values = batch.column(c).to_numpy(zero_copy_only=False)
                nonfinite[c] += int(np.count_nonzero(np.isinf(values)))
            bad = [c for c in _REQUIRED if batch.column(c).null_count]
            if bad:
                raise ValueError(f"null values in required columns {bad} near row {offset}")
            cols["start"].append(batch.column("FLOW_START_MILLISECONDS").to_numpy())
            cols["end"].append(batch.column("FLOW_END_MILLISECONDS").to_numpy())
            cols["label"].append(batch.column("Label").to_numpy())
            cols["attack"].append(_global_codes(batch.column("Attack"), lookup, class_names))
            frame = _hash_frame(batch)
            cols["hash_all"].append(_row_hash(frame, CSV_COLUMNS))
            cols["hash_features"].append(_row_hash(frame, FLOW_FEATURES))
            narrow = _narrow(batch, l7_raw, offset)
            row_id = pa.array(np.arange(offset, offset + n, dtype=np.int64))
            buffer.append(pa.RecordBatch.from_arrays([row_id, *narrow.columns], names=["row_id", *SCHEMA.names]))
            buffered += n
            offset += n
            if buffered >= rows_per_part:
                flush()
                buffered = 0
            if log and time.time() > next_report:
                log(f"[{bag.name}] {offset:,} rows, {source.nbytes / 1e9:.2f} / {bag.csv_size / 1e9:.2f} GB")
                next_report = time.time() + 30
        if buffer:
            flush()
        while source.read(1 << 20):
            pass
    finally:
        source.close()

    if offset == 0:
        raise ValueError(f"{bag.csv_rel} has a header but no rows")
    if source.nbytes != bag.csv_size:
        raise ChecksumError(f"{bag.csv_rel}: read {source.nbytes} bytes, zip says {bag.csv_size}")
    _check_sha1(bag.csv_rel, bag.manifest[bag.csv_rel], source.sha1.hexdigest())

    arrays = {k: np.concatenate(v) if v else np.empty(0) for k, v in cols.items()}
    return {
        "header": header,
        "arrays": arrays,
        "class_names": class_names,
        "l7_raw": l7_raw,
        "nulls": {c: v for c, v in nulls.items() if v},
        "nonfinite": {c: v for c, v in nonfinite.items() if v},
        "parts": parts,
        "sha1": source.sha1.hexdigest(),
        "bytes": source.nbytes,
    }


def temporal_thresholds(starts, fractions=(0.6, 0.2, 0.2)):
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"fractions must be three positive numbers summing to 1, got {fractions}")
    n = len(starts)
    ordered = np.sort(starts)
    thresholds = [int(ordered[min(int(np.floor(q * n)), n - 1)]) for q in np.cumsum(fractions)[:-1]]
    if not thresholds[0] < thresholds[1] or thresholds[0] <= ordered[0]:
        raise ValueError(f"split thresholds collapse on tied start times: {thresholds}")
    return thresholds


def temporal_split(starts, ends, thresholds, purge_gap_ms):
    codes = np.searchsorted(np.asarray(thresholds), starts, side="right").astype(np.int8)
    eff_end = np.maximum(starts, ends)
    dropped = []
    for k, threshold in enumerate(thresholds):
        in_split = codes == k
        straddle = in_split & (eff_end >= threshold)
        purge = in_split & ~straddle & (eff_end >= threshold - purge_gap_ms)
        codes[straddle | purge] = PURGED
        dropped.append({
            "boundary": f"{SPLITS[k]}->{SPLITS[k + 1]}",
            "threshold_ms": int(threshold),
            "straddle": int(straddle.sum()),
            "purge_gap": int(purge.sum()),
        })
    return codes, dropped


def _allocate(n, weights):
    raw = n * np.asarray(weights, dtype=float) / np.sum(weights)
    counts = np.floor(raw).astype(np.int64)
    rest = n - int(counts.sum())
    if rest:
        counts[np.argsort(-(raw - counts), kind="stable")[:rest]] += 1
    return counts


def stratified_random_split(strata, keep, sizes, rng):
    codes = np.full(len(strata), PURGED, dtype=np.int8)
    for stratum in np.unique(strata[keep]):
        idx = np.flatnonzero(keep & (strata == stratum))
        labels = np.repeat(np.arange(len(sizes), dtype=np.int8), _allocate(len(idx), sizes))
        rng.shuffle(labels)
        codes[idx] = labels
    return codes


def _duplicates(hashes, attack=None):
    n = len(hashes)
    if n == 0:
        return {"duplicate_rows": 0, "fraction": 0.0}
    order = np.argsort(hashes, kind="stable")
    ordered = hashes[order]
    first = np.ones(n, dtype=bool)
    first[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(first)
    out = {"duplicate_rows": int(n - len(starts)), "fraction": float((n - len(starts)) / n)}
    if attack is not None:
        labels = attack[order]
        conflict = np.minimum.reduceat(labels, starts) != np.maximum.reduceat(labels, starts)
        sizes = np.diff(np.append(starts, n))
        out["groups_with_conflicting_attack"] = int(conflict.sum())
        out["rows_in_conflicting_groups"] = int(sizes[conflict].sum())
    return out


def _test_rows_seen_in_train(hashes, codes):
    train = np.unique(hashes[codes == 0])
    return int(np.isin(hashes[codes == 2], train).sum())


def _class_counts(counts, class_names):
    return {class_names[i]: int(counts[i]) for i in np.argsort(-counts, kind="stable") if counts[i]}


def _split_summary(codes, attack, labels, class_names, starts=None):
    k = len(class_names)
    per_split, counts = {}, {}
    for code, split in enumerate(SPLIT_NAMES):
        mask = codes == code
        counts[split] = np.bincount(attack[mask], minlength=k)
        entry = {
            "rows": int(mask.sum()),
            "attack_fraction": float(np.mean(labels[mask] != 0)) if mask.any() else None,
            "class_counts": _class_counts(counts[split], class_names),
        }
        if starts is not None and mask.any():
            entry["start_utc"] = [_utc(starts[mask].min()), _utc(starts[mask].max())]
        per_split[split] = entry
    absent = {f"absent_from_{s}": sorted(class_names[i] for i in range(k) if counts[s][i] == 0) for s in SPLITS}
    return {"splits": per_split, **absent}


def _largest_gaps(ordered, top=5):
    if len(ordered) < 2:
        return []
    gaps = np.diff(ordered)
    top = min(top, len(gaps))
    idx = np.argpartition(gaps, -top)[-top:]
    idx = idx[np.argsort(-gaps[idx], kind="stable")]
    return [{"gap_s": float(gaps[i] / 1000), "from_utc": _utc(ordered[i]), "to_utc": _utc(ordered[i + 1])} for i in idx]


def _profile(pass1):
    a = pass1["arrays"]
    starts, ends, labels, attack = a["start"], a["end"], a["label"], a["attack"]
    names = pass1["class_names"]
    n = len(starts)
    ordered = np.sort(starts)
    class_counts = np.bincount(attack, minlength=len(names))
    benign = [i for i, c in enumerate(names) if c.lower() == "benign"]
    is_benign = np.isin(attack, benign)
    by_attack = {}
    for i in np.argsort(-class_counts):
        values, counts = np.unique(labels[attack == i], return_counts=True)
        by_attack[names[i]] = {str(v): int(c) for v, c in zip(values, counts)}
    l7_floats = {}
    for raw in pass1["l7_raw"]:
        l7_floats.setdefault(float(np.float32(raw)), []).append(raw)
    return {
        "rows": n,
        "time_range": {
            "start_min_ms": int(ordered[0]),
            "start_max_ms": int(ordered[-1]),
            "end_max_ms": int(ends.max()),
            "start_min_utc": _utc(ordered[0]),
            "start_max_utc": _utc(ordered[-1]),
            "end_max_utc": _utc(ends.max()),
            "span_hours": float((ends.max() - ordered[0]) / 3.6e6),
            "largest_start_gaps": _largest_gaps(ordered),
        },
        "end_before_start": {"rows": int(np.sum(ends < starts)), "fraction": float(np.mean(ends < starts))},
        "file_order": {
            "start_decreases": int(np.sum(np.diff(starts) < 0)),
            "sorted_by_start": bool(np.all(np.diff(starts) >= 0)),
        },
        "class_counts": _class_counts(class_counts, names),
        "label_counts": {str(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True))},
        "label_attack_consistency": {
            "benign_classes": [names[i] for i in benign],
            "benign_with_nonzero_label": int(np.sum(is_benign & (labels != 0))),
            "attack_with_label_0": int(np.sum(~is_benign & (labels == 0))),
            "consistent": bool(np.all(is_benign == (labels == 0))),
            "labels_by_attack": by_attack,
        },
        "duplicates": {
            "method": (
                "64-bit row hash (pandas.util.hash_pandas_object) of the parsed values before float32 "
                "narrowing; exact up to hash collisions"
            ),
            "expected_hash_collisions": float(n * n / 2**65),
            "all_columns": _duplicates(a["hash_all"]),
            "flow_features": {
                "columns": "FLOW_FEATURES (all columns except timestamps, IPs, ports, Label, Attack)",
                **_duplicates(a["hash_features"], attack),
            },
        },
        "nulls": pass1["nulls"],
        "nonfinite": pass1["nonfinite"],
        "l7_proto": {
            "distinct_raw_codes": len(pass1["l7_raw"]),
            "float32_collisions": {str(k): sorted(v) for k, v in l7_floats.items() if len(v) > 1},
        },
    }


def _write_final(parts, stage, partial, temporal, random, row_group_rows, compression):
    dictionary = pa.array(SPLIT_NAMES, pa.string())
    files, offset = [], 0
    for name in parts:
        table = pq.read_table(stage / name)
        n = table.num_rows
        for column, codes in (("split_temporal", temporal), ("split_random", random)):
            array = pa.DictionaryArray.from_arrays(pa.array(codes[offset:offset + n], pa.int8()), dictionary)
            table = table.append_column(pa.field(column, _SPLIT_TYPE), array)
        if not table.schema.equals(STORED_SCHEMA):
            raise TypeError(f"stored schema drifted in {name}: {table.schema}")
        pq.write_table(table, partial / name, compression=compression, row_group_size=row_group_rows)
        del table
        (stage / name).unlink()
        info = pq.ParquetFile(partial / name).metadata
        files.append({"file": name, "rows": n, "row_groups": info.num_row_groups,
                      "bytes": (partial / name).stat().st_size})
        offset += n
    return files


def prepare_netflow(zip_path, out_dir=NETFLOW_PROCESSED, name=None, fractions=(0.6, 0.2, 0.2),
                    purge_gap_s=60, seed=0, block_size=32 << 20, rows_per_part=2_000_000,
                    row_group_rows=250_000, compression="zstd", log=None):
    started = time.time()
    out_dir = ensure_dir(out_dir)
    with Bag(zip_path, name) as bag:
        final = out_dir / bag.name
        stage = out_dir / f".{bag.name}.stage"
        partial = out_dir / f".{bag.name}.partial"
        for path in (stage, partial):
            shutil.rmtree(path, ignore_errors=True)
        try:
            payload = bag.verify_payload(skip={bag.csv_rel})
            pass1 = _stream(bag, ensure_dir(stage), block_size, rows_per_part, log)
            if log:
                log(f"[{bag.name}] sha1 verified, {len(pass1['arrays']['start']):,} rows; profiling and splitting")
            a = pass1["arrays"]
            profile = _profile(pass1)
            purge_gap_ms = int(round(purge_gap_s * 1000))
            thresholds = temporal_thresholds(a["start"], fractions)
            temporal, dropped = temporal_split(a["start"], a["end"], thresholds, purge_gap_ms)
            keep = temporal != PURGED
            sizes = np.bincount(temporal[keep], minlength=3)
            rng = spawn_generators(seed, ["random_split"])["random_split"]
            random = stratified_random_split(a["attack"], keep, sizes, rng)
            names = pass1["class_names"]
            split_meta = {
                "temporal": {
                    "fractions": [float(f) for f in fractions],
                    "purge_gap_s": float(purge_gap_s),
                    "thresholds_ms": thresholds,
                    "thresholds_utc": [_utc(t) for t in thresholds],
                    "rule": (
                        "train: start < t1; val: t1 <= start < t2; test: start >= t2. A train/val flow is "
                        "purged unless max(start, end) < next threshold - purge gap."
                    ),
                    "dropped": dropped,
                    "dropped_total": int((~keep).sum()),
                    **_split_summary(temporal, a["attack"], a["label"], names, a["start"]),
                    "test_rows_with_feature_duplicate_in_train": _test_rows_seen_in_train(
                        a["hash_features"], temporal),
                },
                "random": {
                    "seed": seed,
                    "rule": (
                        "stratified by Attack over the rows kept by the temporal split; each Attack class is "
                        "split in proportion to the temporal kept sizes (largest remainder within the class), "
                        "so each split's size differs from the temporal one by fewer rows than there are "
                        "Attack classes; temporally purged rows stay purged"
                    ),
                    **_split_summary(random, a["attack"], a["label"], names),
                    "test_rows_with_feature_duplicate_in_train": _test_rows_seen_in_train(
                        a["hash_features"], random),
                },
            }
            del pass1["arrays"], a
            files = _write_final(pass1["parts"], stage, ensure_dir(partial), temporal, random,
                                 row_group_rows, compression)
            meta = {
                "name": bag.name,
                "source": {
                    "zip": bag.path.name,
                    "zip_bytes": bag.path.stat().st_size,
                    "member": bag.member,
                    "member_bytes": pass1["bytes"],
                    "sha1": pass1["sha1"],
                    "sha1_verified": True,
                    "other_payload_sha1_verified": payload,
                },
                "header": pass1["header"],
                "schema": {f.name: str(f.type) for f in STORED_SCHEMA},
                **profile,
                "split": split_meta,
                "files": files,
                "params": {
                    "block_size": block_size,
                    "rows_per_part": rows_per_part,
                    "row_group_rows": row_group_rows,
                    "compression": compression,
                },
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(time.time() - started, 1),
            }
            (partial / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if final.exists():
                shutil.rmtree(final)
            partial.rename(final)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return meta


def load_meta(name, root=None):
    path = Path(root or NETFLOW_PROCESSED) / name / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_netflow(name, split, columns=None, split_kind="temporal", root=None, as_pandas=True):
    if split_kind not in SPLIT_KINDS:
        raise ValueError(f"split_kind must be one of {SPLIT_KINDS}, got {split_kind!r}")
    if split not in SPLITS + ("all",):
        raise ValueError(f"split must be one of {SPLITS + ('all',)}, got {split!r}")
    path = Path(root or NETFLOW_PROCESSED) / name
    files = sorted(path.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet parts under {path}; run scripts/prepare_netflow.py")
    dataset = ds.dataset(files, format="parquet")
    if columns is not None:
        columns = list(columns)
        missing = [c for c in columns if c not in dataset.schema.names]
        if missing:
            raise KeyError(f"unknown columns {missing}")
    condition = None if split == "all" else ds.field(f"split_{split_kind}") == split
    table = dataset.to_table(columns=columns, filter=condition)
    return table.to_pandas() if as_pandas else table
