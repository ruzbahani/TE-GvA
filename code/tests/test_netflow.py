import datetime as dt
import hashlib
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from cpsdefense.data import netflow as nf

NAME = "NF-TEST-v3"
BASE = 1_600_000_000_000
N = 1000
GAP_S = 60
T1, T2 = BASE + 600_000, BASE + 800_000
EXPECTED_ROWS = {"NF-UNSW-NB15-v3": 2_365_424, "NF-ToN-IoT-v3": 27_520_260}


def synthetic_frame(seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(N)
    frame = pd.DataFrame({c: rng.integers(0, 200, N) for c in nf.CSV_COLUMNS})
    frame["FLOW_START_MILLISECONDS"] = BASE + i * 1000
    frame["FLOW_END_MILLISECONDS"] = BASE + i * 1000 + 10
    frame["IPV4_SRC_ADDR"] = [f"10.0.0.{k % 5}" for k in i]
    frame["IPV4_DST_ADDR"] = [f"192.168.1.{k % 7}" for k in i]
    frame["L7_PROTO"] = np.array(["7.37", "5.0", "0.0"], dtype=object)[i % 3]
    frame["SRC_TO_DST_SECOND_BYTES"] = [repr(float(v)) for v in rng.random(N) * 1000]
    frame["DST_TO_SRC_SECOND_BYTES"] = rng.random(N) * 1000
    frame.loc[3, "SRC_TO_DST_SECOND_BYTES"] = ""
    frame.loc[4, "SRC_TO_DST_SECOND_BYTES"] = "inf"
    attack = np.full(N, "Benign", dtype=object)
    attack[i % 10 == 3] = "DoS"
    attack[(i >= 100) & (i < 400) & (i % 10 == 7)] = "Reconnaissance"
    attack[(i >= 900) & (i % 10 == 5)] = "Worms"
    attack[950] = "Worms"
    frame["Attack"] = attack
    frame["Label"] = (attack != "Benign").astype(int)
    frame.loc[250, "Label"] = 1
    frame.loc[5, "FLOW_END_MILLISECONDS"] = BASE + 5000 - 5
    frame.loc[500, "FLOW_END_MILLISECONDS"] = BASE + 650_000
    frame.loc[700, "FLOW_END_MILLISECONDS"] = BASE + 900_000
    frame.loc[11] = frame.loc[10]
    frame.loc[950, list(nf.FLOW_FEATURES)] = frame.loc[30, list(nf.FLOW_FEATURES)]
    rows = np.arange(N)
    rows[[20, 21]] = [21, 20]
    frame = frame.iloc[rows].reset_index(drop=True)
    order = list(nf.CSV_COLUMNS)
    rng.shuffle(order)
    return frame[order]


def make_bag(directory, frame, name=NAME, csv_sha=None, include_csv_entry=True):
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode()
    features = b"Feature,Description\nIN_BYTES,Incoming number of bytes\n"
    lines = [f"{hashlib.sha1(features).hexdigest()} data/NetFlow_v3_Features.csv"]
    if include_csv_entry:
        lines.insert(0, f"{csv_sha or hashlib.sha1(csv_bytes).hexdigest()} data/{name}.csv")
    bag = "0123abcd_NFV3DATA-TEST"
    path = directory / f"{name}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{bag}/bagit.txt", "BagIt-Version: 0.96\n")
        archive.writestr(f"{bag}/manifest-sha1.txt", "\n".join(lines) + "\n")
        archive.writestr(f"{bag}/data/{name}.csv", csv_bytes)
        archive.writestr(f"{bag}/data/NetFlow_v3_Features.csv", features)
    return path, csv_bytes


def convert(zip_path, out_dir, **kwargs):
    params = dict(purge_gap_s=GAP_S, block_size=16_384, rows_per_part=250, row_group_rows=100)
    params.update(kwargs)
    return nf.prepare_netflow(zip_path, out_dir, name=NAME, **params)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("netflow")
    zip_path, csv_bytes = make_bag(directory, synthetic_frame())
    out = directory / "processed"
    meta = convert(zip_path, out)
    return {"zip": zip_path, "csv": csv_bytes, "out": out, "meta": meta}


def test_conversion_writes_verified_parts(prepared):
    meta, out = prepared["meta"], prepared["out"]
    assert meta["rows"] == N
    assert meta["source"]["sha1"] == hashlib.sha1(prepared["csv"]).hexdigest()
    assert meta["source"]["sha1_verified"]
    assert "data/NetFlow_v3_Features.csv" in meta["source"]["other_payload_sha1_verified"]
    parts = sorted((out / NAME).glob("part-*.parquet"))
    assert len(parts) >= 3 and [f["file"] for f in meta["files"]] == [p.name for p in parts]
    assert sum(f["rows"] for f in meta["files"]) == N
    for part in parts:
        info = pq.ParquetFile(part).metadata
        assert all(info.row_group(g).num_rows <= 100 for g in range(info.num_row_groups))
    assert not list(out.glob(".*"))
    assert set(meta["header"]) == set(nf.CSV_COLUMNS)


def test_schema_and_values(prepared):
    out = prepared["out"]
    schema = pq.read_schema(next((out / NAME).glob("part-*.parquet")))
    assert schema.remove_metadata().equals(nf.STORED_SCHEMA)
    assert schema.field("FLOW_START_MILLISECONDS").type == pa.int64()
    assert schema.field("L4_DST_PORT").type == pa.uint16()
    assert schema.field("IN_BYTES").type == pa.int64()
    assert schema.field("SRC_TO_DST_SECOND_BYTES").type == pa.float32()
    assert schema.field("IPV4_SRC_ADDR").type == pa.dictionary(pa.int32(), pa.string())
    assert schema.field("Label").type == pa.int8()
    assert schema.field("Attack").type == pa.dictionary(pa.int32(), pa.string())

    table = nf.load_netflow(NAME, "all", root=out, as_pandas=False)
    assert table.column("row_id").to_pylist() == list(range(N))
    frame = table.to_pandas()
    source = synthetic_frame()
    for column in ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IN_BYTES", "Label"):
        np.testing.assert_array_equal(frame[column].to_numpy(), source[column].to_numpy())
    np.testing.assert_array_equal(frame["Attack"].astype(str), source["Attack"])
    np.testing.assert_array_equal(frame["IPV4_DST_ADDR"].astype(str), source["IPV4_DST_ADDR"])
    l7 = frame["L7_PROTO"].to_numpy()
    assert l7.dtype == np.float32 and l7[0] == np.float32(7.37) and l7[1] == np.float32(5.0)
    rate = frame["SRC_TO_DST_SECOND_BYTES"]
    assert np.isnan(rate[3]) and np.isinf(rate[4])
    np.testing.assert_allclose(rate[10], float(source.loc[10, "SRC_TO_DST_SECOND_BYTES"]), rtol=1e-6)


def test_profile(prepared):
    meta = prepared["meta"]
    assert meta["time_range"]["start_min_ms"] == BASE
    assert meta["time_range"]["start_max_ms"] == BASE + 999_000
    assert meta["time_range"]["start_min_utc"] == "2020-09-13T12:26:40.000+00:00"
    assert meta["end_before_start"]["rows"] == 1
    assert meta["file_order"] == {"start_decreases": 1, "sorted_by_start": False}
    assert meta["nulls"] == {"SRC_TO_DST_SECOND_BYTES": 1}
    assert meta["nonfinite"] == {"SRC_TO_DST_SECOND_BYTES": 1}
    assert meta["duplicates"]["all_columns"]["duplicate_rows"] == 1
    features = meta["duplicates"]["flow_features"]
    assert features["duplicate_rows"] == 2
    assert features["groups_with_conflicting_attack"] == 1
    assert features["rows_in_conflicting_groups"] == 2
    consistency = meta["label_attack_consistency"]
    assert consistency["benign_with_nonzero_label"] == 1
    assert consistency["attack_with_label_0"] == 0
    assert not consistency["consistent"]
    assert meta["class_counts"]["Worms"] == 11
    assert meta["l7_proto"] == {"distinct_raw_codes": 3, "float32_collisions": {}}


def test_temporal_split_thresholds_and_drops(prepared):
    temporal = prepared["meta"]["split"]["temporal"]
    assert temporal["thresholds_ms"] == [T1, T2]
    first, second = temporal["dropped"]
    assert (first["straddle"], first["purge_gap"]) == (1, 60)
    assert (second["straddle"], second["purge_gap"]) == (1, 60)
    rows = {k: v["rows"] for k, v in temporal["splits"].items()}
    assert rows == {"train": 539, "val": 139, "test": 200, "purged": 122}
    assert temporal["dropped_total"] == 122
    assert temporal["absent_from_train"] == ["Worms"]
    assert temporal["absent_from_val"] == ["Reconnaissance", "Worms"]
    assert temporal["absent_from_test"] == ["Reconnaissance"]
    assert temporal["test_rows_with_feature_duplicate_in_train"] == 1


def test_temporal_split_has_purge_gap(prepared):
    out = prepared["out"]
    cols = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "row_id"]
    split = {s: nf.load_netflow(NAME, s, columns=cols, root=out) for s in nf.SPLITS}
    eff_end = {s: np.maximum(f.iloc[:, 0], f.iloc[:, 1]) for s, f in split.items()}
    assert eff_end["train"].max() < T1 - GAP_S * 1000 <= split["val"].iloc[:, 0].min()
    assert eff_end["val"].max() < T2 - GAP_S * 1000 <= split["test"].iloc[:, 0].min()
    assert 500 not in set(split["train"]["row_id"]) and 700 not in set(split["val"]["row_id"])
    assert list(split["train"].columns) == cols


def test_random_split_is_stratified_over_kept_rows(prepared):
    out = prepared["out"]
    frame = nf.load_netflow(NAME, "all", columns=["Attack", "split_temporal", "split_random"], root=out)
    temporal = frame["split_temporal"].astype(str)
    random = frame["split_random"].astype(str)
    assert ((temporal == "purged") == (random == "purged")).all()
    sizes = temporal.value_counts()
    counts = random.value_counts()
    n_classes = frame["Attack"].nunique()
    for split in nf.SPLITS:
        assert abs(counts[split] - sizes[split]) < n_classes
    kept = frame[random != "purged"]
    share = pd.crosstab(kept["Attack"].astype(str), kept["split_random"].astype(str), normalize="index")
    expected = sizes[list(nf.SPLITS)] / sizes[list(nf.SPLITS)].sum()
    for cls in ("Benign", "DoS"):
        np.testing.assert_allclose(share.loc[cls, list(nf.SPLITS)], expected, atol=0.02)
    assert set(kept.loc[kept["Attack"] == "Worms", "split_random"].astype(str)) >= {"train", "test"}
    random_meta = prepared["meta"]["split"]["random"]
    assert random_meta["absent_from_train"] == []
    test_ids = nf.load_netflow(NAME, "test", columns=["row_id"], split_kind="random", root=out)
    assert len(test_ids) == counts["test"]


def test_random_split_seeded():
    strata = np.repeat(np.arange(3), [50, 30, 20])
    keep = np.ones(100, dtype=bool)
    keep[:5] = False
    a = nf.stratified_random_split(strata, keep, [60, 20, 20], np.random.default_rng(1))
    b = nf.stratified_random_split(strata, keep, [60, 20, 20], np.random.default_rng(1))
    c = nf.stratified_random_split(strata, keep, [60, 20, 20], np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert (a != c).any()
    assert (a[:5] == nf.PURGED).all()
    assert np.bincount(a[strata == 1], minlength=4)[:3].tolist() == [18, 6, 6]


def test_random_split_sizes_are_proportional_not_equal(prepared):
    strata = np.repeat(np.arange(3), 5)
    codes = nf.stratified_random_split(strata, np.ones(15, dtype=bool), [5, 5, 5], np.random.default_rng(0))
    totals = np.bincount(codes, minlength=3)
    assert totals.tolist() != [5, 5, 5] and np.abs(totals - 5).max() < 3
    rule = prepared["meta"]["split"]["random"]["rule"]
    assert "equal" not in rule and "fewer rows than there are Attack classes" in rule
    assert "same split sizes" not in " ".join((nf.__doc__ or "").split())


def test_temporal_split_unit():
    starts = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int64)
    ends = starts + 1
    ends[3] = 65
    thresholds = nf.temporal_thresholds(starts, (0.6, 0.2, 0.2))
    assert thresholds == [60, 80]
    codes, dropped = nf.temporal_split(starts, ends, thresholds, purge_gap_ms=15)
    assert codes.tolist() == [0, 0, 0, 3, 0, 3, 1, 3, 2, 2]
    assert dropped[0]["straddle"] == 1 and dropped[0]["purge_gap"] == 1
    assert dropped[1]["straddle"] == 0 and dropped[1]["purge_gap"] == 1
    with pytest.raises(ValueError):
        nf.temporal_thresholds(np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError):
        nf.temporal_thresholds(starts, (0.5, 0.5))


def test_checksum_mismatch_publishes_nothing(tmp_path):
    zip_path, _ = make_bag(tmp_path, synthetic_frame(), csv_sha="0" * 40)
    out = tmp_path / "processed"
    with pytest.raises(nf.ChecksumError, match="sha1"):
        convert(zip_path, out)
    assert list(out.iterdir()) == []


def test_checksum_mismatch_keeps_previous_output(tmp_path):
    out = tmp_path / "processed"
    zip_path, csv_bytes = make_bag(tmp_path, synthetic_frame())
    convert(zip_path, out)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_zip, _ = make_bag(bad_dir, synthetic_frame(seed=1), csv_sha="f" * 40)
    with pytest.raises(nf.ChecksumError):
        convert(bad_zip, out)
    assert nf.load_meta(NAME, out)["source"]["sha1"] == hashlib.sha1(csv_bytes).hexdigest()
    assert len(nf.load_netflow(NAME, "all", columns=["row_id"], root=out)) == N
    assert sorted(p.name for p in out.iterdir()) == [NAME]


def test_missing_manifest_entry(tmp_path):
    zip_path, _ = make_bag(tmp_path, synthetic_frame(), include_csv_entry=False)
    with pytest.raises(nf.ChecksumError, match="manifest"):
        convert(zip_path, tmp_path / "processed")


def test_header_mismatch(tmp_path):
    zip_path, _ = make_bag(tmp_path, synthetic_frame().drop(columns=["MIN_TTL"]))
    with pytest.raises(ValueError, match="MIN_TTL"):
        convert(zip_path, tmp_path / "processed")
    assert list((tmp_path / "processed").iterdir()) == []


def test_header_only_csv(tmp_path):
    zip_path, _ = make_bag(tmp_path, synthetic_frame().iloc[:0])
    with pytest.raises(ValueError, match="no rows"):
        convert(zip_path, tmp_path / "processed")


def test_out_of_range_value_fails_instead_of_truncating(tmp_path):
    frame = synthetic_frame()
    frame.loc[42, "L4_DST_PORT"] = 70_000
    zip_path, _ = make_bag(tmp_path, frame)
    with pytest.raises(pa.ArrowInvalid):
        convert(zip_path, tmp_path / "processed")


@pytest.mark.parametrize("column, value", [
    ("DST_TO_SRC_SECOND_BYTES", 1e39),
    ("SRC_TO_DST_SECOND_BYTES", "-3.5e38"),
    ("L7_PROTO", "1e39"),
])


def test_float32_overflow_fails_instead_of_becoming_inf(tmp_path, column, value):
    frame = synthetic_frame()
    frame.loc[42, column] = value
    zip_path, _ = make_bag(tmp_path, frame)
    with pytest.raises(ValueError, match=f"{column}.*float32.*row 42"):
        convert(zip_path, tmp_path / "processed")
    assert list((tmp_path / "processed").iterdir()) == []


def test_float32_max_and_inf_tokens_are_stored(tmp_path):
    frame = synthetic_frame()
    frame["DST_TO_SRC_SECOND_BYTES"] = frame["DST_TO_SRC_SECOND_BYTES"].astype(object)
    frame.loc[42, "DST_TO_SRC_SECOND_BYTES"] = "3.4028235e38"
    frame.loc[43, "DST_TO_SRC_SECOND_BYTES"] = "-inf"
    zip_path, _ = make_bag(tmp_path, frame)
    meta = convert(zip_path, tmp_path / "processed")
    assert meta["nonfinite"] == {"SRC_TO_DST_SECOND_BYTES": 1, "DST_TO_SRC_SECOND_BYTES": 1}
    rate = nf.load_netflow(NAME, "all", columns=["DST_TO_SRC_SECOND_BYTES"], root=tmp_path / "processed")
    rate = rate.iloc[:, 0].to_numpy()
    assert rate[42] == np.finfo(np.float32).max and rate[43] == -np.inf
    assert int(np.isinf(rate).sum()) == meta["nonfinite"]["DST_TO_SRC_SECOND_BYTES"]


def test_duplicate_counts_do_not_depend_on_block_size(tmp_path):
    frame = synthetic_frame()
    frame.loc[800] = frame.loc[10]
    frame["IN_PKTS"] = frame["IN_PKTS"].astype(object)
    frame.loc[805, "IN_PKTS"] = ""
    zip_path, _ = make_bag(tmp_path, frame)
    stats = []
    for block_size in (16_384, 16 << 20):
        meta = convert(zip_path, tmp_path / f"out-{block_size}", block_size=block_size)
        assert meta["nulls"] == {"IN_PKTS": 1, "SRC_TO_DST_SECOND_BYTES": 1}
        duplicates = meta["duplicates"]
        stats.append((duplicates["all_columns"]["duplicate_rows"],
                      duplicates["flow_features"]["duplicate_rows"],
                      meta["split"]["temporal"]["test_rows_with_feature_duplicate_in_train"],
                      meta["split"]["random"]["test_rows_with_feature_duplicate_in_train"]))
    assert stats[0] == stats[1]
    assert stats[0][:2] == (2, 3)


def test_row_hash_ignores_nulls_elsewhere_in_the_batch():
    names = ["IN_PKTS", "L4_DST_PORT", "SRC_TO_DST_SECOND_BYTES"]
    clean = pa.record_batch([pa.array([5, 0], pa.int32()), pa.array([80, 443], pa.uint16()),
                             pa.array([1.5, 2.5])], names=names)
    nulled = pa.record_batch([pa.array([5, None], pa.int32()), pa.array([80, 443], pa.uint16()),
                              pa.array([1.5, 2.5])], names=names)
    a = nf._row_hash(nf._hash_frame(clean), names)
    b = nf._row_hash(nf._hash_frame(nulled), names)
    assert a[0] == b[0]
    assert a[1] != b[1]


def test_load_netflow_arguments(prepared):
    out = prepared["out"]
    with pytest.raises(ValueError):
        nf.load_netflow(NAME, "purged", root=out)
    with pytest.raises(ValueError):
        nf.load_netflow(NAME, "train", split_kind="shuffled", root=out)
    with pytest.raises(KeyError):
        nf.load_netflow(NAME, "train", columns=["NOPE"], root=out)
    with pytest.raises(FileNotFoundError):
        nf.load_netflow("NF-MISSING-v3", "train", root=out)


@pytest.mark.data
def test_heldout_guard_refuses_netflow_test(processed_dir):
    for split in ("test", "all"):
        with pytest.raises(AssertionError, match="held-out"):
            nf.load_netflow(nf.DATASETS[0], split, root=processed_dir / "netflow")


@pytest.mark.data
@pytest.mark.parametrize("name", nf.DATASETS)
def test_real_prepared_dataset(name, processed_dir):
    root = processed_dir / "netflow"
    if not (root / name / "meta.json").exists():
        pytest.skip(f"{name} not prepared")
    meta = nf.load_meta(name, root)
    rows = meta["rows"]
    assert rows == EXPECTED_ROWS[name]
    assert meta["source"]["sha1_verified"]
    assert sum(f["rows"] for f in meta["files"]) == rows
    temporal = meta["split"]["temporal"]
    splits = temporal["splits"]
    assert sum(s["rows"] for s in splits.values()) == rows
    for split, fraction in zip(nf.SPLITS, temporal["fractions"]):
        assert abs(splits[split]["rows"] / rows - fraction) < 0.02
    t1, t2 = temporal["thresholds_ms"]
    gap = temporal["purge_gap_s"] * 1000
    cols = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
    tables = {s: nf.load_netflow(name, s, columns=cols, root=root, as_pandas=False) for s in ("train", "val")}
    assert {s: t.num_rows for s, t in tables.items()} == {s: splits[s]["rows"] for s in ("train", "val")}

    def eff_end_max(table):
        return max(pc.max(table.column(0)).as_py(), pc.max(table.column(1)).as_py())

    def start_min(table):
        return pc.min(table.column(0)).as_py()

    test_start, t2_utc = (dt.datetime.fromisoformat(s) for s in (splits["test"]["start_utc"][0],
                                                                  temporal["thresholds_utc"][1]))
    assert eff_end_max(tables["train"]) < t1 - gap and start_min(tables["val"]) >= t1
    assert eff_end_max(tables["val"]) < t2 - gap and test_start >= t2_utc
    random = meta["split"]["random"]["splits"]
    assert random["purged"]["rows"] == splits["purged"]["rows"]
    n_classes = len(meta["class_counts"])
    for split in nf.SPLITS:
        assert abs(random[split]["rows"] - splits[split]["rows"]) < n_classes
