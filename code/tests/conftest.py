import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpsdefense.paths import PROCESSED_DIR, RAW_DIR


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


@pytest.fixture
def raw_dir():
    if not RAW_DIR.exists():
        pytest.skip("raw data not downloaded")
    return RAW_DIR


@pytest.fixture
def processed_dir():
    if not PROCESSED_DIR.exists():
        pytest.skip("processed data not prepared")
    return PROCESSED_DIR


def _heldout_stems():
    from cpsdefense.data import hai
    from cpsdefense.utils.config import load_config

    names = set()
    for release in load_config(hai.CONFIG)["releases"].values():
        for stem in release.get("test") or []:
            names.add(stem)
            spec = (release.get("label_files") or {}).get(stem)
            summary = (release.get("summaries") or {}).get(stem)
            names.update(Path(name).stem for name in (spec and spec["file"], summary) if name)
    return names


@pytest.fixture
def heldout_names():
    return _heldout_stems()


@pytest.fixture(autouse=True)
def _no_heldout_reads(request, monkeypatch):
    if request.node.get_closest_marker("data") is None:
        return
    from cpsdefense.data import hai, netflow

    names = _heldout_stems()

    def refuse(what):
        raise AssertionError(f"data test read the held-out test split: {what}")

    def by_path(func):
        def guarded(path, *args, **kwargs):
            if Path(path).stem in names:
                refuse(path)
            return func(path, *args, **kwargs)
        return guarded

    load_split, load_netflow = hai.load_split, netflow.load_netflow

    def guarded_split(version, split, *args, **kwargs):
        if split == "test":
            refuse(f"hai.load_split({version!r}, 'test')")
        return load_split(version, split, *args, **kwargs)

    def guarded_netflow(name, split, *args, **kwargs):
        if split in ("test", "all"):
            refuse(f"netflow.load_netflow({name!r}, {split!r})")
        return load_netflow(name, split, *args, **kwargs)

    monkeypatch.setattr(hai, "load_split", guarded_split)
    for name in ("read_csv_frame", "read_label_file", "parse_summary"):
        monkeypatch.setattr(hai, name, by_path(getattr(hai, name)))
    monkeypatch.setattr(pd, "read_parquet", by_path(pd.read_parquet))
    monkeypatch.setattr(netflow, "load_netflow", guarded_netflow)
