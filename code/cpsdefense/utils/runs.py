import datetime as dt
import json
import platform
import sys
from pathlib import Path

from cpsdefense.paths import ROOT, RUNS_DIR, ensure_dir
from cpsdefense.utils.hashing import sha256_json, source_digest

SOURCE_AT_IMPORT = source_digest(ROOT)


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "torch", "pyarrow", "scipy"):
        try:
            info[name] = __import__(name).__version__
        except ImportError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


class Run:
    def __init__(self, kind, config, seed, name=None, root=None):
        started = _now()
        stamp = started.strftime("%Y%m%d-%H%M%S")
        tag = f"{stamp}-{name}" if name else stamp
        self.dir = ensure_dir(Path(root or RUNS_DIR) / kind / tag)
        self.record = {
            "kind": kind,
            "name": name,
            "seed": seed,
            "config": config,
            "config_sha256": sha256_json(config),
            "started_utc": started.isoformat(timespec="seconds"),
            "argv": sys.argv,
            "environment": _environment(),
            "source": SOURCE_AT_IMPORT,
        }
        self._write()

    def _write(self):
        (self.dir / "run.json").write_text(
            json.dumps(self.record, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def save_json(self, name, obj):
        path = self.dir / name
        path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    def finish(self, metrics=None):
        self.record["finished_utc"] = _now().isoformat(timespec="seconds")
        self.record["source_changed_during_run"] = (
            source_digest(ROOT)["combined"] != SOURCE_AT_IMPORT["combined"]
        )
        if metrics is not None:
            self.record["metrics"] = metrics
            self.save_json("metrics.json", metrics)
        self._write()


def start_run(kind, config, seed=0, name=None, root=None):
    return Run(kind, config, seed, name=name, root=root)
