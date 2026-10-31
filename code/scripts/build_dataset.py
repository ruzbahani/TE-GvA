import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.experience import dataset
from cpsdefense.experience.tensors import CONFIG as TENSOR_CONFIG
from cpsdefense.paths import CONFIG_DIR, ROOT, RUNS_DIR
from cpsdefense.utils.hashing import sha256_file
from cpsdefense.utils.runs import SOURCE_AT_IMPORT, start_run

DATA_CONFIGS = ("sim/tep.yaml", "sim/telemetry.yaml", "sim/costs.yaml", "sim/tep_env.yaml", "defense/tools.yaml",
                "defense/policy.yaml", TENSOR_CONFIG)


def rel(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def config_file(name):
    path = Path(name)
    for p in ([path] if path.is_absolute() else [CONFIG_DIR / path, ROOT / path]):
        if p.exists():
            return p
    return path if path.is_absolute() else CONFIG_DIR / path


def argv_config(argv, default="experience/generate.yaml"):
    argv = list(argv or ())
    return argv[argv.index("--config") + 1] if "--config" in argv[:-1] else default


def manifest(root):
    files = sorted(Path(root).glob("ep_*.pkl"))
    digest = hashlib.sha256()
    for p in files:
        digest.update(f"{p.name}:{sha256_file(p)}".encode())
    return {"files": len(files), "sha256": digest.hexdigest()}


def generation_record(root):
    root = Path(root)
    gen_file = root / "generation.json"
    gen = json.loads(gen_file.read_text(encoding="utf-8")) if gen_file.exists() else {}
    record = ROOT / gen["run"] / "run.json" if gen.get("run") else RUNS_DIR / "experience" / root.name / "run.json"
    run = json.loads(record.read_text(encoding="utf-8")) if record.exists() else None
    if not gen and run is None:
        return None
    rec = run or {}
    config = gen.get("config") or rec.get("config") or {}
    return {"generation_json": rel(gen_file) if gen else None, "run_record": rel(record) if run else None,
            "config": config, "config_sha256": gen.get("config_sha256") or rec.get("config_sha256"),
            "config_file": gen.get("config_file") or argv_config(rec.get("argv")),
            "incidents_file": config.get("incidents", "sim/incidents.yaml"),
            "incidents_sha256": gen.get("incidents_sha256"),
            "source_combined": gen.get("source_combined") or rec.get("source", {}).get("combined"),
            "started_utc": rec.get("started_utc"), "finished_utc": rec.get("finished_utc"),
            "source_changed_during_run": rec.get("source_changed_during_run"),
            "files": rec.get("source", {}).get("files")}


def config_hashes(generation):
    names = list(DATA_CONFIGS)
    if generation is not None:
        names += [generation["incidents_file"], generation["config_file"]]
    then = (generation or {}).get("files") or {}
    out = {}
    for name in dict.fromkeys(names):
        path = config_file(name)
        key = rel(path)
        out[key] = {"at_generation": then.get(key), "now": sha256_file(path) if path.exists() else None}
    return out


def provenance(episodes, run_dir):
    generation = generation_record(episodes)
    configs = config_hashes(generation)
    if generation is not None:
        generation = {k: v for k, v in generation.items() if k != "files"}
    return {
        "build_run": rel(run_dir), "build_source_combined": SOURCE_AT_IMPORT["combined"],
        "episodes": rel(episodes), "episode_manifest": manifest(episodes), "generation": generation,
        "config_files": configs,
        "configs_changed_since_generation": sorted(k for k, v in configs.items()
                                                   if v["at_generation"] not in (None, v["now"])),
    }


def build_with_record(episodes, out, name, runs_root=None, force=False):
    episodes, out = Path(episodes), Path(out)
    if (out / "meta.json").exists() and not force:
        raise FileExistsError(f"{out / 'meta.json'} exists; pass --force to rebuild over it")
    if not any(episodes.glob("ep_*.pkl")):
        raise FileNotFoundError(f"no ep_*.pkl episodes in {episodes}")
    cfg = {"episodes": rel(episodes), "out": rel(out), "fractions": [0.6, 0.2, 0.2], "tensors": TENSOR_CONFIG}
    run = start_run("data", cfg, seed=None, name=name, root=runs_root)
    prov = provenance(episodes, run.dir)
    meta = dataset.build(episodes, out, tuple(cfg["fractions"]), provenance=prov)
    run.save_json("provenance.json", prov)
    run.finish({"splits": {s: {k: v[k] for k in ("episodes", "decisions", "candidates")}
                           for s, v in meta["splits"].items()},
                "outputs": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return meta, run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, help="directory of ep_*.pkl from generate_experience.py")
    parser.add_argument("--out", required=True, help="output directory of the arrays")
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--force", action="store_true", help="rebuild over an existing meta.json")
    args = parser.parse_args()
    meta, run = build_with_record(args.episodes, args.out, args.name, force=args.force)
    for s, v in meta["splits"].items():
        print(f"{s}: {v['episodes']} episodes, {v['decisions']} decisions, {v['candidates']} candidates")
    print(f"-> {args.out} (run record {run.dir})")

if __name__ == "__main__":
    main()
