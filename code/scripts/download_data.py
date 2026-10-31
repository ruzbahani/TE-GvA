import argparse
import datetime as dt
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.data import download as dl
from cpsdefense.paths import RAW_DIR, ensure_dir
from cpsdefense.utils.hashing import sha256_file


def load_manifest(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}, "datasets": {}}


def save_manifest(manifest, path):
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def fetch_hai(version, out_dir, manifest):
    target_dir = ensure_dir(out_dir / "hai" / f"hai-{version}")
    files = dl.list_hai_files(version)
    for item in files:
        dest = target_dir / item["name"]
        print(f"[hai-{version}] {item['name']} ({(item['size'] or 0) / 1e6:.1f} MB)", flush=True)
        dl.download_file(item["url"], dest, expected_sha256=item["sha256"], expected_size=item["size"])
        key = dest.relative_to(out_dir).as_posix()
        manifest["files"][key] = {
            "url": item["url"],
            "size": dest.stat().st_size,
            "sha256": item["sha256"] or sha256_file(dest),
            "checksum_source": item["checksum_source"],
        }
    manifest["datasets"][f"hai-{version}"] = {
        "source": "https://github.com/icsdataset/hai",
        "citation": (
            "Shin, Lee, Choi, Yun, Kim. HAI 1.0: HIL-based Augmented ICS Security "
            "Dataset. CSET 2020. Dataset release " + version + "."
        ),
        "license_note": "See the README and technical details document in the source repository.",
    }


def fetch_netflow(name, out_dir, manifest):
    spec = dl.NETFLOW_DATASETS[name]
    target_dir = ensure_dir(out_dir / "netflow")
    card = dl.uq_dataset_card(spec["uuid"])
    dest = target_dir / f"{name}.zip"
    print(f"[{name}] requesting signed download url", flush=True)
    url = dl.uq_download_url(spec["uuid"])
    dl.download_file(url, dest)
    digest = sha256_file(dest)
    with zipfile.ZipFile(dest) as archive:
        members = [{"name": info.filename, "size": info.file_size} for info in archive.infolist()]
    key = dest.relative_to(out_dir).as_posix()
    manifest["files"][key] = {
        "url": f"https://rdm.uq.edu.au/files/{spec['uuid']}",
        "size": dest.stat().st_size,
        "sha256": digest,
        "checksum_source": "computed after download",
        "members": members,
    }
    manifest["datasets"][name] = {**card, "citation": spec["citation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hai", nargs="*", default=["22.04", "23.05"])
    parser.add_argument("--netflow", nargs="*", default=["NF-UNSW-NB15-v3", "NF-ToN-IoT-v3"])
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    manifest_path = out_dir / "MANIFEST.json"
    manifest = load_manifest(manifest_path)

    for version in args.hai:
        fetch_hai(version, out_dir, manifest)
        save_manifest(manifest, manifest_path)
    for name in args.netflow:
        fetch_netflow(name, out_dir, manifest)
        save_manifest(manifest, manifest_path)

    manifest["retrieved_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    save_manifest(manifest, manifest_path)
    print(f"manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
