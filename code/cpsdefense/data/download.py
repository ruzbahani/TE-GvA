import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from cpsdefense.utils.hashing import sha256_file

USER_AGENT = "cpsdefense-research-downloader/0.1"
HAI_REPO_API = "https://api.github.com/repos/icsdataset/hai/contents"
HAI_RAW = "https://raw.githubusercontent.com/icsdataset/hai/master"
HAI_MEDIA = "https://media.githubusercontent.com/media/icsdataset/hai/master"
UQ_API = "https://api.rdm.uq.edu.au/datasets"

NETFLOW_DATASETS = {
    "NF-ToN-IoT-v3": {
        "uuid": "343e2e8c-6e6e-4a0c-813d-a46acea1b7f4",
        "citation": (
            "Luay, Layeghy, Hosseininoorbin, Sarhan, Moustafa, Portmann. "
            "Temporal Analysis of NetFlow Datasets for Network Intrusion "
            "Detection Systems. arXiv:2503.04404, 2025."
        ),
    },
    "NF-UNSW-NB15-v3": {
        "uuid": "abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f",
        "citation": (
            "Luay, Layeghy, Hosseininoorbin, Sarhan, Moustafa, Portmann. "
            "Temporal Analysis of NetFlow Datasets for Network Intrusion "
            "Detection Systems. arXiv:2503.04404, 2025."
        ),
    },
}


def http_get(url, headers=None, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def http_json(url):
    return json.loads(http_get(url).decode("utf-8"))


def download_file(url, dest, expected_sha256=None, expected_size=None, retries=6, chunk_size=1 << 20):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_sha256 is None or sha256_file(dest) == expected_sha256:
            return dest
        dest.unlink()

    partial = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if offset and response.status != 206:
                    offset = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                total = response.headers.get("Content-Length")
                total = int(total) + offset if total else None
                written = offset
                last_report = time.time()
                with open(partial, mode) as handle:
                    for block in iter(lambda: response.read(chunk_size), b""):
                        handle.write(block)
                        written += len(block)
                        if time.time() - last_report > 15:
                            report_progress(dest.name, written, total)
                            last_report = time.time()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            attempt += 1
            if attempt > retries:
                raise
            wait = min(60, 5 * attempt)
            print(f"[download] {dest.name}: {error}; retry {attempt}/{retries} in {wait}s", flush=True)
            time.sleep(wait)

    if expected_size is not None and partial.stat().st_size != expected_size:
        raise IOError(f"{dest.name}: size {partial.stat().st_size} != expected {expected_size}")
    if expected_sha256 is not None:
        actual = sha256_file(partial)
        if actual != expected_sha256:
            partial.unlink()
            raise IOError(f"{dest.name}: sha256 mismatch ({actual} != {expected_sha256})")
    os.replace(partial, dest)
    return dest


def report_progress(name, written, total):
    if total:
        print(f"[download] {name}: {written / 1e6:.1f} / {total / 1e6:.1f} MB", flush=True)
    else:
        print(f"[download] {name}: {written / 1e6:.1f} MB", flush=True)


def parse_lfs_pointer(text):
    fields = {}
    for line in text.strip().splitlines():
        key, value = line.split(" ", 1)
        fields[key] = value
    oid = fields.get("oid", "")
    return {"sha256": oid.split(":", 1)[1] if oid.startswith("sha256:") else None,
            "size": int(fields["size"]) if "size" in fields else None}


def list_hai_files(version):
    entries = http_json(f"{HAI_REPO_API}/hai-{version}")
    files = []
    for entry in entries:
        if entry["type"] != "file":
            continue
        name = entry["name"]
        raw_url = f"{HAI_RAW}/hai-{version}/{name}"
        if entry["size"] < 1024:
            pointer = http_get(raw_url).decode("utf-8", errors="replace")
            if pointer.startswith("version https://git-lfs"):
                lfs = parse_lfs_pointer(pointer)
                files.append({
                    "name": name,
                    "url": f"{HAI_MEDIA}/hai-{version}/{name}",
                    "sha256": lfs["sha256"],
                    "size": lfs["size"],
                    "checksum_source": "git-lfs pointer",
                })
                continue
        files.append({
            "name": name,
            "url": raw_url,
            "sha256": None,
            "size": entry["size"],
            "checksum_source": "computed after download",
        })
    return files


def uq_download_url(uuid):
    info = http_json(f"{UQ_API}/{uuid}/download")
    if info.get("accessType") != "OPEN" or not info.get("url"):
        raise PermissionError(f"dataset {uuid} is not openly downloadable")
    return info["url"]


def uq_dataset_card(uuid):
    details = http_json(f"{UQ_API}/{uuid}/file-details")
    license_info = details.get("datasetLicense") or {}
    return {
        "name": details.get("name"),
        "description": details.get("description"),
        "project": details.get("projectName"),
        "status": details.get("status"),
        "access_type": details.get("accessType"),
        "license_title": license_info.get("title"),
        "license_terms_url": license_info.get("description"),
        "repository_record": details.get("espacePid"),
    }
