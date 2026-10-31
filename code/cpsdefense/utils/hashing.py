import hashlib
import json
from pathlib import Path

SOURCE_PATTERNS = (
    "cpsdefense/**/*.py",
    "scripts/*.py",
    "configs/**/*.yaml",
    "tests/*.py",
)


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj):
    return sha256_text(json.dumps(obj, sort_keys=True, default=str))


def source_digest(root, patterns=SOURCE_PATTERNS):
    root = Path(root)
    files = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                key = path.relative_to(root).as_posix()
                files[key] = sha256_file(path)
    combined = hashlib.sha256()
    for name in sorted(files):
        combined.update(name.encode("utf-8"))
        combined.update(files[name].encode("utf-8"))
    return {"combined": combined.hexdigest(), "files": files}
