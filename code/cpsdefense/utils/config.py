import copy
from pathlib import Path

import yaml

from cpsdefense.paths import CONFIG_DIR


def load_config(path, overrides=None):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in overrides or ():
        key, value = item.split("=", 1)
        set_dotted(config, key, yaml.safe_load(value))
    return config


def set_dotted(config, key, value):
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def merged(base, update):
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merged(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
