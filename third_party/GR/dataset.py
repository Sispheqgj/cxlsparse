"""Load catalog titles used as readable text material, independently of user heat."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/mnt/nfs/share/archive/260808_old_nfs/share/wsh/Bi-KV/Bi-KV/data")


class _DataUnpickler(pickle.Unpickler):
    # The archived catalog files contain plain dict/list/str/int data.
    def find_class(self, module: str, name: str):
        raise ValueError(f"unsupported pickle object: {module}.{name}; expected plain data")


def load_titles(path: Path) -> dict[int, str]:
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as source:
            raw = json.load(source)
    else:
        with path.open("rb") as source:
            raw = _DataUnpickler(source).load()
    titles = {int(item): title for item, title in raw["meta"].items()}
    if not titles or any(
        not isinstance(title, str) or not title.strip() for title in titles.values()
    ):
        raise ValueError("catalog contains empty or non-text titles")
    return titles
