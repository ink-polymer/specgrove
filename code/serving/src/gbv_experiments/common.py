from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc
    return rows


def prompt_seed(seed: int, dataset: str, source_id: str) -> int:
    return int(digest([seed, dataset, source_id])[:15], 16) % (2**31 - 1)


def source_hashes() -> dict:
    paths = sorted((ROOT / "src/gbv_experiments").glob("*.py"))
    paths += sorted((ROOT / "third_party/ddtree_official/model").glob("*.py"))
    paths += sorted((ROOT / "third_party/livecodebench_official").glob("*.py"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
