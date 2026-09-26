from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import unicodedata

from .common import digest, file_hash, read_jsonl, write_json
from .data import load_prepared


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def audit(cfg, data_dir: Path, training_files: list[Path], output: Path):
    manifest, rows = load_prepared(data_dir, cfg["datasets"], cfg.get("evaluation", {}))
    normalized = defaultdict(list)
    for row in rows:
        normalized[digest(normalize(row["source_question"]))].append([row["dataset"], row["source_id"]])
    duplicates = [value for value in normalized.values() if len(value) > 1]
    inspected, collisions, unknown_fields = [], [], []
    for path in training_files:
        records = read_jsonl(path)
        inspected.append({"path": str(path.resolve()), "sha256": file_hash(path), "rows": len(records)})
        for i, r in enumerate(records):
            text = next((r[k] for k in ("source_question", "question", "problem", "prompt", "text")
                         if isinstance(r.get(k), str) and r[k]), None)
            if text is None:
                unknown_fields.append({"path": str(path), "row": i})
                continue
            matches = normalized.get(digest(normalize(text)), [])
            # Training artifacts may retain the exact evaluation prompt wrapper.
            if not matches:
                matches = [[row["dataset"], row["source_id"]] for row in rows
                           if normalize(row["prompt"]) == normalize(text)]
            if matches:
                collisions.append({"path": str(path), "row": i, "evaluation_ids": matches})
    math_errors, empty_references = [], []
    from math_verify import LatexExtractionConfig, parse
    for row in rows:
        evaluation = row["evaluation"]
        if evaluation["kind"] == "math":
            if not parse("$" + evaluation["answer"] + "$", extraction_config=[LatexExtractionConfig()]):
                math_errors.append([row["dataset"], row["source_id"]])
        elif evaluation["kind"] in {"humaneval", "mbpp"} and not evaluation.get("reference_code", "").strip():
            empty_references.append([row["dataset"], row["source_id"]])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest_sha256": digest(manifest),
        "coverage": manifest["coverage"], "evaluation": manifest.get("evaluation", {"protocol": "full"}),
        "evaluation_counts": {name: info["count"] for name, info in manifest["datasets"].items()},
        "training_policy": "frozen_target_and_frozen_published_draft; no local training or fitting",
        "locally_trained_components_used": [],
        "pretrained_training_corpus_overlap": "not independently verified; no contamination-free claim",
        "training_files_checked": inspected, "exact_text_overlaps": collisions,
        "training_rows_without_text": unknown_fields, "duplicate_evaluation_questions": duplicates,
        "unparseable_math_gold": math_errors, "missing_code_reference": empty_references,
        "reference_free_datasets": [name for name in cfg["datasets"] if name in {"livecodebench", "mt-bench"}],
        "scope": "Exact normalized question matching only; semantic/near duplicates require additional review.",
        "checks_passed": not (collisions or unknown_fields or math_errors or empty_references),
    }
    write_json(output, result)
    if not result["checks_passed"]:
        raise ValueError(f"Data audit requires review; see {output}")
    return result
