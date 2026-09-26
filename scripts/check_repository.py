#!/usr/bin/env python3
"""CPU-only syntax, source-fingerprint, and release-content checks."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache',
           'data', 'datasets', 'results', 'outputs', 'checkpoints'}
SECRET_PATTERNS = (
    re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(rb'\b(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{25,})\b'),
    re.compile(rb'''(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*["'][A-Za-z0-9_./+-]{8,}["']'''),
)


def source_files(root):
    for path in sorted(root.rglob('*')):
        if path.is_file() and not any(part in IGNORED for part in path.relative_to(root).parts):
            yield path


def verify_manifest(root):
    checked = 0
    names = set()
    for line in (root / 'SOURCE_MANIFEST.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'code':
            raise ValueError(f'Unsafe source manifest path: {name}')
        if name in names:
            raise ValueError(f'Duplicate source manifest entry: {name}')
        names.add(name)
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen source fingerprint mismatch: {name}')
        checked += 1
    present = {p.relative_to(root).as_posix() for p in source_files(root / 'code')}
    if present != names:
        raise ValueError('Source manifest membership differs from the code snapshots')
    return checked


def check_repository(root=ROOT):
    python_files = checked_files = 0
    for path in source_files(root):
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError(f'Possible credential in release file: {path.relative_to(root)}')
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8'), filename=str(path.relative_to(root)))
            python_files += 1
        checked_files += 1
    return {'passed': True, 'python_sources_parsed': python_files,
            'frozen_source_files_verified': verify_manifest(root),
            'release_files_scanned': checked_files, 'gpu_inference_run': False}


if __name__ == '__main__':
    print(json.dumps(check_repository(), indent=2))
