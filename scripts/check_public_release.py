#!/usr/bin/env python3
"""Scan publishable files without printing sensitive matched text.

This is a defense-in-depth check, not a guarantee that arbitrary PII is absent.
Ignored local datasets and environments are never included in the publication set.
"""
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'provider credential': re.compile(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b'),
    'credential URL': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    'personal home directory': re.compile(r'/(?:Users|home)/[A-Za-z][A-Za-z0-9_.-]+/'),
}
TOP = {'example_id', 'response_status', 'response_text', 'chosen', 'correct', 'is_correct', 'extraction_source', 'extraction_classification', 'meta'}
META = {'example_id', 'question_order', 'category', 'dimension', 'answer_key', 'split', 'task'}
FORBIDDEN = {'.pem', '.key', '.p12', '.pfx', '.wav', '.mp3', '.flac', '.safetensors', '.pt', '.pth', '.zip', '.log', '.pyc'}


def scan(path: Path):
    relative = path.relative_to(ROOT).as_posix()
    if path.is_symlink():
        yield 0, 'symlink requires explicit review'
        return
    if path.suffix in FORBIDDEN or path.name in {'.env', 'local.env'}:
        yield 0, 'forbidden release artifact'
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError:
        yield 0, 'binary file requires explicit review'
        return
    for n, line in enumerate(lines, 1):
        for rule, pattern in RULES.items():
            if pattern.search(line):
                # Deliberately synthetic fixture tests the archive's PII detector.
                if rule == 'personal home directory' and relative == 'tests/test_validate_archive.py' and ('/Users/' + 'jane/raw/clip.wav') in line:
                    continue
                yield n, rule
        if relative.startswith('reference_trajectories/') and path.suffix == '.jsonl':
            try:
                row = json.loads(line)
                if set(row) - TOP or set(row.get('meta', {})) - META:
                    yield n, 'unexpected reference field'
                if row.get('response_text') not in ('', '[response withheld]'):
                    yield n, 'unredacted response'
            except (ValueError, TypeError):
                yield n, 'invalid reference record'


def main():
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT, check=True, capture_output=True)
    files = sorted(set(result.stdout.decode().split('\0')) - {''})
    failures = 0
    for name in files:
        path = ROOT / name
        if not path.exists():
            continue
        for line, rule in scan(path):
            print(f'{name}:{line}: {rule}')
            failures += 1
    print(f'{"FAIL" if failures else "PASS"}: {len(files)} publishable files scanned; {failures} findings. Matched values are never printed.')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
