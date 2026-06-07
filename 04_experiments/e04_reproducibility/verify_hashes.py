"""Verify current 05_results/ outputs match expected_hashes.txt.

Returns exit code 0 if all match, 1 if any mismatch.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    expected_file = paths.EXPERIMENTS / "e04_reproducibility" / "expected_hashes.txt"
    if not expected_file.exists():
        print("FAIL: expected_hashes.txt not found. "
              "Run compute_hashes.py to create snapshot first.")
        sys.exit(2)

    expected = {}
    for line in expected_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, path = line.split(None, 1)
        expected[path] = sha

    mismatches = []
    missing = []
    for path, want in expected.items():
        full = paths.RESULTS / path
        if not full.exists():
            missing.append(path); continue
        got = sha256(full)
        if got != want:
            mismatches.append((path, want, got))

    print(f"Checked {len(expected)} files.")
    print(f"  missing:    {len(missing)}")
    print(f"  mismatched: {len(mismatches)}")
    if missing:
        for m in missing[:10]:
            print(f"    MISSING {m}")
    if mismatches:
        for p, w, g in mismatches[:10]:
            print(f"    DIFF    {p}\n      expected {w}\n      actual   {g}")
        sys.exit(1)
    if missing:
        sys.exit(1)
    print("ALL HASHES MATCH.")
    sys.exit(0)


if __name__ == "__main__":
    main()
