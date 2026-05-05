#!/usr/bin/env python3
"""Rehydrate the paper payload set from MalwareBazaar by SHA256.

The anonymous GitHub artifact intentionally does not ship real malware
payload binaries. This helper lets reviewers with an approved isolated
environment reproduce the paper payload directory from public MalwareBazaar
hashes.

Usage:
    python data/download_paper_payloads.py
    python data/download_paper_payloads.py --appendix

Environment:
    MALWAREBAZAAR_API_KEY: Optional MalwareBazaar API key. Some deployments
        require it for downloads.

Safety:
    Downloaded files are real malware samples. Never execute them. Treat them
    as opaque bytes for LSB injection experiments only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


MAIN_PAYLOADS = [
    {
        "sha256": "c37c0db91ab188c2fe01642e04e0db9186bc5bf54ad8b6b72512ad5aab921a88",
        "role": "main",
        "label": "low-entropy representative payload",
        "type": "js",
    },
    {
        "sha256": "5704fabda6a0851ea156d1731b4ed4383ce102ec3a93f5d7109cc2f47f8196d0",
        "role": "main",
        "label": "high-entropy representative payload",
        "type": "exe",
    },
]

APPENDIX_PAYLOADS = [
    {
        "sha256": "5b6787d5068199e43006b0918523a3685d9962f5c0a75113656051b3aa74b360",
        "role": "appendix",
        "label": "supplementary payload",
        "type": "xlsx",
    },
    {
        "sha256": "1368f3a8a8254feea14af7dc928af6847cab8fcceec4f21e0166843a75e81964",
        "role": "appendix",
        "label": "supplementary payload",
        "type": "elf",
    },
    {
        "sha256": "31f110e228d8faca59619ec4f77105438f1b318c3c17779cdc9ab4fc16d6147f",
        "role": "appendix",
        "label": "supplementary payload",
        "type": "zip",
    },
    {
        "sha256": "9d5b59d0ee9914baaa0c45b0748126c3bf585fc8380b34270dc03aca9bdcb46e",
        "role": "appendix",
        "label": "supplementary payload",
        "type": "apk",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_existing(output_dir: Path, sha256_hash: str) -> Path | None:
    matches = sorted(output_dir.glob(f"{sha256_hash}.*.malware"))
    return matches[0] if matches else None


def verify_existing(path: Path, sha256_hash: str) -> bool:
    return path.is_file() and sha256_file(path) == sha256_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "malware"),
        help="Directory where .malware files will be written.",
    )
    parser.add_argument(
        "--appendix",
        action="store_true",
        help="Also download the four supplementary appendix payloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if a matching local file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected hashes without downloading anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    selected: list[dict[str, str]] = list(MAIN_PAYLOADS)
    if args.appendix:
        selected.extend(APPENDIX_PAYLOADS)

    print("WARNING: this script downloads REAL MALWARE samples.")
    print("Use only in an isolated research environment; never execute them.")
    print(f"Output directory: {output_dir}")
    print(f"Selected payloads: {len(selected)}")

    if args.dry_run:
        for item in selected:
            print(f"{item['sha256']}  {item['role']}  {item['type']}  {item['label']}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Imported lazily so dry-run works without optional downloader dependencies.
    from grayshield.payload.malwarebazaar import download_by_hash

    downloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for item in selected:
        sha256_hash = item["sha256"]
        existing = find_existing(output_dir, sha256_hash)
        if existing and not args.force:
            if verify_existing(existing, sha256_hash):
                print(f"Skipping verified existing payload: {existing.name}")
                skipped.append({**item, "path": str(existing)})
                continue
            print(f"Existing file failed hash verification, re-downloading: {existing.name}")

        print(f"Downloading {sha256_hash[:16]}... ({item['role']}, {item['type']})")
        result = download_by_hash(sha256_hash, str(output_dir))
        if result is None:
            failed.append({**item, "error": "download failed"})
            continue

        path = Path(result.local_path)
        if not verify_existing(path, sha256_hash):
            failed.append({**item, "path": str(path), "error": "sha256 verification failed"})
            continue

        downloaded.append({**item, "path": str(path), "size_bytes": path.stat().st_size})

    manifest = {
        "created": datetime.utcnow().isoformat() + "Z",
        "source": "MalwareBazaar",
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
    manifest_path = output_dir / "paper_payload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")

    if failed:
        print(f"Failed downloads: {len(failed)}")
        return 1

    print("Paper payload rehydration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
