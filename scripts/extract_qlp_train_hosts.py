#!/usr/bin/env python3
"""Extract only host-disjoint training TIC light curves from the official QLP tar."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_ARCHIVE_MD5 = "6b3cff6aff56d39fd281e2bfd805b7a7"
TIC_RE = re.compile(r"(?<!\d)(\d{13,16})(?!\d)")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-md5-check", action="store_true", help="Skip the official Zenodo MD5 check.")
    args = parser.parse_args()

    if not args.archive.is_file() or not args.split_manifest.is_file():
        raise SystemExit("Archive and split manifest must both exist.")

    archive_md5 = file_hash(args.archive, "md5")
    if not args.skip_md5_check and archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise SystemExit(f"Archive MD5 mismatch: expected {EXPECTED_ARCHIVE_MD5}, got {archive_md5}")

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_tics = {
        int(tic)
        for tic in split.get("metadata", {}).get("host_ids_by_split", {}).get("train", [])
    }
    if not train_tics:
        raise SystemExit("Split manifest has no metadata.host_ids_by_split.train list.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(p.is_file() and p.suffix.lower() in {".fits", ".fit"} for p in args.output_dir.iterdir()):
        raise SystemExit(f"Output directory already contains FITS files; choose a fresh directory: {args.output_dir}")
    found: set[int] = set()
    extracted: list[dict[str, object]] = []
    seen_names: set[str] = set()

    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member_index, member in enumerate(archive, start=1):
            if not member.isfile() or not member.name.lower().endswith((".fits", ".fit")):
                continue
            base = Path(member.name).name
            match = TIC_RE.findall(Path(base).stem)
            if not match:
                continue
            tic_id = int(match[-1])
            if tic_id not in train_tics:
                continue
            if base in seen_names:
                raise SystemExit(f"Duplicate archive basename would overwrite an extracted curve: {base}")
            seen_names.add(base)
            source = archive.extractfile(member)
            if source is None:
                continue
            target = args.output_dir / base
            digest = hashlib.sha256()
            with source, target.open("wb") as destination:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
            found.add(tic_id)
            extracted.append({
                "tic_id": tic_id,
                "archive_member": member.name,
                "file_name": base,
                "size_bytes": member.size,
                "sha256": digest.hexdigest(),
            })
            if len(extracted) % 1000 == 0:
                print(f"Extracted {len(extracted):,} training-host curves; scanned {member_index:,} archive members.", flush=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(args.archive.resolve()),
        "archive_md5": archive_md5,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": file_hash(args.split_manifest),
        "output_dir": str(args.output_dir.resolve()),
        "train_host_count_requested": len(train_tics),
        "train_host_count_found": len(found),
        "missing_train_tic_ids": sorted(train_tics - found),
        "extracted_file_count": len(extracted),
        "files": extracted,
    }
    manifest_path = args.output_dir / "qlp_train_hosts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Extracted {len(extracted):,} real QLP curves covering {len(found):,}/{len(train_tics):,} training TICs.")
    print(f"Extraction manifest: {manifest_path}")


if __name__ == "__main__":
    main()
