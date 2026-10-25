"""
Provenance, hashing, manifest generation, and reproducibility tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_sha256(file_path: str | Path) -> str:
    """Calculate the hex SHA-256 checksum of a file."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {file_path}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Calculate the hex SHA-256 checksum of raw in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def get_runtime_environment() -> Dict[str, Any]:
    """Capture execution environment metadata for reproducibility."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def create_manifest(
    manifest_id: str,
    manifest_type: str,
    files: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a structured, immutable manifest dictionary."""
    return {
        "manifest_id": manifest_id,
        "manifest_type": manifest_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": get_runtime_environment(),
        "file_count": len(files),
        "files": files,
        "metadata": metadata or {},
    }


def save_manifest(manifest: Dict[str, Any], output_path: str | Path) -> Path:
    """Save manifest to a JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return p
