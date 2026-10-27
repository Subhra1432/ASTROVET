"""
Host-grouped dataset partitioning preventing cross-candidate and cross-sector leakage.

Scientific safeguards:
- All events for the same host system / TIC are strictly grouped into the SAME partition.
- Train, validation, calibration, and test partitions are 100% group-disjoint.
- Prevents data leakage from multi-sector observations, multiple planet candidates, and augmentations.
- Generates an immutable, versioned split manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from astrovet.provenance import create_manifest, save_manifest


def make_host_grouped_split(
    candidate_records: List[Dict[str, Any]],
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    cal_frac: float = 0.10,
    test_frac: float = 0.15,
    random_seed: int = 42,
    split_manifest_path: Optional[str | Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Partition candidate records into train/val/calibration/test partitions strictly
    grouped by host star (tic_id).
    """
    assert abs(train_frac + val_frac + cal_frac + test_frac - 1.0) < 1e-4, "Partition fractions must sum to 1.0"

    # Group candidate IDs by TIC
    tic_to_candidates: Dict[int, List[Dict[str, Any]]] = {}
    for rec in candidate_records:
        tic = int(rec["tic_id"])
        tic_to_candidates.setdefault(tic, []).append(rec)

    unique_tics = sorted(list(tic_to_candidates.keys()))
    rng = np.random.default_rng(random_seed)
    shuffled_tics = unique_tics.copy()
    rng.shuffle(shuffled_tics)

    n_total_tics = len(shuffled_tics)
    n_train = int(n_total_tics * train_frac)
    n_val = int(n_total_tics * val_frac)
    n_cal = int(n_total_tics * cal_frac)

    train_tics = set(shuffled_tics[:n_train])
    val_tics = set(shuffled_tics[n_train:n_train + n_val])
    cal_tics = set(shuffled_tics[n_train + n_val:n_train + n_val + n_cal])
    test_tics = set(shuffled_tics[n_train + n_val + n_cal:])

    # Strictly verify zero overlap
    assert len(train_tics.intersection(val_tics)) == 0, "Leakage detected: train and val share TICs"
    assert len(train_tics.intersection(cal_tics)) == 0, "Leakage detected: train and cal share TICs"
    assert len(train_tics.intersection(test_tics)) == 0, "Leakage detected: train and test share TICs"
    assert len(val_tics.intersection(test_tics)) == 0, "Leakage detected: val and test share TICs"

    split_results: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "cal": [],
        "test": [],
    }

    for tic, recs in tic_to_candidates.items():
        if tic in train_tics:
            split_results["train"].extend(recs)
        elif tic in val_tics:
            split_results["val"].extend(recs)
        elif tic in cal_tics:
            split_results["cal"].extend(recs)
        elif tic in test_tics:
            split_results["test"].extend(recs)

    # Manifest creation
    if split_manifest_path:
        manifest_meta = {
            "total_candidates": len(candidate_records),
            "total_host_stars": n_total_tics,
            "train_hosts": len(train_tics),
            "val_hosts": len(val_tics),
            "cal_hosts": len(cal_tics),
            "test_hosts": len(test_tics),
            "train_candidates": len(split_results["train"]),
            "val_candidates": len(split_results["val"]),
            "cal_candidates": len(split_results["cal"]),
            "test_candidates": len(split_results["test"]),
            "random_seed": random_seed,
            "leakage_check_passed": True,
        }
        manifest = create_manifest(
            manifest_id=f"manifest_split_seed{random_seed}",
            manifest_type="DATASET_HOST_GROUPED_SPLIT",
            files=[],
            metadata=manifest_meta,
        )
        save_manifest(manifest, split_manifest_path)

    return split_results
