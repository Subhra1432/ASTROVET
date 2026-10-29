"""
CSV catalog auditing workflow and report generation.

Scientific safeguards:
- Never infer schema or label meanings from filename alone.
- Report real row counts, candidate identifiers, labels, missing values, and duplicates.
- Disclose absence of expected datasets without inventing statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from astrovet.provenance import compute_sha256, get_runtime_environment


def audit_catalog_csv(
    file_path: str | Path,
    output_report_path: Optional[str | Path] = None,
    label_col_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a comprehensive audit of a candidate CSV catalog.
    Writes a machine-readable JSON report if output_report_path is supplied.
    """
    p = Path(file_path)
    if not p.is_file():
        audit_result = {
            "status": "FILE_NOT_FOUND",
            "file_path": str(p.resolve()),
            "message": f"Target CSV catalog file does not exist: {p}",
            "runtime": get_runtime_environment(),
        }
        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(audit_result, f, indent=2)
        return audit_result

    # Basic file metadata
    file_size_bytes = p.stat().st_size
    file_sha256 = compute_sha256(p)

    # Read dataframe with comment header and unit row detection
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f_in:
            first_line = f_in.readline().strip()
        if first_line.startswith('#'):
            df = pd.read_csv(p, comment='#', low_memory=False)
        else:
            df = pd.read_csv(p, low_memory=False)
    except Exception:
        df = pd.read_csv(p, low_memory=False)

    # Detect and drop unit specification rows (e.g., TIC ID is NaN and units present)
    if len(df) > 0 and (pd.isna(df.iloc[0, 0]) or str(df.iloc[0].get('Epoch', '')).strip() in ['BTJD', 'days', 'ppm']):
        df = df.iloc[1:].reset_index(drop=True)

    row_count, col_count = df.shape
    columns = list(df.columns)

    # Null analysis
    null_counts = df.isnull().sum().to_dict()
    null_pct = (df.isnull().sum() / max(1, row_count) * 100.0).round(2).to_dict()

    # Column type summary
    col_dtypes = {col: str(df[col].dtype) for col in columns}

    # Identifier analysis
    tic_col = None
    for candidate_name in ["TIC ID", "tic_id", "target_id", "TIC", "tid"]:
        if candidate_name in df.columns:
            tic_col = candidate_name
            break

    tce_col = None
    for candidate_name in ["TCE ID", "tce_id", "uid", "TCE"]:
        if candidate_name in df.columns:
            tce_col = candidate_name
            break

    unique_tics = int(df[tic_col].nunique()) if tic_col else None
    unique_tces = int(df[tce_col].nunique()) if tce_col else None

    # Multiplicity check: how many stars have >1 candidate?
    multiple_candidates_per_tic = None
    if tic_col:
        counts_per_tic = df[tic_col].value_counts()
        multiple_candidates_per_tic = int((counts_per_tic > 1).sum())

    # Label column detection & analysis
    potential_label_cols = []
    if label_col_hint and label_col_hint in df.columns:
        potential_label_cols.append(label_col_hint)
    for col in columns:
        low = col.lower()
        if any(term in low for term in ["label", "disp", "disposition", "prediction", "tfop"]):
            if col not in potential_label_cols:
                potential_label_cols.append(col)

    label_distributions = {}
    for lcol in potential_label_cols:
        val_counts = df[lcol].value_counts(dropna=False).to_dict()
        label_distributions[lcol] = {str(k): int(v) for k, v in val_counts.items()}

    # Duplicate rows check
    exact_duplicates = int(df.duplicated().sum())
    tic_duplicates = int(df.duplicated(subset=[tic_col]).sum()) if tic_col else None
    tce_duplicates = int(df.duplicated(subset=[tce_col]).sum()) if tce_col else None

    audit_result = {
        "status": "SUCCESS",
        "file_name": p.name,
        "file_path": str(p.resolve()),
        "file_size_bytes": file_size_bytes,
        "sha256": file_sha256,
        "row_count": row_count,
        "column_count": col_count,
        "columns": columns,
        "column_dtypes": col_dtypes,
        "null_counts": null_counts,
        "null_percentages": null_pct,
        "identifiers": {
            "tic_column": tic_col,
            "unique_tics": unique_tics,
            "tce_column": tce_col,
            "unique_tces": unique_tces,
            "stars_with_multiple_candidates": multiple_candidates_per_tic,
            "tic_alone_is_unique_event": (unique_tics == row_count) if unique_tics else None,
        },
        "duplicates": {
            "exact_duplicate_rows": exact_duplicates,
            "tic_duplicate_rows": tic_duplicates,
            "tce_duplicate_rows": tce_duplicates,
        },
        "labels": {
            "detected_label_columns": potential_label_cols,
            "distributions": label_distributions,
        },
        "runtime": get_runtime_environment(),
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(audit_result, f, indent=2)

    return audit_result
