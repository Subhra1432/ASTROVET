"""Create a traceable TESS TCE ↔ NASA TOI disposition cohort.

Only period-and-epoch matches are admitted. Candidate, ambiguous and unset TOI
dispositions are retained in the full match audit but are excluded from the
hard-label classifier cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_PAGE = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html"
TOI_TAP = "https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html"
POSITIVE = {"CP", "KP"}
NEGATIVE = {"FP", "FA"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tce_files(folder: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No sector TCE CSV files found in {folder}")
    frames = []
    provenance = []
    for path in files:
        frame = pd.read_csv(path, comment="#", low_memory=False)
        if frame.empty:
            continue
        sector_match = re.search(r"-s(\d{4})-s\d{4}_dvr-tcestats\.csv$", path.name)
        frame["_source_sector"] = int(sector_match.group(1)) if sector_match else -1
        frame["_source_file"] = path.name
        frames.append(frame)
        provenance.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    if not frames:
        raise ValueError(f"All TCE CSV files in {folder} were empty")
    combined = pd.concat(frames, ignore_index=True)
    required = {"tceid", "ticid", "tce_period", "tce_time0bt", "tce_duration", "tce_depth"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"TCE data missing required columns: {sorted(missing)}")
    return combined, provenance


def build_matches(tce: pd.DataFrame, toi: pd.DataFrame, period_tolerance: float, epoch_floor_days: float) -> pd.DataFrame:
    required_toi = {"tid", "toi", "tfopwg_disp", "pl_orbper", "pl_tranmid", "pl_trandurh", "rowupdate"}
    missing = required_toi - set(toi.columns)
    if missing:
        raise ValueError(f"TOI data missing required columns: {sorted(missing)}")

    toi = toi.copy()
    for col in ("tid", "pl_orbper", "pl_tranmid", "pl_trandurh"):
        toi[col] = pd.to_numeric(toi[col], errors="coerce")
    toi["_epoch_btjd"] = toi["pl_tranmid"] - 2457000.0
    by_tic = {int(tic): group for tic, group in toi.dropna(subset=["tid"]).groupby("tid")}

    rows: list[dict[str, Any]] = []
    for _, row in tce.iterrows():
        tic = pd.to_numeric(row["ticid"], errors="coerce")
        period = pd.to_numeric(row["tce_period"], errors="coerce")
        epoch = pd.to_numeric(row["tce_time0bt"], errors="coerce")
        if pd.isna(tic) or pd.isna(period) or pd.isna(epoch) or period <= 0:
            continue
        possibilities = []
        for _, candidate in by_tic.get(int(tic), pd.DataFrame()).iterrows():
            toi_period = candidate["pl_orbper"]
            toi_epoch = candidate["_epoch_btjd"]
            if pd.isna(toi_period) or pd.isna(toi_epoch) or toi_period <= 0:
                continue
            relative_period_error = abs(float(period) / float(toi_period) - 1.0)
            if relative_period_error > period_tolerance:
                continue
            phase_error = abs(((float(epoch) - float(toi_epoch) + float(toi_period) / 2) % float(toi_period)) - float(toi_period) / 2)
            duration_days = candidate["pl_trandurh"] / 24 if pd.notna(candidate["pl_trandurh"]) else 0.1
            if phase_error > max(epoch_floor_days, float(duration_days)):
                continue
            possibilities.append((relative_period_error, phase_error, candidate))
        if not possibilities:
            continue
        possibilities.sort(key=lambda item: (item[0], item[1]))
        best = possibilities[0]
        ambiguous = len(possibilities) > 1 and abs(possibilities[1][0] - best[0]) < 1e-5 and abs(possibilities[1][1] - best[1]) < 0.02
        candidate = best[2]
        disposition = str(candidate["tfopwg_disp"]).strip().upper() if pd.notna(candidate["tfopwg_disp"]) else "UNSET"
        if ambiguous:
            label = "AMBIGUOUS_MATCH"
        elif disposition in POSITIVE:
            label = "PLANET_DISPOSED"
        elif disposition in NEGATIVE:
            label = "NON_PLANET_DISPOSED"
        else:
            label = "UNLABELED"

        record = row.to_dict()
        record.update({
            "tic_id": int(tic),
            "toi": str(candidate["toi"]),
            "disposition": disposition,
            "label": label,
            "toi_rowupdate": str(candidate["rowupdate"]),
            "toi_period_days": float(candidate["pl_orbper"]),
            "toi_epoch_btjd": float(candidate["_epoch_btjd"]),
            "relative_period_error": float(best[0]),
            "epoch_phase_error_days": float(best[1]),
            "candidate_match_count": len(possibilities),
        })
        rows.append(record)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tce-dir", type=Path, required=True, help="Folder of downloaded MAST per-sector TCE statistics CSVs")
    parser.add_argument("--toi-csv", type=Path, required=True, help="NASA Exoplanet Archive TOI CSV with dispositions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--period-tolerance", type=float, default=0.005, help="Maximum fractional period mismatch (default 0.5%%)")
    parser.add_argument("--epoch-floor-days", type=float, default=0.08, help="Minimum folded-epoch tolerance (default 0.08 d)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tce, tce_files = read_tce_files(args.tce_dir)
    toi = pd.read_csv(args.toi_csv, low_memory=False)
    matches = build_matches(tce, toi, args.period_tolerance, args.epoch_floor_days)
    if matches.empty:
        raise SystemExit("No period-and-epoch TOI/TCE matches found; no cohort written.")

    all_matches_path = args.output_dir / "tce_toi_all_matches.csv"
    matches.to_csv(all_matches_path, index=False)

    # One representative single-sector TCE observation per astrophysical TOI.
    # Earliest sector is selected deterministically, avoiding max-SNR selection bias.
    labeled = matches[matches["label"].isin(["PLANET_DISPOSED", "NON_PLANET_DISPOSED"])].copy()
    labeled = labeled.sort_values(["tic_id", "toi", "_source_sector", "relative_period_error", "epoch_phase_error_days"])
    cohort = labeled.drop_duplicates(["tic_id", "toi"], keep="first").copy()
    cohort["target"] = (cohort["label"] == "PLANET_DISPOSED").astype(int)
    cohort_path = args.output_dir / "planet_disposition_cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "tce_source_page": SOURCE_PAGE,
        "toi_source_api_guide": TOI_TAP,
        "toi_file": str(args.toi_csv.resolve()),
        "toi_sha256": sha256(args.toi_csv),
        "tce_directory": str(args.tce_dir.resolve()),
        "tce_files": tce_files,
        "tce_input_rows": len(tce),
        "toi_rows": len(toi),
        "period_match_tolerance_fraction": args.period_tolerance,
        "epoch_fold_tolerance_days": "max(epoch_floor_days, TOI duration in days)",
        "epoch_floor_days": args.epoch_floor_days,
        "period_harmonics_accepted": False,
        "hard_label_mapping": {"CP/KP": "PLANET_DISPOSED (target=1)", "FP/FA": "NON_PLANET_DISPOSED (target=0)"},
        "unlabeled_dispositions": ["PC", "APC", "UNSET", "other"],
        "representative_observation_rule": "one record per TIC+TOI; lowest sector number, then closest period/epoch match",
        "matched_rows": len(matches),
        "matched_disposition_counts": matches["disposition"].value_counts(dropna=False).to_dict(),
        "labeled_unique_TIC_TOI_events": len(cohort),
        "labeled_event_counts": cohort["disposition"].value_counts(dropna=False).to_dict(),
        "unique_hosts": int(cohort["tic_id"].nunique()),
        "output_sha256": {"all_matches": sha256(all_matches_path), "labeled_cohort": sha256(cohort_path)},
        "limitations": [
            "TOI follow-up sample is selected and is not representative of all TESS TCEs.",
            "Disposition labels describe current archive status; candidate and ambiguous rows are not forced into either class.",
            "This cohort supports research evaluation, not a deployment-ready planet probability.",
        ],
    }
    manifest_path = args.output_dir / "planet_disposition_cohort_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"matches": len(matches), "cohort_rows": len(cohort), "hosts": cohort["tic_id"].nunique(), "labels": manifest["labeled_event_counts"], "cohort_csv": str(cohort_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
