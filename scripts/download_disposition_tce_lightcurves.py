"""Download one real SPOC light-curve product for each labeled TOI/TCE event.

Products are matched to the sector used by the MAST TCE statistics, prefer
120-second cadence, and are SHA-256 recorded. This can download several GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _cadence_rank(seconds: float) -> tuple[float, float]:
    if seconds <= 0:
        return (1e9, 1e9)
    # 2-minute data is preferred; otherwise use the nearest supported cadence.
    return (abs(seconds - 120.0), seconds)


def discover_products(cohort: pd.DataFrame, batch_size: int = 50) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from astroquery.mast import Observations

    pair_frame = cohort[["tic_id", "_source_sector"]].drop_duplicates().copy()
    pair_frame["tic_id"] = pd.to_numeric(pair_frame.tic_id, errors="raise").astype(int)
    pair_frame["_source_sector"] = pd.to_numeric(pair_frame._source_sector, errors="raise").astype(int)
    pair_to_rows = {(int(tic), int(sector)): [] for tic, sector in pair_frame[["tic_id", "_source_sector"]].itertuples(index=False, name=None)}
    by_tic = pair_frame.groupby("tic_id")["_source_sector"].apply(set).to_dict()
    tics = sorted(by_tic)
    selected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for start in range(0, len(tics), batch_size):
        batch = tics[start:start + batch_size]
        print(f"MAST metadata batch {start // batch_size + 1}/{(len(tics) + batch_size - 1) // batch_size}: TICs {start + 1}-{start + len(batch)}", flush=True)
        try:
            observations = Observations.query_criteria(
                obs_collection="TESS", target_name=[str(tic) for tic in batch], dataproduct_type="timeseries"
            )
            if not len(observations):
                errors.append({"batch_start": start, "tics": batch, "error": "no observations"})
                continue
            obs_df = observations.to_pandas()
            obs_df["_tic_int"] = pd.to_numeric(obs_df["target_name"], errors="coerce")
            obs_df["_sector_int"] = pd.to_numeric(obs_df["sequence_number"], errors="coerce")
            valid_obs = obs_df[
                obs_df.provenance_name.astype(str).str.upper().eq("SPOC")
                & obs_df.dataproduct_type.astype(str).str.lower().eq("timeseries")
                & obs_df.apply(lambda r: pd.notna(r["_tic_int"]) and pd.notna(r["_sector_int"]) and int(r["_sector_int"]) in by_tic.get(int(r["_tic_int"]), set()), axis=1)
            ]
            if valid_obs.empty:
                continue
            obs_ids = set(pd.to_numeric(valid_obs.obsid, errors="coerce").dropna().astype(int))
            sub_obs = observations[[int(x) in obs_ids for x in observations["obsid"]]]
            products = Observations.get_product_list(sub_obs)
            if not len(products):
                continue
            product_df = products.to_pandas()
            product_df = product_df[product_df.productFilename.astype(str).str.lower().str.endswith("_lc.fits")]
            if product_df.empty:
                continue
            obs_lookup = {
                int(record["obsid"]): record
                for record in valid_obs.drop_duplicates("obsid").to_dict("records")
                if pd.notna(pd.to_numeric(record.get("obsid"), errors="coerce"))
            }
            for _, product in product_df.iterrows():
                parent = pd.to_numeric(product.get("parent_obsid"), errors="coerce")
                if pd.isna(parent) or int(parent) not in obs_lookup:
                    continue
                obs = obs_lookup[int(parent)]
                key = (int(obs["_tic_int"]), int(obs["_sector_int"]))
                selected.append({
                    "tic_id": key[0], "sector": key[1], "obsid": int(parent),
                    "cadence_seconds": float(obs.get("t_exptime", 0) or 0),
                    "product_filename": str(product["productFilename"]),
                    "data_uri": str(product["dataURI"]),
                    "size_bytes": int(pd.to_numeric(product.get("size"), errors="coerce")) if pd.notna(pd.to_numeric(product.get("size"), errors="coerce")) else 0,
                })
        except Exception as exc:
            errors.append({"batch_start": start, "tics": batch, "error": f"{type(exc).__name__}: {exc}"})

    # One product per TIC and sector, preferring 120-s cadence then most recent product filename.
    selected.sort(key=lambda x: (x["tic_id"], x["sector"], _cadence_rank(x["cadence_seconds"]), x["product_filename"]))
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for product in selected:
        best.setdefault((product["tic_id"], product["sector"]), product)
    missing = [{"tic_id": tic, "sector": sector} for tic, sector in pair_to_rows if (tic, sector) not in best]
    return list(best.values()), errors + [{"missing_pairs": missing}] if missing else errors


def _download_one(product: dict[str, Any], out_dir: Path, context: ssl.SSLContext) -> dict[str, Any]:
    filename = f"tic{product['tic_id']}_s{product['sector']:04d}_{product['product_filename']}"
    destination = out_dir / filename
    if destination.is_file() and destination.stat().st_size > 0:
        data_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        return {**product, "path": str(destination), "bytes": destination.stat().st_size, "sha256": data_hash, "status": "CACHED"}
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + urllib.parse.quote(product["data_uri"], safe="")
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ASTROVET real-data cohort downloader/0.1"})
            with urllib.request.urlopen(request, timeout=120, context=context) as response:
                data = response.read()
            if len(data) < 2880 or not data.startswith(b"SIMPLE"):
                raise ValueError(f"Response did not look like FITS (bytes={len(data)})")
            temp = destination.with_suffix(destination.suffix + ".part")
            temp.write_bytes(data)
            temp.replace(destination)
            return {**product, "path": str(destination), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "status": "DOWNLOADED"}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
    return {**product, "status": "FAILED", "error": error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--product-manifest", type=Path, default=None, help="Optional JSON path to cache the selected MAST product metadata")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metadata-only", action="store_true", help="Only query and save products; do not download FITS files")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort, low_memory=False)
    metadata_path = args.product_manifest or args.output_dir / "mast_product_selection.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        products = metadata["products"]
        print(f"Reusing cached MAST product selection: ${metadata_path}", flush=True)
    else:
        products, errors = discover_products(cohort, batch_size=args.batch_size)
        metadata = {"queried_utc": datetime.now(timezone.utc).isoformat(), "cohort": str(args.cohort.resolve()), "selected_count": len(products), "estimated_bytes": sum(p["size_bytes"] for p in products), "products": products, "query_errors": errors}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected {len(products)} MAST SPOC light curves; estimated {sum(p['size_bytes'] for p in products)/1e9:.2f} GB")
    print(f"Product manifest: {metadata_path}")
    if args.metadata_only:
        return

    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download_one, product, args.output_dir, context) for product in products]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 50 == 0 or index == len(futures):
                print(f"Completed {index}/{len(futures)} light curves; failures={sum(r['status']=='FAILED' for r in results)}")
    results.sort(key=lambda r: (r["tic_id"], r["sector"]))
    download_manifest = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "product_manifest": str(metadata_path.resolve()),
        "requested_count": len(products),
        "downloaded_count": sum(r["status"] == "DOWNLOADED" for r in results),
        "cached_count": sum(r["status"] == "CACHED" for r in results),
        "failed_count": sum(r["status"] == "FAILED" for r in results),
        "files": results,
    }
    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(download_manifest, indent=2), encoding="utf-8")
    print(f"Download manifest: {manifest_path}")


if __name__ == "__main__":
    main()
