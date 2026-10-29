"""
Command-line interface (CLI) for ASTROVET.

Scientific safeguards:
- All commands enforce explicit data provenance, versioning, and disclaimers.
- Doctor check verifies runtime dependencies and reports declared upstream release metadata.
- Data audit inspects real catalog structure without assuming schema or labels.
- Explain command discloses demonstration mode and uncalibrated heuristic scores.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional
import click
import numpy as np
import pandas as pd

import astrovet
from astrovet.ids import resolve_candidate_identity
from astrovet.provenance import get_runtime_environment, compute_sha256
from astrovet.catalogs.audit import audit_catalog_csv
from astrovet.catalogs.resolver import dataframe_to_candidate_records
from astrovet.ingest.mast import fetch_tess_lightcurve, is_fits_synthetic_fixture
from astrovet.lightcurves.reader import read_tess_fits_lightcurve
from astrovet.lightcurves.quality import filter_quality_cadences
from astrovet.lightcurves.normalization import segment_and_normalize_lightcurve
from astrovet.views.folding import extract_transit_views
from astrovet.physics.diagnostics import run_physics_diagnostics
from astrovet.baseline.adapter import (
    ExoMinerBaselineAdapter,
    EXOMINER_RELEASE_TAG,
    EXOMINER_RELEASE_COMMIT,
    EXOMINER_MODEL_NAME,
    find_default_catalog_path,
)
from astrovet.ssl.encoder import LightCurvePCAPrototypeEncoder
from astrovet.models.fusion import AstrovetFusionModel
from astrovet.models.calibration import PlattCalibrator
from astrovet.evaluation.splits import make_host_grouped_split
from astrovet.evaluation.metrics import compute_comprehensive_metrics
from astrovet.explain.plots import generate_candidate_diagnostic_plot
from astrovet.explain.report import generate_candidate_markdown_report


@click.group()
@click.version_option(version=astrovet.__version__, prog_name="astrovet")
def cli() -> None:
    """ASTROVET: Hybrid, explainable framework for vetting TESS transit candidates."""
    pass


@cli.command("doctor")
def doctor_cmd() -> None:
    """Check environment health, dependencies, and recorded baseline metadata."""
    click.echo("=" * 60)
    click.echo(f"ASTROVET Doctor — System Diagnostic (v{astrovet.__version__})")
    click.echo("=" * 60)
    
    env = get_runtime_environment()
    click.echo(f"Python Version : {env['python_version'].split()[0]}")
    click.echo(f"Platform       : {env['platform']}")
    click.echo(f"Machine        : {env['machine']}")
    
    deps = [
        "numpy", "pandas", "scipy", "astropy", "lightkurve", "astroquery",
        "sklearn", "matplotlib", "pydantic", "pyarrow", "click", "yaml", "jinja2",
        "transitleastsquares"
    ]
    click.echo("\n[Core Dependencies]")
    all_ok = True
    for d in deps:
        try:
            import importlib.metadata
            v = importlib.metadata.version(d)
            click.echo(f"  [OK] {d:<12}: {v}")
        except Exception:
            try:
                mod = __import__(d)
                v = getattr(mod, "__version__", "installed")
                click.echo(f"  [OK] {d:<12}: {v}")
            except ImportError:
                click.echo(f"  [MISSING] {d:<12}")
                all_ok = False

    click.echo("\n[Upstream NASA ExoMiner++ Baseline]")
    click.echo(f"  Model Name             : {EXOMINER_MODEL_NAME}")
    click.echo(f"  Recorded Release Tag   : {EXOMINER_RELEASE_TAG}")
    click.echo(f"  Recorded Git Commit    : {EXOMINER_RELEASE_COMMIT}")
    
    project_root = Path(__file__).resolve().parents[2]
    upstream_dir = project_root.parent / "ExoMiner-main"
    catalog_path = find_default_catalog_path(project_root)
    if catalog_path:
        click.echo(f"  [FOUND] ExoMiner++ prediction catalog: {catalog_path.resolve()}")
    else:
        click.echo("  [UNAVAILABLE] Prediction catalog not found; supply --catalog for reference scores.")
    if upstream_dir.is_dir() and (upstream_dir / "exominer_pipeline").is_dir():
        click.echo(f"  [FOUND] Upstream ExoMiner checkout: {upstream_dir.resolve()}")
        click.echo("  [NOTE] Live container execution also requires an available Docker/Podman daemon.")
    else:
        click.echo("  [INFO] No upstream source checkout; catalog mode remains available when a catalog is present.")

    click.echo("=" * 60)
    if all_ok:
        click.echo("[SUCCESS] All declared runtime dependencies are installed.")
    else:
        click.echo("[WARNING] Some dependencies are missing. Run pip install.")


@cli.group("data")
def data_group() -> None:
    """Data catalog audit, ingestion, and light-curve fetching."""
    pass


@data_group.command("audit")
@click.option("--input", "input_path", required=True, type=click.Path(), help="Path to input CSV catalog.")
@click.option("--output", "output_path", default=None, type=click.Path(), help="Path for JSON audit report.")
def data_audit_cmd(input_path: str, output_path: Optional[str]) -> None:
    """Audit a candidate catalog CSV and output a machine-readable report."""
    click.echo(f"Auditing catalog: {input_path}")
    report = audit_catalog_csv(input_path, output_report_path=output_path)
    
    if report.get("status") == "FILE_NOT_FOUND":
        click.secho(f"Error: {report.get('message')}", fg="red")
        sys.exit(1)

    click.secho("\nAudit Summary:", fg="green", bold=True)
    click.echo(f"  File Size      : {report['file_size_bytes']:,} bytes")
    click.echo(f"  SHA-256        : {report['sha256']}")
    click.echo(f"  Row Count      : {report['row_count']:,}")
    click.echo(f"  Column Count   : {report['column_count']}")
    
    ident = report["identifiers"]
    click.echo(f"  Unique TICs    : {ident['unique_tics']}")
    click.echo(f"  Unique TCEs    : {ident['unique_tces']}")
    click.echo(f"  Stars with >1  : {ident['stars_with_multiple_candidates']}")
    click.echo(f"  TIC is unique? : {ident['tic_alone_is_unique_event']} (TIC alone is NEVER unique event key)")
    
    if output_path:
        click.echo(f"\nMachine-readable report written to: {output_path}")


@data_group.command("fetch")
@click.option("--tic", required=True, type=int, help="Target TIC ID.")
@click.option("--sector", default=23, type=int, help="TESS Sector number.")
@click.option("--output-dir", default="data/raw/mast", type=click.Path(), help="Directory to save FITS product.")
@click.option("--manifest", default="data/manifests/mast_ingest.json", type=click.Path(), help="Output manifest path.")
@click.option(
    "--offline/--online",
    "offline",
    default=False,
    help="Force offline mode using synthetic fixture (--offline) or attempt official online MAST SPOC retrieval (--online, default).",
)
@click.option(
    "--fallback-fixture/--no-fallback-fixture",
    "fallback_fixture",
    default=False,
    help="Explicit opt-in to generate synthetic test fixture when online retrieval fails (default: False).",
)
def data_fetch_cmd(
    tic: int,
    sector: int,
    output_dir: str,
    manifest: str,
    offline: bool,
    fallback_fixture: bool,
) -> None:
    """Fetch TESS SPOC light-curve product and record provenance manifest."""
    mode_desc = "explicit offline fixture mode (--offline)" if offline else "official online MAST retrieval (--online)"
    click.echo(f"Fetching TESS SPOC light curve for TIC {tic} (Sector {sector}) using {mode_desc}...")
    try:
        res = fetch_tess_lightcurve(
            tic_id=tic,
            output_dir=output_dir,
            sector=sector,
            manifest_output_path=manifest,
            offline_mode=offline,
            fixture_if_offline=fallback_fixture,
        )
    except Exception as e:
        raise click.ClickException(f"Light curve retrieval failed: {e}")

    if res.get("is_synthetic_fixture"):
        if offline:
            click.secho("[SUCCESS] Deterministic synthetic test fixture generated (explicit offline mode).", fg="green")
        else:
            click.secho(f"[FALLBACK] Online MAST retrieval unsuccessful ({res.get('error')}); generated synthetic test fixture.", fg="yellow")
    else:
        click.secho("[SUCCESS] Official SPOC light curve retrieved successfully from MAST.", fg="green")

    click.echo(f"  Product File     : {res['product_file']}")
    click.echo(f"  SHA-256 Checksum : {res['sha256']}")
    click.echo(f"  Synthetic Fixture: {res['is_synthetic_fixture']}")
    click.echo(f"  Cadence Seconds  : {res.get('cadence_seconds', 120.0):.1f}s")
    click.echo(f"  Retrieval Mode   : {res.get('retrieval_mode', 'N/A')}")
    click.echo(f"  Manifest Saved   : {manifest}")




@data_group.command("validate-fits")
@click.option("--fits-file", required=True, type=click.Path(exists=True), help="Path to TESS SPOC or QLP FITS light curve.")
@click.option("--output", "output_path", default=None, type=click.Path(), help="Optional JSON path to save audit.")
def data_validate_fits_cmd(fits_file: str, output_path: Optional[str]) -> None:
    """Validate a TESS FITS light curve, check finite cadences, and report quality attrition audit."""
    from astrovet.lightcurves.validator import audit_lightcurve_fits
    click.echo(f"Validating FITS light curve: {fits_file}")
    try:
        rep = audit_lightcurve_fits(fits_file)
    except Exception as e:
        raise click.ClickException(f"Validation failed: {e}")

    click.secho("\nFITS Light Curve Audit Summary:", fg="green", bold=True)
    click.echo(f"  File Name      : {rep['file_name']}")
    click.echo(f"  File Size      : {rep['file_size_bytes']:,} bytes")
    click.echo(f"  SHA-256        : {rep['sha256']}")
    click.echo(f"  Pipeline Origin: {rep['pipeline_origin']}")
    meta = rep["metadata"]
    click.echo(f"  Target TIC ID  : {meta.get('tic_id')}")
    click.echo(f"  Observing Sector: {meta.get('sector')}")
    click.echo(f"  Median Cadence : {meta.get('cadence_seconds')}s")

    ca = rep["cadence_audit"]
    click.secho("\n--- Cadence Attrition & Quality Filtering ---", bold=True)
    click.echo(f"  Raw Total Cadences : {ca['raw_total_cadences']:,}")
    click.echo(f"  Finite & Positive  : {ca['finite_positive_cadences']:,}")
    click.echo(f"  Non-Finite Rejected: {ca['non_finite_rejected']:,}")
    click.echo(f"  Bitmask Filtered   : {ca['quality_bitmask_retained']:,} retained ({ca['retention_percentage']}%)")
    click.echo(f"  Usable Cadences    : {ca['usable_cadences']:,}")
    click.echo(f"  Observing Segments : {len(rep['segments'])}")

    if rep.get("warnings"):
        click.secho("\nWarnings:", fg="yellow", bold=True)
        for w in rep["warnings"]:
            click.echo(f"  - {w}")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        click.echo(f"\nAudit saved to: {out_p.resolve()}")


@cli.group("baseline")
def baseline_group() -> None:
    """Upstream NASA ExoMiner++ baseline commands."""
    pass


@baseline_group.command("run")
@click.option("--catalog", default=None, type=click.Path(), help="Precomputed baseline scores catalog.")
@click.option("--candidate-id", "candidate_ids", multiple=True, help="Candidate ID(s) to score.")
@click.option("--synthetic", is_flag=True, default=False, help="Run in synthetic fixture mode.")
def baseline_run_cmd(catalog: Optional[str], candidate_ids: tuple, synthetic: bool) -> None:
    """Run ExoMiner++ baseline adapter (catalog lookup & fixture mode) and verify reconciliation."""
    adapter = ExoMinerBaselineAdapter(catalog_path=catalog, is_synthetic=synthetic)
    cids = list(candidate_ids) if candidate_ids else ["390651552-1-S23"]
    click.echo(f"Running baseline catalog adapter for {len(cids)} candidate(s)...")
    preds, recon = adapter.reconcile_batch(cids)
    
    click.echo("\nReconciliation Report:")
    click.echo(f"  Requested    : {recon['requested_count']}")
    click.echo(f"  Success      : {recon['reconciled_success_count']}")
    click.echo(f"  Missing/Fail : {recon['missing_or_failed_count']}")
    
    for p in preds:
        score_str = f"{p.score:.4f}" if p.score is not None else "None (NOT_FOUND)"
        click.echo(f"  Candidate: {p.candidate_id} | Score: {score_str} | Mode: {p.execution_mode} | Status: {p.status}")


@cli.command("explain")
@click.option("--candidate-id", required=True, type=str, help="Target compound candidate ID (e.g. 390651552-1-S23).")
@click.option("--fits-file", default=None, type=click.Path(), help="Path to SPOC light curve FITS.")
@click.option("--output-dir", default="artifacts/reports", type=click.Path(), help="Output directory.")
@click.option("--run-id", default="run_vet_001", help="Execution run identifier.")
@click.option(
    "--baseline-mode",
    type=click.Choice(["catalog", "container", "synthetic"]),
    default="catalog",
    help="ExoMiner baseline execution branch (catalog lookup, container process, or synthetic fixture)."
)
@click.option(
    "--container-fallback/--no-container-fallback",
    default=True,
    help="When container daemon is unavailable, fall back to catalog lookup instead of failing."
)
@click.option(
    "--encoder-mode",
    type=click.Choice(["ssl", "pca"]),
    default="ssl",
    help="Representation encoder branch: self-supervised neural encoder (ssl) or linear PCA prototype (pca)."
)
@click.option(
    "--ssl-checkpoint",
    default=None,
    type=click.Path(),
    help="Optional path to trained SSL encoder .npz checkpoint."
)
@click.option(
    "--period",
    default=None,
    type=float,
    help="Candidate orbital period in days (if verifying/overriding catalog)."
)
@click.option(
    "--epoch",
    default=None,
    type=float,
    help="Candidate transit epoch in BTJD (BJD - 2457000.0)."
)
@click.option(
    "--duration",
    default=None,
    type=float,
    help="Candidate transit duration in hours."
)
@click.option(
    "--depth",
    default=None,
    type=float,
    help="Candidate transit depth in ppm."
)
@click.option(
    "--search-periods/--no-search-periods",
    default=True,
    help="Execute BLS and TLS transit periodicity search."
)
@click.option(
    "--min-period",
    default=0.5,
    type=float,
    help="Minimum period in days for periodogram search."
)
@click.option(
    "--max-period",
    default=None,
    type=float,
    help="Maximum period in days for periodogram search."
)
@click.option(
    "--allow-synthetic",
    is_flag=True,
    default=False,
    help="Explicit opt-in to allow synthetic demo fixture when real FITS is missing."
)
@click.option(
    "--model-artifact",
    default=None,
    type=click.Path(),
    help="Optional path to a compatible fitted fusion-model bundle (.joblib); catalog E-vs-B/J artifacts are not planet-vetting models."
)
def explain_cmd(
    candidate_id: str,
    fits_file: Optional[str],
    output_dir: str,
    run_id: str,
    baseline_mode: str,
    container_fallback: bool,
    encoder_mode: str,
    ssl_checkpoint: Optional[str],
    period: Optional[float],
    epoch: Optional[float],
    duration: Optional[float],
    depth: Optional[float],
    search_periods: bool,
    min_period: float,
    max_period: Optional[float],
    allow_synthetic: bool,
    model_artifact: Optional[str] = None,
) -> None:
    """Run full ASTROVET vetting slice with BLS/TLS verification and generate evidence report."""
    from astrovet.explain.report import vet_and_generate_report

    click.echo("=" * 60)
    click.echo(f"ASTROVET Vetting & Evidence Generation: {candidate_id}")
    if encoder_mode == "ssl":
        if ssl_checkpoint:
            click.secho("Loading trained self-supervised representation checkpoint...", fg="blue")
        else:
            click.secho("SSL representation unavailable: no trained checkpoint supplied; this modality will be omitted.", fg="yellow")
    if baseline_mode == "container":
        click.secho("Executing official ExoMiner++ via container runner...", fg="blue")

    try:
        res = vet_and_generate_report(
            candidate_id=candidate_id,
            output_dir=output_dir,
            fits_file=fits_file,
            run_id=run_id,
            period=period,
            epoch=epoch,
            duration=duration,
            depth=depth,
            search_periods=search_periods,
            min_period_days=min_period,
            max_period_days=max_period,
            baseline_mode=baseline_mode,
            container_fallback=container_fallback,
            encoder_mode=encoder_mode,
            ssl_checkpoint=ssl_checkpoint,
            allow_synthetic=allow_synthetic,
            model_artifact=model_artifact,
        )
    except FileNotFoundError as fnf_err:
        raise click.ClickException(f"{fnf_err}")
    except Exception as e:
        raise click.ClickException(f"Vetting pipeline error: {e}")

    cand = res["candidate"]
    pred = res["prediction"]
    report_file = res["report_file"]
    plot_file = res["plot_file"]
    pgram = res.get("periodogram_results", {})
    bls = pgram.get("bls", {})
    tls = pgram.get("tls", {})

    click.secho("\n[SUCCESS] Candidate Vetting Completed!", fg="green", bold=True)
    click.echo(f"  Candidate ID     : {cand.candidate_id}")
    click.echo(f"  Host TIC / Sector: TIC {cand.tic_id} | Sector {cand.sector_run}")
    click.echo(f"  Ephemeris Used   : P={cand.ephemeris.period_days:.5f} d | T0={cand.ephemeris.epoch_bjd_tdb:.4f} BTJD | Dur={cand.ephemeris.duration_hours:.3f} h")
    click.echo(f"  Baseline Score   : {pred.baseline_score if pred.baseline_score is not None else 'N/A'}")

    if search_periods:
        click.secho("\n--- Periodicity Verification (BLS & TLS) ---", bold=True)
        if bls.get("is_available"):
            if bls.get("compared_to_ephemeris"):
                b_match = "MATCH" if bls.get("is_period_match") else ("HARMONIC" if bls.get("is_harmonic") else "DIFF")
                diff_val = bls.get("period_diff_percent")
                diff_str = f"{diff_val:.2f}%" if diff_val is not None else "N/A"
                click.echo(f"  BLS Best Period  : {bls['best_period_days']:.5f} d (SNR={bls.get('peak_snr', 0):.1f}) -> {b_match} ({diff_str})")
            else:
                click.echo(f"  BLS Best Period  : {bls['best_period_days']:.5f} d (SNR={bls.get('peak_snr', 0):.1f}) -> RECOVERED")
        else:
            click.echo(f"  BLS Periodogram  : Unavailable ({bls.get('reason')})")

        if tls.get("is_available"):
            oe_str = f"{tls['odd_even_mismatch']:.2f}" if tls.get("odd_even_mismatch") is not None else "N/A"
            if tls.get("compared_to_ephemeris"):
                t_match = "MATCH" if tls.get("is_period_match") else ("HARMONIC" if tls.get("is_harmonic") else "DIFF")
                diff_val = tls.get("period_diff_percent")
                diff_str = f"{diff_val:.2f}%" if diff_val is not None else "N/A"
                click.echo(f"  TLS Best Period  : {tls['best_period_days']:.5f} d (SDE={tls.get('sde', 0):.1f}, Odd/Even={oe_str}) -> {t_match} ({diff_str})")
            else:
                click.echo(f"  TLS Best Period  : {tls['best_period_days']:.5f} d (SDE={tls.get('sde', 0):.1f}, Odd/Even={oe_str}) -> RECOVERED")
        else:
            click.echo(f"  TLS Periodogram  : Unavailable ({tls.get('reason')})")

        if bls.get("is_available") and tls.get("is_available"):
            agree_str = "YES" if pgram.get("period_agreement") else "NO"
            click.echo(f"  BLS/TLS Agreement: {agree_str} (Diff: {pgram.get('period_agreement_diff_percent', 'N/A')}%)")

    click.secho("\n--- Vetting Decision ---", bold=True)
    click.echo(f"  ASTROVET Score   : {pred.astrovet_raw_score:.4f} ({pred.score_type})")
    click.echo(f"  Calibrated Prob  : {pred.calibrated_planet_like_probability or 'N/A (uncalibrated heuristic)'}")
    click.echo(f"  Triage Decision  : {pred.triage_state.value}")
    click.echo(f"  Active Reasons   : {', '.join(pred.reason_codes) if pred.reason_codes else 'None'}")
    click.echo(f"  Diagnostic Plot  : {plot_file.resolve()}")
    click.echo(f"  Report Saved To  : {report_file.resolve()}")


@cli.group("periodogram")
def periodogram_group() -> None:
    """Box Least Squares (BLS) and Transit Least Squares (TLS) transit periodogram commands."""
    pass


@periodogram_group.command("search")
@click.option("--fits-file", required=True, type=click.Path(exists=True), help="Path to real observed TESS SPOC FITS file.")
@click.option("--min-period", default=0.5, type=float, help="Minimum search period in days.")
@click.option("--max-period", default=None, type=float, help="Maximum search period in days.")
@click.option("--period", default=None, type=float, help="Known candidate period (days) to verify/compare.")
@click.option("--epoch", default=None, type=float, help="Known candidate epoch (BTJD).")
@click.option("--duration", default=None, type=float, help="Known candidate duration (hours).")
@click.option("--method", type=click.Choice(["both", "bls", "tls"]), default="both", help="Algorithm to run.")
def periodogram_search_cmd(
    fits_file: str,
    min_period: float,
    max_period: Optional[float],
    period: Optional[float],
    epoch: Optional[float],
    duration: Optional[float],
    method: str,
) -> None:
    """Run BLS and/or TLS transit periodogram search on a TESS FITS light curve."""
    from astrovet.lightcurves.reader import read_tess_fits_lightcurve
    from astrovet.lightcurves.quality import filter_quality_cadences
    from astrovet.lightcurves.normalization import segment_and_normalize_lightcurve
    from astrovet.views.periodogram import run_periodicity_search

    click.secho("=== ASTROVET Transit Periodicity Search (BLS & TLS) ===", bold=True)
    click.echo(f"Reading real observed light curve: {fits_file}")
    raw_lc = read_tess_fits_lightcurve(fits_file)
    clean_t, clean_f, clean_fe, q_rep = filter_quality_cadences(raw_lc.time, raw_lc.flux, raw_lc.flux_err, raw_lc.quality)
    click.echo(f"Filtered cadences: {q_rep['retained_fraction']:.1%} valid ({q_rep['retained_cadences']:,} of {q_rep['total_cadences']:,} cadences)")

    norm_lc = segment_and_normalize_lightcurve(clean_t, clean_f, clean_fe)

    run_bls = method in ["both", "bls"]
    run_tls = method in ["both", "tls"]

    res = run_periodicity_search(
        time=norm_lc.time,
        flux=norm_lc.flux_detrended,
        flux_err=norm_lc.flux_err_norm,
        min_period_days=min_period,
        max_period_days=max_period,
        candidate_period=period,
        candidate_epoch=epoch,
        candidate_duration=duration,
        run_bls=run_bls,
        run_tls=run_tls,
    )

    bls = res.get("bls", {})
    tls = res.get("tls", {})

    if run_bls:
        click.secho("\n--- Box Least Squares (BLS) ---", bold=True)
        if bls.get("is_available"):
            click.echo(f"  Best Period   : {bls['best_period_days']:.5f} days")
            click.echo(f"  Peak SNR      : {bls['peak_snr']:.2f}")
            click.echo(f"  Epoch (T0)    : {bls['best_t0_btjd']:.4f} BTJD")
            click.echo(f"  Duration      : {bls['best_duration_hours']:.3f} hours")
            click.echo(f"  Transit Depth : {bls['best_depth_ppm']:.1f} ppm")
            if bls.get("compared_to_ephemeris"):
                status = "MATCH" if bls.get("is_period_match") else ("HARMONIC" if bls.get("is_harmonic") else "MISMATCH")
                click.echo(f"  Ephemeris Comp: {status} (Diff: {bls.get('period_diff_percent')}%, Harmonic: {bls.get('harmonic_relation')})")
        else:
            click.secho(f"  BLS Unavailable: {bls.get('reason')}", fg="yellow")

    if run_tls:
        click.secho("\n--- Transit Least Squares (TLS) ---", bold=True)
        if tls.get("is_available"):
            click.echo(f"  Best Period   : {tls['best_period_days']:.5f} days")
            click.echo(f"  Signal SDE    : {tls['sde']:.2f}")
            click.echo(f"  Transit SNR   : {tls['snr'] if tls['snr'] is not None else 'N/A'}")
            click.echo(f"  Epoch (T0)    : {tls['best_t0_btjd']:.4f} BTJD")
            click.echo(f"  Duration      : {tls['best_duration_hours']:.3f} hours")
            click.echo(f"  Transit Depth : {tls['best_depth_ppm']:.1f} ppm")
            click.echo(f"  Odd/Even Diff : {tls.get('odd_even_mismatch', 'N/A')}")
            click.echo(f"  Transit Count : {tls.get('transit_count')} observed ({tls.get('distinct_transit_count')} distinct)")
            if tls.get("compared_to_ephemeris"):
                status = "MATCH" if tls.get("is_period_match") else ("HARMONIC" if tls.get("is_harmonic") else "MISMATCH")
                click.echo(f"  Ephemeris Comp: {status} (Diff: {tls.get('period_diff_percent')}%, Harmonic: {tls.get('harmonic_relation')})")
        else:
            click.secho(f"  TLS Unavailable: {tls.get('reason')}", fg="yellow")

    if run_bls and run_tls:
        click.secho("\n--- Algorithm Agreement ---", bold=True)
        agree_str = "YES (Consistent)" if res.get("period_agreement") else "NO (Discrepant)"
        click.echo(f"  BLS vs TLS Agreement: {agree_str} (Diff: {res.get('period_agreement_diff_percent', 'N/A')}%)")

    click.echo("\nScientific Notice: Periodicity detection confirms signal recurrence, NOT exoplanetary origin.")



@cli.command("ui")
@click.option("--host", default="127.0.0.1", help="Host address to bind to.")
@click.option("--port", default=8080, type=int, help="Port to listen on.")
def ui_command(host: str, port: int) -> None:
    """Start interactive local browser UI dashboard for human review and candidate vetting."""
    from astrovet.ui.server import run_dashboard_server
    run_dashboard_server(host=host, port=port)




@cli.group("ssl")
def ssl_group() -> None:
    """Self-supervised representation learning and checkpointing commands."""
    pass


@ssl_group.command("pretrain")
@click.option("--input-fits", default=None, type=str, help="Path to single real unlabeled TESS FITS light curve.")
@click.option("--input-fits-dir", default=None, type=str, help="Directory containing real unlabeled TESS FITS files.")
@click.option("--train-host-split-manifest", default=None, type=click.Path(exists=True), help="Host-grouped model-training manifest; when supplied, only train-partition TICs are used for SSL.")
@click.option("--max-input-files", default=None, type=click.IntRange(min=1), help="Optional deterministic sample cap for a FITS directory (applied after training-host filtering).")
@click.option("--input-sample-seed", default=42, type=int, help="Seed used when --max-input-files samples a FITS directory.")
@click.option("--prepared-views", default=None, type=str, help="Path to pre-extracted .npz views file.")
@click.option("--synthetic-demo", is_flag=True, default=False, help="Explicit opt-in to generate synthetic transit curves for demo/testing only.")
@click.option("--epochs", default=5, type=int, help="Pretraining epochs.")
@click.option("--batch-size", default=16, type=int, help="Batch size.")
@click.option("--lr", default=0.005, type=float, help="Learning rate.")
@click.option("--output-checkpoint", default="data/checkpoints/ssl_transit_encoder.npz", help="Output .npz checkpoint.")
@click.option("--num-samples", default=64, type=int, help="Number of samples if running in --synthetic-demo mode.")
def ssl_pretrain_cmd(
    input_fits: Optional[str],
    input_fits_dir: Optional[str],
    train_host_split_manifest: Optional[str],
    max_input_files: Optional[int],
    input_sample_seed: int,
    prepared_views: Optional[str],
    synthetic_demo: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    output_checkpoint: str,
    num_samples: int,
) -> None:
    """Pretrain self-supervised neural encoder on real TESS FITS views or synthetic demo views and save checkpoint."""
    from datetime import datetime, timezone
    from astrovet.ssl.neural_encoder import SelfSupervisedTransitEncoder
    from astrovet.ssl.dataset import extract_views_from_fits, extract_views_from_directory, load_prepared_views

    click.secho(f"=== ASTROVET Self-Supervised Pretraining (MAE + View Consistency) ===", bold=True)

    if not input_fits and not input_fits_dir and not prepared_views and not synthetic_demo:
        raise click.UsageError(
            "Unconditional synthetic training is disabled. "
            "Please provide real TESS light curves using --input-fits, --input-fits-dir, or --prepared-views. "
            "To run a synthetic CPU verification test instead, pass --synthetic-demo explicitly."
        )

    allowed_tic_ids = None
    split_manifest_sha256 = None
    if train_host_split_manifest:
        if not input_fits_dir:
            raise click.UsageError("--train-host-split-manifest requires --input-fits-dir so only the training-host light curves can be selected.")
        split_path = Path(train_host_split_manifest)
        with open(split_path, "r", encoding="utf-8") as f:
            split_doc = json.load(f)
        allowed_tic_ids = set(split_doc.get("metadata", {}).get("host_ids_by_split", {}).get("train", []))
        if not allowed_tic_ids:
            raise click.ClickException("The split manifest contains no metadata.host_ids_by_split.train IDs.")
        split_manifest_sha256 = compute_sha256(split_path)

    provenance: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }

    if synthetic_demo:
        click.secho(
            "[DEMO NOTICE] Pretraining on synthetic transit population for testing/demonstration only (--synthetic-demo enabled).",
            fg="yellow",
            bold=True,
        )
        rng = np.random.default_rng(42)
        phase = np.linspace(-0.5, 0.5, 200)
        views = []
        for _ in range(num_samples):
            depth = rng.uniform(0.0005, 0.015)
            duration = rng.uniform(0.04, 0.18)
            flux = np.ones(200)
            in_transit = np.abs(phase) < (duration / 2.0)
            flux[in_transit] -= depth
            flux += rng.normal(0, 0.0001, size=200)
            views.append(flux)
        views_arr = np.array(views, dtype=np.float64)
        provenance["data_source"] = "SYNTHETIC_DEMO"
        provenance["num_views"] = len(views_arr)
        provenance["num_synthetic_samples"] = num_samples
    elif input_fits:
        click.echo(f"Extracting real training views from TESS FITS: {input_fits}")
        views_arr, meta = extract_views_from_fits(input_fits)
        provenance["data_source"] = "REAL_TESS_FITS"
        provenance["input_mode"] = "single_fits"
        provenance["input_files"] = [meta["file_path"]]
        provenance["files_metadata"] = [meta]
        provenance["num_views"] = len(views_arr)
        provenance["tic_id"] = meta.get("tic_id")
        provenance["sector"] = meta.get("sector")
        provenance["cadence_seconds"] = meta.get("cadence_seconds")
        click.echo(
            f"  Extracted {len(views_arr)} real views (TIC {meta.get('tic_id')}, Sector {meta.get('sector')}, {meta.get('cadence_seconds')}s cadence, {meta.get('retained_fraction')*100:.1f}% valid cadences)."
        )
    elif input_fits_dir:
        click.echo(f"Scanning directory for real TESS FITS curves: {input_fits_dir}")
        views_arr, meta_list = extract_views_from_directory(
            input_fits_dir,
            max_files=max_input_files,
            allowed_tic_ids=allowed_tic_ids,
            sample_seed=input_sample_seed,
        )
        provenance["data_source"] = "REAL_TESS_FITS"
        provenance["input_mode"] = "directory"
        provenance["input_dir"] = str(Path(input_fits_dir).resolve())
        used_meta = [m for m in meta_list if m.get("views_extracted", 0) > 0]
        skipped_meta = [m for m in meta_list if m.get("skipped")]
        provenance["input_files"] = [m["file_path"] for m in used_meta]
        provenance["files_metadata"] = meta_list
        provenance["num_views"] = len(views_arr)
        provenance["num_light_curve_files_used"] = len(used_meta)
        provenance["num_light_curve_files_skipped"] = len(skipped_meta)
        provenance["skipped_file_audit"] = skipped_meta
        provenance["max_input_files"] = max_input_files
        provenance["input_sample_seed"] = input_sample_seed
        if train_host_split_manifest:
            provenance["training_host_split_manifest"] = str(Path(train_host_split_manifest).resolve())
            provenance["training_host_split_manifest_sha256"] = split_manifest_sha256
            provenance["training_host_tic_count_in_manifest"] = len(allowed_tic_ids)
            provenance["ssl_partition_policy"] = "train hosts only; validation/calibration/test host curves excluded"
        click.echo(
            f"  Extracted {len(views_arr)} real views from {len(used_meta)} FITS files; "
            f"skipped {len(skipped_meta)} files (reasons are recorded in checkpoint provenance)."
        )
    elif prepared_views:
        click.echo(f"Loading prepared training views from: {prepared_views}")
        views_arr, meta = load_prepared_views(prepared_views)
        provenance["data_source"] = "PREPARED_VIEWS_FILE"
        provenance["input_file"] = meta["file_path"]
        provenance["file_metadata"] = meta
        provenance["num_views"] = len(views_arr)
        click.echo(f"  Loaded {len(views_arr)} pre-extracted views from {meta['file_name']}.")

    encoder = SelfSupervisedTransitEncoder(input_dim=200, hidden_dim=64, embedding_dim=16, seed=42)
    click.echo(f"Training encoder for {epochs} epochs on {len(views_arr)} views (batch_size={batch_size}, lr={lr})...")

    history = encoder.fit(views_arr, epochs=epochs, lr=lr, batch_size=batch_size)

    for ep in range(epochs):
        click.echo(
            f"  Epoch {ep+1:2d}/{epochs:2d} | "
            f"Total Loss: {history['loss'][ep]:.6f} | "
            f"Recon Loss: {history['recon_loss'][ep]:.6f} | "
            f"View Loss: {history['view_loss'][ep]:.6f}"
        )

    provenance["final_loss"] = float(history["loss"][-1])
    provenance["final_recon_loss"] = float(history["recon_loss"][-1])
    provenance["final_view_loss"] = float(history["view_loss"][-1])

    out_p = encoder.save_checkpoint(output_checkpoint, provenance=provenance)
    click.secho(f"[SUCCESS] Trained SSL Checkpoint Saved To: {out_p.resolve()}", fg="green", bold=True)
    click.echo(f"  Checkpoint Data Source: {provenance['data_source']}")
    click.echo(f"  Trained Views Count:    {provenance['num_views']}")
    click.echo("Provenance note: Pretrained on local CPU v1 dataset. Large-scale pretraining on full archive requires GPU cluster.")


@cli.command("ui")
@click.option("--host", default="127.0.0.1", help="Host interface to bind.")
@click.option("--port", default=8080, type=int, help="Port to listen on.")
def ui_cmd(host: str, port: int) -> None:
    """Start local interactive ASTROVET vetting dashboard server."""
    from astrovet.ui.server import run_dashboard_server
    run_dashboard_server(host=host, port=port)




@cli.group("model")
def model_group() -> None:
    """Model training, probability calibration, and evaluation commands."""
    pass


@model_group.command("train")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Path to TCE catalog CSV (e.g. tces_with_labels_v3.csv).")
@click.option("--artifacts-dir", default="artifacts", type=click.Path(), help="Directory to save models, manifests, and reports.")
@click.option("--seed", default=42, type=int, help="Random seed for host-grouped splitting.")
def model_train_cmd(data_path: str, artifacts_dir: str, seed: int) -> None:
    """Train the tabular TCE E-versus-B/J baseline with leakage-safe splits and calibration."""
    from astrovet.models.training import run_training_pipeline

    click.secho("=== ASTROVET Model Training Workflow ===", bold=True)
    click.echo(f"  Dataset Path  : {data_path}")
    click.echo(f"  Artifacts Dir : {artifacts_dir}")
    click.echo(f"  Random Seed   : {seed}")

    try:
        res = run_training_pipeline(
            dataset_path=data_path,
            artifacts_dir=artifacts_dir,
            random_seed=seed,
        )
    except Exception as e:
        raise click.ClickException(f"Training failed: {e}")

    click.secho("\n[SUCCESS] Model Training & Evaluation Completed!", fg="green", bold=True)
    audit = res["dataset_audit"]["valid_labeled_cohort"]
    click.echo(f"  Valid Labeled Candidates : {audit['total_candidates']:,} across {audit['unique_host_stars']:,} unique host stars")
    click.echo(f"  Positive (E: eclipsing signal): {audit['positive_count']:,} ({audit['positive_prevalence']:.1%})")
    click.echo(f"  Negative (B/J: contact binary or junk): {audit['negative_count']:,}")

    cal = res["test_evaluation"]
    click.secho("\n--- Untouched Test Partition Metrics ---", bold=True)
    click.echo(f"  ROC-AUC         : {cal['ranking']['roc_auc']:.4f}")
    click.echo(f"  PR-AUC          : {cal['ranking']['pr_auc']:.4f}")
    click.echo(f"  Raw Brier Score : {cal['calibration']['raw_brier_score']:.4f}")
    click.echo(f"  Calibrated Brier: {cal['calibration']['calibrated_brier_score']:.4f}")
    click.echo(f"  Raw ECE         : {cal['calibration']['raw_ece']:.4f}")
    click.echo(f"  Calibrated ECE  : {cal['calibration']['calibrated_ece']:.4f}")
    cls_rep = cal.get("classification_at_p05", {})
    click.echo(f"  Precision (p>=0.5): {cls_rep.get('precision', 'N/A')}")
    click.echo(f"  Recall (p>=0.5)   : {cls_rep.get('recall', 'N/A')}")
    click.echo(f"  F1 Score (p>=0.5) : {cls_rep.get('f1_score', 'N/A')}")

    click.echo(f"\n  Model Artifact Saved  : {res['model_artifact']}")
    click.echo(f"  Split Manifest Saved  : {res['split_manifest_file']}")


@cli.command("train")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Path to TCE catalog CSV (e.g. tces_with_labels_v3.csv).")
@click.option("--artifacts-dir", default="artifacts", type=click.Path(), help="Directory to save models, manifests, and reports.")
@click.option("--seed", default=42, type=int, help="Random seed for host-grouped splitting.")
def root_train_cmd(data_path: str, artifacts_dir: str, seed: int) -> None:
    """Train the tabular TCE E-versus-B/J baseline with leakage-safe splits and calibration."""
    from astrovet.models.training import run_training_pipeline

    click.secho("=== ASTROVET Model Training Workflow ===", bold=True)
    click.echo(f"  Dataset Path  : {data_path}")
    click.echo(f"  Artifacts Dir : {artifacts_dir}")
    click.echo(f"  Random Seed   : {seed}")

    try:
        res = run_training_pipeline(
            dataset_path=data_path,
            artifacts_dir=artifacts_dir,
            random_seed=seed,
        )
    except Exception as e:
        raise click.ClickException(f"Training failed: {e}")

    click.secho("\n[SUCCESS] Model Training & Evaluation Completed!", fg="green", bold=True)
    audit = res["dataset_audit"]["valid_labeled_cohort"]
    click.echo(f"  Valid Labeled Candidates : {audit['total_candidates']:,} across {audit['unique_host_stars']:,} unique host stars")
    click.echo(f"  Positive (E: eclipsing signal): {audit['positive_count']:,} ({audit['positive_prevalence']:.1%})")
    click.echo(f"  Negative (B/J: contact binary or junk): {audit['negative_count']:,}")

    cal = res["test_evaluation"]
    click.secho("\n--- Untouched Test Partition Metrics ---", bold=True)
    click.echo(f"  ROC-AUC         : {cal['ranking']['roc_auc']:.4f}")
    click.echo(f"  PR-AUC          : {cal['ranking']['pr_auc']:.4f}")
    click.echo(f"  Raw Brier Score : {cal['calibration']['raw_brier_score']:.4f}")
    click.echo(f"  Calibrated Brier: {cal['calibration']['calibrated_brier_score']:.4f}")
    click.echo(f"  Raw ECE         : {cal['calibration']['raw_ece']:.4f}")
    click.echo(f"  Calibrated ECE  : {cal['calibration']['calibrated_ece']:.4f}")
    cls_rep = cal.get("classification_at_p05", {})
    click.echo(f"  Precision (p>=0.5): {cls_rep.get('precision', 'N/A')}")
    click.echo(f"  Recall (p>=0.5)   : {cls_rep.get('recall', 'N/A')}")
    click.echo(f"  F1 Score (p>=0.5) : {cls_rep.get('f1_score', 'N/A')}")

    click.echo(f"\n  Model Artifact Saved  : {res['model_artifact']}")
    click.echo(f"  Split Manifest Saved  : {res['split_manifest_file']}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
