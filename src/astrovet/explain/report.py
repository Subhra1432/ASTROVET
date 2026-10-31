"""
Reviewable candidate evidence report generator (Markdown & JSON).

Scientific safeguards:
- Clear header statement:
  "ASTROVET is a candidate-prioritization and triage aid for human review.
  A high score is NOT a planet confirmation. A low score is NOT proof of absence."
- Integrates Box Least Squares (BLS) and Transit Least Squares (TLS) periodograms.
- Evaluates period agreement and harmonic relations against provided candidate ephemeris.
- Recovers candidate parameters when ephemeris is missing without fabricating example values.
- Real observed data is strictly prioritized; synthetic fixtures require explicit opt-in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from astrovet.schemas.candidate import CandidateRecord, Ephemeris
from astrovet.schemas.prediction import AstrovetPrediction, PhysicsDiagnostics
from astrovet.explain.reasons import get_reason_explanation


def generate_candidate_markdown_report(
    candidate: CandidateRecord,
    prediction: AstrovetPrediction,
    plot_relative_path: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Format a comprehensive, reviewable Markdown vetting card including BLS/TLS verification.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    triage = prediction.triage_state.value
    p_diag = prediction.physics_diagnostics
    pgram = getattr(prediction, "periodogram_diagnostics", None) or p_diag.periodogram_summary or {}

    baseline_score_display = (
        f"`{prediction.baseline_score:.4f}`"
        if prediction.baseline_score is not None
        else "*Unavailable (NOT_FOUND in reference catalog)*"
    )

    bls = pgram.get("bls", {})
    tls = pgram.get("tls", {})

    lines = [
        f"# ASTROVET Candidate Vetting Report: `{candidate.candidate_id}`",
        "",
        "> [!IMPORTANT]",
        "> **Scientific Status:** ASTROVET is a candidate-prioritization and triage aid for human review.",
        "> A high model score is **NOT** a planet confirmation or statistical validation.",
        "> A low model score is **NOT** proof that the target star hosts no planets.",
        "",
        f"**Report Generated:** {now_utc}  ",
        f"**Execution Run ID:** `{prediction.run_id}`  ",
        f"**ASTROVET Model Version:** `{prediction.model_version}`  ",
        f"**Synthetic / Demo Fixture:** `{'YES — DEMO / FIXTURE (NO SCIENTIFIC CLAIMS)' if prediction.is_synthetic else 'NO (OBSERVED FLIGHT DATA)'}`  ",
        "",
        "---",
        "",
        "## 1. Candidate Identity & Ephemeris",
        "",
        "| Parameter | Value | Unit / Standard |",
        "|---|---|---|",
        f"| **Candidate ID** | `{candidate.candidate_id}` | Canonical Compound Key |",
        f"| **Host Star (TIC ID)** | `{candidate.tic_id}` | TESS Input Catalog |",
        f"| **TCE Planet Number** | `{candidate.planet_num}` | SPOC Integer Event # |",
        f"| **Sector Run** | `{candidate.sector_run}` | TESS Observing Sector(s) |",
        f"| **Orbital Period** | `{candidate.ephemeris.period_days:.5f}` | Days |",
        f"| **Transit Epoch** | `{candidate.ephemeris.epoch_bjd_tdb:.4f}` | {candidate.ephemeris.time_convention} |",
        f"| **Transit Duration** | `{candidate.ephemeris.duration_hours:.3f}` | Hours |",
        f"| **Reported / Fitted Depth** | `{candidate.ephemeris.transit_depth_ppm or 'N/A'}` | ppm |",
        f"| **Gaia DR3 RUWE** | `{candidate.gaia_ruwe or 'N/A'}` | Astrometric Quality |",
        f"| **Ground Truth Source Label** | `{candidate.label_raw or 'UNLABELED'}` | Source Catalog Record |",
        "",
        "## 2. Vetting Classification & Model Scores",
        "",
        "| Model Branch | Score | Confidence / Uncertainty | Status / Source |",
        "|---|---|---|---|",
        f"| **NASA ExoMiner++ (Baseline)** | {baseline_score_display} | `+/- {prediction.baseline_uncertainty if prediction.baseline_uncertainty is not None else 'N/A'}` | {prediction.baseline_prediction.upstream_model if prediction.baseline_prediction else 'N/A'} ({prediction.baseline_prediction.execution_mode if prediction.baseline_prediction else 'N/A'}) |",
    ]

    # Render calibrated probability only when is_calibrated and value is present
    if prediction.is_calibrated and prediction.calibrated_planet_like_probability is not None:
        calib_prov = prediction.calibration_provenance if getattr(prediction, "calibration_provenance", None) else "Calibration provenance not recorded"
        score_label = "TOI-Cohort Calibrated Planet-Like Score" if "selected TOI" in str(calib_prov) else "Calibrated Planet-Like Probability"
        lines.extend([
            f"| **ASTROVET Raw Score** | `{prediction.astrovet_raw_score:.4f}` | Predictive Unc: `+/- {prediction.predictive_uncertainty:.3f}` | `{prediction.score_type}` |",
            f"| **{score_label}** | **`{prediction.calibrated_planet_like_probability:.4f}`** | Fitted Platt Calibrator | {calib_prov} |",
            *(["> **Calibration limit:** This score is calibrated on a selected TOI follow-up cohort and is not a population-wide planet probability."] if "selected TOI" in str(calib_prov) else []),
        ])
    else:
        score_label = "Uncalibrated Fitted Fusion Score" if prediction.score_type == "fitted_fusion_score" else "ASTROVET Score (Uncalibrated Heuristic)"
        lines.extend([
            f"| **{score_label}** | `{prediction.astrovet_raw_score:.4f}` | Predictive Unc: `+/- {prediction.predictive_uncertainty:.3f}` | Calibration was not applied |",
        ])

    lines.extend([
        "",
        f"### Actionable Triage Recommendation: **`{triage}`**",
        "",
        f"- **Out-Of-Distribution (OOD) Flag:** `{'YES (ANOMALY)' if prediction.ood_flag else 'NO'}`",
        f"- **Missing Diagnostic Modalities:** `{', '.join(prediction.missing_modalities) if prediction.missing_modalities else 'None (All Available)'}`",
        "",
        "## 3. Physics-Informed Diagnostics",
        "",
        "| Diagnostic | Value | Availability | Scientific Interpretation |",
        "|---|---|---|---|",
        f"| **Fitted Transit Depth** | `{p_diag.transit_depth_fitted_ppm or 'N/A'} ppm` | `{p_diag.modality_availability.get('transit_model_fit', False)}` | Approximate limb-darkened template fit |",
        f"| **Fitted Duration** | `{p_diag.transit_duration_fitted_hours or 'N/A'} hours` | `{p_diag.modality_availability.get('transit_model_fit', False)}` | Contact I-IV approximate duration |",
        f"| **Transit SNR** | `{p_diag.transit_snr or 'N/A'}` | `{p_diag.modality_availability.get('transit_model_fit', False)}` | Signal-to-noise of transit signature |",
        f"| **Reduced Chi-Squared** | `{p_diag.fit_chi2_reduced or 'N/A'}` | `{p_diag.modality_availability.get('transit_model_fit', False)}` | Template residual goodness-of-fit |",
        f"| **Odd vs Even Consistency** | `{p_diag.odd_even_consistent if p_diag.odd_even_consistent is not None else 'N/A'}` | `{p_diag.modality_availability.get('odd_even_comparison', False)}` | Depth difference: `{p_diag.odd_even_depth_diff_ppm or 'N/A'} ppm` |",
        f"| **Secondary Eclipse Detected** | `{p_diag.secondary_eclipse_detected if p_diag.secondary_eclipse_detected is not None else 'N/A'}` | `{p_diag.modality_availability.get('secondary_eclipse_search', False)}` | Phase 0.5 depth: `{p_diag.secondary_depth_ppm or 'N/A'} ppm` |",
        f"| **Event Consistency** | `{p_diag.transit_count_observed or 0} events observed` | `{p_diag.modality_availability.get('cross_event_consistency', False)}` | Depth scatter: `{p_diag.transit_depth_scatter_ppm or 'N/A'} ppm` |",
        f"| **Physical Plausibility** | `{'PASSED' if p_diag.plausibility_all_passed else 'FAILED'}` | Validated | Period > 0, Dur < Period, Feasible Depth |",
        "",
        "## 4. Periodicity Search & Verification (BLS & TLS)",
        "",
        "> [!NOTE]",
        "> **Periodicity Interpretation Notice:** Box Least Squares (BLS) and Transit Least Squares (TLS) detect repeating",
        "> flux decrements. A high SDE or SNR confirms signal periodicity, but does **NOT** validate a planet;",
        "> stellar pulsations, rotational modulation, and eclipsing binaries also produce prominent periodogram peaks.",
        "",
        "| Parameter / Metric | Candidate Ephemeris | Box Least Squares (BLS) | Transit Least Squares (TLS) |",
        "|---|---|---|---|",
        f"| **Orbital Period (Days)** | `{candidate.ephemeris.period_days:.5f}` | `{bls.get('best_period_days', 'N/A')}` | `{tls.get('best_period_days', 'N/A')}` |",
        f"| **Detection Statistic** | Reference Baseline | `SNR = {bls.get('peak_snr', 'N/A')}` | `SDE = {tls.get('sde', 'N/A')}` |",
        f"| **Epoch (BTJD)** | `{candidate.ephemeris.epoch_bjd_tdb:.4f}` | `{bls.get('best_t0_btjd', 'N/A')}` | `{tls.get('best_t0_btjd', 'N/A')}` |",
        f"| **Duration (Hours)** | `{candidate.ephemeris.duration_hours:.3f}` | `{bls.get('best_duration_hours', 'N/A')}` | `{tls.get('best_duration_hours', 'N/A')}` |",
        f"| **Fitted Depth (ppm)** | `{candidate.ephemeris.transit_depth_ppm or 'N/A'}` | `{bls.get('best_depth_ppm', 'N/A')}` | `{tls.get('best_depth_ppm', 'N/A')}` |",
        f"| **Odd/Even Mismatch** | Standard Screen | N/A | `{tls.get('odd_even_mismatch', 'N/A')}` |",
        f"| **Period Agreement** | Ground Truth | `{'MATCH (' + str(bls.get('period_diff_percent')) + '%)' if bls.get('is_period_match') else ('HARMONIC (' + str(bls.get('harmonic_relation')) + ')' if bls.get('is_harmonic') else ('DIFF ' + str(bls.get('period_diff_percent')) + '%' if bls.get('period_diff_percent') is not None else 'N/A'))}` | `{'MATCH (' + str(tls.get('period_diff_percent')) + '%)' if tls.get('is_period_match') else ('HARMONIC (' + str(tls.get('harmonic_relation')) + ')' if tls.get('is_harmonic') else ('DIFF ' + str(tls.get('period_diff_percent')) + '%' if tls.get('period_diff_percent') is not None else 'N/A'))}` |",
        "",
        f"- **Cross-Algorithm (BLS vs TLS) Period Agreement:** `{'YES (AGREEMENT)' if pgram.get('period_agreement') else 'NO / UNTESTED'}` "
        f"(Diff: `{pgram.get('period_agreement_diff_percent', 'N/A')}%`)",
        "",
        "## 5. Human-Review Evidence & Reason Codes",
        "",
    ])

    if prediction.reason_codes:
        for rcode in prediction.reason_codes:
            desc = get_reason_explanation(rcode)
            lines.append(f"- **`{rcode}`**: {desc}")
    else:
        lines.append("- *No negative diagnostic anomalies detected. Candidate exhibits consistent transit-like characteristics.*")

    lines.append("")
    explanation = getattr(prediction, "explanation", {}) or {}
    lines.extend(["## 6. Score Explanation", ""])
    basis = explanation.get("basis", "unavailable")
    lines.append(f"- **Explanation basis:** `{basis}`")
    if "top_contributions" in explanation:
        lines.append("- **Largest additive logit contributions:**")
        for item in explanation["top_contributions"]:
            lines.append(f"  - `{item['feature']}`: `{item['logit_contribution']:+.5f}`")
        lines.append(f"- **Classifier intercept:** `{explanation.get('intercept', 'N/A')}`")
        lines.append(f"- **Reconstructed classifier logit:** `{explanation.get('logit_reconstruction', 'N/A')}`")
    if "positive_signal" in explanation:
        signal = explanation["positive_signal"]
        lines.append(f"- **Positive heuristic input:** `{signal['feature']}` = `{signal['value']:.4f}`")
        for name, value in explanation.get("penalties", {}).items():
            lines.append(f"- **Heuristic penalty ({name.replace('_', ' ')}):** `{value:.4f}`")
        lines.append(f"- **Score after clipping:** `{explanation.get('raw_score_after_clipping', 'N/A')}`")
    if explanation.get("note"):
        lines.append(f"- {explanation['note']}")
    if prediction.score_type == "fitted_fusion_score":
        lines.append("- SSL embeddings are learned representations; their contribution is shown above, but it is an association learned by this research head, not direct physical evidence or causal proof.")
    else:
        lines.append("- SSL embeddings are representation features; without a trained downstream decision head, they do not independently classify a planet candidate.")
    lines.append("")
    if plot_relative_path:
        lines.extend([
            "## 7. Diagnostic Plots",
            "",
            f"![Vetting Diagnostic Plots]({plot_relative_path})",
            "",
        ])

    lines.extend([
        "---",
        "**Provenance & Attribution Notice:**",
        "ASTROVET combines available reference-baseline scores with physics diagnostics and BLS/TLS periodicity screening; a trained self-supervised representation is optional.",
        "Attributions describe model sensitivity patterns, not causal proof of physical reality.",
    ])

    report_text = "\n".join(lines)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text


def vet_and_generate_report(
    candidate_id: str = "390651552-1-S23",
    output_dir: str | Path = "artifacts/reports",
    fits_file: Optional[str | Path] = None,
    run_id: str = "run_vet_001",
    period: Optional[float] = None,
    epoch: Optional[float] = None,
    duration: Optional[float] = None,
    depth: Optional[float] = None,
    search_periods: bool = True,
    min_period_days: float = 0.5,
    max_period_days: Optional[float] = None,
    baseline_mode: str = "catalog",
    container_fallback: bool = True,
    encoder_mode: str = "ssl",
    ssl_checkpoint: Optional[str | Path] = None,
    allow_synthetic: bool = False,
    synthetic_fixture_ok: bool = False,
    model_artifact: Optional[str | Path] = None,
) -> dict:
    """
    End-to-end programmatic pipeline for candidate vetting and report generation.
    Integrates BLS/TLS periodograms and enforces real observed data discipline.
    """
    import pandas as pd
    from astrovet.ids import resolve_candidate_identity
    from astrovet.ingest.mast import fetch_tess_lightcurve, is_fits_synthetic_fixture
    from astrovet.lightcurves.reader import read_tess_fits_lightcurve
    from astrovet.lightcurves.quality import filter_quality_cadences
    from astrovet.lightcurves.normalization import segment_and_normalize_lightcurve
    from astrovet.views.folding import extract_transit_views
    from astrovet.views.periodogram import run_periodicity_search
    from astrovet.physics.diagnostics import run_physics_diagnostics
    from astrovet.baseline.adapter import ExoMinerBaselineAdapter
    from astrovet.ssl.encoder import LightCurvePCAPrototypeEncoder
    from astrovet.models.fusion import AstrovetFusionModel
    from astrovet.explain.plots import generate_candidate_diagnostic_plot, generate_lightcurve_explainer_plot
    from astrovet.schemas.candidate import CandidateRecord, Ephemeris
    from astrovet.baseline.adapter import find_default_catalog_path

    # Reject synthetic baseline selection before attempting data retrieval so
    # the opt-in guard is deterministic even on machines without local FITS.
    if baseline_mode == "synthetic" and not (allow_synthetic or synthetic_fixture_ok):
        raise ValueError(
            "Baseline mode 'synthetic' requires explicit opt-in (--allow-synthetic). "
            "Normal scientific workflows require a real catalog or container baseline."
        )

    project_root = Path(__file__).resolve().parents[3]
    if not Path(output_dir).is_absolute():
        output_dir = project_root / output_dir

    cand_ident = resolve_candidate_identity(candidate_id=candidate_id)
    req_candidate_id = cand_ident.candidate_id
    req_tic = cand_ident.tic_id
    req_plnt = cand_ident.planet_num
    req_sec = cand_ident.sector_run

    try:
        sec_num = int(req_sec.split("-")[0].replace("S", ""))
    except Exception:
        sec_num = 23

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Obtain real observed light curve
    is_synth_lightcurve = False
    resolved_fits: Optional[Path] = None

    if fits_file and Path(fits_file).is_file():
        resolved_fits = Path(fits_file)
    else:
        # Check standard local MAST storage for this TIC
        local_data_root = project_root / "data" / "raw" / "mast"
        local_candidates = list(local_data_root.glob(f"*{req_tic}*.fits"))
        if local_candidates:
            resolved_fits = local_candidates[0]
        else:
            # Check nested mastDownload directory
            nested = list(local_data_root.glob(f"**/*{req_tic}*_lc.fits"))
            if nested:
                resolved_fits = nested[0]

    if resolved_fits is None:
        if allow_synthetic or synthetic_fixture_ok:
            fetch_res = fetch_tess_lightcurve(
                tic_id=req_tic,
                output_dir=project_root / "data" / "fixtures",
                sector=sec_num,
                offline_mode=True,
                fixture_if_offline=True,
            )
            resolved_fits = Path(fetch_res["product_file"])
            is_synth_lightcurve = bool(fetch_res.get("is_synthetic_fixture", False))
        else:
            raise FileNotFoundError(
                f"No observed light curve FITS provided or found locally for TIC {req_tic}. "
                f"Please supply a valid FITS file via fits_file parameter or download using 'astrovet data fetch'. "
                f"Silent synthetic fallback is disabled."
            )
    else:
        is_synth_lightcurve = is_fits_synthetic_fixture(resolved_fits)
        if is_synth_lightcurve and not (allow_synthetic or synthetic_fixture_ok):
            raise ValueError(
                f"The provided FITS light curve '{resolved_fits}' is a synthetic test fixture, but allow_synthetic is False. "
                f"Normal scientific workflows require real observed data. To evaluate synthetic fixtures for testing, pass --allow-synthetic explicitly."
            )

    raw_lc = read_tess_fits_lightcurve(resolved_fits)
    clean_t, clean_f, clean_fe, q_rep = filter_quality_cadences(raw_lc.time, raw_lc.flux, raw_lc.flux_err, raw_lc.quality)

    # 2. Ephemeris Resolution: Supplied args -> Catalog Lookup -> BLS/TLS Recovery
    dash_csv = find_default_catalog_path(project_root)
    cat_df = pd.read_csv(dash_csv, low_memory=False) if dash_csv and dash_csv.is_file() else pd.DataFrame()
    matched_row = None
    if len(cat_df) > 0 and "TCE ID" in cat_df.columns:
        match_mask = cat_df["TCE ID"] == req_candidate_id
        if match_mask.any():
            matched_row = cat_df[match_mask].iloc[0]

    resolved_period = period
    resolved_epoch = epoch
    resolved_duration = duration
    resolved_depth = depth
    ruwe = 1.0
    label_val = "UNREVIEWED"

    if resolved_period is None:
        if matched_row is not None:
            resolved_period = float(matched_row.get("Orbital Period [day]", 0.0))
            resolved_epoch = float(matched_row.get("Transit Epoch [BTJD]", 0.0))
            resolved_duration = float(matched_row.get("Transit Duration [hour]", 0.0))
            resolved_depth = float(matched_row.get("Transit Depth [ppm]", 0.0))
            ruwe = float(matched_row.get("Gaia RUWE", 1.0))
            label_val = "PC"

    # 3. Periodicity Search & Ephemeris Recovery (BLS & TLS)
    periodogram_res: Dict[str, Any] = {}
    if search_periods:
        periodogram_res = run_periodicity_search(
            time=clean_t,
            flux=clean_f,
            flux_err=clean_fe,
            min_period_days=min_period_days,
            max_period_days=max_period_days,
            candidate_period=resolved_period,
            candidate_epoch=resolved_epoch,
            candidate_duration=resolved_duration,
        )

    # Where no ephemeris existed, recover from BLS/TLS search
    if resolved_period is None or resolved_period <= 0.0:
        rec_eph = periodogram_res.get("recovered_ephemeris")
        if rec_eph:
            resolved_period = float(rec_eph["period_days"])
            resolved_epoch = float(rec_eph["epoch_btjd"])
            resolved_duration = float(rec_eph["duration_hours"])
            resolved_depth = float(rec_eph["depth_ppm"])
            label_val = f"RECOVERED_{rec_eph['source_algorithm']}"
        else:
            raise ValueError(
                f"Candidate '{req_candidate_id}' not found in catalog, no ephemeris was provided, "
                f"and automated BLS/TLS search was unable to recover a periodic transit signal."
            )

    if resolved_epoch is None:
        rec = periodogram_res.get("recovered_ephemeris")
        if rec and rec.get("epoch_btjd") is not None:
            resolved_epoch = float(rec["epoch_btjd"])
        else:
            raise ValueError(f"Candidate transit epoch (T0) is missing and could not be recovered from periodogram search.")
    if resolved_duration is None:
        rec = periodogram_res.get("recovered_ephemeris")
        if rec and rec.get("duration_hours") is not None:
            resolved_duration = float(rec["duration_hours"])
        else:
            raise ValueError(f"Candidate transit duration is missing and could not be recovered from periodogram search.")
    if resolved_depth is None:
        rec = periodogram_res.get("recovered_ephemeris")
        if rec and rec.get("depth_ppm") is not None:
            resolved_depth = float(rec["depth_ppm"])
        else:
            resolved_depth = 0.0

    # 4. Phase Folding & Multi-View Extraction
    norm_lc = segment_and_normalize_lightcurve(
        clean_t, clean_f, clean_fe,
        ephemeris_period_days=resolved_period,
        ephemeris_epoch_btjd=resolved_epoch,
        ephemeris_duration_hours=resolved_duration,
    )
    primary_zoom, full_orbit, odd_zoom, even_zoom, stamps = extract_transit_views(
        norm_lc.time, norm_lc.flux_detrended, norm_lc.flux_err_norm,
        resolved_period, resolved_epoch, resolved_duration,
    )

    # 5. Physics Diagnostics (with Periodogram integration)
    phys_diag, fit_res = run_physics_diagnostics(
        primary_zoom=primary_zoom,
        full_orbit=full_orbit,
        stamps=stamps,
        time=norm_lc.time,
        flux=norm_lc.flux_detrended,
        flux_err=norm_lc.flux_err_norm,
        period_days=resolved_period,
        epoch_btjd=resolved_epoch,
        duration_hours=resolved_duration,
        reported_depth_ppm=resolved_depth,
        periodogram_results=periodogram_res,
    )

    # 6. Baseline Reference Branch (Catalog or Container)
    if baseline_mode == "synthetic":
        if not (allow_synthetic or synthetic_fixture_ok):
            raise ValueError(
                "Baseline mode 'synthetic' requires explicit opt-in (--allow-synthetic). "
                "Normal scientific workflows require real catalog or container baseline."
            )
        adapter = ExoMinerBaselineAdapter(catalog_path=None, is_synthetic=True)
        b_pred = adapter.predict_candidate(req_candidate_id, req_tic, req_plnt, req_sec)
    elif baseline_mode == "container":
        adapter = ExoMinerBaselineAdapter(catalog_path=dash_csv if dash_csv and dash_csv.is_file() else None, is_synthetic=False)
        b_pred = adapter.predict_via_container(
            candidate_id=req_candidate_id,
            fits_path=resolved_fits,
            period=resolved_period,
            epoch=resolved_epoch,
            duration_hours=resolved_duration,
            fallback_to_catalog=container_fallback,
        )
    else:  # catalog
        adapter = ExoMinerBaselineAdapter(catalog_path=dash_csv if dash_csv and dash_csv.is_file() else None, is_synthetic=False)
        b_pred = adapter.predict_candidate(req_candidate_id, req_tic, req_plnt, req_sec)

    # Load the opt-in fusion artifact before embedding so its training-time
    # encoder fingerprint can be checked before any prediction is produced.
    from astrovet.models.fusion import AstrovetFusionModel, validate_ssl_checkpoint_compatibility

    configured_fusion = os.environ.get("ASTROVET_FUSION_MODEL")
    if model_artifact is None and configured_fusion:
        model_artifact = configured_fusion
        if not Path(model_artifact).is_file():
            raise FileNotFoundError(f"ASTROVET_FUSION_MODEL points to a missing file: {model_artifact}")
    if model_artifact and Path(model_artifact).is_file():
        fusion = AstrovetFusionModel.load(model_artifact)
    else:
        default_art = project_root / "artifacts" / "models" / "fusion_classifier.joblib"
        if default_art.is_file():
            try:
                fusion = AstrovetFusionModel.load(default_art)
            except Exception:
                fusion = AstrovetFusionModel()
        else:
            fusion = AstrovetFusionModel()
    validate_ssl_checkpoint_compatibility(fusion, encoder_mode, ssl_checkpoint)

    # 7. Representation Branch (SSL Neural Encoder vs PCA Prototype)
    if encoder_mode == "ssl":
        # Never feed random, untrained weights into the fusion features. A missing
        # checkpoint means this modality is unavailable, not a random embedding.
        emb = None
        if ssl_checkpoint:
            checkpoint_path = Path(ssl_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"SSL checkpoint does not exist: {checkpoint_path}")
            from astrovet.ssl.neural_encoder import SelfSupervisedTransitEncoder
            encoder = SelfSupervisedTransitEncoder.load_checkpoint(checkpoint_path)
            if not encoder.is_fitted:
                raise ValueError(f"SSL checkpoint is not trained: {checkpoint_path}")
            if getattr(encoder, "data_source", "").upper() == "SYNTHETIC_DEMO":
                raise ValueError("Synthetic-demo SSL checkpoints cannot be used for observed-data vetting.")
            flux_2d = primary_zoom.flux if primary_zoom.flux.ndim == 2 else primary_zoom.flux.reshape(1, -1)
            emb = encoder.encode(flux_2d)[0]
    else:
        pca_enc = LightCurvePCAPrototypeEncoder(embedding_dim=16)
        emb = pca_enc.encode(primary_zoom.flux)
    phys_diag.modality_availability["ssl_representation"] = emb is not None

    # 8. Late Fusion & Triage
    pred = fusion.predict_single(
        candidate_id=req_candidate_id,
        tic_id=req_tic,
        baseline_pred=b_pred,
        physics_diag=phys_diag,
        ssl_embedding=emb,
        calibrator=None,
        run_id=run_id,
    )
    if is_synth_lightcurve or (b_pred and b_pred.is_synthetic):
        pred.is_synthetic = True
        # Keep synthetic demo/test outputs isolated from normal scientific reports
        out_dir_str = str(out_dir).replace("\\", "/")
        if out_dir_str.endswith("artifacts/reports"):
            out_dir = out_dir.parent / "synthetic_demo"
            out_dir.mkdir(parents=True, exist_ok=True)

    # Attach periodogram diagnostics to prediction
    pred.periodogram_diagnostics = periodogram_res

    # Add reason codes for periodogram results if diagnostic anomalies exist
    bls = periodogram_res.get("bls", {})
    tls = periodogram_res.get("tls", {})
    if tls.get("is_harmonic") or bls.get("is_harmonic"):
        if "PERIODICITY_HARMONIC_DETECTED" not in pred.reason_codes:
            pred.reason_codes.append("PERIODICITY_HARMONIC_DETECTED")
    if (tls.get("compared_to_ephemeris") and not tls.get("is_period_match") and tls.get("sde", 0) > 8.0) or \
       (bls.get("compared_to_ephemeris") and not bls.get("is_period_match") and bls.get("peak_snr", 0) > 6.0):
        if "PERIODICITY_MISMATCH" not in pred.reason_codes:
            pred.reason_codes.append("PERIODICITY_MISMATCH")
    if tls.get("odd_even_mismatch") is not None and tls.get("odd_even_mismatch", 0) > 3.0:
        if "TLS_ODD_EVEN_MISMATCH" not in pred.reason_codes:
            pred.reason_codes.append("TLS_ODD_EVEN_MISMATCH")
    if search_periods and not bls.get("is_available") and not tls.get("is_available"):
        if "PERIODICITY_NOT_DETECTED" not in pred.reason_codes:
            pred.reason_codes.append("PERIODICITY_NOT_DETECTED")

    # 9. Diagnostic Plot Generation (6-Panel including Periodograms)
    plot_file = out_dir / f"{req_candidate_id}_diagnostics.png"
    generate_candidate_diagnostic_plot(
        candidate_id=req_candidate_id,
        time=norm_lc.time,
        flux=norm_lc.flux_detrended,
        primary_zoom=primary_zoom,
        fit_res=fit_res,
        odd_zoom=odd_zoom,
        even_zoom=even_zoom,
        full_orbit=full_orbit,
        output_path=plot_file,
        periodogram_results=periodogram_res,
        candidate_period=resolved_period,
    )
    lightcurve_plot_file = out_dir / f"{req_candidate_id}_lightcurve.png"
    generate_lightcurve_explainer_plot(
        time=norm_lc.time,
        flux=norm_lc.flux_detrended,
        folded_view=primary_zoom,
        fit_res=fit_res,
        output_path=lightcurve_plot_file,
    )

    # 10. Markdown Vetting Report
    cand_rec = CandidateRecord(
        candidate_id=req_candidate_id,
        tic_id=req_tic,
        planet_num=req_plnt,
        sector_run=req_sec,
        ephemeris=Ephemeris(
            period_days=resolved_period,
            epoch_bjd_tdb=resolved_epoch,
            duration_hours=resolved_duration,
            transit_depth_ppm=resolved_depth,
        ),
        gaia_ruwe=ruwe,
        label_raw=label_val,
    )
    report_file = out_dir / f"{req_candidate_id}_report.md"
    generate_candidate_markdown_report(
        candidate=cand_rec,
        prediction=pred,
        plot_relative_path=plot_file.name,
        output_path=report_file,
    )

    return {
        "candidate": cand_rec,
        "prediction": pred,
        "report_file": report_file,
        "plot_file": plot_file,
        "lightcurve_plot_file": lightcurve_plot_file,
        "periodogram_results": periodogram_res,
    }

# Backward compatibility alias
run_explain_and_report = vet_and_generate_report
