# ASTROVET Model Card

**Model Name:** ASTROVET (Astronomical Vetting & Explainable Triage)  
**Version:** 0.1.0  
**Date:** 2026-10-21  
**Architecture:** Late Fusion of Baseline Scores, Physics Diagnostics, and Unsupervised Light-Curve Representations  

---

## 1. Model Details

- **Developer:** ASTROVET Research & Engineering Team
- **Baseline Adapter:** The code records an ExoMiner++ release tag/commit and supports catalog lookup and container execution. No upstream checkout or prediction catalog is bundled in this workspace; no live baseline result or upstream preservation claim is made without those files.
- **Periodicity Verification (BLS & TLS):** Automated transit periodicity screening using Box Least Squares (Astropy BLS; Kovacs et al. 2002) and Transit Least Squares (TLS; Hippke & Heller 2019, `transitleastsquares>=2.0.0`). Detects periodic dips, compares recovered parameters ($P, T_0, \text{duration}$) against candidate ephemerides, flags harmonic/subharmonic relations (0.5x, 2x, 0.333x, 3x), and recovers ephemerides for uncharacterized candidates without inserting demo values.
- **Physics Diagnostics:** Approximate limb-darkened transit template fitting (for residual baseline estimation), cadence-level and per-event odd-even depth testing with propagated observational uncertainty, secondary eclipse search at phase 0.5, and transit-to-transit consistency.
- **Representation Branch:** Self-supervised neural encoder with a local pretraining command. A run without a trained checkpoint omits this modality; synthetic-demo checkpoints are rejected for observed-data vetting. Available real-data checkpoints here were smoke-trained on one local TIC/sector and are not scientifically validated representations.
- **Fusion Layer:** Logistic-regression fitting and Platt calibration primitives exist, but a full labeled training, calibration, held-out evaluation, and deployment workflow is not implemented. Default reported scores are heuristics, not planet probabilities.

---

## 2. Intended Use

- **Primary Purpose:** Prioritizing and triaging TESS SPOC transit-like threshold-crossing events (TCEs) for human review and telescope follow-up allocation.
- **Out of Scope & Non-Goals:**
  - Claiming exoplanet confirmation or statistical validation.
  - Automatically re-labeling ambiguous or unreviewed signals as planets.
  - Operational scheduling without human subject matter expert oversight.

---

## 3. Scientific Safeguards & Data Discipline

1. **Host-Grouped Leakage Prevention:** All candidate events for the same stellar host (TIC ID) are grouped into the same partition. Zero cross-sector or multi-planet leakage into test sets.
2. **Explicit Modality Availability:** Missing data products (e.g. difference images, target pixel files) or missing baseline scores (`NOT_FOUND`) are explicitly tracked in `modality_availability` and NEVER replaced with fabricated zeros or neutral defaults like 0.50.
3. **Advisory Attribution:** Explanations and reason codes reflect model sensitivity patterns, not physical causal proof.
4. **Disjoint Calibration:** Probability calibration is fitted exclusively on an independent calibration partition. Unfitted execution paths emit uncalibrated heuristic triage scores, not formal probabilities.
5. **Baseline Checkpoint Overlap Audit:** Training overlap for public ExoMiner++ checkpoints has not been audited in this project.

---

## 4. Limitations & Scientific Caveats

- **Approximate Transit Template:** The transit fit utilizes an empirical parameterized limb-darkened profile for fast screening and residual calculation, not a full numerical Mandel & Agol (2002) orbit integration.
- **Periodicity Detection vs Validation:** Box Least Squares (BLS) and Transit Least Squares (TLS) detect periodic flux decrements. High SDE or SNR confirms signal recurrence and shape match, but never proves planetary status; astrophysical false positives (eclipsing binaries, blend scenarios) and stellar variability also produce prominent periodogram peaks.
- **Self-Supervised Representation Scale:** The local encoder has a CPU training loop and checkpoint workflow. The available local smoke checkpoints do not represent full-archive training or scientific validation. Full TESS archive pretraining requires substantially more data and compute.
- **Containerized Baseline Daemon Dependency:** Live raw-pixel ExoMiner baseline container inference requires an active local Podman or Docker daemon. Where container engines are absent, ASTROVET records `CONTAINER_RUNTIME_UNAVAILABLE` with the exact constructed command rather than falsely claiming live container execution.
- **Instrument Systematics:** Blended eclipsing binaries and stellar activity (spots, flares) can occasionally mimic planetary transit signatures.
- **Cadence Scope:** First version is configured for TESS SPOC 2-minute cadence; Full Frame Images (FFI) require dedicated detrending adapters.
