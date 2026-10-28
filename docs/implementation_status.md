# ASTROVET Implementation Status

Last audited: 2026-10-28

## Operational in this checkout

- Local UI for a candidate ID, local FITS file, or explicit MAST retrieval.
- Real TESS SPOC FITS ingestion and provenance manifests.
- Light-curve quality filtering and normalization.
- BLS and TLS periodicity search, plus transit and eclipsing-binary screening diagnostics.
- Evidence report and diagnostic plot generation.
- CPU self-supervised encoder training from real FITS data. Vetting omits this modality unless a trained checkpoint is supplied; synthetic-demo checkpoints are rejected for observed-data vetting.
- Host-grouped split, metric, fusion, and calibration code primitives.

## Not scientifically complete or validated

- `tces_with_labels_v3.csv` and a NASA ExoMiner++ prediction catalog are not in this checkout. No full-catalog data audit or labeled training run has been performed here.
- The fusion classifier and calibration helpers are not wired into a complete train, save, load, held-out evaluate, and inference workflow. Current candidate scores are uncalibrated heuristic triage scores.
- Local SSL checkpoints were trained for smoke verification on one TIC and sector. They are not evidence of archive-scale representation quality.
- The upstream NASA ExoMiner checkout and container image are unavailable here. The declared release metadata is not proof that an upstream model was executed.
- The available single-target run cannot establish population-level precision, recall, calibration, or generalization to unseen stars/sectors.

These limits require real labeled TCEs, a verified baseline release/catalog, and sufficient compute. They cannot be resolved by generating synthetic examples or by software-only changes without making unsupported scientific claims.

## Acceptance gate for a research result

1. Audit and version the actual labeled catalog; document label mapping and unresolved-label handling.
2. Produce light-curve features for every labeled event and record file/checksum provenance.
3. Make host-star-disjoint train, validation, calibration, and test partitions.
4. Fit the fusion model on training data, select settings on validation data, calibrate only on calibration data, and report metrics once on the untouched test set.
5. Compare against the verified ExoMiner++ baseline and simple physics-only controls; report cohort counts, missing modalities, confidence intervals, and subgroup limitations.
6. Save the trained artifact, split manifest, source hashes, environment, and evaluation report together.
