# ADR 001: Initial Scope, Baseline Pinning, and Isolation

**Date:** 2026-10-21  
**Status:** Historical / Superseded by v1 implementation

> [!NOTE]
> **Historical Record:** This ADR documents the initial project scoping decisions formulated at inception.
> In ASTROVET v1, several components marked as out-of-scope or future work have been actively implemented:
> 1. **Deep Self-Supervised Learning:** Implemented via `SelfSupervisedTransitEncoder` (joint Masked Cadence Autoencoding and view consistency) and real TESS FITS view extraction (`astrovet ssl pretrain --input-fits`).
> 2. **Containerized Baseline:** Implemented via `ExoMinerContainerRunner` (`--baseline-mode container`), with honest `CONTAINER_RUNTIME_UNAVAILABLE` reporting when host container engines (Podman/Docker) are absent.
> 3. **Interactive Local UI:** Implemented via `astrovet ui` local review server.  
>
> **Audit correction:** Earlier versions of this document asserted that the upstream ExoMiner checkout and a prediction catalog were present and verified. They are not present in the current workspace, so those statements and the associated catalog counts are unsupported and withdrawn. The fusion primitives also do not constitute a complete trained/calibrated model workflow.

## Context

ASTROVET is designed as a hybrid vetting and candidate prioritization framework extending NASA ExoMiner++. The upstream ExoMiner codebase resides in `ExoMiner-main`. A clean scientific and software engineering separation is required to preserve the baseline without modification and ensure 100% reproducible benchmarking.

## Decisions

1. **Workspace Boundary:**
   - All ASTROVET implementation resides exclusively in `astrovet/`.
   - The current checkout contains ASTROVET only. There is no separate upstream checkout here, so its state cannot be assessed.
   - This workspace has no Git metadata; release hashes recorded in source are metadata, not proof of a checked out or verified release.
2. **Pinned Baseline:**
   - Official NASA software release tag: `v1.0.0` (Git commit `d877a23`).
   - Model Name: `ExoMiner++ (TESS 2-min S1-S67)`.
   - Adapter Scope: The adapter supports optional catalog lookup and a container runner. The upstream catalog and checkout are not bundled in the current workspace.
   - If candidate is not found in the baseline catalog, `score=None` (never fabricated as 0.50).
3. **Audit First Gate:**
   - The requested file `tces_with_labels_v3.csv` was verified absent in the workspace. An audit workflow is provided to audit any incoming catalog.
   - No labeled TCE dataset or baseline prediction catalog is bundled; dataset statistics are therefore not reported.
4. **Target-Star Grouped Splitting:**
   - Grouping unit is `tic_id`. Under no circumstances may sectors or candidates of the same TIC be partitioned across train and test.
5. **Physics Diagnostics & SSL Scope:**
   - Transit fits and odd-even tests are framed as approximate diagnostic screening tools, not formal physical parameter inversions.
   - The neural encoder can be self-supervised-pretrained on real FITS data. Its embedding is omitted unless a trained checkpoint is explicitly provided. Full archive pretraining and downstream scientific validation remain unperformed.
6. **Probability & Calibration:**
   - Scores produced without a fitted calibrator are explicitly reported as uncalibrated heuristic triage scores. Formal probabilities are emitted only when a calibrator fitted on disjoint calibration data is supplied.
