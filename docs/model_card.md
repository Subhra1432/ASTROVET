# ASTROVET Model Card

**Model Name:** ASTROVET TCE Catalog Baseline (E vs B/J)
**Version:** 0.1.0
**Date:** 2026-10-28
**Architecture:** Standardized tabular features, class-weighted logistic regression, and Platt calibration

---

## 1. Model Details

- **Developer:** ASTROVET Research & Engineering Team
- **Training Dataset:** `D:\Final Year Projet\tces_with_labels_v3.csv` (Astronet-Triage-v2 catalog of 24,926 Threshold-Crossing Events).
- **Source SHA-256:** `8c4412b073ac3fb02b8b8ee61be32d25d002cd569d4975f848cf8deddcadc1bb`
- **Trained Artifact:** `ASTROVET_data/artifacts/models/tce_e_vs_bj_classifier.joblib` (local run artifact; separate from the untrained hybrid fusion model).
- **Target Semantics:** Binary catalog triage, not planet identification:
  - Positive Class (1): `E`, periodic eclipsing signal (N=2,613). This category includes transiting planets and non-contact eclipsing binaries.
  - Negative Class (0): `B`, contact eclipsing binary (N=741), or `J`, junk/noise/systematics (N=15,803).
  - Excluded: `S`, single transit or incorrect period (N=275); `N`, inconclusive (N=154); and missing consensus label (N=5,340).
- **Partitioning:** 100% Group-Disjoint by Host Star (`TIC ID`) across Train (60%), Validation (15%), Calibration (10%), and Test (15%). Strictly verified zero host star overlap across all partition intersections.
- **Preprocessing:** `TabularFeaturePreprocessor` fitted strictly on the `train` partition (train medians for continuous features and `StandardScaler`). No test or calibration data was exposed during preprocessing.
- **Model Tuning:** Regularization parameter $C$ selected strictly on the `validation` partition based on PR-AUC ($C=10.0$).
- **Probability Calibration:** `PlattCalibrator` (logistic scaling) fitted strictly on the held-out `calibration` partition.
- **Test Evaluation:** Evaluated once on the untouched `test` partition (2,878 candidates across 2,873 host stars).

---

## 2. Test Set Evaluation Metrics

Evaluated on untouched test partition ($N=2,878$, Prevalence = 13.6%):
- **ROC-AUC:** `0.8901`
- **PR-AUC:** `0.5368`
- **Raw Brier Score:** `0.1417` $\rightarrow$ **Calibrated Brier Score:** `0.0825`
- **Raw Expected Calibration Error (ECE):** `0.2027` $\rightarrow$ **Calibrated ECE:** `0.0222` (order-of-magnitude calibration improvement)
- **Precision (at threshold 0.50):** `0.6069`
- **Recall (at threshold 0.50):** `0.4077`
- **F1 Score:** `0.4877`
- **Precision@10:** `0.5000` | **Precision@25:** `0.6000` | **Precision@50:** `0.6800`

---

## 3. Subgroup Performance & Physical Boundaries

- **Orbital Period:**
  - Short Period ($P < 3$ days): ROC-AUC = `0.7994`, PR-AUC = `0.5533`
  - Medium Period ($3 \le P < 15$ days): ROC-AUC = `0.8807`, PR-AUC = `0.5904`
  - Long Period ($P \ge 15$ days): ROC-AUC = `0.8790`, PR-AUC = `0.1296` (low prevalence, only 15 positives out of 969 candidates)
- **Transit Depth:**
  - Shallow ($< 500$ ppm): ROC-AUC = `0.8580`, PR-AUC = `0.1782`
  - Medium ($500 - 2,500$ ppm): ROC-AUC = `0.8454`, PR-AUC = `0.2151`
  - Deep ($\ge 2,500$ ppm): ROC-AUC = `0.8465`, PR-AUC = `0.6284`
- **Host Star Brightness:**
  - Bright Stars ($T_\text{mag} < 9$): ROC-AUC = `0.9174`, PR-AUC = `0.6227`
  - Faint Stars ($T_\text{mag} \ge 9$): ROC-AUC = `0.8613`, PR-AUC = `0.5034`

---

## 4. Scientific Safeguards & Data Discipline

1. **Host-Grouped Leakage Prevention:** All candidates for the same stellar host (`TIC ID`) are strictly grouped into the same partition. Zero cross-sector or multi-candidate leakage into test or calibration sets.
2. **Missing Stellar Values:** Missing stellar mass/radius values are represented by explicit missingness flags; imputation medians and scaling are fit only on training hosts.
3. **Target / Leakage Column Exclusion:** Identifier columns (`TIC ID`), human labeler annotations (`L1`–`L8`), notes, external split tags (`Split`), and temporal calendar boundaries (`Epoch`, `MinT`, `MaxT`, `Year`) are strictly excluded from predictive features.
4. **Disjoint Calibration:** E-versus-B/J probability calibration is fitted exclusively on an independent calibration partition. These calibrated values remain probabilities for the catalog label task only.
5. **Clear Scientific Scope:** The model estimates E versus B/J from catalog summary parameters only. E is not a planet disposition; this artifact is blocked from planet-vetting inference and is not the ASTROVET hybrid fusion model.

The dataset and taxonomy are documented by [Zenodo record 7411579](https://zenodo.org/records/7411579) and [Tey et al. (2023)](https://doi.org/10.3847/1538-3881/acad85). Tey et al. state that Astronet-Triage-v2 does not distinguish eclipsing binaries from transiting planets.

---

## 6. SSL + Physics Fusion Research Head (separate model)

- **Artifact:** `ASTROVET_data/artifacts/models/planet_disposition_prototype/ssl_physics_fusion/astrovet_ssl_physics_fusion_research.joblib`
- **Target:** CP/KP TOI dispositions versus FP/FA; PC/APC/unset are excluded. This is a selected follow-up cohort, not a representative sample of all TESS targets.
- **Inputs:** 16-dimensional embedding from the real-data self-supervised encoder plus 11 physics/missingness features from the same feature-vector builder used by FITS upload inference. ExoMiner catalog score is deliberately omitted.
- **Cohort:** 1,771 real matched light curves from the 1,782-event hard-label cohort; 11 events without a matching downloaded FITS are audited and excluded.
- **Partitions:** Host-grouped and held out by TIC; after FITS availability filtering: 1,061 train, 270 validation, 177 calibration, 263 test events. Feature scaling is fit on train only; C is chosen on validation; Platt calibration is fit on calibration; test remains untouched.
- **Test results:** ROC-AUC `0.7768`, PR-AUC `0.8251` at test prevalence `0.5817`; raw Brier `0.1909`, ECE `0.0882`; calibrated Brier `0.1903`, ECE `0.0510`. At calibrated-score threshold 0.5: precision `0.7074`, recall `0.8693`, F1 `0.7801`.
- **Interpretation:** The calibrated output is a selected-TOI-cohort planet-like score, not a population-wide planet probability, statistical validation, or autonomous disposition. Always retain human review and test broader independent sectors/pipelines before making generalization claims.
- **Audit:** Feature failures, source/checkpoint hashes, split IDs and full metrics are stored beside the artifact. Rebuild with `scripts/train_ssl_physics_fusion.py`; launch the explicit dashboard path with `scripts/start_dashboard_with_trained_model.ps1`.
