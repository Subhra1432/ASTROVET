"""
End-to-end reproducible training workflow for the tabular TCE E-versus-B/J baseline.

Scientific safeguards:
- Strict host-grouped (TIC-disjoint) dataset partitioning preventing cross-candidate and cross-system leakage.
- Explicit label mapping with unresolved, ambiguous, and non-consensus labels excluded.
- Preprocessing (imputation and feature scaling) fitted strictly on the train partition.
- Model hyperparameters selected strictly on the validation partition.
- Probability calibration fitted strictly on the calibration partition.
- Untouched test partition evaluated once, reporting comprehensive metrics, calibration errors, and subgroup limitations.
- Target, identifier, labeller vote, and leakage columns are strictly excluded.
- This is not the ASTROVET hybrid fusion model and its positive E class is not a planet label.
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from astrovet.provenance import compute_sha256, get_runtime_environment, create_manifest, save_manifest
from astrovet.models.calibration import PlattCalibrator, compute_calibration_metrics
from astrovet.evaluation.metrics import compute_comprehensive_metrics


# Excluded columns that MUST NOT be used as predictive features
EXCLUDED_LEAKAGE_COLUMNS = [
    "TIC ID",
    "Consensus Label",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "Notes",
    "Split",
    "Epoch",
    "MinT",
    "MaxT",
    "Year",
    "target",
]

# Physical base feature columns from TCE catalog
CORE_FEATURE_COLUMNS = [
    "Period",
    "Duration",
    "Depth",
    "SMass",
    "SRad",
    "SRadEst",
    "Tmag",
]


class TabularFeaturePreprocessor:
    """
    Leakage-safe feature extraction and normalization pipeline.
    All state (medians, scaling parameters) is fitted strictly on the training partition.
    """

    def __init__(self) -> None:
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.train_medians: Dict[str, float] = {}
        self.scaler = StandardScaler()

    def _extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute base and physically motivated derived features."""
        feats = pd.DataFrame(index=df.index)

        # 1. Base numeric conversions
        period = pd.to_numeric(df["Period"], errors="coerce")
        duration = pd.to_numeric(df["Duration"], errors="coerce")
        depth = pd.to_numeric(df["Depth"], errors="coerce")
        smass = pd.to_numeric(df["SMass"], errors="coerce")
        srad = pd.to_numeric(df["SRad"], errors="coerce")
        sradest = pd.to_numeric(df["SRadEst"], errors="coerce")
        tmag = pd.to_numeric(df["Tmag"], errors="coerce")

        # 2. Missingness indicators
        feats["smass_missing"] = smass.isna().astype(np.float32)
        feats["srad_missing"] = srad.isna().astype(np.float32)
        feats["sradest_missing"] = sradest.isna().astype(np.float32)

        # 3. Log-scale transit parameters
        feats["log10_period"] = np.log10(np.clip(period.fillna(1.0), 1e-4, 1e4)).astype(np.float32)
        feats["log10_duration"] = np.log10(np.clip(duration.fillna(0.1), 1e-4, 1e2)).astype(np.float32)
        feats["log10_depth"] = np.log10(np.clip(depth.fillna(100.0), 1.0, 1e9)).astype(np.float32)

        # 4. Duty cycle: transit duration / orbital period
        period_safe = np.clip(period.fillna(1.0), 1e-4, 1e4)
        duration_safe = np.clip(duration.fillna(0.1), 1e-4, 1e2)
        feats["duty_cycle"] = (duration_safe / period_safe).astype(np.float32)

        # 5. Effective stellar radius (SRad preferred, fallback to SRadEst)
        eff_srad = srad.combine_first(sradest)
        feats["eff_srad_raw"] = eff_srad.astype(np.float32)

        # 6. Estimated planetary radius in Earth radii: Rp ~ 109.1 * R_star * sqrt(Depth / 1e6)
        srad_safe = np.clip(eff_srad.fillna(1.0), 0.05, 500.0)
        depth_fraction = np.clip(depth.fillna(100.0), 0.0, 1e9) / 1e6
        rp_est_rearth = 109.076 * srad_safe * np.sqrt(np.maximum(0.0, depth_fraction))
        feats["log10_rp_est"] = np.log10(np.clip(rp_est_rearth, 0.01, 1e5)).astype(np.float32)

        # 7. Raw physical features for imputation
        feats["smass_raw"] = smass.astype(np.float32)
        feats["srad_raw"] = srad.astype(np.float32)
        feats["tmag_raw"] = tmag.astype(np.float32)

        return feats

    def fit(self, df_train: pd.DataFrame) -> TabularFeaturePreprocessor:
        """Compute training statistics (medians and scaler) strictly on train data."""
        raw_train = self._extract_raw_features(df_train)
        self.feature_names = list(raw_train.columns)

        # Compute medians on training data only
        self.train_medians = {}
        for col in self.feature_names:
            med_val = float(raw_train[col].median(skipna=True))
            if np.isnan(med_val):
                med_val = 0.0
            self.train_medians[col] = med_val

        # Impute training features with training medians
        imputed_train = raw_train.copy()
        for col, med in self.train_medians.items():
            imputed_train[col] = imputed_train[col].fillna(med)

        # Fit StandardScaler strictly on training set
        self.scaler.fit(imputed_train.values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply train-fitted imputation and scaling."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training partition before transform.")
        raw = self._extract_raw_features(df)
        imputed = raw.copy()
        for col, med in self.train_medians.items():
            if col in imputed.columns:
                imputed[col] = imputed[col].fillna(med)
            else:
                imputed[col] = med
        ordered = imputed[self.feature_names].values
        return self.scaler.transform(ordered).astype(np.float32)


def load_and_audit_dataset(file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load candidate catalog, audit integrity, verify schema, and extract the valid labeled cohort.
    Excludes unresolved, ambiguous, or absent consensus labels.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {p}")

    file_size = p.stat().st_size
    file_sha256 = compute_sha256(p)

    # Read CSV handling comment headers and unit rows
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    if first_line.startswith("#"):
        df_raw = pd.read_csv(p, comment="#", low_memory=False)
    else:
        df_raw = pd.read_csv(p, low_memory=False)

    # Detect and drop unit specification rows
    if len(df_raw) > 0 and (pd.isna(df_raw.iloc[0, 0]) or str(df_raw.iloc[0].get("Epoch", "")).strip() in ["BTJD", "days", "ppm"]):
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    raw_row_count = len(df_raw)
    raw_col_count = len(df_raw.columns)

    if "TIC ID" not in df_raw.columns or "Consensus Label" not in df_raw.columns:
        raise ValueError(
            f"Dataset at {p} lacks required columns 'TIC ID' or 'Consensus Label'. Found: {list(df_raw.columns)}"
        )

    # Ensure TIC ID is integer host star key
    df_raw["TIC ID"] = pd.to_numeric(df_raw["TIC ID"], errors="coerce").astype(np.int64)

    # Label audit
    label_distribution = df_raw["Consensus Label"].value_counts(dropna=False).to_dict()
    label_distribution = {str(k): int(v) for k, v in label_distribution.items()}

    # E means a periodic eclipsing signal, including planet transits and
    # non-contact eclipsing binaries. It is not a planet label. This binary
    # target is E versus B/J, not planet probability. S, N, and missing labels
    # are excluded rather than forced into either class.
    valid_labels = ["E", "B", "J"]
    valid_mask = df_raw["Consensus Label"].isin(valid_labels)
    df_valid = df_raw[valid_mask].copy().reset_index(drop=True)

    df_valid["target"] = (df_valid["Consensus Label"] == "E").astype(int)

    n_valid = len(df_valid)
    n_unique_hosts = int(df_valid["TIC ID"].nunique())
    n_pos = int((df_valid["target"] == 1).sum())
    n_neg = int((df_valid["target"] == 0).sum())
    n_excluded = raw_row_count - n_valid

    audit_summary = {
        "file_path": str(p.resolve()),
        "file_size_bytes": file_size,
        "sha256": file_sha256,
        "raw_total_rows": raw_row_count,
        "raw_total_columns": raw_col_count,
        "raw_columns": list(df_raw.columns),
        "raw_label_distribution": label_distribution,
        "label_mapping": {
            "positive_eclipsing_signal": ["E"],
            "negative_contact_binary_or_junk": ["B", "J"],
            "excluded_single_transit_inconclusive_or_unlabeled": ["S", "N", "NaN"],
            "interpretation_limit": "E includes both planet transits and non-contact eclipsing binaries; this target is not planet identification.",
        },
        "valid_labeled_cohort": {
            "total_candidates": n_valid,
            "unique_host_stars": n_unique_hosts,
            "positive_count": n_pos,
            "negative_count": n_neg,
            "positive_prevalence": round(n_pos / max(1, n_valid), 4),
            "excluded_unresolved_count": n_excluded,
        },
        "leakage_columns_excluded": EXCLUDED_LEAKAGE_COLUMNS,
    }

    # Validation safeguards
    if n_unique_hosts < 50:
        raise ValueError(
            f"Insufficient independent labeled host stars for host-disjoint splits: found {n_unique_hosts} (minimum 50 required)."
        )
    if n_pos < 10 or n_neg < 10:
        raise ValueError(
            f"Severe class collapse: positive={n_pos}, negative={n_neg}. Both classes must have at least 10 instances."
        )

    return df_valid, audit_summary


def partition_host_disjoint(
    df: pd.DataFrame,
    random_seed: int = 42,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    cal_frac: float = 0.10,
    test_frac: float = 0.15,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Split dataframe into train, val, cal, and test strictly grouped by host TIC ID.
    Guarantees 0 host star leakage across all 6 pairwise partition intersections.
    """
    assert abs(train_frac + val_frac + cal_frac + test_frac - 1.0) < 1e-4, "Split fractions must sum to 1.0"

    # Unique host stars
    unique_hosts = sorted(df["TIC ID"].unique().tolist())
    rng = np.random.default_rng(random_seed)
    shuffled_hosts = unique_hosts.copy()
    rng.shuffle(shuffled_hosts)

    n_total = len(shuffled_hosts)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_cal = int(n_total * cal_frac)

    train_tics = set(shuffled_hosts[:n_train])
    val_tics = set(shuffled_hosts[n_train:n_train + n_val])
    cal_tics = set(shuffled_hosts[n_train + n_val:n_train + n_val + n_cal])
    test_tics = set(shuffled_hosts[n_train + n_val + n_cal:])

    # Strictly assert zero overlap across all pairs
    assert len(train_tics.intersection(val_tics)) == 0, "Train and Val share TICs"
    assert len(train_tics.intersection(cal_tics)) == 0, "Train and Cal share TICs"
    assert len(train_tics.intersection(test_tics)) == 0, "Train and Test share TICs"
    assert len(val_tics.intersection(cal_tics)) == 0, "Val and Cal share TICs"
    assert len(val_tics.intersection(test_tics)) == 0, "Val and Test share TICs"
    assert len(cal_tics.intersection(test_tics)) == 0, "Cal and Test share TICs"

    splits = {
        "train": df[df["TIC ID"].isin(train_tics)].copy().reset_index(drop=True),
        "val": df[df["TIC ID"].isin(val_tics)].copy().reset_index(drop=True),
        "cal": df[df["TIC ID"].isin(cal_tics)].copy().reset_index(drop=True),
        "test": df[df["TIC ID"].isin(test_tics)].copy().reset_index(drop=True),
    }

    manifest_meta = {
        "random_seed": random_seed,
        "total_hosts": n_total,
        "host_ids_by_split": {
            "train": sorted(int(x) for x in train_tics),
            "val": sorted(int(x) for x in val_tics),
            "cal": sorted(int(x) for x in cal_tics),
            "test": sorted(int(x) for x in test_tics),
        },
        "splits": {
            k: {
                "hosts": len(df_split["TIC ID"].unique()),
                "candidates": len(df_split),
                "positive": int((df_split["target"] == 1).sum()),
                "negative": int((df_split["target"] == 0).sum()),
                "prevalence": round(float(np.mean(df_split["target"])), 4),
            }
            for k, df_split in splits.items()
        },
        "leakage_zero_overlap_verified": True,
    }

    return splits, manifest_meta


def evaluate_subgroups(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_prob_cal: np.ndarray,
) -> Dict[str, Any]:
    """
    Subgroup and slice evaluation to reveal uncertainty and physical performance boundaries.
    """
    subgroups = {}

    # 1. Period slices
    period = pd.to_numeric(df_test["Period"], errors="coerce").fillna(10.0)
    p_short = period < 3.0
    p_med = (period >= 3.0) & (period < 15.0)
    p_long = period >= 15.0

    for name, mask in [("period_short_under_3d", p_short), ("period_med_3_to_15d", p_med), ("period_long_ge_15d", p_long)]:
        if mask.sum() >= 10 and len(np.unique(y_test[mask])) > 1:
            subgroups[name] = {
                "count": int(mask.sum()),
                "positive": int((y_test[mask] == 1).sum()),
                "roc_auc": round(float(roc_auc_score(y_test[mask], y_prob_cal[mask])), 4),
                "pr_auc": round(float(average_precision_score(y_test[mask], y_prob_cal[mask])), 4),
                "brier": round(float(np.mean((y_prob_cal[mask] - y_test[mask])**2)), 4),
            }
        else:
            subgroups[name] = {"count": int(mask.sum()), "note": "Insufficient samples or single class"}

    # 2. Depth slices
    depth = pd.to_numeric(df_test["Depth"], errors="coerce").fillna(1000.0)
    d_shallow = depth < 500.0
    d_med = (depth >= 500.0) & (depth < 2500.0)
    d_deep = depth >= 2500.0

    for name, mask in [("depth_shallow_under_500ppm", d_shallow), ("depth_med_500_to_2500ppm", d_med), ("depth_deep_ge_2500ppm", d_deep)]:
        if mask.sum() >= 10 and len(np.unique(y_test[mask])) > 1:
            subgroups[name] = {
                "count": int(mask.sum()),
                "positive": int((y_test[mask] == 1).sum()),
                "roc_auc": round(float(roc_auc_score(y_test[mask], y_prob_cal[mask])), 4),
                "pr_auc": round(float(average_precision_score(y_test[mask], y_prob_cal[mask])), 4),
                "brier": round(float(np.mean((y_prob_cal[mask] - y_test[mask])**2)), 4),
            }
        else:
            subgroups[name] = {"count": int(mask.sum()), "note": "Insufficient samples or single class"}

    # 3. Brightness slices
    tmag = pd.to_numeric(df_test["Tmag"], errors="coerce").fillna(10.0)
    bright_mask = tmag < 9.0
    faint_mask = tmag >= 9.0
    for name, mask in [("bright_hosts_tmag_under_9", bright_mask), ("faint_hosts_tmag_ge_9", faint_mask)]:
        if mask.sum() >= 10 and len(np.unique(y_test[mask])) > 1:
            subgroups[name] = {
                "count": int(mask.sum()),
                "positive": int((y_test[mask] == 1).sum()),
                "roc_auc": round(float(roc_auc_score(y_test[mask], y_prob_cal[mask])), 4),
                "pr_auc": round(float(average_precision_score(y_test[mask], y_prob_cal[mask])), 4),
                "brier": round(float(np.mean((y_prob_cal[mask] - y_test[mask])**2)), 4),
            }

    return subgroups


def run_training_pipeline(
    dataset_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    random_seed: int = 42,
    c_candidates: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Execute full leakage-safe training, validation tuning, calibration, and test evaluation.
    Saves trained model bundle, manifests, and reports.
    """
    artifacts_root = Path(artifacts_dir)
    models_dir = artifacts_root / "models"
    manifests_dir = artifacts_root / "manifests"
    reports_dir = artifacts_root / "reports"
    for d in [models_dir, manifests_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if c_candidates is None:
        c_candidates = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    # Step 1: Audit and load valid cohort
    df_valid, audit_report = load_and_audit_dataset(dataset_path)

    # Step 2: Leakage-safe Host-Disjoint Partitioning
    splits, split_manifest_meta = partition_host_disjoint(df_valid, random_seed=random_seed)

    split_manifest = create_manifest(
        manifest_id=f"manifest_split_seed{random_seed}_{audit_report['sha256'][:8]}",
        manifest_type="DATASET_HOST_GROUPED_SPLIT",
        files=[str(Path(dataset_path).resolve())],
        metadata=split_manifest_meta,
    )
    manifest_path = manifests_dir / "dataset_host_split_manifest.json"
    save_manifest(split_manifest, manifest_path)

    df_train = splits["train"]
    df_val = splits["val"]
    df_cal = splits["cal"]
    df_test = splits["test"]

    y_train = df_train["target"].values.astype(int)
    y_val = df_val["target"].values.astype(int)
    y_cal = df_cal["target"].values.astype(int)
    y_test = df_test["target"].values.astype(int)

    # Step 3: Train-only Preprocessing
    preprocessor = TabularFeaturePreprocessor()
    preprocessor.fit(df_train)

    X_train = preprocessor.transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_cal = preprocessor.transform(df_cal)
    X_test = preprocessor.transform(df_test)

    # Step 4: Validation-only Model Tuning & Selection
    best_c = 1.0
    best_val_prauc = -1.0
    val_records = []

    for c_val in c_candidates:
        candidate_model = LogisticRegression(
            C=c_val,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_seed,
            solver="lbfgs",
        )
        candidate_model.fit(X_train, y_train)

        val_probs = candidate_model.predict_proba(X_val)[:, 1]
        val_prauc = float(average_precision_score(y_val, val_probs))
        val_rocauc = float(roc_auc_score(y_val, val_probs))

        val_records.append({
            "C": c_val,
            "val_pr_auc": round(val_prauc, 4),
            "val_roc_auc": round(val_rocauc, 4),
        })

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_c = c_val

    # Final selected classifier trained on train set with best C
    selected_classifier = LogisticRegression(
        C=best_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=random_seed,
        solver="lbfgs",
    )
    selected_classifier.fit(X_train, y_train)

    # Step 5: Calibration-only Probability Calibration
    cal_raw_probs = selected_classifier.predict_proba(X_cal)[:, 1]
    calibrator = PlattCalibrator()
    calibrator.fit(cal_raw_probs, y_cal)
    calibrator.provenance = f"Platt Scaling fitted on disjoint calibration partition (N={len(y_cal)}, seed={random_seed})"

    # Step 6: Untouched Test Partition Evaluation (Single Evaluation)
    test_raw_probs = selected_classifier.predict_proba(X_test)[:, 1]
    test_cal_probs = calibrator.calibrate_batch(test_raw_probs)

    # Evaluation metrics
    raw_eval = compute_comprehensive_metrics(y_test, test_raw_probs)
    cal_eval = compute_comprehensive_metrics(y_test, test_cal_probs)

    test_preds_binary = (test_cal_probs >= 0.50).astype(int)
    test_precision = float(precision_score(y_test, test_preds_binary, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds_binary, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds_binary, zero_division=0))
    cm = confusion_matrix(y_test, test_preds_binary).tolist()

    subgroups = evaluate_subgroups(df_test, y_test, test_cal_probs)

    # Feature importance (logistic regression standardized weights)
    coefficients = {
        feat: round(float(coef), 4)
        for feat, coef in zip(preprocessor.feature_names, selected_classifier.coef_[0])
    }

    # Step 7: Save Model Artifact & Metadata
    model_bundle = {
        "classifier": selected_classifier,
        "preprocessor": preprocessor,
        "calibrator": calibrator,
        "feature_names": preprocessor.feature_names,
        "best_hyperparameters": {"C": best_c, "solver": "lbfgs", "class_weight": "balanced"},
        "provenance": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset_path": str(Path(dataset_path).resolve()),
            "source_dataset_sha256": audit_report["sha256"],
            "random_seed": random_seed,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "cal_samples": len(y_cal),
            "test_samples": len(y_test),
            "classification_target": "E periodic eclipsing signal versus B contact binary or J junk",
            "target_is_planet_identification": False,
            "input_modality": "tabular TCE catalog features only; no light-curve, physics-diagnostic, or SSL embeddings",
            "environment": get_runtime_environment(),
        },
    }
    model_artifact_path = models_dir / "tce_e_vs_bj_classifier.joblib"
    joblib.dump(model_bundle, model_artifact_path)

    # Save summary report
    full_report = {
        "status": "COMPLETED",
        "dataset_audit": audit_report,
        "split_manifest": split_manifest_meta,
        "hyperparameter_selection": {
            "tested_candidates": val_records,
            "selected_best_c": best_c,
            "best_validation_pr_auc": round(best_val_prauc, 4),
        },
        "test_evaluation": {
            "sample_counts": cal_eval["sample_counts"],
            "ranking": cal_eval["ranking"],
            "classification_at_p05": {
                "precision": round(test_precision, 4),
                "recall": round(test_recall, 4),
                "f1_score": round(test_f1, 4),
                "confusion_matrix_tn_fp_fn_tp": cm,
            },
            "calibration": {
                "raw_brier_score": raw_eval["calibration"]["brier_score"],
                "raw_ece": raw_eval["calibration"]["expected_calibration_error"],
                "calibrated_brier_score": cal_eval["calibration"]["brier_score"],
                "calibrated_ece": cal_eval["calibration"]["expected_calibration_error"],
                "calibration_improvement": {
                    "brier_reduction": round(raw_eval["calibration"]["brier_score"] - cal_eval["calibration"]["brier_score"], 4),
                    "ece_reduction": round(raw_eval["calibration"]["expected_calibration_error"] - cal_eval["calibration"]["expected_calibration_error"], 4),
                },
            },
            "subgroup_performance": subgroups,
        },
        "feature_coefficients": coefficients,
        "model_artifact": str(model_artifact_path.resolve()),
        "split_manifest_file": str(manifest_path.resolve()),
    }

    report_json_path = reports_dir / "training_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Write Markdown summary
    md_content = f"""# ASTROVET TCE Catalog Model Training & Evaluation Report

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
**Source Dataset:** `{audit_report['file_path']}`
**Source Dataset SHA-256:** `{audit_report['sha256']}`
**Random Seed:** `{random_seed}`
**Model Artifact:** `{model_artifact_path.resolve()}`
**Task:** E periodic eclipsing signal versus B contact binary or J junk; this is not planet identification.
**Inputs:** Tabular TCE catalog features only; this run does not use light curves, BLS/TLS outputs, physics diagnostics, or self-supervised embeddings.

---

## 1. Dataset Cohort & Split Partitioning

- **Total Catalog Rows:** {audit_report['raw_total_rows']:,} candidates
- **Valid Labeled Cohort:** {audit_report['valid_labeled_cohort']['total_candidates']:,} candidates across {audit_report['valid_labeled_cohort']['unique_host_stars']:,} unique host stars
- **Positive (E: periodic eclipsing signal):** {audit_report['valid_labeled_cohort']['positive_count']:,} ({audit_report['valid_labeled_cohort']['positive_prevalence']:.1%}); includes planets and non-contact eclipsing binaries
- **Negative (B: contact binary, J: junk/noise):** {audit_report['valid_labeled_cohort']['negative_count']:,} ({1.0 - audit_report['valid_labeled_cohort']['positive_prevalence']:.1%})
- **Excluded Unresolved ('S', 'N', NaN):** {audit_report['valid_labeled_cohort']['excluded_unresolved_count']:,}

### Partitions (100% Group-Disjoint by Host TIC ID)

| Partition | Host Stars | Candidates | Positives | Negatives | Prevalence |
|---|---|---|---|---|---|
| **Train** | {split_manifest_meta['splits']['train']['hosts']:,} | {split_manifest_meta['splits']['train']['candidates']:,} | {split_manifest_meta['splits']['train']['positive']:,} | {split_manifest_meta['splits']['train']['negative']:,} | {split_manifest_meta['splits']['train']['prevalence']:.1%} |
| **Validation** | {split_manifest_meta['splits']['val']['hosts']:,} | {split_manifest_meta['splits']['val']['candidates']:,} | {split_manifest_meta['splits']['val']['positive']:,} | {split_manifest_meta['splits']['val']['negative']:,} | {split_manifest_meta['splits']['val']['prevalence']:.1%} |
| **Calibration** | {split_manifest_meta['splits']['cal']['hosts']:,} | {split_manifest_meta['splits']['cal']['candidates']:,} | {split_manifest_meta['splits']['cal']['positive']:,} | {split_manifest_meta['splits']['cal']['negative']:,} | {split_manifest_meta['splits']['cal']['prevalence']:.1%} |
| **Test** | {split_manifest_meta['splits']['test']['hosts']:,} | {split_manifest_meta['splits']['test']['candidates']:,} | {split_manifest_meta['splits']['test']['positive']:,} | {split_manifest_meta['splits']['test']['negative']:,} | {split_manifest_meta['splits']['test']['prevalence']:.1%} |

---

## 2. Validation Model Tuning

- **Hyperparameter:** Regularization inverse strength $C \\in {c_candidates}$
- **Selected Best C:** `{best_c}` (Validation PR-AUC: `{best_val_prauc:.4f}`)

---

## 3. Untouched Test Set Evaluation

- **ROC-AUC:** `{cal_eval['ranking']['roc_auc']:.4f}`
- **PR-AUC:** `{cal_eval['ranking']['pr_auc']:.4f}`
- **Brier Score:** Raw: `{raw_eval['calibration']['brier_score']:.4f}` -> Calibrated: `{cal_eval['calibration']['brier_score']:.4f}`
- **Expected Calibration Error (ECE):** Raw: `{raw_eval['calibration']['expected_calibration_error']:.4f}` -> Calibrated: `{cal_eval['calibration']['expected_calibration_error']:.4f}`
- **Precision:** `{test_precision:.4f}` | **Recall:** `{test_recall:.4f}` | **F1 Score:** `{test_f1:.4f}`

### Precision@K (Reviewer Capacity)
{json.dumps(cal_eval['ranking']['precision_at_k'], indent=2)}

### Recall at Fixed Precision
{json.dumps(cal_eval['ranking']['recall_at_fixed_precision'], indent=2)}

---

## 4. Subgroup & Uncertainty Limitations

```json
{json.dumps(subgroups, indent=2)}
```

---

## 5. Feature Importances (Standardized Logistic Coefficients)

```json
{json.dumps(coefficients, indent=2)}
```

## Scientific scope

This is a tabular baseline only. It does not consume light curves, physics diagnostics, or self-supervised embeddings, and label E is not a confirmed or candidate planet class.
"""
    report_md_path = reports_dir / "training_evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return full_report
