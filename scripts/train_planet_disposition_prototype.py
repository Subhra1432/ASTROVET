"""Train an explicitly research-only planet-disposition baseline on a prepared cohort.

This is a candidate-level supervised downstream head, not a self-supervised
model and not an ASTROVET upload-path model. TOI follow-up selection bias means
its probability calibration cannot be interpreted as population prevalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "tce_period", "tce_duration", "tce_depth", "tce_model_snr", "tce_num_transits",
    "tce_chisq2", "tce_bin_oedp_stat", "tce_bin_spc_stat", "tce_bin_lpc_stat",
    "tce_ws_maxmes", "tce_dicco_msky", "tce_steff", "tce_slogg", "tce_sradius",
]
LOG_FEATURES = {"tce_period", "tce_duration", "tce_depth", "tce_model_snr", "tce_num_transits", "tce_sradius"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def features_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for name in FEATURES:
        values = pd.to_numeric(frame.get(name, pd.Series(index=frame.index, dtype=float)), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        if name in LOG_FEATURES:
            values = np.log10(values.where(values > 0))
        result[name] = values
    return result


def make_group_splits(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    hosts = np.array(sorted(frame["tic_id"].astype(int).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(hosts)
    n = len(hosts)
    n_train, n_val, n_cal = int(n * .60), int(n * .15), int(n * .10)
    groups = {
        "train": set(hosts[:n_train]),
        "validation": set(hosts[n_train:n_train + n_val]),
        "calibration": set(hosts[n_train + n_val:n_train + n_val + n_cal]),
        "test": set(hosts[n_train + n_val + n_cal:]),
    }
    pairs = list(groups)
    for i, first in enumerate(pairs):
        for second in pairs[i + 1:]:
            if groups[first] & groups[second]:
                raise RuntimeError(f"Host leakage: {first}/{second}")
    splits = {name: frame[frame.tic_id.isin(host_set)].copy().reset_index(drop=True) for name, host_set in groups.items()}
    for name, split in splits.items():
        if split.target.nunique() < 2:
            raise ValueError(f"The {name} split does not contain both labels; use a different seed or gather more dispositions.")
    return splits


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            error += mask.sum() / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.cohort, low_memory=False)
    required = {"tic_id", "toi", "label", "target", *FEATURES}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Cohort is missing columns: {sorted(missing)}")
    data = data[data.label.isin(["PLANET_DISPOSED", "NON_PLANET_DISPOSED"])].copy()
    data["tic_id"] = pd.to_numeric(data.tic_id, errors="raise").astype(int)
    data["target"] = pd.to_numeric(data.target, errors="raise").astype(int)
    if data.duplicated(["tic_id", "toi"]).any():
        raise ValueError("Cohort must contain one representative record per TIC+TOI event.")
    if data.target.nunique() != 2:
        raise ValueError("Both hard-label classes are required.")

    splits = make_group_splits(data, args.seed)
    X = {name: features_from_frame(split) for name, split in splits.items()}
    y = {name: split.target.to_numpy(dtype=int) for name, split in splits.items()}

    best_pipeline = None
    best_c = None
    best_ap = -1.0
    tuning = []
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=args.seed)),
        ])
        model.fit(X["train"], y["train"])
        val_p = model.predict_proba(X["validation"])[:, 1]
        ap = float(average_precision_score(y["validation"], val_p))
        tuning.append({"C": c, "validation_pr_auc": ap})
        if ap > best_ap:
            best_pipeline, best_c, best_ap = model, c, ap
    assert best_pipeline is not None

    # Platt calibration is fitted only on the dedicated calibration hosts.
    cal_p = np.clip(best_pipeline.predict_proba(X["calibration"])[:, 1], 1e-6, 1 - 1e-6)
    cal_logits = np.log(cal_p / (1 - cal_p)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=args.seed)
    calibrator.fit(cal_logits, y["calibration"])

    test_p_raw = best_pipeline.predict_proba(X["test"])[:, 1]
    test_p_clip = np.clip(test_p_raw, 1e-6, 1 - 1e-6)
    test_logits = np.log(test_p_clip / (1 - test_p_clip)).reshape(-1, 1)
    test_p = calibrator.predict_proba(test_logits)[:, 1]
    test_pred = (test_p >= 0.5).astype(int)
    y_test = y["test"]
    metrics = {
        "test_event_count": len(y_test),
        "test_host_count": int(splits["test"].tic_id.nunique()),
        "test_planet_disposed": int(y_test.sum()),
        "test_non_planet_disposed": int((y_test == 0).sum()),
        "test_positive_prevalence": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, test_p)),
        "pr_auc": float(average_precision_score(y_test, test_p)),
        "brier_score": float(brier_score_loss(y_test, test_p)),
        "ece_10_bins": expected_calibration_error(y_test, test_p),
        "precision_at_0_5": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_test, test_pred, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, test_pred, labels=[0, 1], target_names=["non_planet_disposed", "planet_disposed"], output_dict=True, zero_division=0),
    }

    split_manifest = {
        name: {"events": len(split), "hosts": int(split.tic_id.nunique()), "positive": int(split.target.sum()), "negative": int((split.target == 0).sum()), "host_ids": sorted(int(x) for x in split.tic_id.unique())}
        for name, split in splits.items()
    }
    bundle_path = args.output_dir / "research_only_planet_disposition_tabular.joblib"
    joblib.dump({
        "model": best_pipeline,
        "calibrator": calibrator,
        "features": FEATURES,
        "positive_label": "CP/KP (confirmed/known planet disposition)",
        "negative_label": "FP/FA (false-positive/false-alarm disposition)",
        "research_only": True,
        "not_for_upload_inference": "Training inputs are MAST DV TCE statistics; no validated adapter exists from arbitrary uploaded light curves to these exact features.",
        "label_selection_warning": "TOI follow-up labels are a selected sample and calibrated outputs are not population probabilities.",
        "seed": args.seed,
        "selected_C": best_c,
    }, bundle_path)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_file": str(args.cohort.resolve()),
        "cohort_sha256": sha256(args.cohort),
        "event_count": len(data),
        "unique_host_count": int(data.tic_id.nunique()),
        "disposition_counts": data.label.value_counts().to_dict(),
        "class_mapping": {"CP/KP": 1, "FP/FA": 0, "PC/APC/UNSET": "excluded from supervised fit"},
        "feature_columns": FEATURES,
        "log10_transformed_features": sorted(LOG_FEATURES),
        "split_seed": args.seed,
        "split_manifest": split_manifest,
        "host_leakage": False,
        "validation_tuning": tuning,
        "selected_C": best_c,
        "test_metrics": metrics,
        "model_bundle": str(bundle_path.resolve()),
        "model_bundle_sha256": sha256(bundle_path),
        "scientific_limitations": [
            "The cohort contains TOI-followed-up targets and is not a random sample of all TESS TCEs.",
            "The task is current CP/KP versus FP/FA disposition within this selected cohort; it does not establish statistical validation.",
            "The model uses catalog DV features and is not connected to arbitrary FITS upload inference.",
            "Despite Platt scaling on disjoint hosts, the probability is not calibrated to the prevalence of planets among all TESS targets.",
        ],
    }
    report_path = args.output_dir / "research_only_planet_disposition_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    split_path = args.output_dir / "research_only_planet_disposition_splits.json"
    split_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"cohort_events": len(data), "cohort_hosts": data.tic_id.nunique(), "selected_C": best_c, "test_metrics": metrics, "model_bundle": str(bundle_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
