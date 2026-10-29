"""
NASA ExoMiner++ baseline adapter and candidate reconciliation.

Scientific safeguards & scope:
- Pinned upstream release tag: v1.0.0 (Git commit d877a23) from official NASA release.
- Model / checkpoint name: "ExoMiner++ (TESS 2-min S1-S67)".
- Adapter scope: Implements catalog lookup against published ExoMiner++ prediction tables,
  explicit synthetic fixtures for tests, and optional container execution through the official
  upstream pipeline interface. Live container execution requires a working Docker/Podman daemon.
- If a candidate is not found, score is None (NEVER fabricated as 0.50).
- Checkpoint overlap audit: records whether evaluation targets were potentially seen in
  the baseline model's training set. If unknown, marks status as UNKNOWN to prevent inflated
  generalization claims.
- Request-to-result reconciliation: verifies one-to-one mapping for requested candidate IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from astrovet.ids import make_canonical_candidate_id, resolve_candidate_identity
from astrovet.schemas.baseline import BaselinePrediction


EXOMINER_RELEASE_TAG = "v1.0.0"
EXOMINER_RELEASE_COMMIT = "d877a23"
EXOMINER_MODEL_NAME = "ExoMiner++ (TESS 2-min S1-S67)"


def find_default_catalog_path(project_root: Optional[str | Path] = None) -> Optional[Path]:
    """Find ASTROVET-local or sibling upstream NASA ExoMiner prediction data."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
    filename = "exominer_vetting_tess-spoc-2-min-s1s67_dashtable_dvm-url_scoregt0.1.csv"
    candidates = (
        root / "data" / "catalogs" / "exominer_s1s67_catalog.csv",
        root.parent / "ExoMiner-main" / "exominer_vetting_pc_catalog_dash-render-web-app" / "data" / filename,
    )
    return next((path for path in candidates if path.is_file()), None)


class ExoMinerBaselineAdapter:
    """
    Thin, version-pinned adapter for NASA ExoMiner++.
    Supports catalog lookup against published prediction catalogs and synthetic fixture mode.
    """

    def __init__(
        self,
        catalog_path: Optional[str | Path] = None,
        release_tag: str = EXOMINER_RELEASE_TAG,
        release_commit: str = EXOMINER_RELEASE_COMMIT,
        model_name: str = EXOMINER_MODEL_NAME,
        is_synthetic: bool = False,
    ) -> None:
        self.catalog_path = Path(catalog_path) if catalog_path else find_default_catalog_path()
        self.release_tag = release_tag
        self.release_commit = release_commit
        self.model_name = model_name
        self.is_synthetic = is_synthetic
        self._catalog_cache: Optional[pd.DataFrame] = None

    def _load_catalog(self) -> pd.DataFrame:
        if self._catalog_cache is not None:
            return self._catalog_cache

        if self.catalog_path and self.catalog_path.is_file():
            df = pd.read_csv(self.catalog_path, low_memory=False)
            if "TCE ID" in df.columns:
                df["canonical_id"] = df["TCE ID"].astype(str)
            elif "uid" in df.columns:
                df["canonical_id"] = df["uid"].astype(str)
            else:
                tic_col = next((c for c in df.columns if "tic" in c.lower()), None)
                plnt_col = next((c for c in df.columns if "planet" in c.lower() or "plnt" in c.lower()), None)
                sec_col = next((c for c in df.columns if "sector" in c.lower()), None)
                if tic_col and plnt_col and sec_col:
                    df["canonical_id"] = [
                        make_canonical_candidate_id(int(r[tic_col]), int(r[plnt_col]), str(r[sec_col]))
                        for _, r in df.iterrows()
                    ]
            self._catalog_cache = df
            return self._catalog_cache
        else:
            return pd.DataFrame()

    def predict_candidate(
        self,
        candidate_id: str,
        tic_id: int,
        planet_num: int,
        sector_run: str,
    ) -> BaselinePrediction:
        """
        Query baseline prediction for a single candidate from the published catalog.
        """
        canonical_id = make_canonical_candidate_id(tic_id, planet_num, sector_run)
        
        if self.is_synthetic:
            # Explicit synthetic demonstration fixture
            return BaselinePrediction(
                candidate_id=canonical_id,
                upstream_model=f"{self.model_name} [SYNTHETIC_FIXTURE]",
                score=0.88,
                score_pc=0.88,
                score_afp=0.08,
                score_ntp=0.04,
                score_uncertainty=0.02,
                status="SUCCESS",
                execution_mode="SYNTHETIC_FIXTURE",
                is_synthetic=True,
                checkpoint_overlap_audited=True,
                checkpoint_overlap_status="KNOWN_DISJOINT",
                raw_output={"note": "Synthetic test fixture baseline score"},
            )

        cat_df = self._load_catalog()
        if len(cat_df) > 0 and "canonical_id" in cat_df.columns:
            matches = cat_df[cat_df["canonical_id"] == canonical_id]
            if len(matches) > 0:
                row = matches.iloc[0]
                raw_score = row.get("ExoMiner Score", row.get("score"))
                score_val = float(raw_score) if (raw_score is not None and not pd.isna(raw_score)) else None
                
                raw_unc = row.get("ExoMiner Unc. Score", row.get("score_std"))
                unc_val = float(raw_unc) if (raw_unc is not None and not pd.isna(raw_unc)) else None
                
                return BaselinePrediction(
                    candidate_id=canonical_id,
                    upstream_model=f"{self.model_name} (Release {self.release_tag})",
                    score=score_val,
                    score_pc=score_val,
                    score_uncertainty=unc_val,
                    status="SUCCESS" if score_val is not None else "NOT_FOUND",
                    execution_mode="CATALOG",
                    is_synthetic=False,
                    checkpoint_overlap_audited=False,
                    checkpoint_overlap_status="UNKNOWN",
                    raw_output=row.to_dict(),
                )

        # Candidate not present in baseline catalog: score is None (NEVER fabricated 0.50)
        return BaselinePrediction(
            candidate_id=canonical_id,
            upstream_model=f"{self.model_name} (Release {self.release_tag})",
            score=None,
            score_uncertainty=None,
            status="NOT_FOUND",
            execution_mode="CATALOG",
            is_synthetic=False,
            checkpoint_overlap_audited=False,
            checkpoint_overlap_status="UNKNOWN",
            raw_output={"error": f"Candidate {canonical_id} not found in reference catalog."},
        )

    def reconcile_batch(
        self,
        candidate_ids: List[str],
    ) -> Tuple[List[BaselinePrediction], Dict[str, Any]]:
        """
        Execute baseline queries across a batch and audit request-to-result completeness.
        """
        predictions: List[BaselinePrediction] = []
        found_count = 0
        missing_ids = []

        for cid in candidate_ids:
            try:
                ident = resolve_candidate_identity(candidate_id=cid)
                pred = self.predict_candidate(
                    candidate_id=ident.candidate_id,
                    tic_id=ident.tic_id,
                    planet_num=ident.planet_num,
                    sector_run=ident.sector_run,
                )
                predictions.append(pred)
                if pred.status == "SUCCESS" and pred.score is not None:
                    found_count += 1
                else:
                    missing_ids.append(cid)
            except Exception as e:
                predictions.append(BaselinePrediction(
                    candidate_id=cid,
                    upstream_model=self.model_name,
                    score=None,
                    status="ERROR",
                    execution_mode="CATALOG",
                    raw_output={"exception": str(e)},
                ))
                missing_ids.append(cid)

        reconciliation_report = {
            "requested_count": len(candidate_ids),
            "reconciled_success_count": found_count,
            "missing_or_failed_count": len(missing_ids),
            "missing_candidate_ids": missing_ids,
            "release_tag": self.release_tag,
            "release_commit": self.release_commit,
            "model_name": self.model_name,
        }

        return predictions, reconciliation_report

    def predict_via_container(
        self,
        candidate_id: str,
        fits_path: Optional[str | Path] = None,
        period: Optional[float] = None,
        epoch: Optional[float] = None,
        duration_hours: Optional[float] = None,
        image_name: Optional[str] = None,
        work_dir: Optional[str | Path] = None,
        fallback_to_catalog: bool = True,
    ) -> BaselinePrediction:
        """
        Execute official ExoMiner++ baseline via containerized process (Podman/Docker).
        Falls back to catalog prediction if daemon is not running.
        """
        from astrovet.baseline.container import ExoMinerContainerRunner, EXOMINER_DEFAULT_IMAGE
        runner = ExoMinerContainerRunner(image_name=image_name or EXOMINER_DEFAULT_IMAGE)
        return runner.run(
            candidate_id=candidate_id,
            fits_path=fits_path,
            period=period,
            epoch=epoch,
            duration_hours=duration_hours,
            work_dir=work_dir,
            fallback_adapter=self if fallback_to_catalog else None,
        )
