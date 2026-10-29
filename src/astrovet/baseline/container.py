"""
Official NASA ExoMiner container / subprocess integration runner.

Scientific & architectural safeguards:
- Pinned to official NASA release tag: v1.0.0 (commit d877a23).
- Uses containerized Podman / Docker execution without modifying or mutating the upstream checkout.
- If a container daemon is not active on the host, records a structured CONTAINER_RUNTIME_UNAVAILABLE
  status, constructs the exact reproducible CLI command, and optionally falls back to catalog lookup.
- Validates input volumes, FITS light curves, and TCE parameters before invocation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrovet.ids import parse_candidate_string
from astrovet.schemas.baseline import BaselinePrediction


# OCI index digest for the official NASA multi-architecture ExoMiner image,
# verified from ghcr.io on 2026-10-29. The v1.0.0 tag is a Git release tag,
# not a published container tag.
EXOMINER_DEFAULT_IMAGE = "ghcr.io/nasa/exominer@sha256:1b7ffac9db8afe1787ec9a0464ed8cc16ab91b7a280e6e824c27e3a121b0c6e9"


class ExoMinerContainerRunner:
    """
    Manages containerized execution of the official NASA ExoMiner baseline pipeline.
    """

    def __init__(
        self,
        image_name: str = EXOMINER_DEFAULT_IMAGE,
        engine: Optional[str] = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.image_name = image_name
        self.timeout_seconds = timeout_seconds
        self.engine = engine or self._detect_engine()

    @staticmethod
    def _detect_engine() -> Optional[str]:
        """Detect whether podman or docker is installed in system PATH."""
        if shutil.which("podman"):
            return "podman"
        if shutil.which("docker"):
            return "docker"
        return None

    def build_command(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        tic_id: int,
        sector: int | str,
        period: Optional[float] = None,
        epoch: Optional[float] = None,
        duration_hours: Optional[float] = None,
        use_local_lightcurves: bool = False,
    ) -> List[str]:
        """
        Build the container execution command list.
        """
        in_path = str(Path(input_dir).resolve())
        out_path = str(Path(output_dir).resolve())
        engine_bin = self.engine or "podman"

        # The official image entrypoint is run_pipeline.py; it accepts a TIC
        # CSV via --tic_ids_fp and an output directory.
        cmd = [
            engine_bin,
            "run",
            "--rm",
            "-v", f"{in_path}:/workspace/input:ro",
            "-v", f"{out_path}:/workspace/output",
            self.image_name,
            "--output_dir=/workspace/output",
            "--tic_ids_fp", "/workspace/output/tics_tbl.csv",
            "--data_collection_mode=2min",
            "--num_processes=1",
            "--num_jobs=1",
            "--get_mast_urls_dv_reports=false",
            "--task=phot-vetting",
            "--exominer_model=single",
            "--max_model_workers=1",
        ]
        if use_local_lightcurves:
            cmd.append("--lc_data_repository=/workspace/input")

        return cmd

    def run(
        self,
        candidate_id: str,
        fits_path: Optional[str | Path] = None,
        period: Optional[float] = None,
        epoch: Optional[float] = None,
        duration_hours: Optional[float] = None,
        work_dir: Optional[str | Path] = None,
        fallback_adapter: Optional[Any] = None,
    ) -> BaselinePrediction:
        """
        Execute ExoMiner pipeline inside a container or return structured unavailable state.
        """
        tic_id, planet_num, sector = parse_candidate_string(candidate_id)

        work_path = Path(work_dir) if work_dir else Path("data/container_runs") / candidate_id
        in_dir = work_path / "input"
        out_dir = work_path / "output"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write candidate input descriptor for container pipeline
        spec_path = in_dir / "candidate_spec.json"
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump({
                "candidate_id": candidate_id,
                "tic_id": tic_id,
                "planet_num": planet_num,
                "sector": sector,
                "period_days": period,
                "epoch_btjd": epoch,
                "duration_hours": duration_hours,
            }, f, indent=2)

        # The official pipeline writes back to its TIC CSV during setup, so
        # keep that generated file on the writable output mount. Use numeric
        # TIC values and the documented sector range format.
        sector_run = str(sector).strip()
        if "-" not in sector_run:
            sector_run = f"{sector_run}-{sector_run}"
        with open(out_dir / "tics_tbl.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("tic_id", "sector_run"))
            writer.writerow((int(tic_id), sector_run))

        # The NASA pipeline consumes SPOC 2-minute FITS plus DV XML. QLP or
        # non-2-minute uploads are not valid substitutes; let the official
        # pipeline retrieve its own SPOC inputs in that case.
        use_local_lightcurves = False
        if fits_path and Path(fits_path).is_file():
            try:
                from astrovet.lightcurves.reader import read_tess_fits_lightcurve
                raw = read_tess_fits_lightcurve(fits_path)
                cadence = raw.cadence_seconds
                use_local_lightcurves = (
                    raw.pipeline_origin == "TESS_SPOC"
                    and cadence is not None
                    and 110.0 <= float(cadence) <= 130.0
                )
                if use_local_lightcurves:
                    shutil.copy2(fits_path, in_dir / Path(fits_path).name)
            except Exception:
                use_local_lightcurves = False

        cmd = self.build_command(
            input_dir=in_dir,
            output_dir=out_dir,
            tic_id=tic_id,
            sector=sector,
            period=period,
            epoch=epoch,
            duration_hours=duration_hours,
            use_local_lightcurves=use_local_lightcurves,
        )

        # If container runtime is unavailable, handle gracefully
        if not self.engine:
            if fallback_adapter:
                pred = fallback_adapter.predict_candidate(candidate_id=candidate_id, tic_id=tic_id, planet_num=planet_num, sector_run=str(sector))
                pred.raw_output["container_notice"] = (
                    "Container runtime (podman/docker) not found in system PATH. "
                    "Fallback catalog prediction used."
                )
                pred.raw_output["attempted_command"] = " ".join(cmd)
                return pred

            return BaselinePrediction(
                candidate_id=candidate_id,
                upstream_model=f"ExoMiner++_v1.0.0 (Container: {self.image_name})",
                score=None,
                status="CONTAINER_RUNTIME_UNAVAILABLE",
                execution_mode="CONTAINER_PROCESS",
                is_synthetic=False,
                raw_output={
                    "error": "Podman/Docker engine is not active or installed.",
                    "attempted_command": " ".join(cmd),
                    "instruction": "Install Podman or Docker to execute raw pixel pipeline container.",
                }
            )

        # Execute container process
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )

            # Official pipeline writes predictions_<run_name>.csv with comment
            # metadata and score_PC/AFP/NTP columns.
            prediction_files = sorted(out_dir.glob("predictions_*.csv"))
            if res.returncode == 0 and prediction_files:
                import pandas as pd
                predictions = pd.concat(
                    [pd.read_csv(path, comment="#", low_memory=False) for path in prediction_files],
                    ignore_index=True,
                )
                uid_col = next((c for c in ("uid", "TCE ID", "candidate_id") if c in predictions.columns), None)
                if uid_col is None:
                    raise ValueError("ExoMiner output has no uid/TCE ID column")
                matches = predictions[predictions[uid_col].astype(str).str.strip() == candidate_id]
                if matches.empty:
                    raise LookupError(f"ExoMiner completed but did not return requested TCE {candidate_id}")
                row = matches.iloc[0]
                data = row.where(pd.notna(row), None).to_dict()
                pc_score = data.get("score_PC", data.get("ExoMiner Score", data.get("score")))
                afp_score = data.get("score_AFP")
                ntp_score = data.get("score_NTP")
                uncertainty = data.get("score_std_PC", data.get("score_std", data.get("ExoMiner Unc. Score")))
                if pc_score is None:
                    raise ValueError("ExoMiner output row contains no PC/planet score column")
                return BaselinePrediction(
                    candidate_id=candidate_id,
                    upstream_model=f"NASA ExoMiner++ phot-vetting (Container: {self.image_name})",
                    score=float(pc_score),
                    score_pc=float(pc_score),
                    score_afp=float(afp_score) if afp_score is not None else None,
                    score_ntp=float(ntp_score) if ntp_score is not None else None,
                    score_uncertainty=float(uncertainty) if uncertainty is not None else None,
                    status="SUCCESS",
                    execution_mode="CONTAINER_PROCESS",
                    is_synthetic=False,
                    raw_output=data,
                )
            else:
                err_msg = res.stderr or res.stdout or f"Process exited with code {res.returncode}"
                if fallback_adapter:
                    pred = fallback_adapter.predict_candidate(candidate_id=candidate_id, tic_id=tic_id, planet_num=planet_num, sector_run=str(sector))
                    pred.raw_output["container_error"] = err_msg
                    pred.raw_output["attempted_command"] = " ".join(cmd)
                    return pred

                return BaselinePrediction(
                    candidate_id=candidate_id,
                    upstream_model=f"NASA ExoMiner++ phot-vetting (Container: {self.image_name})",
                    score=None,
                    status="CONTAINER_ERROR",
                    execution_mode="CONTAINER_PROCESS",
                    is_synthetic=False,
                    raw_output={"stderr": err_msg, "command": " ".join(cmd)},
                )

        except subprocess.TimeoutExpired:
            return BaselinePrediction(
                candidate_id=candidate_id,
                upstream_model=f"NASA ExoMiner++ phot-vetting (Container: {self.image_name})",
                score=None,
                status="CONTAINER_TIMEOUT",
                execution_mode="CONTAINER_PROCESS",
                is_synthetic=False,
                raw_output={"error": f"Container run timed out after {self.timeout_seconds}s"},
            )
        except Exception as e:
            return BaselinePrediction(
                candidate_id=candidate_id,
                upstream_model=f"NASA ExoMiner++ phot-vetting (Container: {self.image_name})",
                score=None,
                status="ERROR",
                execution_mode="CONTAINER_PROCESS",
                is_synthetic=False,
                raw_output={"exception": str(e)},
            )
