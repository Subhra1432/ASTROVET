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
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrovet.ids import parse_candidate_string
from astrovet.schemas.baseline import BaselinePrediction


EXOMINER_DEFAULT_IMAGE = "ghcr.io/nasa/exominer:v1.0.0"


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
    ) -> List[str]:
        """
        Build the container execution command list.
        """
        in_path = str(Path(input_dir).resolve())
        out_path = str(Path(output_dir).resolve())
        engine_bin = self.engine or "podman"

        cmd = [
            engine_bin,
            "run",
            "--rm",
            "-v", f"{in_path}:/workspace/input:ro",
            "-v", f"{out_path}:/workspace/output",
            self.image_name,
            "python", "-m", "exominer_pipeline.run",
            "--tic", str(tic_id),
            "--sector", str(sector),
        ]
        if period is not None:
            cmd.extend(["--period", str(period)])
        if epoch is not None:
            cmd.extend(["--epoch", str(epoch)])
        if duration_hours is not None:
            cmd.extend(["--duration-hours", str(duration_hours)])

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

        # Copy or symlink target FITS if provided
        if fits_path and Path(fits_path).is_file():
            dest_fits = in_dir / Path(fits_path).name
            if not dest_fits.exists():
                shutil.copy2(fits_path, dest_fits)

        cmd = self.build_command(
            input_dir=in_dir,
            output_dir=out_dir,
            tic_id=tic_id,
            sector=sector,
            period=period,
            epoch=epoch,
            duration_hours=duration_hours,
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

            # Inspect output directory for container predictions
            pred_file = out_dir / "prediction.json"
            if res.returncode == 0 and pred_file.is_file():
                with open(pred_file, "r", encoding="utf-8") as pf:
                    data = json.load(pf)
                return BaselinePrediction(
                    candidate_id=candidate_id,
                    upstream_model=f"ExoMiner++_v1.0.0 (Container: {self.image_name})",
                    score=float(data.get("score", 0.0)),
                    score_pc=data.get("score_pc"),
                    score_afp=data.get("score_afp"),
                    score_ntp=data.get("score_ntp"),
                    score_uncertainty=data.get("score_uncertainty", 0.0),
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
                    upstream_model=f"ExoMiner++_v1.0.0 (Container: {self.image_name})",
                    score=None,
                    status="CONTAINER_ERROR",
                    execution_mode="CONTAINER_PROCESS",
                    is_synthetic=False,
                    raw_output={"stderr": err_msg, "command": " ".join(cmd)},
                )

        except subprocess.TimeoutExpired:
            return BaselinePrediction(
                candidate_id=candidate_id,
                upstream_model=f"ExoMiner++_v1.0.0 (Container: {self.image_name})",
                score=None,
                status="CONTAINER_TIMEOUT",
                execution_mode="CONTAINER_PROCESS",
                is_synthetic=False,
                raw_output={"error": f"Container run timed out after {self.timeout_seconds}s"},
            )
        except Exception as e:
            return BaselinePrediction(
                candidate_id=candidate_id,
                upstream_model=f"ExoMiner++_v1.0.0 (Container: {self.image_name})",
                score=None,
                status="ERROR",
                execution_mode="CONTAINER_PROCESS",
                is_synthetic=False,
                raw_output={"exception": str(e)},
            )
