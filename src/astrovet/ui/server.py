"""
Lightweight local HTTP dashboard server for ASTROVET.
Provides interactive human review, candidate vetting, and secure TESS FITS upload interface.

Scientific safeguards:
- Preserves uploaded FITS originals unchanged under an isolated data/uploads directory.
- Rejects arbitrary server file paths requested from the browser.
- Enforces strict file type, magic byte, and size limits.
- Audits and displays SHA-256, metadata, cadence attrition, quality flag rejections, and segments.
- Automatically executes full vetting pipeline reusing vet_and_generate_report.
"""

from __future__ import annotations

import json
import mimetypes
import os
import math
import email
from email.policy import default as email_default_policy
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, Optional, Tuple

from astrovet.baseline.adapter import ExoMinerBaselineAdapter
from astrovet.explain.report import vet_and_generate_report
from astrovet.ids import parse_candidate_string, make_canonical_candidate_id, resolve_candidate_identity
from astrovet.lightcurves.validator import (
    store_uploaded_lightcurve,
    audit_lightcurve_fits,
    MAX_UPLOAD_SIZE_BYTES,
)


UI_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def resolve_ssl_checkpoint() -> Optional[Path]:
    """Resolve an explicitly configured or project-local real-data encoder."""
    configured = os.environ.get("ASTROVET_SSL_CHECKPOINT")
    if configured:
        checkpoint = Path(configured).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ASTROVET_SSL_CHECKPOINT points to a missing file: {checkpoint}")
        return checkpoint
    candidates = (
        PROJECT_ROOT / "artifacts" / "models" / "ssl_encoder_real_qlp_trainhosts_v2.npz",
        PROJECT_ROOT / "data" / "checkpoints" / "ssl_transit_encoder.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


class AstrovetDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output concise
        pass

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/" or path == "/index.html":
            index_file = UI_DIR / "index.html"
            if index_file.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(index_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Index file not found")
            return

        elif path.startswith("/artifacts/reports/"):
            rel_path = path[len("/artifacts/reports/"):]
            report_root = (PROJECT_ROOT / "artifacts" / "reports").resolve()
            report_file = (report_root / rel_path).resolve()
            if report_file != report_root and report_root not in report_file.parents:
                self.send_error(404, "Artifact file not found")
                return
            if report_file.is_file():
                mime, _ = mimetypes.guess_type(str(report_file))
                self.send_response(200)
                self.send_header("Content-Type", mime or "application/octet-stream")
                self.end_headers()
                with open(report_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Artifact file not found")
            return

        elif path == "/api/vet":
            query = parse_qs(parsed_url.query)
            candidate_id_param = query.get("candidate_id", [None])[0]
            if not candidate_id_param or not candidate_id_param.strip():
                self._send_json_error(400, "candidate_id query parameter is required. A real candidate ID must be specified (e.g. 390651552-1-S23).")
                return

            candidate_id = candidate_id_param.strip()
            fits_file_param = query.get("fits_file", [None])[0]
            fits_file = fits_file_param.strip() if fits_file_param and fits_file_param.strip() else None
            retrieve_mast = query.get("retrieve_mast", ["false"])[0].lower() in ["true", "1", "yes"]

            # Security: prevent arbitrary filesystem access from browser
            if fits_file:
                p_fits = Path(fits_file).resolve()
                allowed_roots = [
                    (PROJECT_ROOT / "data").resolve(),
                    (PROJECT_ROOT / "artifacts").resolve(),
                ]
                # Also allow parent directory if inside the known project dataset directory
                external_dataset = Path(r"D:\Final Year Projet").resolve()
                if external_dataset.is_dir():
                    allowed_roots.append(external_dataset)

                is_allowed = any(root == p_fits or root in p_fits.parents for root in allowed_roots)
                if not is_allowed:
                    self._send_json_error(400, "Arbitrary server file paths are not permitted. Please upload your FITS light curve using the upload control.")
                    return

            try:
                period = float(query["period"][0]) if "period" in query and query["period"][0] else None
                epoch = float(query["epoch"][0]) if "epoch" in query and query["epoch"][0] else None
                duration = float(query["duration"][0]) if "duration" in query and query["duration"][0] else None
            except (TypeError, ValueError):
                self._send_json_error(400, "period, epoch, and duration must be numeric values.")
                return
            if any(value is not None and not math.isfinite(value) for value in (period, epoch, duration)):
                self._send_json_error(400, "period, epoch, and duration must be finite numeric values.")
                return
            if period is not None and period <= 0 or duration is not None and duration <= 0:
                self._send_json_error(400, "period and duration must be greater than zero.")
                return

            try:
                cand_ident = resolve_candidate_identity(candidate_id=candidate_id)
            except Exception as e:
                self._send_json_error(400, f"Invalid candidate ID format: {e}")
                return

            # Real data verification: check local FITS or explicit online MAST retrieval
            if not fits_file:
                local_data_root = (PROJECT_ROOT / "data" / "raw" / "mast")
                local_candidates = list(local_data_root.glob(f"*{cand_ident.tic_id}*.fits"))
                if not local_candidates:
                    nested = list(local_data_root.glob(f"**/*{cand_ident.tic_id}*_lc.fits"))
                    if nested:
                        local_candidates = nested

                if local_candidates:
                    fits_file = str(local_candidates[0])
                elif retrieve_mast:
                    from astrovet.ingest.mast import fetch_tess_lightcurve
                    try:
                        sec_num = int(cand_ident.sector_run.split("-")[0].replace("S", ""))
                    except Exception:
                        sec_num = 23
                    try:
                        fetch_res = fetch_tess_lightcurve(
                            tic_id=cand_ident.tic_id,
                            output_dir=local_data_root,
                            sector=sec_num,
                            offline_mode=False,
                            fixture_if_offline=False,
                        )
                        fits_file = str(fetch_res["product_file"])
                    except Exception as e:
                        self._send_json_error(502,
                            f"Official MAST retrieval failed for TIC {cand_ident.tic_id} Sector {sec_num}: {e}. "
                            f"Silent synthetic fallback is disabled."
                        )
                        return
                else:
                    self._send_json_error(400,
                        f"No real observed FITS light curve found locally for TIC {cand_ident.tic_id}. "
                        f"Please supply an explicit fits_file path or pass retrieve_mast=true to download official SPOC data. "
                        f"Silent synthetic fallback is disabled."
                    )
                    return

            try:
                payload = self._execute_vetting(
                    candidate_id=candidate_id,
                    fits_file=fits_file,
                    period=period,
                    epoch=epoch,
                    duration=duration,
                )
                self._send_json_success(payload)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json_error(500, str(e))
            return

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path in ["/api/upload_and_vet", "/api/upload"]:
            content_length_header = self.headers.get("Content-Length")
            if not content_length_header:
                self._send_json_error(411, "Content-Length header required.")
                return

            try:
                content_length = int(content_length_header)
            except ValueError:
                self._send_json_error(400, "Invalid Content-Length header.")
                return

            if content_length > MAX_UPLOAD_SIZE_BYTES:
                self._send_json_error(413, f"Upload size exceeds {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB limit.")
                return

            body_bytes = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "")

            file_bytes: Optional[bytes] = None
            filename = "uploaded_lightcurve.fits"
            candidate_id: Optional[str] = None
            period: Optional[float] = None
            epoch: Optional[float] = None
            duration: Optional[float] = None

            if "multipart/form-data" in content_type:
                # Parse standard multipart/form-data using standard library email parser
                header_lines = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
                msg = email.message_from_bytes(header_lines + body_bytes, policy=email_default_policy)
                if not msg.is_multipart():
                    self._send_json_error(400, "Malformed multipart form-data payload.")
                    return

                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    fn = part.get_filename()
                    payload = part.get_payload(decode=True)

                    if name == "file" or fn:
                        file_bytes = payload
                        if fn:
                            filename = fn
                    elif name == "candidate_id" and payload:
                        candidate_id = payload.decode("utf-8", errors="ignore").strip()
                    elif name == "period" and payload:
                        try:
                            val = float(payload.decode("utf-8", errors="ignore").strip())
                            if math.isfinite(val) and val > 0:
                                period = val
                        except Exception:
                            pass
                    elif name == "epoch" and payload:
                        try:
                            val = float(payload.decode("utf-8", errors="ignore").strip())
                            if math.isfinite(val):
                                epoch = val
                        except Exception:
                            pass
                    elif name == "duration" and payload:
                        try:
                            val = float(payload.decode("utf-8", errors="ignore").strip())
                            if math.isfinite(val) and val > 0:
                                duration = val
                        except Exception:
                            pass
            else:
                # Raw binary upload with query parameters
                file_bytes = body_bytes
                query = parse_qs(parsed_url.query)
                if "filename" in query and query["filename"][0]:
                    filename = query["filename"][0]
                if "candidate_id" in query and query["candidate_id"][0]:
                    candidate_id = query["candidate_id"][0].strip()
                try:
                    if "period" in query and query["period"][0]:
                        period = float(query["period"][0])
                    if "epoch" in query and query["epoch"][0]:
                        epoch = float(query["epoch"][0])
                    if "duration" in query and query["duration"][0]:
                        duration = float(query["duration"][0])
                except Exception:
                    pass

            if not file_bytes:
                self._send_json_error(400, "No file content was provided in the upload request.")
                return

            # Validate and store file in ignored uploads directory
            uploads_dir = PROJECT_ROOT / "data" / "uploads"
            try:
                stored_path, sha256_hash = store_uploaded_lightcurve(
                    data=file_bytes,
                    filename=filename,
                    upload_dir=uploads_dir,
                )
            except ValueError as val_err:
                self._send_json_error(400, str(val_err))
                return
            except Exception as e:
                self._send_json_error(500, f"Failed to store uploaded file safely: {e}")
                return

            # Perform comprehensive FITS audit
            try:
                audit_result = audit_lightcurve_fits(stored_path)
            except ValueError as audit_err:
                self._send_json_error(400, f"FITS Validation Failed: {audit_err}")
                return
            except Exception as e:
                self._send_json_error(500, f"FITS Light Curve Audit Error: {e}")
                return

            # Auto-resolve candidate_id if user left it blank
            if not candidate_id:
                meta = audit_result.get("metadata", {})
                tic_val = meta.get("tic_id")
                sec_val = meta.get("sector") or 1
                if tic_val:
                    candidate_id = f"{tic_val}-1-S{sec_val}"
                else:
                    candidate_id = f"UPLOAD-{sha256_hash[:8]}-1-S23"

            try:
                cand_ident = resolve_candidate_identity(candidate_id=candidate_id)
            except Exception as e:
                self._send_json_error(400, f"Invalid Candidate ID '{candidate_id}': {e}")
                return

            # Execute vetting pipeline
            try:
                payload = self._execute_vetting(
                    candidate_id=cand_ident.candidate_id,
                    fits_file=str(stored_path),
                    period=period,
                    epoch=epoch,
                    duration=duration,
                )
                payload["upload_audit"] = audit_result
                self._send_json_success(payload)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json_error(500, f"Vetting Pipeline Failure on uploaded file: {e}")
            return

        else:
            self.send_error(404, "Endpoint not found")

    def _execute_vetting(
        self,
        candidate_id: str,
        fits_file: str,
        period: Optional[float] = None,
        epoch: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run complete vetting pipeline strictly with real observed data."""
        res = vet_and_generate_report(
            candidate_id=candidate_id,
            output_dir=PROJECT_ROOT / "artifacts" / "reports",
            fits_file=fits_file,
            period=period,
            epoch=epoch,
            duration=duration,
            search_periods=True,
            synthetic_fixture_ok=False,
            allow_synthetic=False,
            encoder_mode="ssl",
            ssl_checkpoint=resolve_ssl_checkpoint(),
        )
        pred = res["prediction"]
        b_pred = pred.baseline_prediction
        b_mode = b_pred.execution_mode if (b_pred and b_pred.score is not None) else "BASELINE_UNAVAILABLE"
        is_synthetic = bool(pred.is_synthetic or (b_pred and b_pred.is_synthetic))

        pdiag = pred.physics_diagnostics
        pgram = getattr(pred, "periodogram_diagnostics", None) or pdiag.periodogram_summary or {}

        return {
            "candidate_id": pred.candidate_id,
            "baseline_score": pred.baseline_score,
            "baseline_execution_mode": b_mode,
            "astrovet_raw_score": pred.astrovet_raw_score,
            "score_type": pred.score_type,
            "is_calibrated": pred.is_calibrated,
            "calibration_provenance": pred.calibration_provenance,
            "calibrated_planet_like_probability": pred.calibrated_planet_like_probability,
            "class_or_triage_state": pred.triage_state.value,
            "predictive_uncertainty": pred.predictive_uncertainty,
            "ssl_representation_available": bool(pdiag.modality_availability.get("ssl_representation", False)),
            "score_explanation": pred.explanation,
            "is_synthetic": is_synthetic,
            "synthetic_disclaimer": (
                "DEMONSTRATION / SYNTHETIC FIXTURE: Scores are generated from a synthetic test fixture. "
                "They are NOT real flight model predictions and carry NO scientific claims."
                if is_synthetic else "Real observed TESS light-curve product; source and cadence are shown in the ingestion audit."
            ),
            "fits_file_used": str(fits_file),
            "plot_url": f"/artifacts/reports/{pred.candidate_id}_diagnostics.png",
            "report_url": f"/artifacts/reports/{pred.candidate_id}_report.md",
            "diagnostics": pred.physics_diagnostics.model_dump(),
            "periodogram": {
                "bls": pgram.get("bls", {"is_available": False, "reason": "BLS not executed"}),
                "tls": pgram.get("tls", {"is_available": False, "reason": "TLS not executed"}),
                "period_agreement": pgram.get("period_agreement", False),
                "period_agreement_diff_percent": pgram.get("period_agreement_diff_percent"),
            },
            "active_reasons": pred.reason_codes,
        }

    def _send_json_error(self, status: int, message: str) -> None:
        data = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json_success(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_dashboard_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start local interactive HTTP dashboard."""
    server = HTTPServer((host, port), AstrovetDashboardHandler)
    print(f"[ASTROVET Dashboard] Listening at http://{host}:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping dashboard server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASTROVET Dashboard Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()
    run_dashboard_server(host=args.host, port=args.port)
