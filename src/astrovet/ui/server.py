"""
Lightweight local HTTP dashboard server for ASTROVET.
Provides interactive human review and candidate vetting interface.
"""

from __future__ import annotations

import json
import mimetypes
import os
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict

from astrovet.baseline.adapter import ExoMinerBaselineAdapter
from astrovet.explain.report import vet_and_generate_report
from astrovet.ids import parse_candidate_string, make_canonical_candidate_id


UI_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
                err_data = json.dumps({
                    "error": "candidate_id query parameter is required. A real candidate ID must be specified (e.g. 390651552-1-S23)."
                }).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)
                return

            candidate_id = candidate_id_param.strip()
            fits_file_param = query.get("fits_file", [None])[0]
            fits_file = fits_file_param.strip() if fits_file_param and fits_file_param.strip() else None
            retrieve_mast = query.get("retrieve_mast", ["false"])[0].lower() in ["true", "1", "yes"]

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

            from astrovet.ids import resolve_candidate_identity
            try:
                cand_ident = resolve_candidate_identity(candidate_id=candidate_id)
            except Exception as e:
                err_data = json.dumps({"error": f"Invalid candidate ID format: {e}"}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)
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
                # Run complete vetting pipeline strictly with real observed data
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
                )
                pred = res["prediction"]
                cand = res.get("candidate")
                b_pred = pred.baseline_prediction
                b_mode = b_pred.execution_mode if (b_pred and b_pred.score is not None) else "BASELINE_UNAVAILABLE"
                is_synthetic = bool(pred.is_synthetic or (b_pred and b_pred.is_synthetic))

                pdiag = pred.physics_diagnostics
                pgram = getattr(pred, "periodogram_diagnostics", None) or pdiag.periodogram_summary or {}

                payload = {
                    "candidate_id": pred.candidate_id,
                    "baseline_score": pred.baseline_score,
                    "baseline_execution_mode": b_mode,
                    "astrovet_raw_score": pred.astrovet_raw_score,
                    "score_type": pred.score_type,
                    "is_calibrated": pred.is_calibrated,
                    "calibrated_planet_like_probability": pred.calibrated_planet_like_probability,
                    "class_or_triage_state": pred.triage_state.value,
                    "predictive_uncertainty": pred.predictive_uncertainty,
                    "is_synthetic": is_synthetic,
                    "synthetic_disclaimer": (
                        "DEMONSTRATION / SYNTHETIC FIXTURE: Scores are generated from a synthetic test fixture. "
                        "They are NOT real flight model predictions and carry NO scientific claims."
                        if is_synthetic else "Official TESS SPOC 2-Minute Cadence Observation"
                    ),
                    "fits_file_used": str(fits_file),
                    "plot_url": f"/artifacts/reports/{pred.candidate_id}_diagnostics.png",
                    "diagnostics": pred.physics_diagnostics.model_dump(),
                    "periodogram": {
                        "bls": pgram.get("bls", {"is_available": False, "reason": "BLS not executed"}),
                        "tls": pgram.get("tls", {"is_available": False, "reason": "TLS not executed"}),
                        "period_agreement": pgram.get("period_agreement", False),
                        "period_agreement_diff_percent": pgram.get("period_agreement_diff_percent"),
                    },
                    "active_reasons": pred.reason_codes,
                }
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json_error(500, str(e))
            return

        else:
            self.send_error(404, "Endpoint not found")

    def _send_json_error(self, status: int, message: str) -> None:
        data = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
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
