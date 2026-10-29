import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from astrovet.ui.server import AstrovetDashboardHandler, resolve_ssl_checkpoint


class TestDashboardUI(unittest.TestCase):
    def test_ssl_checkpoint_environment_override(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "encoder.npz"
            checkpoint.write_bytes(b"checkpoint placeholder")
            with patch.dict(os.environ, {"ASTROVET_SSL_CHECKPOINT": str(checkpoint)}):
                self.assertEqual(resolve_ssl_checkpoint(), checkpoint.resolve())
            with patch.dict(os.environ, {"ASTROVET_SSL_CHECKPOINT": str(checkpoint.with_name("missing.npz"))}):
                with self.assertRaises(FileNotFoundError):
                    resolve_ssl_checkpoint()

    @classmethod
    def setUpClass(cls):
        # Bind to ephemeral port
        cls.server = HTTPServer(("127.0.0.1", 0), AstrovetDashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_index_html(self):
        """Verify dashboard index.html loads and starts with neutral placeholders."""
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("ASTROVET", content)
            self.assertIn("Candidate Prioritization", content)

            # Ensure inputs are blank by default
            self.assertIn('id="candidateIdInput" value=""', content)
            self.assertIn('id="fitsPathInput" value=""', content)
            self.assertIn('id="retrieveMastCheck"', content)

            # Ensure pre-run display shows neutral NOT RUN and N/A placeholders
            self.assertIn('id="triageBadge" class="triage-badge badge-not-run">NOT RUN</span>', content)
            self.assertIn('id="valBaselineScore">N/A</span>', content)
            self.assertIn('id="valBaselineMode">[NOT_RUN]</span>', content)
            self.assertIn('id="valAstrovetScore">N/A</span>', content)
            self.assertIn('id="valCalibratedProb">N/A</span>', content)
            self.assertIn('id="valUncertainty">N/A</span>', content)

            # Ensure NO pre-run demo scores or synthetic fixture tags exist in the initial HTML
            self.assertNotIn("0.8800", content)
            self.assertNotIn("[SYNTHETIC_FIXTURE]", content)
            self.assertNotIn("Demonstration score", content)

    def test_api_vet_candidate(self):
        """Test running real candidate through the dashboard API."""
        if not Path("data/raw/mast/tess_s0023_390651552_spoc_lc.fits").is_file():
            self.skipTest("Optional observed SPOC FITS fixture is not installed")
        url = f"http://127.0.0.1:{self.port}/api/vet?candidate_id=390651552-1-S23"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["candidate_id"], "390651552-1-S23")
            self.assertIn("baseline_score", data)
            self.assertIn("astrovet_raw_score", data)
            self.assertIn("class_or_triage_state", data)
            self.assertIn("plot_url", data)
            self.assertFalse(data["is_synthetic"])
            self.assertIn("periodogram", data)
            self.assertTrue(data["periodogram"]["bls"]["is_available"])
            self.assertTrue(data["periodogram"]["tls"]["is_available"])
            self.assertIn("transit_depth_fitted_ppm", data["diagnostics"])
            self.assertEqual(
                data["diagnostics"]["modality_availability"]["ssl_representation"],
                resolve_ssl_checkpoint() is not None,
            )

    def test_api_vet_missing_candidate_id_fails_400(self):
        """Verify request without candidate_id or with blank string returns HTTP 400."""
        # Omitted query param
        url1 = f"http://127.0.0.1:{self.port}/api/vet"
        with self.assertRaises(urllib.error.HTTPError) as ctx1:
            urllib.request.urlopen(url1)
        self.assertEqual(ctx1.exception.code, 400)
        err1 = json.loads(ctx1.exception.read().decode("utf-8"))
        self.assertIn("candidate_id query parameter is required", err1.get("error", ""))

        # Blank string query param
        url2 = f"http://127.0.0.1:{self.port}/api/vet?candidate_id=%20%20"
        with self.assertRaises(urllib.error.HTTPError) as ctx2:
            urllib.request.urlopen(url2)
        self.assertEqual(ctx2.exception.code, 400)
        err2 = json.loads(ctx2.exception.read().decode("utf-8"))
        self.assertIn("candidate_id query parameter is required", err2.get("error", ""))

    def test_api_vet_rejects_nonfinite_period(self):
        url = f"http://127.0.0.1:{self.port}/api/vet?candidate_id=390651552-1-S23&period=nan"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url)
        self.assertEqual(ctx.exception.code, 400)
        body = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("finite numeric values", body["error"])

    def test_api_vet_missing_fits_fails_closed(self):
        """Verify request without local FITS or retrieve_mast fails closed without synthetic fallback."""
        url = f"http://127.0.0.1:{self.port}/api/vet?candidate_id=987654321-1-S23"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url)
        self.assertEqual(ctx.exception.code, 400)
        err_msg = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("No real observed FITS", err_msg.get("error", ""))
        self.assertIn("Silent synthetic fallback is disabled", err_msg.get("error", ""))

    def test_api_vet_explicit_retrieve_mast_handling(self):
        """Verify explicit retrieve_mast=true contacts MAST and fails closed with 502 on retrieval failure."""
        with patch("astrovet.ingest.mast.fetch_tess_lightcurve", side_effect=RuntimeError("MAST connection timeout")):
            url = f"http://127.0.0.1:{self.port}/api/vet?candidate_id=987654321-1-S23&retrieve_mast=true"
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(url)
            self.assertEqual(ctx.exception.code, 502)
            err_msg = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertIn("Official MAST retrieval failed", err_msg.get("error", ""))
            self.assertIn("Silent synthetic fallback is disabled", err_msg.get("error", ""))

    def test_api_vet_bls_tls_unavailable_reporting(self):
        """Verify dashboard API gracefully reports unavailable BLS/TLS with explicit reasons."""
        if not Path("data/raw/mast/tess_s0023_390651552_spoc_lc.fits").is_file():
            self.skipTest("Optional observed SPOC FITS fixture is not installed")
        mock_pgram = {
            "bls": {"is_available": False, "reason": "Insufficient non-gap flux points for box model"},
            "tls": {"is_available": False, "reason": "Light curve too sparse for limb-darkened search"},
            "period_agreement": False,
            "period_agreement_diff_percent": None,
        }
        with patch("astrovet.views.periodogram.run_periodicity_search", return_value=mock_pgram):
            url = f"http://127.0.0.1:{self.port}/api/vet?candidate_id=390651552-1-S23&period=1.46644&epoch=1931.1654&duration=1.023"
            with urllib.request.urlopen(url) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                pgram = data.get("periodogram", {})
                self.assertFalse(pgram["bls"]["is_available"])
                self.assertEqual(pgram["bls"]["reason"], "Insufficient non-gap flux points for box model")
                self.assertFalse(pgram["tls"]["is_available"])
                self.assertEqual(pgram["tls"]["reason"], "Light curve too sparse for limb-darkened search")



    def test_upload_and_vet_valid_fits(self):
        """Test uploading a real FITS light curve and executing vetting in one action."""
        sample_fits = Path("data/raw/mast/tess_s0023_390651552_spoc_lc.fits")
        if not sample_fits.is_file():
            self.skipTest("Optional local SPOC FITS integration fixture is not installed")
        fits_bytes = sample_fits.read_bytes()

        boundary = "---------------------------974767299852498929531610575"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="uploaded_spoc_test.fits"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + fits_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="candidate_id"\r\n\r\n'
            f"390651552-1-S23\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/upload_and_vet",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["candidate_id"], "390651552-1-S23")
            self.assertIn("upload_audit", data)
            aud = data["upload_audit"]
            self.assertEqual(aud["pipeline_origin"], "TESS_SPOC")
            self.assertEqual(aud["metadata"]["tic_id"], 390651552)
            self.assertGreater(aud["cadence_audit"]["usable_cadences"], 1000)
            self.assertIn("class_or_triage_state", data)
            self.assertIn("plot_url", data)

    def test_upload_rejects_empty_file(self):
        boundary = "---------------------------974767299852498929531610575"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="empty.fits"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/upload_and_vet",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_upload_rejects_invalid_extension(self):
        boundary = "---------------------------974767299852498929531610575"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
            f"Content-Type: text/plain\r\n\r\n"
            f"Hello world\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/upload_and_vet",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        err = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("Invalid file extension", err["error"])

    def test_api_vet_rejects_arbitrary_server_paths(self):
        url = f"http://127.0.0.1:{self.port}/api/vet?candidate_id=390651552-1-S23&fits_file=/etc/passwd"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url)
        self.assertEqual(ctx.exception.code, 400)
        err = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("Arbitrary server file paths are not permitted", err["error"])


if __name__ == "__main__":
    unittest.main()
