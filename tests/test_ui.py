import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from astrovet.ui.server import AstrovetDashboardHandler


class TestDashboardUI(unittest.TestCase):
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
            self.assertFalse(data["diagnostics"]["modality_availability"]["ssl_representation"])

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


if __name__ == "__main__":
    unittest.main()
