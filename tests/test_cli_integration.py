import json
import tempfile
import unittest
from pathlib import Path
from click.testing import CliRunner

from astrovet.cli import cli
from astrovet.ssl.neural_encoder import SelfSupervisedTransitEncoder


class TestCLIIntegrationBranches(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_cli_ssl_pretrain_unconditional_synthetic_fails(self):
        """Verify astrovet ssl pretrain without inputs or --synthetic-demo is rejected."""
        res = self.runner.invoke(cli, ["ssl", "pretrain"])
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("Unconditional synthetic training is disabled", res.output)

    def test_cli_ssl_pretrain_synthetic_demo_flag(self):
        """Verify astrovet ssl pretrain --synthetic-demo trains and writes synthetic provenance."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "synth_encoder.npz"
            res = self.runner.invoke(
                cli,
                [
                    "ssl", "pretrain",
                    "--synthetic-demo",
                    "--epochs", "2",
                    "--batch-size", "16",
                    "--num-samples", "32",
                    "--output-checkpoint", str(ckpt_path),
                ],
            )
            self.assertEqual(res.exit_code, 0, f"Command failed: {res.output}")
            self.assertIn("[DEMO NOTICE]", res.output)
            self.assertIn("SYNTHETIC_DEMO", res.output)
            self.assertTrue(ckpt_path.is_file())

            enc = SelfSupervisedTransitEncoder.load_checkpoint(ckpt_path)
            self.assertTrue(enc.is_fitted)
            self.assertEqual(enc.data_source, "SYNTHETIC_DEMO")

    def test_cli_ssl_pretrain_real_fits(self):
        """Verify astrovet ssl pretrain --input-fits extracts real views and logs provenance."""
        fits_file = Path("data/raw/mast/tess_s0023_390651552_spoc_lc.fits")
        if not fits_file.is_file():
            self.skipTest("Local MAST FITS file not found")

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "real_encoder.npz"
            res = self.runner.invoke(
                cli,
                [
                    "ssl", "pretrain",
                    "--input-fits", str(fits_file),
                    "--epochs", "2",
                    "--batch-size", "16",
                    "--output-checkpoint", str(ckpt_path),
                ],
            )
            self.assertEqual(res.exit_code, 0, f"Command failed: {res.output}")
            self.assertIn("Extracting real training views from TESS FITS", res.output)
            self.assertIn("REAL_TESS_FITS", res.output)
            self.assertTrue(ckpt_path.is_file())

            enc = SelfSupervisedTransitEncoder.load_checkpoint(ckpt_path)
            self.assertTrue(enc.is_fitted)
            self.assertEqual(enc.data_source, "REAL_TESS_FITS")
            self.assertEqual(enc.provenance.get("tic_id"), 390651552)
            self.assertEqual(enc.provenance.get("sector"), 23)

    def test_cli_explain_with_ssl_encoder_branch(self):
        """Verify explain command invokes SelfSupervisedTransitEncoder branch."""
        res = self.runner.invoke(
            cli,
            [
                "explain",
                "--candidate-id", "390651552-1-S23",
                "--encoder-mode", "ssl",
                "--baseline-mode", "synthetic",
                "--allow-synthetic",
            ],
        )
        self.assertEqual(res.exit_code, 0, f"Command failed: {res.output}")
        self.assertIn("no trained checkpoint supplied", res.output)
        self.assertIn("Candidate Vetting Completed!", res.output)

    def test_cli_explain_synthetic_mode_without_opt_in_fails(self):
        """Verify baseline-mode synthetic without --allow-synthetic fails closed."""
        res = self.runner.invoke(
            cli,
            [
                "explain",
                "--candidate-id", "390651552-1-S23",
                "--baseline-mode", "synthetic",
            ],
        )
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("--allow-synthetic", res.output)

    def test_cli_explain_with_container_baseline_branch_fallback(self):
        """Verify explain command invokes container runner and handles runtime-unavailable fallback."""
        res = self.runner.invoke(
            cli,
            [
                "explain",
                "--candidate-id", "390651552-1-S23",
                "--baseline-mode", "container",
                "--container-fallback",
            ],
        )
        self.assertEqual(res.exit_code, 0, f"Command failed: {res.output}")
        self.assertIn("Executing official ExoMiner++ via container runner", res.output)
        self.assertIn("Candidate Vetting Completed!", res.output)

    def test_cli_explain_with_container_baseline_no_fallback(self):
        """Verify explain command without fallback marks baseline as unavailable without inventing scores."""
        res = self.runner.invoke(
            cli,
            [
                "explain",
                "--candidate-id", "390651552-1-S23",
                "--baseline-mode", "container",
                "--no-container-fallback",
            ],
        )
        self.assertEqual(res.exit_code, 0, f"Command failed: {res.output}")
        self.assertIn("BASELINE_UNAVAILABLE", res.output)
        self.assertIn("Baseline Score   : N/A", res.output)


if __name__ == "__main__":
    unittest.main()
