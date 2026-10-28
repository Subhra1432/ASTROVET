import unittest
from pathlib import Path
import numpy as np

from astrovet.ssl.neural_encoder import SelfSupervisedTransitEncoder
from astrovet.ssl.augmentations import random_cadence_mask, bounded_photometric_jitter
from astrovet.ssl.dataset import extract_views_from_fits, extract_views_from_directory


class TestSelfSupervisedLearning(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        phase = np.linspace(-0.5, 0.5, 200)
        self.views = []
        for i in range(32):
            depth = self.rng.uniform(0.001, 0.01)
            duration = self.rng.uniform(0.05, 0.15)
            flux = np.ones(200)
            in_transit = np.abs(phase) < (duration / 2.0)
            flux[in_transit] -= depth
            flux += self.rng.normal(0, 0.0001, size=200)
            self.views.append(flux)
        self.views = np.array(self.views)

    def test_encoder_initial_pass_and_shape(self):
        encoder = SelfSupervisedTransitEncoder(input_dim=200, hidden_dim=32, embedding_dim=16, seed=42)
        embeddings = encoder.encode(self.views[:4])
        self.assertEqual(embeddings.shape, (4, 16))
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_ssl_training_cpu_smoke_test(self):
        """CPU smoke test verifying loss convergence under masked autoencoding and view consistency."""
        encoder = SelfSupervisedTransitEncoder(
            input_dim=200,
            hidden_dim=32,
            embedding_dim=16,
            view_loss_weight=0.5,
            seed=42,
        )

        history = encoder.fit(self.views, epochs=6, lr=0.01, batch_size=16)

        self.assertIn("loss", history)
        self.assertIn("recon_loss", history)
        self.assertIn("view_loss", history)
        self.assertEqual(len(history["loss"]), 6)

        initial_loss = history["loss"][0]
        final_loss = history["loss"][-1]
        self.assertLess(final_loss, initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}")

        self.assertTrue(encoder.is_fitted)
        encoded = encoder.encode(self.views)
        self.assertEqual(encoded.shape, (32, 16))
        self.assertFalse(np.any(np.isnan(encoded)))

    def test_masked_input_handling(self):
        """Verify encoder gracefully handles cadences with missing/NaN values."""
        encoder = SelfSupervisedTransitEncoder(input_dim=200, hidden_dim=32, embedding_dim=16)
        corrupted = self.views[:2].copy()
        corrupted[0, 10:30] = np.nan
        corrupted[1, 150:170] = np.nan

        embeddings = encoder.encode(corrupted)
        self.assertEqual(embeddings.shape, (2, 16))
        self.assertFalse(np.any(np.isnan(embeddings)))

    def test_real_fits_view_extraction(self):
        """Verify real unlabeled TESS FITS light curve is parsed, filtered, and converted to valid views."""
        fits_file = Path("data/raw/mast/tess_s0023_390651552_spoc_lc.fits")
        if not fits_file.is_file():
            self.skipTest("Local MAST FITS file not found")

        views_arr, meta = extract_views_from_fits(fits_file, view_length=200)
        self.assertGreater(len(views_arr), 50)
        self.assertEqual(views_arr.shape[1], 200)
        self.assertFalse(np.any(np.isnan(views_arr)))
        self.assertEqual(meta["tic_id"], 390651552)
        self.assertEqual(meta["sector"], 23)
        self.assertAlmostEqual(meta["cadence_seconds"], 120.0, delta=1.0)
        self.assertIn("sha256", meta)

    def test_checkpoint_provenance_roundtrip(self):
        """Verify encoder save_checkpoint and load_checkpoint record and preserve provenance."""
        import tempfile
        encoder = SelfSupervisedTransitEncoder(input_dim=200, hidden_dim=32, embedding_dim=16)
        encoder.is_fitted = True

        provenance = {
            "data_source": "REAL_TESS_FITS",
            "tic_id": 390651552,
            "sector": 23,
            "num_views": 135,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_prov.npz"
            encoder.save_checkpoint(ckpt_path, provenance=provenance)
            self.assertTrue(ckpt_path.is_file())

            loaded = SelfSupervisedTransitEncoder.load_checkpoint(ckpt_path)
            self.assertEqual(loaded.data_source, "REAL_TESS_FITS")
            self.assertEqual(loaded.provenance["tic_id"], 390651552)
            self.assertEqual(loaded.provenance["sector"], 23)
            self.assertEqual(loaded.provenance["num_views"], 135)


if __name__ == "__main__":
    unittest.main()
