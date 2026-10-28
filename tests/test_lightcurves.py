import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from astropy.io import fits
from astropy.table import Table

from astrovet.ingest.mast import fetch_tess_lightcurve, is_fits_synthetic_fixture
from astrovet.lightcurves.reader import read_tess_fits_lightcurve
from astrovet.lightcurves.quality import filter_quality_cadences
from astrovet.lightcurves.normalization import segment_and_normalize_lightcurve
from astrovet.views.folding import extract_transit_views


def _make_mock_spoc_fits(
    path: Path,
    tic_id: int = 12345678,
    sector: int = 1,
    cadence_seconds: float = 120.0,
    exposure_seconds: float = 20.5346,
) -> Path:
    """Create a mock official SPOC FITS (non-synthetic header cards) for testing."""
    times = np.linspace(1900.0, 1927.0, 200)
    tab = Table({
        "TIME": times,
        "PDCSAP_FLUX": np.ones(200) * 1000.0,
        "PDCSAP_FLUX_ERR": np.ones(200) * 1.0,
        "QUALITY": np.zeros(200, dtype=np.int32),
    })
    pri = fits.PrimaryHDU()
    pri.header["TELESCOP"] = "TESS"
    pri.header["TICID"] = tic_id
    pri.header["SECTOR"] = sector
    pri.header["CREATOR"] = "Lightkurve / SPOC"
    pri.header["ORIGIN"] = "NASA/Ames"
    tbl = fits.table_to_hdu(tab)
    tbl.header["EXTNAME"] = "LIGHTCURVE"
    tbl.header["TIMEDEL"] = cadence_seconds / 86400.0
    tbl.header["EXPOSURE"] = exposure_seconds
    tbl.header["NUM_FRM"] = 60
    tbl.header["FRAMETIM"] = 2.0
    hdul = fits.HDUList([pri, tbl])
    hdul.writeto(path, overwrite=True)
    return path


class TestLightCurves(unittest.TestCase):
    def test_lightcurve_pipeline(self):
        """Pipeline test running strictly in offline mode on deterministic fixture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Fetch synthetic fixture in explicit offline mode (zero network)
            res = fetch_tess_lightcurve(
                tic_id=12345678,
                output_dir=tmp_dir,
                sector=1,
                offline_mode=True,
                fixture_if_offline=True,
            )
            fits_path = Path(res["product_file"])
            self.assertTrue(fits_path.exists())
            self.assertTrue(res["is_synthetic_fixture"])
            self.assertEqual(res["retrieval_mode"], "EXPLICIT_OFFLINE_FIXTURE")

            # 2. Read FITS
            raw_lc = read_tess_fits_lightcurve(fits_path)
            self.assertEqual(raw_lc.tic_id, 12345678)
            self.assertEqual(raw_lc.sector, 1)
            self.assertGreater(len(raw_lc.time), 1000)
            self.assertEqual(raw_lc.time_convention, "BJD - 2457000 (BTJD)")

            # 3. Quality filter
            clean_t, clean_f, clean_fe, q_rep = filter_quality_cadences(
                raw_lc.time, raw_lc.flux, raw_lc.flux_err, raw_lc.quality
            )
            self.assertGreater(q_rep["retained_fraction"], 0.8)
            self.assertGreater(q_rep["rejected_cadences"], 0)

            # 4. Segmented normalization
            norm_lc = segment_and_normalize_lightcurve(
                clean_t, clean_f, clean_fe,
                ephemeris_period_days=3.5,
                ephemeris_epoch_btjd=1901.2,
                ephemeris_duration_hours=2.5,
            )
            self.assertAlmostEqual(float(np.median(norm_lc.flux_raw_norm)), 1.0, places=2)
            self.assertAlmostEqual(float(np.median(norm_lc.flux_detrended)), 1.0, places=2)

            # 5. Extract views
            pri, full, odd, even, stamps = extract_transit_views(
                norm_lc.time, norm_lc.flux_detrended, norm_lc.flux_err_norm,
                period_days=3.5, epoch_btjd=1901.2, duration_hours=2.5,
            )
            self.assertEqual(len(pri.phase), 200)
            self.assertEqual(len(full.phase), 500)
            self.assertGreater(len(stamps), 3)

    def test_mast_fetch_explicit_offline_mode(self):
        """Explicit offline mode produces synthetic fixture with zero network interaction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("lightkurve.search_lightcurve") as mock_search:
                res = fetch_tess_lightcurve(
                    tic_id=12345678,
                    output_dir=tmp_dir,
                    sector=1,
                    offline_mode=True,
                )
                # Network function should never be called
                mock_search.assert_not_called()
                self.assertTrue(res["is_synthetic_fixture"])
                self.assertEqual(res["retrieval_mode"], "EXPLICIT_OFFLINE_FIXTURE")
                self.assertTrue(is_fits_synthetic_fixture(res["product_file"]))

    def test_mast_fetch_online_success_mocked(self):
        """Online mode with successful Lightkurve download marks official product."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "source_mock"
            source_dir.mkdir(parents=True)
            mock_fits = _make_mock_spoc_fits(source_dir / "official_spoc.fits", tic_id=390651552, sector=23)

            mock_sr = MagicMock()
            mock_sr.__len__.return_value = 1
            mock_lc = MagicMock()
            mock_lc.filename = str(mock_fits)
            mock_sr.download.return_value = mock_lc

            with patch("lightkurve.search_lightcurve", return_value=mock_sr) as mock_search:
                res = fetch_tess_lightcurve(
                    tic_id=390651552,
                    output_dir=Path(tmp_dir) / "dest",
                    sector=23,
                    offline_mode=False,
                    fixture_if_offline=False,
                )
                mock_search.assert_called_once()
                self.assertFalse(res["is_synthetic_fixture"])
                self.assertEqual(res["retrieval_mode"], "ONLINE_MAST")
                self.assertFalse(is_fits_synthetic_fixture(res["product_file"]))

    def test_mast_fetch_online_no_result_fallback_and_error(self):
        """Online mode when Lightkurve finds no products: fails when fallback disabled, creates fixture when enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_sr = MagicMock()
            empty_sr.__len__.return_value = 0

            with patch("lightkurve.search_lightcurve", return_value=empty_sr):
                # Fallback disabled -> RuntimeError
                with self.assertRaises(RuntimeError) as ctx:
                    fetch_tess_lightcurve(
                        tic_id=99999999,
                        output_dir=tmp_dir,
                        sector=99,
                        offline_mode=False,
                        fixture_if_offline=False,
                    )
                self.assertTrue(
                    "No SPOC 2-min light curve found" in str(ctx.exception)
                    or "No official 120-second cadence" in str(ctx.exception)
                )

                # Fallback enabled -> FALLBACK_OFFLINE_FIXTURE
                res = fetch_tess_lightcurve(
                    tic_id=99999999,
                    output_dir=tmp_dir,
                    sector=99,
                    offline_mode=False,
                    fixture_if_offline=True,
                )
                self.assertTrue(res["is_synthetic_fixture"])
                self.assertEqual(res["retrieval_mode"], "FALLBACK_OFFLINE_FIXTURE")
                self.assertTrue(is_fits_synthetic_fixture(res["product_file"]))

    def test_mast_fetch_online_error_fallback_and_error(self):
        """Online mode when network/service raises exception: fails when fallback disabled, creates fixture when enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("lightkurve.search_lightcurve", side_effect=ConnectionError("MAST connection timed out")):
                # Fallback disabled -> RuntimeError
                with self.assertRaises(RuntimeError) as ctx:
                    fetch_tess_lightcurve(
                        tic_id=99999999,
                        output_dir=tmp_dir,
                        sector=99,
                        offline_mode=False,
                        fixture_if_offline=False,
                    )
                self.assertTrue(
                    "MAST query/download failed" in str(ctx.exception)
                    or "MAST query/download" in str(ctx.exception)
                )

                # Fallback enabled -> FALLBACK_OFFLINE_FIXTURE
                res = fetch_tess_lightcurve(
                    tic_id=99999999,
                    output_dir=tmp_dir,
                    sector=99,
                    offline_mode=False,
                    fixture_if_offline=True,
                )
                self.assertTrue(res["is_synthetic_fixture"])
                self.assertEqual(res["retrieval_mode"], "FALLBACK_OFFLINE_FIXTURE")
                self.assertTrue(is_fits_synthetic_fixture(res["product_file"]))

    def test_mast_fetch_online_wrong_cadence_rejected(self):
        """When MAST provides a 20s or non-120s product, it must be rejected and not mislabeled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "source_mock"
            source_dir.mkdir(parents=True)
            mock_fits_20s = _make_mock_spoc_fits(
                source_dir / "spoc_20s.fits",
                tic_id=390651552,
                sector=23,
                cadence_seconds=20.0,
            )

            mock_sr = MagicMock()
            mock_sr.__len__.return_value = 1
            mock_lc = MagicMock()
            mock_lc.filename = str(mock_fits_20s)
            mock_sr.download.return_value = mock_lc

            with patch("lightkurve.search_lightcurve", return_value=mock_sr):
                # When fallback is disabled -> must fail clearly with cadence mismatch
                with self.assertRaises(RuntimeError) as ctx:
                    fetch_tess_lightcurve(
                        tic_id=390651552,
                        output_dir=Path(tmp_dir) / "dest",
                        sector=23,
                        offline_mode=False,
                        fixture_if_offline=False,
                    )
                self.assertIn("FITS cadence mismatch", str(ctx.exception))

                # When fallback is enabled -> falls back to ~120s synthetic fixture
                res = fetch_tess_lightcurve(
                    tic_id=390651552,
                    output_dir=Path(tmp_dir) / "dest",
                    sector=23,
                    offline_mode=False,
                    fixture_if_offline=True,
                )
                self.assertTrue(res["is_synthetic_fixture"])
                self.assertEqual(res["retrieval_mode"], "FALLBACK_OFFLINE_FIXTURE")
                self.assertAlmostEqual(res["cadence_seconds"], 120.0, delta=10.0)

    def test_mast_fetch_online_exposure_20s_timedel_120s_passes(self):
        """Official SPOC FITS with EXPOSURE~20.5s but TIMEDEL=120s (and ~120s row spacing) must pass and manifest ~120s."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "source_mock"
            source_dir.mkdir(parents=True)
            mock_fits = _make_mock_spoc_fits(
                source_dir / "official_spoc_20s_exp.fits",
                tic_id=390651552,
                sector=23,
                cadence_seconds=120.0,
                exposure_seconds=20.5346,
            )

            mock_sr = MagicMock()
            mock_sr.__len__.return_value = 1
            mock_lc = MagicMock()
            mock_lc.filename = str(mock_fits)
            mock_sr.download.return_value = mock_lc

            with patch("lightkurve.search_lightcurve", return_value=mock_sr):
                res = fetch_tess_lightcurve(
                    tic_id=390651552,
                    output_dir=Path(tmp_dir) / "dest",
                    sector=23,
                    offline_mode=False,
                    fixture_if_offline=False,
                )
                self.assertFalse(res["is_synthetic_fixture"])
                self.assertEqual(res["retrieval_mode"], "ONLINE_MAST")
                self.assertAlmostEqual(res["cadence_seconds"], 120.0, delta=1.0)
                self.assertEqual(res["manifest"]["files"][0]["cadence_seconds"], 120.0)


if __name__ == "__main__":
    unittest.main()
