"""
Unit tests for upload validation, secure storage, and FITS audit.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
from astropy.io import fits

from astrovet.lightcurves.validator import (
    sanitize_filename,
    validate_upload_bytes,
    store_uploaded_lightcurve,
    audit_lightcurve_fits,
    MAX_UPLOAD_SIZE_BYTES,
    MINIMUM_USABLE_CADENCES,
)
from astrovet.lightcurves.reader import read_tess_fits_lightcurve


class TestUploadValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sanitize_filename_traversal(self):
        self.assertEqual(sanitize_filename("../../etc/passwd.fits"), "passwd.fits")
        self.assertEqual(sanitize_filename("..\\..\\windows\\system32.fits"), "system32.fits")
        self.assertEqual(sanitize_filename("path/to/my_file.fits"), "my_file.fits")

    def test_sanitize_filename_special_chars(self):
        self.assertEqual(
            sanitize_filename("test file (1) [sector=23]!@#.fits"),
            "test_file__1___sector_23____.fits",
        )
        self.assertEqual(sanitize_filename("...leading_dots.fits"), "leading_dots.fits")
        self.assertEqual(sanitize_filename("a....b.fits"), "a.b.fits")

    def test_sanitize_filename_empty(self):
        self.assertEqual(sanitize_filename(""), "uploaded_lightcurve.fits")
        self.assertEqual(sanitize_filename("..."), "uploaded_lightcurve.fits")

    def test_validate_upload_bytes_empty(self):
        with self.assertRaises(ValueError) as ctx:
            validate_upload_bytes(b"", "test.fits")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_validate_upload_bytes_oversize(self):
        with self.assertRaises(ValueError) as ctx:
            validate_upload_bytes(b"x" * 1024, "test.fits", max_size=500)
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))

    def test_validate_upload_bytes_bad_extension(self):
        with self.assertRaises(ValueError) as ctx:
            validate_upload_bytes(b"data", "test.csv")
        self.assertIn("Invalid file extension", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_upload_bytes(b"data", "test.exe")
        self.assertIn("Invalid file extension", str(ctx.exception))

    def test_validate_upload_bytes_fits_magic(self):
        bad_fits = b"NOT_A_FITS_HEADER" + b" " * 100
        with self.assertRaises(ValueError) as ctx:
            validate_upload_bytes(bad_fits, "test.fits")
        self.assertIn("lacks standard FITS magic header", str(ctx.exception))

        # Valid FITS magic header
        valid_fits = b"SIMPLE  =                    T / file conforms to FITS standard" + b" " * 2000
        validate_upload_bytes(valid_fits, "valid.fits")

    def test_validate_upload_bytes_gzip_magic(self):
        bad_gz = b"SIMPLE  = " + b" " * 100
        with self.assertRaises(ValueError) as ctx:
            validate_upload_bytes(bad_gz, "test.fits.gz")
        self.assertIn("lacks valid gzip magic header", str(ctx.exception))

        # Gzip magic is 0x1f, 0x8b
        valid_gz = bytes([0x1F, 0x8B]) + b"\x08\x00" + b"\x00" * 50
        validate_upload_bytes(valid_gz, "valid.fits.gz")

    def test_store_uploaded_lightcurve(self):
        # Create minimal valid FITS bytes
        import io
        hdr = fits.Header()
        hdr["SIMPLE"] = True
        hdr["BITPIX"] = 8
        hdr["NAXIS"] = 0
        hdu = fits.PrimaryHDU(header=hdr)
        bio = io.BytesIO()
        hdu.writeto(bio)
        fits_bytes = bio.getvalue()

        stored_path, sha_hex = store_uploaded_lightcurve(
            data=fits_bytes,
            filename="../../unsafe/my_tess_lc.fits",
            upload_dir=self.upload_dir,
        )

        self.assertTrue(stored_path.is_file())
        self.assertEqual(stored_path.parent, self.upload_dir)
        self.assertIn("my_tess_lc.fits", stored_path.name)
        self.assertNotIn("unsafe", stored_path.name)
        # Check bytes preserved unchanged
        self.assertEqual(stored_path.read_bytes(), fits_bytes)
        self.assertEqual(len(sha_hex), 64)

    def test_content_addressed_astronet_filename_still_detects_qlp(self):
        """Upload prefixes must not hide QLP origin encoded by the product name."""
        time = np.arange(100, dtype=float) * 0.00694 + 2200.0
        flux = np.ones(100, dtype=float)
        cols = [
            fits.Column(name="TIME", format="D", array=time),
            fits.Column(name="SAP_FLUX", format="D", array=flux),
            fits.Column(name="SAP_FLUX_ERR", format="D", array=np.full(100, 0.001)),
            fits.Column(name="QUALITY", format="J", array=np.zeros(100, dtype=np.int32)),
        ]
        primary = fits.PrimaryHDU()
        primary.header["TELESCOP"] = "TESS"
        primary.header["TICID"] = 183978641
        primary.header["SECTOR"] = 39
        table = fits.BinTableHDU.from_columns(cols)
        path = self.upload_dir / "abcdef123456_20261029_astronet_tess_ffi_s0039_test.fits"
        fits.HDUList([primary, table]).writeto(path)
        raw = read_tess_fits_lightcurve(path)
        self.assertEqual(raw.pipeline_origin, "TESS_QLP")

    def test_audit_lightcurve_fits_real_data(self):
        real_fits = Path("data/raw/mast/tess_s0023_390651552_spoc_lc.fits")
        if not real_fits.is_file():
            self.skipTest("Real MAST FITS fixture not found.")

        audit = audit_lightcurve_fits(real_fits)
        self.assertTrue(audit["is_valid"])
        self.assertEqual(audit["pipeline_origin"], "TESS_SPOC")
        self.assertEqual(audit["metadata"]["tic_id"], 390651552)
        self.assertEqual(audit["metadata"]["sector"], 23)
        self.assertIn("cadence_audit", audit)
        self.assertGreater(audit["cadence_audit"]["usable_cadences"], 1000)
        self.assertIn("segments", audit)
        self.assertGreater(len(audit["segments"]), 0)
        self.assertIn("processing_policy", audit)
        self.assertIn("sha256", audit)

    def test_audit_lightcurve_fits_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            audit_lightcurve_fits(self.upload_dir / "nonexistent.fits")


if __name__ == "__main__":
    unittest.main()
