import tempfile
import unittest
from pathlib import Path
import pandas as pd

from astrovet.catalogs.audit import audit_catalog_csv


class TestCatalogAudit(unittest.TestCase):
    def test_missing_file_handling(self):
        report = audit_catalog_csv("non_existent_file.csv")
        self.assertEqual(report["status"], "FILE_NOT_FOUND")

    def test_audit_csv_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_catalog.csv"
            out_json = Path(tmp_dir) / "audit_report.json"

            df = pd.DataFrame({
                "TIC ID": [1001, 1001, 1002, 1003],
                "TCE ID": ["1001-1-S1", "1001-2-S1", "1002-1-S1", "1003-1-S1"],
                "Period": [1.5, 3.2, 5.0, None],
                "label": ["PC", "FP", "CP", "UNK"],
            })
            df.to_csv(csv_path, index=False)

            report = audit_catalog_csv(csv_path, output_report_path=out_json)
            self.assertEqual(report["status"], "SUCCESS")
            self.assertEqual(report["row_count"], 4)
            self.assertEqual(report["column_count"], 4)
            self.assertEqual(report["identifiers"]["unique_tics"], 3)
            self.assertEqual(report["identifiers"]["unique_tces"], 4)
            self.assertEqual(report["identifiers"]["stars_with_multiple_candidates"], 1)
            self.assertFalse(report["identifiers"]["tic_alone_is_unique_event"])
            self.assertTrue(out_json.exists())


if __name__ == "__main__":
    unittest.main()
