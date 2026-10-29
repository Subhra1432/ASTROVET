import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from astrovet.baseline.container import ExoMinerContainerRunner, EXOMINER_DEFAULT_IMAGE
from astrovet.baseline.adapter import ExoMinerBaselineAdapter
from astrovet.schemas.baseline import BaselinePrediction


class TestExoMinerContainer(unittest.TestCase):
    def setUp(self):
        self.runner = ExoMinerContainerRunner(image_name=EXOMINER_DEFAULT_IMAGE, engine="podman")

    def test_build_command(self):
        cmd = self.runner.build_command(
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            tic_id=390651552,
            sector=23,
            period=1.467,
            epoch=1928.225,
            duration_hours=1.008,
        )
        self.assertEqual(cmd[0], "podman")
        self.assertTrue(any(arg.startswith("ghcr.io/nasa/exominer@sha256:") for arg in cmd))
        self.assertIn("--tic_ids", cmd)
        self.assertIn("390651552_23", cmd)
        self.assertIn("--task=phot-vetting", cmd)
        self.assertIn("--exominer_model=single", cmd)
        self.assertIn("--output_dir=/workspace/output", cmd)
        self.assertNotIn("--period", cmd)  # the official runner obtains ephemerides from SPOC DV

    def test_engine_missing_structured_status(self):
        """Verify clear unavailable status and instructions when no daemon exists."""
        runner_no_engine = ExoMinerContainerRunner(engine=None)
        runner_no_engine.engine = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            pred = runner_no_engine.run(
                candidate_id="390651552-1-S23",
                work_dir=tmp_dir,
                fallback_adapter=None,
            )
            self.assertEqual(pred.status, "CONTAINER_RUNTIME_UNAVAILABLE")
            self.assertEqual(pred.execution_mode, "CONTAINER_PROCESS")
            self.assertIsNone(pred.score)
            self.assertIn("attempted_command", pred.raw_output)

    @patch("subprocess.run")
    def test_container_success_mocked(self, mock_run):
        """Verify parsing of container prediction output when container executes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "predictions_output.csv").write_text(
                "uid,score_PC,score_std_PC,score_AFP,score_NTP,prediction\n"
                "390651552-1-S23,0.892,0.018,0.051,0.057,PC\n",
                encoding="utf-8",
            )

            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_run.return_value = mock_res

            pred = self.runner.run(
                candidate_id="390651552-1-S23",
                work_dir=tmp_dir,
            )
            self.assertEqual(pred.status, "SUCCESS")
            self.assertEqual(pred.execution_mode, "CONTAINER_PROCESS")
            self.assertEqual(pred.score, 0.892)
            self.assertFalse(pred.is_synthetic)

    def test_adapter_container_fallback_to_catalog(self):
        """Verify adapter predict_via_container falls back gracefully to catalog when container missing."""
        adapter = ExoMinerBaselineAdapter(is_synthetic=True)
        # Force runner to have no engine
        with patch.object(ExoMinerContainerRunner, "_detect_engine", return_value=None):
            pred = adapter.predict_via_container("390651552-1-S23", fallback_to_catalog=True)
            self.assertEqual(pred.status, "SUCCESS")
            self.assertEqual(pred.score, 0.88)
            self.assertIn("container_notice", pred.raw_output)


if __name__ == "__main__":
    unittest.main()
