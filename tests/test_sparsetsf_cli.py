"""模块用途：检验服务器入口、路径错误与超时监督；不启动真实训练或 GPU。"""

import importlib.util
from contextlib import redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from energy_forecast.data import PROJECT_ROOT, read_json

ENTRY = PROJECT_ROOT / "reproductions/01_SparseTSF_TPAMI2026/train_erco.py"


class CliTests(unittest.TestCase):
    def entry(self):
        self.assertTrue(ENTRY.is_file(), "ERCO CLI is not implemented")
        spec = importlib.util.spec_from_file_location("erco_training_entry_test", ENTRY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help_from_unrelated_cwd(self):
        self.entry()
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, "-X", "utf8", str(ENTRY), "--help"], cwd=temp,
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--check-only", result.stdout)
            self.assertNotIn("--test", result.stdout)
            self.assertFalse(list(Path(temp).iterdir()))

    def test_bad_source_fails_before_cuda_or_outputs(self):
        self.entry()
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, "-X", "utf8", str(ENTRY), "--check-only", "--code-dir", str(Path(temp) / "absent")],
                                    cwd=temp, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing author source", result.stdout + result.stderr)
            self.assertFalse(list(Path(temp).iterdir()))

    def test_supervisor_timeout_and_failure_are_not_success_or_retried(self):
        m = self.entry()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "timeout"
            captured = io.StringIO()
            with redirect_stderr(captured):
                status = m.supervise([sys.executable, "-c", "import time; time.sleep(5)"], .1, output)
            self.assertIn("TIMED_OUT", captured.getvalue())
            self.assertEqual(status, 124)
            self.assertEqual(read_json(output / "termination.json")["status"], "TIMED_OUT")
            self.assertFalse((output / "run.json").exists())
            failure = m.supervise([sys.executable, "-c", "raise SystemExit(7)"], 10, Path(temp) / "failure")
            self.assertEqual(failure, 7)

    def test_output_must_stay_in_experiments_and_be_new(self):
        m = self.entry()
        for path in (PROJECT_ROOT, PROJECT_ROOT / "data/raw", PROJECT_ROOT / "data/experiments"):
            with self.subTest(path=path), self.assertRaises((ValueError, FileExistsError)):
                m.validate_output(path)


if __name__ == "__main__":
    unittest.main()
