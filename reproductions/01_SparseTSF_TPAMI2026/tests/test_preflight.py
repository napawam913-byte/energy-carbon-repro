"""模块用途：测试准备脚本的目录定位和数据校验；不运行训练，不需要 GPU。"""

import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta


PAPER = Path(__file__).resolve().parents[1]
SCRIPTS = PAPER / "scripts"
HEADER = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


class PreflightTests(unittest.TestCase):
    def run_script(self, name, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPTS / name), *map(str, args)],
            cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=30,
        )

    def test_paths_are_independent_of_current_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "paper"
            (fixture / "scripts").mkdir(parents=True)
            (fixture / "code" / "models").mkdir(parents=True)
            (fixture / "code" / "layers").mkdir(parents=True)
            # 这里只测试路径，不执行模拟的作者文件，也不要求下载作者仓库。
            expected = fixture / "code" / "models" / "SparseTSF.py"
            expected.write_text("# path fixture only", encoding="utf-8")
            (fixture / "code" / "layers" / "Embed.py").write_text("# path fixture only", encoding="utf-8")
            script = fixture / "scripts" / "check_gpu.py"
            shutil.copyfile(SCRIPTS / "check_gpu.py", script)
            result = self.run_script(script, "--paths-only", cwd=folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(expected), result.stdout)
        self.assertNotIn("GPU 前向与反向计算通过", result.stdout)

    def test_missing_code_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_script("check_gpu.py", "--code-dir", folder, "--paths-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("models/SparseTSF.py", result.stderr)

    def test_help_does_not_import_torch_or_install_packages(self):
        for name in ("check_gpu.py", "check_environment.py", "prepare_etth1.py"):
            with self.subTest(script=name):
                result = self.run_script(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_missing_data_check_only_does_not_download(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "ETTh1.csv"
            result = self.run_script("prepare_etth1.py", "--path", target, "--check-only")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())
            self.assertIn("文件不存在", result.stderr)

    def test_invalid_existing_data_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "ETTh1.csv"
            target.write_text("<html>download failed</html>", encoding="utf-8")
            result = self.run_script("prepare_etth1.py", "--path", target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("表头", result.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), "<html>download failed</html>")

    def test_valid_synthetic_structure_reports_hash_and_preserves_file(self):
        # 合成数值仅测试结构，不代表真实 ETTh1 或论文实测数据。
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "ETTh1.csv"
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                start = datetime(2016, 7, 1)
                for hour in range(17420):
                    writer.writerow([str(start + timedelta(hours=hour))] + [1] * 7)
            checksum = hashlib.sha256(target.read_bytes()).hexdigest()
            result = self.run_script("prepare_etth1.py", "--path", target, "--check-only")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(checksum, result.stdout)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), checksum)

    def test_duplicate_hour_is_rejected(self):
        self.check_bad_rows([
            ["2016-07-01 00:00:00"] + [1] * 7,
            ["2016-07-01 00:00:00"] + [1] * 7,
        ], "小时不连续")

    def test_nan_is_rejected(self):
        self.check_bad_rows([["2016-07-01 00:00:00", "nan"] + [1] * 6], "非有限")

    def check_bad_rows(self, rows, message):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "ETTh1.csv"
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                writer.writerows(rows)
            result = self.run_script("prepare_etth1.py", "--path", target, "--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
