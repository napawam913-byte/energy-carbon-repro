"""模块用途：验证 VS Code 训练入口的参数、路径和日志；仅运行轻量测试子进程。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PAPER = Path(__file__).resolve().parents[1]
LAUNCHER = PAPER / "train_etth1_linear.py"


class LauncherTests(unittest.TestCase):
    def load_launcher(self):
        self.assertTrue(LAUNCHER.is_file(), "缺少可直接运行的 train_etth1_linear.py")
        spec = importlib.util.spec_from_file_location("training_launcher", LAUNCHER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def make_code(self, folder):
        folder.mkdir(parents=True)
        # 路径夹具不包含真实模型；不会把它作为训练程序启动。
        (folder / "run_longExp.py").write_text("# path fixture only", encoding="utf-8")
        return folder

    def test_command_matches_successful_single_run(self):
        launcher = self.load_launcher()
        code = Path(tempfile.gettempdir()) / "upstream"
        command = launcher.build_command(code)
        self.assertEqual(command[:3], [sys.executable, "-u", str(code / "run_longExp.py")])
        args = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(args, {
            "--is_training": "1", "--model_id": "ETTh1_720_96", "--model": "SparseTSF",
            "--model_type": "linear", "--data": "ETTh1", "--root_path": str(code / "dataset"),
            "--data_path": "ETTh1.csv", "--features": "M", "--seq_len": "720", "--pred_len": "96",
            "--period_len": "24", "--enc_in": "7", "--train_epochs": "30", "--patience": "5",
            "--itr": "1", "--batch_size": "256", "--learning_rate": "0.02", "--loss": "mse",
            "--lradj": "type3", "--num_workers": "10", "--gpu": "0", "--checkpoints": "./checkpoints/",
        })

    def test_find_existing_sibling_clone(self):
        launcher = self.load_launcher()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paper = root / "energy-carbon-repro" / "reproductions" / PAPER.name
            legacy = self.make_code(root / "reproductions" / PAPER.name / "code")
            self.assertEqual(launcher.find_code_dir(paper), legacy.resolve())

    def test_same_paper_clone_takes_priority(self):
        launcher = self.load_launcher()
        with tempfile.TemporaryDirectory() as folder:
            paper = Path(folder) / "repo" / "reproductions" / PAPER.name
            code = self.make_code(paper / "code")
            self.make_code(Path(folder) / "reproductions" / PAPER.name / "code")
            self.assertEqual(launcher.find_code_dir(paper), code.resolve())

    def test_missing_override_does_not_fall_back_silently(self):
        launcher = self.load_launcher()
        with tempfile.TemporaryDirectory() as folder:
            paper = Path(folder) / "repo" / "reproductions" / PAPER.name
            self.make_code(paper / "code")
            with self.assertRaisesRegex(FileNotFoundError, "run_longExp.py"):
                launcher.find_code_dir(paper, Path(folder) / "missing")

    def test_dry_run_works_from_another_directory_without_writes(self):
        self.load_launcher()
        with tempfile.TemporaryDirectory() as folder:
            code = self.make_code(Path(folder) / "upstream")
            result = subprocess.run(
                [sys.executable, "-B", str(LAUNCHER), "--code-dir", str(code), "--dry-run"],
                cwd=folder, capture_output=True, text=True, encoding="utf-8", timeout=30,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["status"], "PREVIEW_ONLY_NOT_TRAINED")
            self.assertIn("--model_type", plan["command"])
            self.assertEqual(list(code.iterdir()), [code / "run_longExp.py"])

    def test_success_logs_and_fresh_output_directories(self):
        launcher = self.load_launcher()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "runs"
            command = [sys.executable, "-u", "-c", "print('fixture success, NOT model training')"]
            first, exit_code = launcher.execute_training(command, root, {"fixture.txt": "test only"})
            second, _ = launcher.execute_training(command, root, {})
            self.assertEqual(exit_code, 0)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "fixture.txt").read_text(encoding="utf-8"), "test only")
            self.assertIn("fixture success", (first / "train.log").read_text(encoding="utf-8"))
            record = json.loads((first / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["command"], command)
            self.assertEqual(record["exit_code"], 0)
            self.assertEqual(record["status"], "COMPLETED")
            self.assertEqual((first / "exitcode.txt").read_text().strip(), "0")

    def test_failure_is_recorded_and_not_retried(self):
        launcher = self.load_launcher()
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "-u", "-c", "import sys; print('fixture error', file=sys.stderr); sys.exit(7)"]
            run, code = launcher.execute_training(command, Path(folder) / "runs", {})
            self.assertEqual(code, 7)
            self.assertIn("fixture error", (run / "train.log").read_text(encoding="utf-8"))
            record = json.loads((run / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["exit_code"], 7)
            self.assertEqual(len(list(run.parent.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
