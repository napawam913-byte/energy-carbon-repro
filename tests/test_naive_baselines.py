"""模块用途：以手算与合成数据验证朴素基线、原单位指标和仅验证集输出。"""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from test_forecast_preparation import fixture, module, ROOT


def baseline_module():
    return importlib.import_module("energy_forecast.baselines")


class BaselineTests(unittest.TestCase):
    def test_persistence_and_long_seasonal_forecast(self):
        m = baseline_module()
        history = np.arange(60.0).reshape(30, 2)
        np.testing.assert_array_equal(m.predict("persistence", history, 50), np.repeat(history[-1:], 50, axis=0))
        np.testing.assert_array_equal(m.predict("seasonal24", history, 50), history[-24:][np.arange(50) % 24])
        for method, values, horizon in (("bad", history, 6), ("seasonal24", history[:23], 6),
                                         ("persistence", history, 0), ("persistence", history, True),
                                         ("persistence", history * np.nan, 6)):
            with self.subTest(method=method, horizon=horizon), self.assertRaises(ValueError):
                m.predict(method, values, horizon)

    def test_hand_calculated_metrics(self):
        m = baseline_module()
        result = m.metrics(np.array([[[1.0], [3.0]]]), np.array([[[2.0], [1.0]]]), ["factor_generated_kg_per_mwh"])
        scores = result["per_target"]["factor_generated_kg_per_mwh"]
        self.assertEqual(scores["mse"], 2.5)
        self.assertEqual(scores["mae"], 1.5)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(2.5))
        self.assertEqual(scores["unit"], "kg CO2/MWh")
        self.assertEqual(result["per_horizon"]["2"]["factor_generated_kg_per_mwh"]["mae"], 2.0)
        self.assertNotIn("overall", result)
        with self.assertRaises(ValueError):
            m.metrics(np.ones((2, 3, 1)), np.ones((2, 2, 1)), ["generation_wind_mwh"])

    def test_future_cannot_change_baseline_predictions(self):
        f, c = fixture()
        first = module().ForecastData(f, c)
        f.loc[40:, c["input_columns"]] += 99999.0
        second = module().ForecastData(f, c)
        for name in ("persistence", "seasonal24"):
            np.testing.assert_array_equal(baseline_module().predict(name, first.history(40), 6),
                                          baseline_module().predict(name, second.history(40), 6))

    def test_cli_validation_only_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f, c = fixture()
            source = root / c["source"]
            source.parent.mkdir(parents=True)
            f.to_csv(source, index=False)
            prepared, output = root / "prepared", root / "validation"
            module().prepare(root, c, prepared)
            before = source.read_bytes()
            command = [sys.executable, "-X", "utf8", str(ROOT / "run_naive_baselines.py"), "--prepared", str(prepared), "--output", str(output)]
            result = subprocess.run(command, cwd=temp, capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["split"], "val")
            self.assertEqual(saved["status"], "VALIDATION_BASELINES_ONLY")
            self.assertEqual(set(saved["models"]), {"persistence", "seasonal24"})
            self.assertEqual(saved["models"]["persistence"]["origins"], 20)
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "per_horizon_metrics.csv").is_file())
            self.assertFalse(any("test" in p.name for p in output.iterdir()))
            original = (output / "metrics.json").read_bytes()
            again = subprocess.run(command, cwd=temp, capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((output / "metrics.json").read_bytes(), original)
            self.assertEqual(source.read_bytes(), before)

    def test_changing_test_targets_does_not_change_validation_metrics(self):
        m = baseline_module()
        with tempfile.TemporaryDirectory() as temp:
            scores = []
            for changed in (False, True):
                root = Path(temp) / str(changed)
                f, c = fixture()
                if changed:
                    f.loc[65:, c["input_columns"]] += 10000.0
                source = root / c["source"]
                source.parent.mkdir(parents=True)
                f.to_csv(source, index=False)
                module().prepare(root, c, root / "prepared")
                report = m.run_validation(root / "prepared", root / "validation")
                scores.append(report["models"])
            self.assertEqual(*scores)


if __name__ == "__main__":
    unittest.main()
