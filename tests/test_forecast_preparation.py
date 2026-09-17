"""模块用途：以合成小时序列检查预测窗口、时间边界与清单；不是实测预测结果。"""

import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def module():
    return importlib.import_module("energy_forecast.data")


def fixture(n=90):
    m = module()
    config = copy.deepcopy(m.DEFAULT_CONFIG)
    config.update(lookback=24, horizon=6, train_end="2023-01-02T16:00:00Z", val_end="2023-01-03T17:00:00Z")
    frame = pd.DataFrame({"timestamp_utc": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC").astype(str)})
    for k, name in enumerate(config["input_columns"]):
        frame[name] = np.arange(n, dtype=float) + k
    frame["factor_solar_kg_per_mwh"] = np.nan
    return frame, config


class PreparationTests(unittest.TestCase):
    def test_exact_windows_and_boundaries(self):
        frame, config = fixture()
        data = module().ForecastData(frame, config)
        self.assertEqual({k: len(v) for k, v in data.origins.items()}, {"train": 11, "val": 20, "test": 20})
        x, y = data.sample(40, raw=True)
        np.testing.assert_array_equal(x, frame[config["input_columns"]].iloc[16:40])
        np.testing.assert_array_equal(y, frame[config["target_columns"]].iloc[40:46])
        for j in (23, 35, 39, 60, 64, 85, True):
            with self.subTest(j=j), self.assertRaises(ValueError):
                data.sample(j)

    def test_full_year_counts_shapes_and_last_hour(self):
        m = module()
        config = copy.deepcopy(m.DEFAULT_CONFIG)
        frame = pd.DataFrame({"timestamp_utc": pd.date_range("2023-01-01 06:00", periods=8760, freq="h", tz="UTC").astype(str)})
        for name in config["input_columns"]:
            frame[name] = 1.0
        data = m.ForecastData(frame, config)
        self.assertEqual([data.splits[k]["rows"] for k in ("train", "val", "test")], [5826, 1464, 1470])
        self.assertEqual([len(data.origins[k]) for k in ("train", "val", "test")], [5635, 1441, 1447])
        self.assertEqual(data.sample(int(data.origins["test"][-1]))[1].shape, (24, 10))
        self.assertEqual(data.sample(168)[0].shape, (168, 11))
        self.assertEqual(data.splits["test"]["end_exclusive"], "2024-01-01T06:00:00+00:00")
        self.assertTrue(all(data.scalers["input"]["constant"]))

    def test_future_changes_do_not_change_scalers_or_history(self):
        frame, config = fixture()
        first = module().ForecastData(frame, config)
        frame.loc[40:, config["input_columns"]] += 100000.0
        second = module().ForecastData(frame, config)
        self.assertEqual(first.scalers, second.scalers)
        np.testing.assert_array_equal(first.sample(40)[0], second.sample(40)[0])
        np.testing.assert_array_equal(first.history(40), second.history(40))

    def test_invalid_inputs_and_config(self):
        bad_frames = {
            "missing_hour": lambda f: f.drop(index=10),
            "duplicate_hour": lambda f: pd.concat([f.iloc[:1], f]),
            "reverse": lambda f: f.iloc[::-1],
            "naive": lambda f: f.assign(timestamp_utc=pd.to_datetime(f.timestamp_utc).dt.tz_localize(None).astype(str)),
            "half_hour": lambda f: f.assign(timestamp_utc=(pd.to_datetime(f.timestamp_utc) + pd.Timedelta(minutes=30)).astype(str)),
            "nonfinite": lambda f: f.assign(demand_mw=np.inf),
            "missing_field": lambda f: f.drop(columns="demand_mw"),
            "duplicate_column": lambda f: pd.concat([f, f[["demand_mw"]]], axis=1),
        }
        for name, mutate in bad_frames.items():
            with self.subTest(name=name):
                f, c = fixture()
                with self.assertRaises(ValueError):
                    module().ForecastData(mutate(f), c)
        for update in ({"lookback": 0}, {"stride": True}, {"horizon": 6.5}, {"lookback": 1000},
                       {"val_end": "2023-01-01T00:00:00Z"}, {"train_end": "2023-01-02"},
                       {"input_columns": ["reference_consumed_factor_kg_per_mwh"]},
                       {"target_columns": ["demand_mw"]}, {"source": "../raw.csv"}, {"unknown": 1}):
            with self.subTest(update=update):
                f, c = fixture()
                c.update(update)
                with self.assertRaises(ValueError):
                    module().ForecastData(f, c)

    def test_saved_scalers_are_used_without_refitting(self):
        f, c = fixture()
        d = module().ForecastData(f, c)
        saved = copy.deepcopy(d.scalers)
        saved["input"]["mean"] = [7.0] * 11
        reused = module().ForecastData(f, c, saved)
        self.assertEqual(reused.scalers, saved)
        saved["input"]["scale"][0] = 0
        with self.assertRaises(ValueError):
            module().ForecastData(f, c, saved)

    def test_manifest_source_integrity_and_no_overwrite(self):
        f, c = fixture()
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / c["source"]
            source.parent.mkdir(parents=True)
            f.to_csv(source, index=False)
            digest = m.sha256(source)
            output = root / "prepared"
            manifest = m.prepare(root, c, output)
            data, loaded = m.load_prepared(output)
            self.assertEqual(manifest, loaded)
            self.assertEqual(manifest["source_sha256"], digest)
            self.assertEqual(m.sha256(source), digest)
            self.assertEqual(len(data.origins["val"]), 20)
            with self.assertRaises(FileExistsError):
                m.prepare(root, c, output)
            path = output / "manifest.json"
            bad = copy.deepcopy(manifest)
            bad["input_columns"].reverse()
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                m.load_prepared(output)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                m.load_prepared(output)

    def test_cli_independent_of_cwd_and_does_not_import_torch(self):
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f, c = fixture()
            source = root / c["source"]
            source.parent.mkdir(parents=True)
            f.to_csv(source, index=False)
            config = root / "config.json"
            config.write_text(json.dumps(c), encoding="utf-8")
            result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "prepare_forecast_data.py"),
                                     "--root", str(root), "--config", str(config), "--output", str(root / "prepared")],
                                    cwd=temp, capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("train=11 val=20 test=20", result.stdout)
            self.assertIn("no training", result.stdout)
        check = subprocess.run([sys.executable, "-c", "import sys; import energy_forecast.data; assert 'torch' not in sys.modules"], cwd=ROOT, capture_output=True)
        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
