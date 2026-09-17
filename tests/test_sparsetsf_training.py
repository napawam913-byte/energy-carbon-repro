"""模块用途：以临时合成数据验收作者模型训练、最佳权重与验证产物，不运行真实数据。"""

import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from energy_forecast.baselines import metrics
from energy_forecast.data import load_prepared, prepare, read_json, sha256
from test_sparsetsf_adapter import CODE, fixture


def module():
    if importlib.util.find_spec("energy_forecast.training") is None:
        raise AssertionError("training module is not implemented")
    return importlib.import_module("energy_forecast.training")


def prepared_fixture(root, change_test=False):
    frame, config = fixture()
    if change_test:
        frame.loc[84:, config["input_columns"]] += 12345
    path = root / config["source"]
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    prepare(root, config, root / "prepared")
    return root / "prepared"


class TrainingContractTests(unittest.TestCase):
    def test_training_module_exists(self):
        module()


@unittest.skipUnless(importlib.util.find_spec("torch") and CODE.is_dir(), "PyTorch and pinned author source required")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def small_config(self):
        config = copy.deepcopy(module().DEFAULT_TRAINING_CONFIG)
        config.update(epochs=2, patience=2, batch_size=8, period_len=3)
        return config

    def test_configuration_and_learning_rate_and_strict_early_stopping(self):
        m = module()
        c = m.validate_training_config({})
        self.assertEqual((c["epochs"], c["batch_size"], c["seed"], c["model_type"]), (30, 128, 2023, "linear"))
        for bad in ({"epochs": 0}, {"seed": -1}, {"seed": 2**32}, {"seed": True}, {"patience": 2.5},
                    {"learning_rate": float("nan")}, {"learning_rate": 0}, {"batch_size": False},
                    {"model_type": "new_network"}, {"unknown": 1}, {"timeout_seconds": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.validate_training_config(bad)
        self.assertEqual([m.learning_rate(.02, e) for e in (1, 2, 3)], [.02, .02, .02])
        self.assertAlmostEqual(m.learning_rate(.02, 4), .016)
        stopper = m.EarlyStopping(2)
        self.assertTrue(stopper.update(.5, 1))
        self.assertFalse(stopper.update(.5, 2))
        self.assertFalse(stopper.stopped)
        self.assertTrue(stopper.update(.4, 3))
        self.assertFalse(stopper.update(.5, 4))
        self.assertFalse(stopper.update(.4, 5))
        self.assertTrue(stopper.stopped)
        self.assertEqual(stopper.best_epoch, 3)
        with self.assertRaises(ValueError):
            stopper.update(float("nan"), 6)

    def test_windows_and_preflight_and_cuda_does_not_fallback(self):
        import torch
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            prepared = prepared_fixture(Path(temp))
            data, _ = load_prepared(prepared)
            for split in ("train", "val"):
                dataset = m.WindowDataset(data, split)
                x, y = dataset[0]
                expected_x, expected_y = data.sample(int(data.origins[split][0]))
                np.testing.assert_allclose(x.numpy(), expected_x, rtol=1e-6)
                np.testing.assert_allclose(y.numpy(), expected_y, rtol=1e-6)
                self.assertEqual(x.dtype, torch.float32)
            with self.assertRaisesRegex(ValueError, "test|train|val"):
                m.WindowDataset(data, "test")
            files_before = sorted(p.relative_to(temp) for p in Path(temp).rglob("*"))
            info = m.preflight(prepared, CODE, self.small_config(), device="cpu")
            self.assertEqual(info["output_shape"], [1, 6, 10])
            self.assertEqual(info["status"], "PREFLIGHT_ONLY_NO_TRAINING")
            self.assertEqual(files_before, sorted(p.relative_to(temp) for p in Path(temp).rglob("*")))
            if not torch.cuda.is_available():
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    m.preflight(prepared, CODE, self.small_config(), device="cuda")

    def test_train_checkpoint_metrics_timestamps_no_overwrite_and_test_independence(self):
        import torch
        from energy_forecast.sparsetsf import build_model
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = prepared_fixture(root / "first")
            data, manifest = load_prepared(prepared)
            config = self.small_config()
            torch.manual_seed(config["seed"])
            initial, _ = build_model(CODE, data, period_len=3)
            initial_state = {k: v.clone() for k, v in initial.state_dict().items()}
            output = root / "run1"
            report = m.train(prepared, CODE, output, config, device="cpu")
            self.assertEqual(report["status"], "COMPLETED_VALIDATION_ONLY")
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(report["source_sha256"], manifest["source_sha256"])
            history = pd.read_csv(output / "history.csv")
            self.assertEqual(history["train_samples"].tolist(), [19, 19])
            self.assertEqual(history["val_samples"].tolist(), [31, 31])
            self.assertEqual(report["best_epoch"], int(history.loc[history["val_mse"].idxmin(), "epoch"]))
            self.assertEqual(report["best_val_mse"], float(history["val_mse"].min()))
            state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertTrue(any(not torch.equal(state[k], initial_state[k]) for k in state))
            with np.load(output / "validation_predictions.npz", allow_pickle=False) as archive:
                saved = {key: archive[key] for key in archive.files}
            predictions, actual = saved["prediction"], saved["actual"]
            self.assertEqual(predictions.shape, (31, 6, 10))
            self.assertEqual(saved["targets"].tolist(), data.config["target_columns"])
            self.assertEqual(saved["origin_utc"][0], data.times[48].isoformat())
            self.assertEqual(saved["target_utc"][-1, -1], data.times[83].isoformat())
            expected_actual = np.stack([data.sample(int(j), raw=True)[1] for j in data.origins["val"]])
            np.testing.assert_array_equal(actual, expected_actual)
            scores = read_json(output / "metrics.json")
            self.assertEqual(scores["metrics"], metrics(actual, predictions, data.config["target_columns"]))
            self.assertEqual(len(pd.read_csv(output / "metrics.csv")), 10)
            self.assertEqual(len(pd.read_csv(output / "per_horizon_metrics.csv")), 60)
            model, _ = build_model(CODE, data, period_len=3)
            model.load_state_dict(state, strict=True)
            x, _ = m.WindowDataset(data, "val")[0]
            with torch.no_grad():
                normalized = model(x[None])[..., list(range(1, 11))].numpy()
            scale = np.asarray(data.scalers["target"]["scale"])
            mean = np.asarray(data.scalers["target"]["mean"])
            np.testing.assert_allclose(predictions[:1], normalized * scale + mean, rtol=1e-6, atol=1e-6)
            digest = sha256(output / "run.json")
            with self.assertRaises(FileExistsError):
                m.train(prepared, CODE, output, config, device="cpu")
            self.assertEqual(digest, sha256(output / "run.json"))
            other = prepared_fixture(root / "second", change_test=True)
            report2 = m.train(other, CODE, root / "run2", config, device="cpu")
            state2 = torch.load(root / "run2/checkpoint.pt", map_location="cpu", weights_only=True)
            for key in state:
                torch.testing.assert_close(state[key], state2[key], rtol=0, atol=0)
            with np.load(root / "run2/validation_predictions.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(predictions, archive["prediction"])
            self.assertEqual(report["best_val_mse"], report2["best_val_mse"])

    def test_bad_source_and_nonfinite_tensor_fail_visibly(self):
        import torch
        m = module()
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                m.finite_tensor(torch.tensor([value]), "test")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = prepared_fixture(root)
            with self.assertRaises(FileNotFoundError):
                m.train(prepared, root / "absent", root / "failed", self.small_config(), device="cpu")
            self.assertEqual(read_json(root / "failed/run.json")["status"], "FAILED")
            self.assertFalse((root / "failed/checkpoint.pt").exists())

    def test_early_stop_restores_earlier_checkpoint_not_last_epoch(self):
        import torch
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = prepared_fixture(root / "data")
            config = self.small_config()
            config.update(epochs=30, patience=2)
            report = m.train(prepared, CODE, root / "stopped", config, device="cpu")
            self.assertTrue(report["early_stopped"])
            self.assertLess(report["best_epoch"], report["epochs_completed"])
            # 相同随机状态只训练到最佳轮；最终权重必须等于早停运行保存的最佳权重。
            replay_config = dict(config, epochs=report["best_epoch"])
            m.train(prepared, CODE, root / "replay", replay_config, device="cpu")
            best = torch.load(root / "stopped/checkpoint.pt", map_location="cpu", weights_only=True)
            replay = torch.load(root / "replay/checkpoint.pt", map_location="cpu", weights_only=True)
            for key in best:
                torch.testing.assert_close(best[key], replay[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
