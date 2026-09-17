"""模块用途：验证加载的是作者原网络及外部目标映射；仅使用合成 CPU 输入。"""

import copy
import importlib
import importlib.util
import os
import marshal
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from energy_forecast.data import DEFAULT_CONFIG, ForecastData, PROJECT_ROOT

CODE = Path(os.environ.get("SPARSE_TSF_CODE_DIR", PROJECT_ROOT / "reproductions/01_SparseTSF_TPAMI2026/code"))


def fixture():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(lookback=24, horizon=6, train_end="2023-01-03T00:00:00Z", val_end="2023-01-04T12:00:00Z")
    frame = pd.DataFrame({"timestamp_utc": pd.date_range("2023-01-01", periods=120, freq="h", tz="UTC").astype(str)})
    t = np.arange(len(frame), dtype=float)
    for k, name in enumerate(config["input_columns"]):
        frame[name] = 100 * (k + 1) + (k + 1) * np.sin(t / 5 + k) + t / 10
    return frame, config


class AdapterTests(unittest.TestCase):
    def adapter(self):
        self.assertIsNotNone(importlib.util.find_spec("energy_forecast.sparsetsf"), "author-model adapter is not implemented")
        return importlib.import_module("energy_forecast.sparsetsf")

    def test_adapter_exists(self):
        self.adapter()

    def test_source_missing_changed_and_extra_init_rejected(self):
        m = self.adapter()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises((ValueError, FileNotFoundError)):
                m.verify_source(temp)
            if not CODE.is_dir():
                self.skipTest("Set SPARSE_TSF_CODE_DIR to the pinned author source")
            for rel in ("models/SparseTSF.py", "layers/Embed.py"):
                dst = Path(temp) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(CODE / rel, dst)
            before = m.verify_source(temp)
            model = Path(temp) / "models/SparseTSF.py"
            original = model.read_bytes()
            model.write_bytes(original.replace(b"\n", b"\r\n"))
            self.assertEqual(before["normalized_sha256"], m.verify_source(temp)["normalized_sha256"])
            model.write_bytes(original + b"\n# modified model\n")
            with self.assertRaisesRegex(ValueError, "source|SHA256"):
                m.verify_source(temp)
            model.write_bytes(original)
            (Path(temp) / "models/__init__.py").write_text("raise RuntimeError('must not execute')", encoding="utf-8")
            with self.assertRaises(ValueError):
                m.verify_source(temp)

    def test_target_mapping_respects_field_order_and_scalers(self):
        m = self.adapter()
        frame, config = fixture()
        config["input_columns"].reverse()
        data = ForecastData(frame, config)
        indices = m.target_indices(data)
        self.assertEqual([config["input_columns"][k] for k in indices], config["target_columns"])
        data.scalers["target"]["mean"][0] += 1
        with self.assertRaisesRegex(ValueError, "scaler"):
            m.target_indices(data)
        data = ForecastData(frame, config)
        data.config["input_columns"].remove(config["target_columns"][0])
        with self.assertRaises(ValueError):
            m.target_indices(data)

    @unittest.skipUnless(importlib.util.find_spec("torch") and CODE.is_dir(), "PyTorch and pinned source required")
    def test_original_class_forward_shapes_gradients_and_demand_isolation(self):
        import torch
        torch.set_num_threads(1)
        m = self.adapter()
        data = ForecastData(*fixture())
        for kind in ("linear", "mlp"):
            with self.subTest(model_type=kind):
                model, record = m.build_model(CODE, data, model_type=kind, period_len=3, d_model=8)
                author = importlib.import_module("models.SparseTSF").Model
                self.assertIs(type(model), author)
                self.assertIs(type(model).forward, author.forward)
                self.assertEqual(record["commit"], "b8c2740eecc84d8095ffce49ba5acafe68e53bb8")
                x = torch.randn(2, 24, 11, requires_grad=True)
                y = model(x)
                self.assertEqual(tuple(y.shape), (2, 6, 11))
                indices = m.target_indices(data)
                y[..., indices].square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertEqual(torch.count_nonzero(x.grad[..., 0]).item(), 0)
                changed = x.detach().clone()
                changed[..., 0] += 1234
                torch.testing.assert_close(model(changed)[..., indices], y.detach()[..., indices], rtol=0, atol=0)
                direct = author(model.configs) if hasattr(model, "configs") else author(type("Config", (), {
                    "seq_len": 24, "pred_len": 6, "enc_in": 11, "period_len": 3, "d_model": 8, "model_type": kind})())
                direct.load_state_dict(model.state_dict())
                torch.testing.assert_close(direct(x.detach()), y.detach(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            m.build_model(CODE, data, period_len=5)
        with self.assertRaises(ValueError):
            m.build_model(CODE, data, model_type="custom")

    @unittest.skipUnless(importlib.util.find_spec("torch") and CODE.is_dir(), "PyTorch and pinned source required")
    def test_approved_shape_parameter_count(self):
        m = self.adapter()
        frame, config = fixture()
        data = ForecastData(frame, config)
        data.config.update(lookback=168, horizon=24)
        model, _ = m.build_model(CODE, data)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 32)

    @unittest.skipUnless(importlib.util.find_spec("torch") and CODE.is_dir(), "PyTorch and pinned source required")
    def test_cached_bytecode_and_reused_modules_cannot_replace_verified_source(self):
        with tempfile.TemporaryDirectory() as temp:
            for relative in ("models/SparseTSF.py", "layers/Embed.py"):
                source = Path(temp) / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(CODE / relative, source)
                stat = source.stat()
                cache = Path(importlib.util.cache_from_source(str(source)))
                cache.parent.mkdir(parents=True)
                header = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, int(stat.st_mtime), stat.st_size)
                poison = compile("raise RuntimeError('unverified-bytecode-executed')", str(source), "exec")
                cache.write_bytes(header + marshal.dumps(poison))
            code = """
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + '/tests')
from test_sparsetsf_adapter import fixture
from energy_forecast.data import ForecastData
from energy_forecast.sparsetsf import build_model
data = ForecastData(*fixture())
model, _ = build_model(sys.argv[2], data, period_len=3)
assert type(model).__module__ == 'models.SparseTSF'
import models.SparseTSF as cached
cached.Model = None
again, _ = build_model(sys.argv[2], data, period_len=3)
assert type(again).__module__ == 'models.SparseTSF'
"""
            result = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(PROJECT_ROOT), temp],
                                    capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
