"""模块用途：核验并调用作者原版 SparseTSF，建立模型外部的目标列映射。

模块边界：不定义神经网络、不改作者 forward、不负责优化器或数据划分。
"""

import hashlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from .data import positive_int, sha256


COMMIT = "b8c2740eecc84d8095ffce49ba5acafe68e53bb8"
# 固定源码的 LF 归一化 SHA256；只容许 Git 换行转换，不接受模型内容变化。
SOURCE_HASHES = {
    "models/SparseTSF.py": "7f4afe758bd7ad71b939b88e633acd271f275f4e85dd73b724233c7b1a041948",
    "layers/Embed.py": "3743a14dd505f9caafb512727b10bec2d4a84d84cacc6f8936f33a5071e2a0a2",
}


def verify_source(code_dir):
    code = Path(code_dir).expanduser().resolve()
    normalized, raw = {}, {}
    for relative, expected in SOURCE_HASHES.items():
        path = code / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing author source: {path}; specify --code-dir")
        normalized[relative] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if normalized[relative] != expected:
            raise ValueError(f"Author source SHA256 mismatch: {path}; expected commit {COMMIT}. No file was changed.")
        raw[relative] = sha256(path)
    # 固定版本使用 namespace package；额外初始化脚本不在已核验的导入链内。
    for package in ("models", "layers"):
        if (code / package / "__init__.py").exists():
            raise ValueError(f"Unexpected author package initializer: {code / package / '__init__.py'}")
    return {"repository": "https://github.com/lss-1138/SparseTSF", "commit": COMMIT,
            "code_dir": str(code), "normalized_sha256": normalized, "raw_sha256": raw}


def target_indices(data):
    inputs, targets = data.config["input_columns"], data.config["target_columns"]
    if not set(targets).issubset(inputs):
        raise ValueError("All SparseTSF targets must be present in historical inputs")
    indices = [inputs.index(name) for name in targets]
    for key in ("mean", "scale"):
        a = np.asarray(data.scalers["input"][key])[indices]
        b = np.asarray(data.scalers["target"][key])
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Input/target scaler mismatch for {key}")
    return indices


def _load_verified_source(code, name, relative):
    """直接执行本次已校验源码，绕过旧 pyc 与同路径 sys.modules 中的旧类。"""
    path = code / relative
    content = path.read_bytes()
    if hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest() != SOURCE_HASHES[relative]:
        raise ValueError(f"Author source changed before execution: {path}")
    parent_name = name.rsplit(".", 1)[0]
    if parent_name not in sys.modules:
        package = ModuleType(parent_name)
        package.__path__ = [str(code / parent_name)]
        package.__package__ = parent_name
        package.__spec__ = ModuleSpec(parent_name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[parent_name] = package
    module = ModuleType(name)
    module.__file__, module.__package__ = str(path), parent_name
    module.__spec__ = ModuleSpec(name, loader=None, origin=str(path))
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(content, str(path), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    setattr(sys.modules[parent_name], name.rsplit(".", 1)[1], module)
    return module


def build_model(code_dir, data, model_type="linear", period_len=24, d_model=128):
    source = verify_source(code_dir)
    if model_type not in ("linear", "mlp"):
        raise ValueError("Only author linear / mlp branches are supported")
    positive_int(period_len, "period_len")
    positive_int(d_model, "d_model")
    c = data.config
    if c["lookback"] % period_len or c["horizon"] % period_len:
        raise ValueError("lookback and horizon must be divisible by period_len")
    target_indices(data)
    code = Path(source["code_dir"])
    for name in ("models", "layers", "models.SparseTSF", "layers.Embed"):
        existing = sys.modules.get(name)
        if existing is not None:
            paths = ([Path(existing.__file__).resolve()] if getattr(existing, "__file__", None)
                     else [Path(p).resolve() for p in getattr(existing, "__path__", [])])
            if not paths or not all(p.is_relative_to(code) for p in paths):
                raise ValueError(f"Conflicting imported module {name}; use a fresh Python process")
    _load_verified_source(code, "layers.Embed", "layers/Embed.py")
    author = _load_verified_source(code, "models.SparseTSF", "models/SparseTSF.py")
    config = SimpleNamespace(seq_len=c["lookback"], pred_len=c["horizon"], enc_in=len(c["input_columns"]),
                             period_len=period_len, d_model=d_model, model_type=model_type)
    model = author.Model(config).float()
    if verify_source(code) != source:
        raise ValueError("Author source changed while loading")
    return model, source
