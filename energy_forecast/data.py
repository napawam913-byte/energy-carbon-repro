"""模块用途：检查小时观测、固定时间划分、训练段标准化和惰性窗口。

模块边界：仅用 NumPy/pandas，不导入模型框架，不训练、不计算测试指标。
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import numbers
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "erco_2023.json"
FUELS = ("biomass", "coal", "hydro", "natural_gas", "nuclear", "other", "petroleum", "solar", "wind")
ALLOWED_TARGETS = tuple(f"generation_{fuel}_mwh" for fuel in FUELS) + ("factor_generated_kg_per_mwh",)
ALLOWED_INPUTS = ("demand_mw",) + ALLOWED_TARGETS


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


DEFAULT_CONFIG = read_json(CONFIG_PATH)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance():
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "code_sha256": {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
                        for path in sorted((PROJECT_ROOT / "energy_forecast").glob("*.py"))},
    }
    try:
        record["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, timeout=10).decode("ascii").strip()
        record["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, timeout=10).strip())
    except (OSError, subprocess.SubprocessError):
        record.update(git_commit=None, git_dirty=None)
    return record


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def finite(values, name):
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")


def hour(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    stamp = stamp.tz_convert("UTC")
    if stamp != stamp.floor("h"):
        raise ValueError("Timestamps must lie on exact hours")
    return stamp


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    result = copy.deepcopy(DEFAULT_CONFIG)
    result.update(copy.deepcopy(config))
    for key in ("lookback", "horizon", "stride"):
        positive_int(result[key], key)
    for key, allowed in (("input_columns", ALLOWED_INPUTS), ("target_columns", ALLOWED_TARGETS)):
        fields = result[key]
        if not isinstance(fields, list) or not fields or not all(isinstance(f, str) for f in fields):
            raise ValueError(f"{key} must be a nonempty list of field names")
        if len(set(fields)) != len(fields) or not set(fields).issubset(allowed):
            raise ValueError(f"{key} has duplicate or unapproved fields")
    if hour(result["train_end"]) >= hour(result["val_end"]):
        raise ValueError("train_end must precede val_end")
    if not isinstance(result["source"], str) or not result["source"]:
        raise ValueError("source must be a relative file path")
    source = Path(result["source"])
    if source.is_absolute() or ".." in source.parts:
        raise ValueError("source must be relative to root without parent traversal")
    return result


def fit_scaler(values):
    with np.errstate(over="ignore", invalid="ignore"):
        mean, scale = values.mean(axis=0), values.std(axis=0)
    finite(mean, "training mean")
    finite(scale, "training standard deviation")
    constant = scale == 0
    scale[constant] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist(), "constant": constant.tolist()}


class ForecastData:
    """保留二维观测与起点索引，按需构建窗口，不生成巨型三维训练文件。"""

    def __init__(self, frame, config, scalers=None):
        self.config = c = validate_config(config)
        if not frame.columns.is_unique or "timestamp_utc" not in frame:
            raise ValueError("A unique timestamp_utc column and unique field names are required")
        self.times = pd.DatetimeIndex([hour(value) for value in frame["timestamp_utc"]])
        if len(self.times) < 2 or not (self.times[1:] - self.times[:-1] == pd.Timedelta(hours=1)).all():
            raise ValueError("Timestamps must be strictly increasing, unique and hourly continuous")
        required = set(c["input_columns"] + c["target_columns"])
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing selected fields: {sorted(required - set(frame.columns))}")
        self.raw_x = frame[c["input_columns"]].to_numpy(dtype=np.float64, copy=True)
        self.raw_y = frame[c["target_columns"]].to_numpy(dtype=np.float64, copy=True)
        finite(self.raw_x, "selected inputs")
        finite(self.raw_y, "selected targets")
        a, b = (int(self.times.searchsorted(hour(c[key]))) for key in ("train_end", "val_end"))
        if not 0 < a < b < len(frame):
            raise ValueError("All three time splits must contain rows")
        self.bounds = ((0, a), (a, b), (b, len(frame)))
        self.origins, self.splits = {}, {}
        for name, (start, end) in zip(("train", "val", "test"), self.bounds):
            origins = np.arange(max(start, c["lookback"]), end - c["horizon"] + 1, c["stride"], dtype=np.int64)
            if len(origins) == 0:
                raise ValueError(f"{name} has no windows: insufficient history or target rows")
            self.origins[name] = origins
            self.splits[name] = {
                "start": self.times[start].isoformat(), "end_exclusive": (self.times[end - 1] + pd.Timedelta(hours=1)).isoformat(),
                "rows": end - start, "samples": len(origins),
                "first_origin": self.times[origins[0]].isoformat(), "last_origin": self.times[origins[-1]].isoformat(),
            }
        self.scalers = copy.deepcopy(scalers) if scalers is not None else {"input": fit_scaler(self.raw_x[:a]), "target": fit_scaler(self.raw_y[:a])}
        normalized = []
        for name, raw in (("input", self.raw_x), ("target", self.raw_y)):
            saved = self.scalers[name]
            mean, scale = np.asarray(saved["mean"], dtype=float), np.asarray(saved["scale"], dtype=float)
            if mean.shape != (raw.shape[1],) or scale.shape != mean.shape:
                raise ValueError("Scaler dimensions do not match fields")
            finite(mean, "saved mean")
            finite(scale, "saved scale")
            if (scale <= 0).any():
                raise ValueError("Saved scales must be positive")
            with np.errstate(over="ignore", invalid="ignore"):
                values = (raw - mean) / scale
            finite(values, "standardized data")
            normalized.append(values)
        self.x, self.y = normalized

    def sample(self, j, raw=False):
        c = self.config
        if isinstance(j, bool) or not isinstance(j, numbers.Integral) or j < c["lookback"] or not any(start <= j and j + c["horizon"] <= end for start, end in self.bounds):
            raise ValueError("Origin has insufficient history or target window crosses a split boundary")
        x, y = (self.raw_x, self.raw_y) if raw else (self.x, self.y)
        return x[j - c["lookback"]:j], y[j:j + c["horizon"]]

    def history(self, j):
        self.sample(j)
        return self.raw_y[j - self.config["lookback"]:j]


def prepare(root, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = validate_config(read_json(config) if isinstance(config, (str, Path)) else config)
        root = Path(root).resolve()
        source = (root / config["source"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Source must stay within the selected project root")
        digest = sha256(source)
        data = ForecastData(pd.read_csv(source), config)
        if sha256(source) != digest:
            raise ValueError("Source changed during preparation")
        manifest = {
            "schema_version": 1, "status": "PREPARED_NOT_TRAINED",
            "source_path": str(source), "source_sha256": digest,
            "config": config, "input_columns": list(config["input_columns"]), "target_columns": list(config["target_columns"]),
            "splits": data.splits, "scalers": data.scalers, "provenance": provenance(),
            "evaluation_protocol": "Hourly rolling origins; complete target windows stay in one split. Test metrics are not evaluated by preparation.",
            "availability_assumption": "Retrospective benchmark assumes observations through origin-1 are available; real publication lags unvalidated.",
        }
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        write_json(output / "run.json", {"status": "failed", "error": str(error)})
        raise


def load_prepared(path):
    manifest = read_json(Path(path) / "manifest.json")
    if manifest["schema_version"] != 1 or manifest["status"] != "PREPARED_NOT_TRAINED":
        raise ValueError("Unsupported preparation manifest")
    config = validate_config(manifest["config"])
    for key in ("input_columns", "target_columns"):
        if manifest[key] != config[key]:
            raise ValueError(f"Manifest {key} order mismatch")
    source = Path(manifest["source_path"])
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Observation source SHA256 mismatch")
    data = ForecastData(pd.read_csv(source), config, manifest["scalers"])
    if sha256(source) != manifest["source_sha256"] or data.splits != manifest["splits"]:
        raise ValueError("Source changed or manifest splits mismatch")
    return data, manifest
