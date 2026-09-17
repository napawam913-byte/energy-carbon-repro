"""模块用途：只基于已知历史的两个朴素预测器及验证集原单位指标。

模块边界：不拟合神经网络、不评价测试集、不混合不同物理单位的误差。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import finite, load_prepared, positive_int, provenance, sha256, write_json


def predict(method, history, horizon):
    positive_int(horizon, "horizon")
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 2 or not all(history.shape):
        raise ValueError("history must be a nonempty [hours, targets] array")
    finite(history, "baseline history")
    if method == "persistence":
        return np.repeat(history[-1:], horizon, axis=0)
    if method == "seasonal24":
        if len(history) < 24:
            raise ValueError("seasonal24 requires at least 24 historical hours")
        return history[-24:][np.arange(horizon) % 24]
    raise ValueError(f"Unknown baseline: {method}")


def metrics(actual, prediction, targets):
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 3 or not all(actual.shape) or actual.shape[2] != len(targets):
        raise ValueError("Metrics require matching nonempty [origins, horizon, targets] arrays")
    finite(actual, "actual")
    finite(prediction, "prediction")
    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction - actual
        squared = error ** 2
    finite(error, "error")
    finite(squared, "squared error")

    def summarize(errors, squares):
        result = {}
        for k, name in enumerate(targets):
            unit = "kg CO2/MWh" if name == "factor_generated_kg_per_mwh" else "MWh"
            mse = float(squares[..., k].mean())
            result[name] = {"mae": float(np.abs(errors[..., k]).mean()), "mse": mse,
                            "rmse": float(np.sqrt(mse)), "unit": unit, "mse_unit": f"({unit})^2"}
        return result

    return {"per_target": summarize(error, squared),
            "per_horizon": {str(h + 1): summarize(error[:, h], squared[:, h]) for h in range(actual.shape[1])},
            "origins": actual.shape[0], "horizon": actual.shape[1]}


def run_validation(prepared, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest_path = Path(prepared).resolve() / "manifest.json"
        manifest_hash = sha256(manifest_path)
        data, manifest = load_prepared(prepared)
        if data.config["lookback"] < 24:
            raise ValueError("Both baselines require lookback >= 24")
        origins = data.origins["val"]
        actual = np.stack([data.sample(int(j), raw=True)[1] for j in origins])
        report = {
            "status": "VALIDATION_BASELINES_ONLY", "split": "val", "models": {},
            "prepared_manifest_path": str(manifest_path), "prepared_manifest_sha256": manifest_hash,
            "source_sha256": manifest["source_sha256"], "config": data.config,
            "validation_range": data.splits["val"], "provenance": provenance(),
            "notes": ["No model training or test-set metrics.",
                      "Each origin uses only earlier observations; 24-hour repetition is a UTC-duration convention.",
                      "Overlapping rolling windows are not independent statistical samples.",
                      "Each variable retains its physical unit; there is no mixed-unit overall score."],
        }
        overall_rows, horizon_rows = [], []
        for method in ("persistence", "seasonal24"):
            predicted = np.stack([predict(method, data.history(int(j)), data.config["horizon"]) for j in origins])
            scores = metrics(actual, predicted, data.config["target_columns"])
            report["models"][method] = scores
            overall_rows.extend({"model": method, "split": "val", "target": name, **values}
                                for name, values in scores["per_target"].items())
            horizon_rows.extend({"model": method, "split": "val", "horizon": int(h), "target": name, **values}
                                for h, items in scores["per_horizon"].items() for name, values in items.items())
        if sha256(manifest_path) != manifest_hash or sha256(manifest["source_path"]) != manifest["source_sha256"]:
            raise ValueError("Preparation manifest or source changed during validation")
        pd.DataFrame(overall_rows).to_csv(output / "metrics.csv", index=False, mode="x")
        pd.DataFrame(horizon_rows).to_csv(output / "per_horizon_metrics.csv", index=False, mode="x")
        write_json(output / "metrics.json", report)
        return report
    except Exception as error:
        write_json(output / "run.json", {"status": "failed", "error": str(error)})
        raise
