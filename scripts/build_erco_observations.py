"""模块用途：校验并重建 ERCO 2023 历史观测表和碳因子。

模块边界：只读固定版本 EIA/OGE 原文件；不插补、裁剪、划分、标准化或训练。
运行：python scripts/build_erco_observations.py（默认从脚本所在仓库定位数据）。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


FUELS = ["biomass", "coal", "hydro", "natural_gas", "nuclear", "other", "petroleum", "solar", "wind"]
OGE_ROOT = "data/raw/oge/v0.8.0/2023"
POWER_PATH = f"{OGE_ROOT}/power_sector_data/ERCO.csv"
CARBON_PATH = f"{OGE_ROOT}/carbon_accounting/ERCO.csv"
EIA_PATHS = [f"data/raw/eia/EIA930_BALANCE_2023_{half}.csv" for half in ("Jan_Jun", "Jul_Dec")]
SOURCE_SHA256 = {
    POWER_PATH: "42f8086ef1af21e34afb1775e7e9413ada1e5296149fab545693989dc9318a64",
    CARBON_PATH: "2b9d8ef20cc5395179a848f1913003ec04e850c265b80d739735b3437ee2c174",
    EIA_PATHS[0]: "a0e14e9c07ab4cc1de3e384c7be890b0c61d9de3fe04ad3ae5ec739f1f35ec1b",
    EIA_PATHS[1]: "c719fc1b513eec8982ce209b6e3173fc64b2d0d243f83d3bf0d8d1adf6222de2",
}
GENERATION = "net_generation_mwh"
EMISSIONS = "co2_mass_kg_for_electricity"
GENERATED_FACTOR = "generated_co2_rate_kg_per_mwh_for_electricity"
CONSUMED_FACTOR = "consumed_co2_rate_kg_per_mwh_for_electricity"
EIA_TIME = "UTC Time at End of Hour"
EIA_COLUMNS = {
    "Demand (MW)": "demand_mw",
    "Net Generation (MW)": "reference_eia_generation_total_mw",
    **{f"Net Generation (MW) from {label}": f"reference_eia_generation_{fuel}_mw"
       for label, fuel in (("Wind", "wind"), ("Solar", "solar"), ("Coal", "coal"), ("Natural Gas", "natural_gas"))},
}


def sha256_file(path: Path) -> str:
    """流式计算字节哈希，不改变输入文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(project_root: Path) -> dict:
    """固定四个已核验版本；版本变化必须人工审查，不自动接受。"""
    observed = {}
    for relative, expected in SOURCE_SHA256.items():
        path = project_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {relative}; expected={expected}; actual={actual}")
        observed[relative] = actual
    return observed


def _columns(frame: pd.DataFrame, required: list, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def _numeric(frame: pd.DataFrame, columns: list, label: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: missing or non-finite numeric values")


def _utc(values: pd.Series, date_format: str | None = None) -> pd.Series:
    parsed = pd.to_datetime(values, format=date_format, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("Missing timestamp")
    return parsed.astype("datetime64[ns, UTC]")


def _grid(index: pd.DatetimeIndex, expected: pd.DatetimeIndex, label: str) -> None:
    if not index.is_unique:
        raise ValueError(f"{label}: duplicate timestamps")
    if not index.sort_values().equals(expected):
        missing, extra = expected.difference(index), index.difference(expected)
        raise ValueError(f"{label}: hourly grid mismatch; missing={len(missing)}, extra={len(extra)}")


def build_observations(power: pd.DataFrame, carbon: pd.DataFrame, eia: pd.DataFrame,
                       expected_index: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, dict]:
    """纯内存变换；CLI 固定全年网格，测试可传入小型已知小时网格。"""
    if expected_index is None:
        expected_index = pd.date_range("2023-01-01 06:00", periods=8760, freq="h", tz="UTC")
    expected_index = expected_index.astype("datetime64[ns, UTC]")
    power, carbon, eia = power.copy(), carbon.copy(), eia.copy()
    _columns(power, ["fuel_category", "datetime_utc", GENERATION, EMISSIONS, GENERATED_FACTOR], "OGE power")
    _columns(carbon, ["datetime_utc", CONSUMED_FACTOR, CONSUMED_FACTOR + "_adjusted"], "OGE carbon")
    _columns(eia, ["Balancing Authority", EIA_TIME, *EIA_COLUMNS], "EIA")

    power["timestamp_utc"] = _utc(power["datetime_utc"])
    if power.duplicated(["timestamp_utc", "fuel_category"]).any():
        raise ValueError("OGE power: duplicate (timestamp, fuel) keys")
    if set(power["fuel_category"]) != set(FUELS + ["total"]):
        raise ValueError("OGE power: expected nine fuel categories plus total")
    _numeric(power, [GENERATION, EMISSIONS], "OGE power")
    if power[EMISSIONS].lt(0).any():
        raise ValueError("OGE power: negative CO2 mass; manual review required")

    parts = power.loc[power["fuel_category"] != "total"]
    generation = parts.pivot(index="timestamp_utc", columns="fuel_category", values=GENERATION)[FUELS].sort_index()
    emissions = parts.pivot(index="timestamp_utc", columns="fuel_category", values=EMISSIONS)[FUELS].sort_index()
    _grid(generation.index, expected_index, "OGE power")
    if not np.isfinite(generation.to_numpy()).all() or not np.isfinite(emissions.to_numpy()).all():
        raise ValueError("OGE power: incomplete hourly fuel matrix")
    total_generation, total_emissions = generation.sum(axis=1), emissions.sum(axis=1)
    if total_generation.le(0).any():
        raise ValueError("OGE power: non-positive regional generation denominator")

    reference = power.loc[power["fuel_category"] == "total"].set_index("timestamp_utc").sort_index()
    _grid(reference.index, expected_index, "OGE total")
    _numeric(reference, [GENERATED_FACTOR], "OGE total")
    if reference[GENERATION].le(0.005).any():
        raise ValueError("OGE total: generation too small for two-decimal rounding validation")
    differences = {
        GENERATION: float((total_generation - reference[GENERATION]).abs().max()),
        "co2_mass_kg": float((total_emissions - reference[EMISSIONS]).abs().max()),
    }
    # 九项和总计均保留两位小数：最坏舍入差约 (9 + 1) * 0.005。
    if any(difference > 0.051 for difference in differences.values()):
        raise ValueError(f"OGE total: component sums disagree beyond rounding: {differences}")
    reference_ratio = reference[EMISSIONS] / reference[GENERATION]
    # E、G、F 分别保留两位小数，容差必须包括分子与分母的舍入传播。
    rate_tolerance = 0.005 + (0.005 + reference_ratio.abs() * 0.005) / (reference[GENERATION] - 0.005) + 1e-9
    if ((reference_ratio - reference[GENERATED_FACTOR]).abs() > rate_tolerance).any():
        raise ValueError("OGE total: published factor disagrees with mass / generation")

    # 非正分母的单能源因子留空；排放和发电量本身仍参与综合核算。
    source_factors = emissions.div(generation.where(generation > 0))
    data = pd.concat([
        generation.add_prefix("generation_").add_suffix("_mwh"),
        emissions.add_prefix("co2_").add_suffix("_kg"),
        source_factors.add_prefix("factor_").add_suffix("_kg_per_mwh"),
    ], axis=1)
    data["generation_total_mwh"] = total_generation
    data["co2_total_kg"] = total_emissions
    data["factor_generated_kg_per_mwh"] = total_emissions / total_generation
    data["flag_negative_generation"] = generation.lt(0).any(axis=1).astype("int8")
    data["reference_generated_factor_kg_per_mwh"] = reference[GENERATED_FACTOR]

    carbon["timestamp_utc"] = _utc(carbon["datetime_utc"])
    carbon = carbon.set_index("timestamp_utc").sort_index()
    _grid(carbon.index, expected_index, "OGE carbon")
    consumed_columns = [CONSUMED_FACTOR, CONSUMED_FACTOR + "_adjusted"]
    _numeric(carbon, consumed_columns, "OGE carbon")
    consumed = carbon[consumed_columns].rename(columns={
        CONSUMED_FACTOR: "reference_consumed_factor_kg_per_mwh",
        CONSUMED_FACTOR + "_adjusted": "reference_consumed_adjusted_factor_kg_per_mwh",
    })
    data = data.join(consumed, validate="one_to_one")

    eia = eia.loc[eia["Balancing Authority"] == "ERCO"].copy()
    if eia.empty:
        raise ValueError("EIA: no ERCO rows")
    eia["timestamp_utc"] = _utc(eia[EIA_TIME], "%m/%d/%Y %I:%M:%S %p") - pd.Timedelta(hours=1)
    eia = eia.set_index("timestamp_utc").sort_index()
    _grid(eia.index, expected_index, "EIA hour-start")
    _numeric(eia, list(EIA_COLUMNS), "EIA ERCO")
    data = data.join(eia[list(EIA_COLUMNS)].rename(columns=EIA_COLUMNS), validate="one_to_one")
    data.columns.name = None
    data.index.name = "timestamp_utc"
    allowed_nan = {f"factor_{fuel}_kg_per_mwh" for fuel in FUELS}
    if not np.isfinite(data[[column for column in data if column not in allowed_nan]].to_numpy()).all():
        raise ValueError("Final observation table contains unexpected non-finite values")
    negative_values = generation.to_numpy()[generation.to_numpy() < 0]
    report = {
        "status": "OBSERVATIONS_BUILT_NOT_FORECAST_VALIDATED",
        "region": "ERCO", "dataset_year": 2023, "oge_version": "v0.8.0",
        "hour_start_utc_min": str(data.index.min()), "hour_start_utc_max": str(data.index.max()),
        "hourly_rows": len(data), "csv_columns_including_timestamp": len(data.columns) + 1,
        "negative_generation_records": int(negative_values.size),
        "negative_generation_min_mwh": float(negative_values.min()) if negative_values.size else None,
        "negative_generation_max_mwh": float(negative_values.max()) if negative_values.size else None,
        "undefined_source_factor_counts": {fuel: int(source_factors[fuel].isna().sum()) for fuel in FUELS},
        "max_component_sum_difference_from_total": differences,
        "max_factor_difference_from_published_kg_per_mwh": float((data["factor_generated_kg_per_mwh"] - reference[GENERATED_FACTOR]).abs().max()),
        "rounding_tolerance": {
            "component_sum_absolute": 0.051,
            "published_factor_absolute_formula": "0.005 + (0.005 + abs(E/G)*0.005)/(G-0.005) + 1e-9; published total E,G",
            "published_factor_absolute_max": float(rate_tolerance.max()),
        },
        "definitions": {
            "generation_*_mwh": "OGE net electricity generation; negatives retained.",
            "co2_*_kg": "OGE CO2 mass attributed to electricity; unadjusted; not CO2e.",
            "factor_<fuel>_kg_per_mwh": "Per-fuel CO2 mass / net generation; defined only where generation > 0.",
            "factor_generated_kg_per_mwh": "Sum of nine fuel CO2 masses / sum of nine fuel net generation; total rows excluded.",
            "demand_mw": "EIA original demand, no adjusted or imputed values substituted.",
            "reference_eia_*_mw": "EIA original generation; not the OGE factor denominator.",
            "reference_*factor*": "Published OGE references, not independent validation or interchangeable accounting boundaries.",
            "flag_negative_generation": "At least one fuel has negative net generation this hour.",
        },
        "notes": [
            "Regional electric-generation benchmark, not measured industrial-park or multi-carrier heat/gas data.",
            "EIA end-of-hour UTC shifted back one hour to match OGE start-of-hour UTC.",
            "Historical observations, not ready-made forecasting inputs: never expose future actuals or future quality flags.",
            "OGE is retrospective; real-time publication delays and availability have not been validated.",
            "Per-source factors at non-positive generation are intentionally NaN; this differs from some published OGE zero-filling conventions.",
            "No blanket filling, negative-generation clipping, train/test split, scaling, fitting or performance evaluation.",
        ],
    }
    return data, report


def write_observations(data: pd.DataFrame, metadata: dict, output_dir: Path) -> Path:
    """写入新目录后重读核对；拒绝覆盖已有目录与文件。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "observations.csv"
    with np.errstate(invalid="ignore"):
        data.to_csv(path, index=True, encoding="utf-8", mode="x")
    restored = pd.read_csv(path, index_col="timestamp_utc")
    restored.index = pd.to_datetime(restored.index, utc=True).astype("datetime64[ns, UTC]")
    pd.testing.assert_frame_equal(data, restored, check_dtype=False, check_freq=False,
                                  check_exact=False, rtol=1e-12, atol=1e-10)
    saved_metadata = dict(metadata)
    saved_metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    saved_metadata["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    saved_metadata["output_file"] = path.name
    saved_metadata["output_sha256"] = sha256_file(path)
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(saved_metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def run(project_root: Path, output_dir: Path | None = None) -> Path:
    """固定来源校验、读取和输出；不接受原始数据目录作为输出位置。"""
    project_root = project_root.resolve()
    processed = (project_root / "data" / "processed").resolve()
    output_dir = output_dir or processed / "erco_2023_v1"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    if output_dir == processed or not output_dir.is_relative_to(processed):
        raise ValueError("Output must be a new subdirectory under project data/processed")
    if output_dir.exists():
        raise FileExistsError(f"Output directory exists; no overwrite: {output_dir}")
    hashes = verify_sources(project_root)
    power = pd.read_csv(project_root / POWER_PATH, low_memory=False)
    carbon = pd.read_csv(project_root / CARBON_PATH, low_memory=False)
    eia = pd.concat([pd.read_csv(project_root / relative,
                               usecols=["Balancing Authority", EIA_TIME, *EIA_COLUMNS])
                     for relative in EIA_PATHS], ignore_index=True)
    data, report = build_observations(power, carbon, eia)
    report["source_sha256"] = hashes
    report["builder_sha256"] = sha256_file(Path(__file__))
    result = write_observations(data, report, output_dir)
    print(f"Rows: {len(data)}; columns including timestamp: {len(data.columns) + 1}")
    print(f"Hour-start UTC: {data.index.min()} to {data.index.max()}")
    print(f"Negative generation records retained: {report['negative_generation_records']}")
    print("Undefined source factors:", report["undefined_source_factor_counts"])
    print(data[["demand_mw", "generation_wind_mwh", "generation_solar_mwh", "factor_generated_kg_per_mwh"]].head(3).to_string())
    print(f"Saved to: {result}")
    print("PASS: observations built and re-read; no training performed.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="New directory under project data/processed; existing directories are never overwritten")
    args = parser.parse_args()
    try:
        run(args.project_root, args.output_dir)
    except (OSError, ValueError, AssertionError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
