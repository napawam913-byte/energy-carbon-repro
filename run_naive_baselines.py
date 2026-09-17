"""模块用途：在已固定的验证窗口上运行最近值和 24 小时周期基线；不训练。"""

import argparse
from pathlib import Path
import sys

from energy_forecast.baselines import run_validation
from energy_forecast.data import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PROJECT_ROOT / "data/processed/erco_168_24_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/experiments/erco_168_24/naive_validation_v1")
    args = parser.parse_args()
    try:
        report = run_validation(args.prepared, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    for method, scores in report["models"].items():
        factor = scores["per_target"].get("factor_generated_kg_per_mwh")
        if factor is not None:
            print(f"{method}: carbon-factor MAE={factor['mae']:.6f}, RMSE={factor['rmse']:.6f} {factor['unit']}")
    print(f"Results: {args.output.resolve()}")
    print("PASS: validation baselines saved; no training or test-set evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
