"""模块用途：将观测表登记为防泄漏的预测数据协议；不训练、不评价测试集。"""

import argparse
from pathlib import Path
import sys

from energy_forecast.data import CONFIG_PATH, PROJECT_ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "data/processed/erco_168_24_v1"
    try:
        manifest = prepare(args.root, args.config, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    config, splits = manifest["config"], manifest["splits"]
    print("Windows:", " ".join(f"{key}={splits[key]['samples']}" for key in ("train", "val", "test")))
    print(f"X: ({config['lookback']}, {len(config['input_columns'])}); y: ({config['horizon']}, {len(config['target_columns'])})")
    print(f"Manifest: {(output / 'manifest.json').resolve()}")
    print("PASS: forecasting data prepared; no training or test-set evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
