"""模块用途：服务器 SparseTSF ERCO 预检与训练入口，监督单次任务的硬超时。

模块边界：直接调用作者模型，正式训练仅 CUDA，不自动重试，不评价测试集。
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from energy_forecast.data import positive_int, read_json, write_json
from energy_forecast.sparsetsf import verify_source


def validate_output(output):
    output = Path(output).expanduser().resolve()
    allowed = (ROOT / "data/experiments").resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Output must be a new subdirectory inside this repository's data/experiments")
    if output.exists():
        raise FileExistsError(f"Output already exists; no overwrite: {output}")
    return output


def supervise(command, seconds, output):
    """只监督本次唯一子进程；超时/用户中断记录为失败，不自动重跑。"""
    try:
        return subprocess.run(command, timeout=seconds, check=False).returncode
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        timed_out = isinstance(error, subprocess.TimeoutExpired)
        status = "TIMED_OUT" if timed_out else "INTERRUPTED"
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "termination.json", {"status": status, "timeout_seconds": seconds,
                                                "note": "Worker terminated. Partial outputs are not a completed run; no retry."})
        print(f"FAIL: {status}; partial outputs retained in {output}", file=sys.stderr, flush=True)
        return 124 if timed_out else 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent / "code",
                        help="固定版本作者源码；可以指定已有独立克隆目录")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/processed/erco_168_24_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparsetsf_erco.json")
    parser.add_argument("--output", type=Path, help="新运行目录；默认创建带 UTC 时间戳的目录，不覆盖已有结果")
    parser.add_argument("--check-only", action="store_true", help="仅校验来源、数据、GPU 前向；不优化参数，不生成运行目录")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.code_dir, args.prepared, args.config = (p.expanduser().resolve() for p in (args.code_dir, args.prepared, args.config))
    verify_source(args.code_dir)
    config = read_json(args.config)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a JSON object")
    seconds = config.get("timeout_seconds", 1800)
    positive_int(seconds, "timeout_seconds")
    if args.check_only:
        from energy_forecast.training import preflight
        info = preflight(args.prepared, args.code_dir, config, device="cuda")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print("PASS: author source, data mapping and CUDA forward checked; no training or files written.")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = validate_output(args.output or ROOT / "data/experiments/erco_168_24" / f"sparsetsf_{stamp}")
    if args.worker:
        from energy_forecast.training import train
        train(args.prepared, args.code_dir, output, config, device="cuda")
        return 0
    print(f"Output: {output}\nHard timeout: {seconds}s; no automatic retry.", flush=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--code-dir", str(args.code_dir),
               "--prepared", str(args.prepared), "--config", str(args.config), "--output", str(output)]
    result = supervise(command, seconds, output)
    if result == 0:
        if (output / "termination.json").exists() or read_json(output / "run.json").get("status") != "COMPLETED_VALIDATION_ONLY":
            raise RuntimeError("Worker ended without a completed validation-only run record")
        print(f"Finished. Best checkpoint, validation metrics and predictions: {output}", flush=True)
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
