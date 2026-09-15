"""模块用途：将已跑通的 ETTh1-linear 命令封装为 VS Code 可直接运行的入口。

点击“在终端中运行 Python 文件”会开始一次新的正式训练，不是读取旧结果。
只负责参数、路径和日志；模型与训练循环仍由作者 code/ 中的文件执行。
仅查看配置而不训练：python train_etth1_linear.py --dry-run
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PAPER_DIR = Path(__file__).resolve().parent
EXPECTED_COMMIT = "b8c2740eecc84d8095ffce49ba5acafe68e53bb8"
# 锁定用户首轮已检查文件的字节，不宣称这是出版社公布的校验和。
EXPECTED_DATA_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
TRAINING_ARGS = {
    "is_training": 1, "model_id": "ETTh1_720_96", "model": "SparseTSF",
    "model_type": "linear", "data": "ETTh1", "data_path": "ETTh1.csv",
    "features": "M", "seq_len": 720, "pred_len": 96, "period_len": 24,
    "enc_in": 7, "train_epochs": 30, "patience": 5, "itr": 1,
    "batch_size": 256, "learning_rate": 0.02, "loss": "mse", "lradj": "type3",
    "num_workers": 10, "gpu": 0, "checkpoints": "./checkpoints/",
}


def find_code_dir(paper_dir, override=None):
    """只检查两种约定目录，不扫描磁盘；显式指定的目录错误时不偷偷回退。"""
    candidates = [Path(override).expanduser()] if override is not None else [
        paper_dir / "code",
        paper_dir.parents[2] / "reproductions" / paper_dir.name / "code",
    ]
    for candidate in candidates:
        if (candidate / "run_longExp.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "找不到作者 run_longExp.py。请用 --code-dir 指定含该文件的 code 目录。\n"
        + "\n".join(str(path) for path in candidates)
    )


def build_command(code_dir):
    command = [sys.executable, "-u", str(code_dir / "run_longExp.py"),
               "--root_path", str(code_dir / "dataset")]
    for name, value in TRAINING_ARGS.items():
        command.extend([f"--{name}", str(value)])
    return command


def execute_training(command, runs_root, records):
    """在新目录运行一次进程，合并保存输出和退出码；失败不重试。"""
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="etth1_linear_720_96_", dir=runs_root))
    record = {
        "status": "RUNNING", "command": command, "working_directory": str(run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "exit_code": None,
    }
    for name, content in records.items():
        (run_dir / name).write_text(content, encoding="utf-8")
    manifest = run_dir / "run.json"
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("本次实验目录:", run_dir, flush=True)
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    exit_code = 127
    with (run_dir / "train.log").open("x", encoding="utf-8") as log:
        try:
            with subprocess.Popen(
                command, cwd=run_dir, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            ) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    exit_code = process.wait()
                except KeyboardInterrupt:
                    # 用户主动 Ctrl+C 时终止本入口启动的进程，不影响其他训练。
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    exit_code = 130
        except OSError as error:
            log.write(f"启动失败: {error}\n")
            print(f"启动失败: {error}", file=sys.stderr)
    record.update({
        "exit_code": exit_code,
        "status": "COMPLETED" if exit_code == 0 else "INTERRUPTED" if exit_code == 130 else "FAILED",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "exitcode.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    return run_dir, exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, help="可选：手动指定作者代码目录")
    parser.add_argument("--dry-run", action="store_true", help="只预览参数和路径，不检查环境、不创建实验目录、不训练")
    args = parser.parse_args()
    code_dir = find_code_dir(PAPER_DIR, args.code_dir)
    command = build_command(code_dir)
    dataset = code_dir / "dataset" / "ETTh1.csv"
    if args.dry_run:
        print(json.dumps({
            "status": "PREVIEW_ONLY_NOT_TRAINED", "command": command,
            "code_dir": str(code_dir), "dataset": str(dataset), "dataset_exists": dataset.is_file(),
            "runs_root": str(PAPER_DIR / "runs"),
            "note": "预览不代表环境、数据或源码版本检查通过。",
        }, ensure_ascii=False, indent=2))
        return 0

    print("即将重新训练 ETTh1-linear：720 → 96，最多 30 轮；不是读取上次结果。", flush=True)
    # 沿用 VS Code 选中的解释器；选择 base 或不匹配的环境时在训练前停止。
    subprocess.run([sys.executable, str(PAPER_DIR / "scripts" / "check_environment.py")], check=True)
    revision = subprocess.run(["git", "-C", str(code_dir), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != EXPECTED_COMMIT:
        raise ValueError(f"作者提交不匹配：{revision}；不自动切换或覆盖源码。")
    if not dataset.is_file():
        raise FileNotFoundError(f"数据不存在：{dataset}；先运行 scripts/prepare_etth1.py。")
    checksum = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if checksum != EXPECTED_DATA_SHA256:
        raise ValueError(f"数据 SHA256 与首轮记录不同：{checksum}；请先核对文件，不自动替换。")
    subprocess.run([sys.executable, str(PAPER_DIR / "scripts" / "check_gpu.py"),
                    "--code-dir", str(code_dir), "--model-type", "linear"], check=True)
    patch = subprocess.run(["git", "-C", str(code_dir), "diff", "--no-ext-diff", "HEAD"],
                           check=True, capture_output=True, text=True, encoding="utf-8").stdout
    if patch.strip():
        print("注意：作者目录有已跟踪文件改动，会记录差异；该运行不能直接视为未修改源码复现。", flush=True)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            check=True, capture_output=True, text=True).stdout
    records = {
        "environment.txt": frozen, "source-commit.txt": revision + "\n",
        "source-changes.patch": patch, "dataset.sha256": f"{checksum}  {dataset}\n",
    }
    run_dir, exit_code = execute_training(command, PAPER_DIR / "runs", records)
    print(f"运行结束，退出码 {exit_code}；日志：{run_dir / 'train.log'}")
    if exit_code == 0:
        print("本次训练/测试进程完成；是否达到论文指标仍需另行对照。")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"准备检查失败（退出码 {error.returncode}），尚未启动正式训练。") from error
    except (OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}") from error
