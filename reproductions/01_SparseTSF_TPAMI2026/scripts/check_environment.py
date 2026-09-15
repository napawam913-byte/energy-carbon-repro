"""模块用途：只读检查复现环境、指定包版本和 pip 依赖；不安装包、不训练。"""

import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("Python 路径:", sys.executable, flush=True)
    print("Python 版本:", sys.version.split()[0], flush=True)
    print("环境目录:", sys.prefix, flush=True)
    errors = []
    if Path(sys.prefix).name != "sparsetsf-repro":
        errors.append("请先激活独立的 sparsetsf-repro 环境，不要修改共享 base。")
    if sys.version_info[:2] != (3, 10):
        errors.append("本项目兼容性配置使用 Python 3.10。")
    requirements = Path(__file__).resolve().parents[1] / "requirements-repro.txt"
    expected = {"torch": "2.6.0"}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            package, version = line.split("==")
            expected[package] = version
    for package, wanted in expected.items():
        try:
            installed = metadata.version(package)
            print(f"{package}: {installed}（期望 {wanted}）", flush=True)
            if installed.split("+")[0] != wanted:
                errors.append(f"{package} 版本与配置不一致。")
        except metadata.PackageNotFoundError:
            errors.append(f"缺少 {package}。")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    if result.returncode:
        errors.append("pip check 未通过；先处理上面的依赖提示。")
    if errors:
        for error in errors:
            print("FAIL:", error, file=sys.stderr)
        return 1
    print("PASS: 指定包版本与依赖检查通过；GPU 仍需单独检查。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
