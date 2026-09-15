"""模块用途：获取/检查 ETTh1 原始 CSV 并报告哈希；不覆盖文件、不清洗、不训练。"""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import io
import math
from pathlib import Path
import urllib.request


URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
HEADER = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def validate_data(payload):
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
    if next(reader, None) != HEADER:
        raise ValueError("表头不是 ETTh1 的 date + 7 个变量；可能下载到了网页。")
    count, first, previous = 0, None, None
    for line_number, row in enumerate(reader, 2):
        if len(row) != 8:
            raise ValueError(f"第 {line_number} 行列数不是 8。")
        stamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        if stamp.minute or stamp.second or (previous is not None and stamp - previous != timedelta(hours=1)):
            raise ValueError(f"第 {line_number} 行小时不连续或不在整点。")
        for value in row[1:]:
            if not math.isfinite(float(value)):
                raise ValueError(f"第 {line_number} 行存在非有限数值。")
        first = stamp if first is None else first
        previous = stamp
        count += 1
    if count != 17420:
        raise ValueError(f"期望 17420 行数据，实际 {count} 行；不自动补齐或删行。")
    return count, first, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path,
                        default=Path(__file__).resolve().parents[1] / "code" / "dataset" / "ETTh1.csv")
    parser.add_argument("--check-only", action="store_true", help="仅检查本地文件，不下载")
    args = parser.parse_args()
    target = args.path.expanduser().resolve()
    if target.exists():
        payload = target.read_bytes()
        print("使用已有文件（不覆盖）:", target)
    else:
        if args.check_only:
            parser.exit(1, f"FAIL: 文件不存在：{target}\n")
        print("下载来源:", URL, flush=True)
        # 先在内存校验后再以排他创建方式写入，避免覆盖已有文件或保存错误网页。
        with urllib.request.urlopen(URL, timeout=30) as response:
            payload = response.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise ValueError("下载内容超过 ETTh1 的预期大小上限（10 MiB）。")
        validate_data(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
    count, first, last = validate_data(payload)
    print("文件:", target)
    print("数据行数:", count)
    print("原始日期范围（未假定 UTC）:", first, "至", last)
    print("SHA256:", hashlib.sha256(payload).hexdigest())
    print("PASS: ETTh1 结构与小时连续性检查通过；未进行训练。")
    print("哈希用于记录本次文件，不是与已固定的官方校验和比对。")
    print("ETTh1 是负荷/变压器油温基准，不是碳因子或碳排放数据。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}\n下载失败时可人工上传到 --path 指定的位置，再执行 --check-only。") from error
