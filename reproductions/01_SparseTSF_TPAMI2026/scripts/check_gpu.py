"""模块用途：定位作者源码并检查 GPU 前向/反向；不更新参数，不进行正式训练。"""

import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "code",
                        help="作者源码目录；默认使用本论文目录下的 code")
    parser.add_argument("--model-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--paths-only", action="store_true", help="仅检查源码路径，不导入 PyTorch")
    args = parser.parse_args()
    code = args.code_dir.expanduser().resolve()
    model_file = code / "models" / "SparseTSF.py"
    if not model_file.is_file() or not (code / "layers" / "Embed.py").is_file():
        parser.exit(1, f"FAIL: {code} 缺少 models/SparseTSF.py 或 layers/Embed.py。\n"
                    "请先克隆作者源码到 code，或用 --code-dir 指定现有源码目录。\n")
    print("作者模型:", model_file, flush=True)
    if args.paths_only:
        print("PASS: 源码文件存在；尚未执行模型或 GPU 检查。")
        return 0

    # 解决在论文根目录运行时找不到 models；不修改作者源码或系统 PYTHONPATH。
    sys.path.insert(0, str(code))
    sys.dont_write_bytecode = True
    import torch
    model_module = importlib.import_module("models.SparseTSF")
    if Path(model_module.__file__).resolve() != model_file:
        raise RuntimeError("导入了其他目录的 models；请使用新 Python 进程运行此文件。")
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 未识别到 GPU。请检查 nvidia-smi 和 PyTorch CUDA 安装。")
    print("GPU:", torch.cuda.get_device_name(0))
    torch.manual_seed(2023)
    config = SimpleNamespace(seq_len=720, pred_len=96, enc_in=7,
                             period_len=24, d_model=128, model_type=args.model_type)
    model = model_module.Model(config).cuda()
    x = torch.randn(2, 720, 7, device="cuda")
    y = model(x)
    if tuple(y.shape) != (2, 96, 7) or not y.is_cuda or not torch.isfinite(y).all().item():
        raise RuntimeError("模型输出的形状、设备或有限值检查失败。")
    y.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not parameter.grad.is_cuda or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"梯度检查失败：{name}")
    torch.cuda.synchronize()
    print("模型类型:", args.model_type)
    print("输出维度:", tuple(y.shape))
    print("参数量:", sum(parameter.numel() for parameter in model.parameters()))
    print("PASS: SparseTSF GPU 前向与反向计算通过")
    print("仅使用随机输入检查计算；未执行 optimizer.step，未复现论文精度。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, RuntimeError, OSError) as error:
        raise SystemExit(f"FAIL: {error}") from error
