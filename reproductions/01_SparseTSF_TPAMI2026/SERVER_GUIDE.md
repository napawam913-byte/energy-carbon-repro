# SparseTSF：服务器准备脚本

本次只把已讨论的环境、GPU、ETTh1 检查整理成文件，不启动正式训练，也不宣称复现了论文指标。
Python 文件位于 `scripts/`，与作者的 `code/scripts/` 不同；作者源码保持不变。

## 1. 使用你已经创建的独立环境

```bash
conda activate sparsetsf-repro
command -v python
python --version
```

应指向自己的 `sparsetsf-repro` 环境，Python 3.10。若此前用前缀创建而名称无法激活，请使用实际前缀，例如：

```bash
conda activate "$HOME/.conda/envs/sparsetsf-repro"
```

不要在共享的 `base` 环境安装；已有环境不必重复创建。

### 针对当前缺失依赖的提示

你提供的日志显示 `idna`、`pexpect`、`PyYAML`、`ptyprocess` 缺失，涉及 HTTP/Jupyter 等已安装包。
这不等于 SparseTSF 计算失败。确认激活正确环境后执行：

```bash
python -m pip install idna pexpect PyYAML ptyprocess
python -m pip check
```

看到 `No broken requirements found.` 才表示依赖声明检查通过。
若还有提示，保留完整输出，不要通过 `--no-deps`、卸载共享包或忽略报错掩盖问题。
[pip check 官方说明](https://pip.pypa.io/en/stable/cli/pip_check/)。

## 2. 脚本仓库与作者源码分开

公共脚本仓库不内嵌作者源码、不上传论文 PDF、不上传数据和训练产物。
以下用两条路径变量，避免将服务器上的旧目录覆盖掉：

```bash
# 从 xc 下克隆本项目后，脚本位于这里。
paper="$HOME/桌面/xc/energy-carbon-repro/reproductions/01_SparseTSF_TPAMI2026"
# 你已经成功克隆的作者源码，继续使用。
upstream="$HOME/桌面/xc/reproductions/01_SparseTSF_TPAMI2026/code"
ls "$upstream/models/SparseTSF.py"
git -C "$upstream" status --short
git -C "$upstream" rev-parse HEAD
```

本地资料记录的作者提交为 `b8c2740eecc84d8095ffce49ba5acafe68e53bb8`。
若版本不一致或有修改，先记录差异；不要直接覆盖已有工作。
全新服务器获取作者源码的方法见 [README](README.md#云端获取相同源码)。

`No module named 'models'` 的原因：之前的交互式 Python 从论文根目录运行，而作者的
`models/` 在 `code/` 中。新脚本会使用 `--code-dir` 定位，无需 `pip install models`。

## 3. 检查环境，然后检查 GPU

已安装好依赖时直接运行：

```bash
python "$paper/scripts/check_environment.py"
python "$paper/scripts/check_gpu.py" --code-dir "$upstream" --paths-only
python "$paper/scripts/check_gpu.py" --code-dir "$upstream" --model-type linear
```

必须逐条确认通过；`--paths-only` 只检查文件存在，不代表 GPU 通过。
GPU 检查使用随机输入 `(2, 720, 7)`，预期输出 `(2, 96, 7)`；这组配置的 linear 版本有 145 个参数。
它会计算一次前向和梯度，但不会更新参数、读取训练集、保存模型或评估预测精度。
需要检查期刊扩展中的 MLP 结构时：

```bash
python "$paper/scripts/check_gpu.py" --code-dir "$upstream" --model-type mlp
```

这组 MLP 配置预期参数量 4509。这不是选择模型优劣的实验。

### 全新环境才需要的安装命令

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$paper/requirements-repro.txt"
python -m pip check
```

采用 Python 3.10、PyTorch 2.6.0/cu124 及指定科学计算包，是本项目的兼容性配置，
不是作者提供的完整环境锁。`requirements-repro.txt` 没有锁定全部间接依赖，
服务器通过检查后应保留完整环境记录。作者 `utils/tools.py` 使用了 `np.Inf`，因此这里固定 NumPy 1.26.4，
不要直接升级到 NumPy 2。无需安装 TensorFlow，也不需要让 PyTorch CUDA 版本与 `nvidia-smi` 显示的驱动支持版本完全相同。
[PyTorch 官方版本安装命令](https://pytorch.org/get-started/previous-versions/)。

## 4. 准备 ETTh1

```bash
python "$paper/scripts/prepare_etth1.py" --path "$upstream/dataset/ETTh1.csv"
```

- 文件已存在：只读检查，绝不覆盖，即使文件内容损坏也不自动替换。
- 文件不存在：从 [ETDataset 官方仓库](https://github.com/zhouhaoyi/ETDataset) 获取单个 ETTh1 CSV。
- 网络不通：可人工上传到上述位置，然后加 `--check-only` 检查；不要关闭 HTTPS 证书校验。
- 检查项目：表头、17420 行、7 个有限数值变量、小时连续性；输出 SHA256 便于记录。
- SHA256 只记录本次文件，并未与固定官方校验和比较，结构检查也不能证明数值与论文版本完全一致。

ETTh1 是负荷与变压器油温数据，**不是已经核算的碳因子，也不是碳排放量**。
这一阶段是验证论文基准上的复现条件；随后才迁移到统一的能源/碳因子数据做基线对比。

## 5. 正式训练前的边界

当前不运行 `run_all.sh`，不把环境检查 PASS 当作论文复现成功。
下一步确认目标表格、linear/MLP 版本、数据划分和训练参数后，再执行单项训练。
尤其注意固定源码中的 `run_longExp.py` 默认 `model_type=mlp`，而 `linear/etth1.sh` 没有显式覆盖；
复现 linear 时必须显式传 `--model_type linear`，不能仅依据脚本所在文件夹判断。

作者的训练流程每轮打印验证集和测试集结果；只允许依据验证集选择最佳模型和超参数。
这些准备脚本没有改动作者的数据划分、训练循环或评价口径。
正式期刊全文/目标表格尚未在本地核验，不设置或编造所谓“应达到的论文分数”。

## 本地脚本测试

```bash
python -B -m unittest discover -s "$paper/tests" -v
```

测试使用临时合成数据和路径夹具验证文件保护与异常处理，不需要 GPU 或作者仓库，不会训练。
本轮 Windows 自动测试不能代替你服务器上的真实 GPU 检查。
