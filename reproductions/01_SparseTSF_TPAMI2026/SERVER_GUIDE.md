# SparseTSF：服务器准备脚本

`scripts/` 中的环境、GPU、ETTh1 检查不启动正式训练。
现已新增论文目录下的 [train_etth1_linear.py](train_etth1_linear.py)，将用户已执行的单项训练命令封装为普通 Python 入口。
该文件默认会重新训练；作者源码保持不变，训练完成不等于已达到论文指标。

## VS Code 直接运行：现在使用这个入口

先在服务器更新我们的脚本仓库，不需要更新作者仓库：

```bash
git -C "$HOME/桌面/xc/energy-carbon-repro" -c http.version=HTTP/1.1 pull --ff-only
```

1. 在 **Linux 服务器上的 VS Code** 打开 `/home/gdp/桌面/xc/energy-carbon-repro`。
2. 按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`，指定
   `/home/gdp/.conda/envs/sparsetsf-repro/bin/python`。列表没有时使用 `Enter interpreter path` 手动选择。
3. 打开 `/home/gdp/桌面/xc/energy-carbon-repro/reproductions/01_SparseTSF_TPAMI2026/train_etth1_linear.py`。
4. 点击右上角 **Run Python File in Terminal / 在终端中运行 Python 文件**。
   使用 Microsoft Python 扩展的按钮，不是 Code Runner 的 `Run Code`，也不是选择部分代码后 `Shift+Enter`。

[VS Code 运行按钮说明](https://code.visualstudio.com/docs/python/run)；
[解释器选择说明](https://code.visualstudio.com/docs/python/environments)。

本入口沿用选中的解释器，在训练前调用已有环境和 GPU 检查；不自动安装依赖。
从任意当前工作目录点击都可以，路径依据本文件位置解析。
先寻找当前论文目录的 `code/`，没有时寻找你已经克隆的
`/home/gdp/桌面/xc/reproductions/01_SparseTSF_TPAMI2026/code`。
自定义布局可以在终端通过 `--code-dir` 指定；不会全盘搜索或自动克隆。

**点击运行会启动一次新训练，不是打开上次模型，也不是只检查 GPU。**
参数写在入口顶部的 `TRAINING_ARGS`：linear、720→96、7 变量、batch 256、学习率 0.02、
最多 30 轮、早停 5、`itr=1`（固定作者入口中对应种子 2023）。默认参数与刚才命令相同；
如自行改动，要作为新实验记录，不能仍称为原始配置。

入口校验作者 Git 提交及用户首轮数据 SHA256。作者代码存在已跟踪文件改动时会提示并记录差异，
但不会重置你的改动；此类运行不能直接声称是未修改源码的复现。
数据哈希来自用户首轮通过检查的文件记录，不是出版社发布的校验和。

每次新建 `runs/etth1_linear_720_96_<唯一后缀>/`，不覆盖已有实验，保存：

- `train.log`：训练过程与最终测试输出。
- `run.json`、`exitcode.txt`：实际命令、工作目录、开始/结束时间、进程状态与退出码。
- `environment.txt`、`source-commit.txt`、`source-changes.patch`、`dataset.sha256`：环境与来源记录。
- `checkpoints/`、`result.txt`、`test_results/`：由作者程序生成的最佳模型、指标记录与预测图。

这些输出只保存在服务器的本次实验目录，不会自动上传 GitHub。
运行失败不自动重试；在运行终端按 `Ctrl+C` 可主动中止。
这里只配置普通运行；若需要在作者模型内部打断点，需另外配置子进程调试。

仅预览配置而不训练（不验证 GPU、不创建实验目录）：

```bash
python "$HOME/桌面/xc/energy-carbon-repro/reproductions/01_SparseTSF_TPAMI2026/train_etth1_linear.py" --dry-run
```

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

## 5. 单项训练与论文复现的边界

当前入口不运行 `run_all.sh`，不把环境检查 PASS 或单次训练结束当作论文复现成功。
用户已回传 ETTh1-linear 首轮运行结束的截图；完整日志与论文目标表格尚未一起核验。
做论文指标对照前，仍需确认目标表格、linear/MLP 版本、数据划分及参数是否一致。
尤其注意固定源码中的 `run_longExp.py` 默认 `model_type=mlp`，而 `linear/etth1.sh` 没有显式覆盖；
复现 linear 时必须显式传 `--model_type linear`，不能仅依据脚本所在文件夹判断。

作者的训练流程每轮打印验证集和测试集结果；只允许依据验证集选择最佳模型和超参数。
这些准备脚本没有改动作者的数据划分、训练循环或评价口径。
正式期刊全文/目标表格尚未在本地核验，不设置或编造所谓“应达到的论文分数”。

## 本地脚本测试

```bash
python -B -m unittest discover -s "$paper/tests" -v
```

测试使用临时合成数据、路径夹具和仅打印文字的轻量子进程，验证文件保护、参数、日志与退出码，不需要 GPU 或作者仓库，不会训练模型。
本轮 Windows 自动测试不能代替你服务器上的真实 GPU 检查。
