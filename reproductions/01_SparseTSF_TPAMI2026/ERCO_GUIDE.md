# SparseTSF 原模型：ERCO 训练与验证

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-17
- Verification Status: UNVERIFIED（真实 ERCO 神经网络效果及 Linux GPU 运行待服务器回传）
- Version Label: sparsetsf_erco_v1

这是将作者原模型用于本项目数据的基线，不是原论文包含 ERCO/碳因子实验的证据。网络不重写；数据协议、目标选择和训练/验证流程由本项目适配。代码测试与真实训练成绩必须分开记录。

## 1. 先更新和预检，不训练

已有服务器辅助仓库和作者仓库分别位于以下两个目录；不要重新嵌套 clone，不需要重新生成已通过的观测表和准备清单。

```bash
conda activate sparsetsf-repro &&
cd "/home/gdp/桌面/xc/energy-carbon-repro" &&
git -c http.version=HTTP/1.1 pull --ff-only &&
export SPARSE_TSF_CODE_DIR="/home/gdp/桌面/xc/reproductions/01_SparseTSF_TPAMI2026/code" &&
python -m unittest discover -s tests -v &&
python reproductions/01_SparseTSF_TPAMI2026/train_erco.py --code-dir "$SPARSE_TSF_CODE_DIR" --check-only
```

单元测试使用临时合成数据、CPU 和小规模优化，不训练真实 ERCO。若模型测试被 skipped，表示缺少 PyTorch 或作者源码，不能将跳过视为通过。`--check-only` 核对源码、准备清单、字段映射，并在 GPU 做一次真实历史窗口的前向检查；不反向传播、不更新参数、不创建实验结果目录。

默认作者版本：`b8c2740eecc84d8095ffce49ba5acafe68e53bb8`。实际加载的 `models/SparseTSF.py` 和依赖 `layers/Embed.py` 会校验内容 SHA256，允许 CRLF/LF 换行差异，不允许网络代码变化。加载器直接执行本次已核验的源码字节，不使用旧 `.pyc` 或缓存的同名模型类，也不修改作者文件。已有作者源码不符时停止并回传提示，不自动覆盖、更换提交或删除你的修改。

成功预检应出现：

```text
"status": "PREFLIGHT_ONLY_NO_TRAINING"
"parameters": 32
"output_shape": [1, 24, 10]
"device": "cuda"
PASS: author source, data mapping and CUDA forward checked; no training or files written.
```

实际 JSON 还包含来源、配置和划分。GPU 不可用会报错，不会自动降级 CPU。继续使用已安装的 Python 3.10 / PyTorch 2.6.0+cu124 环境，不为这一步盲目升级。

## 2. 再启动一次正式 linear 训练

确认预检通过后执行：

```bash
conda activate sparsetsf-repro &&
cd "/home/gdp/桌面/xc/energy-carbon-repro" &&
python reproductions/01_SparseTSF_TPAMI2026/train_erco.py \
  --code-dir "/home/gdp/桌面/xc/reproductions/01_SparseTSF_TPAMI2026/code"
```

这一步才会反向传播并执行 Adam 参数更新。默认参数保存在 `configs/sparsetsf_erco.json`：linear、周期 24、种子 2023、batch size 128、初始学习率 0.02、最多 30 轮、patience 5。学习率沿用作者 type3；不自动运行 MLP、其他种子或参数搜索。

输入历史 168 小时的 11 列，经作者模型输出 11 列，然后在模型外选择九类发电量及综合发电侧 CO₂ 因子，共 10 列。linear 参数量 32 是该长度配置下的实际值，不是改写网络。作者模型按通道独立计算，负荷通道不进入损失，也不会影响其他目标；该基线不具备跨能源输入融合。

训练/验证窗口数仍为 5635/1441，最后一个小 batch 保留。所有目标在训练段标准化后等权计算 MSE。每轮只打印 Train MSE 和 Val MSE，不打印 Test Loss。最佳 checkpoint 按十目标综合验证标准化 MSE 选择，不能事后按碳因子单项误差挑另一轮。

默认不启动后台任务。请保持该进程运行；关闭终端可能中断训练。入口监督唯一训练子进程，默认 1800 秒硬超时，计入启动开销；超时/中断不重试。若需要改变时限，另存配置并显式 `--config` 指定，改动将记录在运行清单。

## 3. 保存位置与成功条件

启动时打印精确输出目录，默认格式：

```text
/home/gdp/桌面/xc/energy-carbon-repro/data/experiments/erco_168_24/sparsetsf_<UTC时间戳>/
├── run_config.json                # 配置、标准化参数、数据/源码哈希、环境、设备与协议
├── train.log                      # 逐轮日志
├── history.csv                    # train/val MSE、学习率、最佳轮次、耗时和样本数
├── checkpoint.pt                  # 最佳作者模型的 state_dict
├── validation_predictions.npz     # 预测、真实值、字段名、预测起点和目标 UTC 时间
├── metrics.csv                    # 验证集 10 个目标的原单位指标
├── per_horizon_metrics.csv        # 24 步 × 10 目标，240 行
├── metrics.json                   # 同一指标的结构化记录及比较用来源哈希
└── run.json                       # 最终状态、最佳轮次、checkpoint 重载损失与产物哈希
```

第一次有效验证会保存 checkpoint，随后只在验证 MSE 严格下降时更新本轮运行的同一个文件。训练结束重新加载最佳权重，再导出验证预测；不是使用最后一轮权重。保存最佳模型不等于支持中断续训，本入口未实现优化器状态恢复。

成功需同时满足：进程退出码 0；`run.json` 状态为 `COMPLETED_VALIDATION_ONLY`；不存在 `termination.json`。超时留下 `termination.json`，内容为 `TIMED_OUT`，即使存在 checkpoint 也不能算完整成功。运行异常会尽可能留下 `FAILED` 的 run.json；校验阶段提前失败可能尚无运行目录。失败产物保留便于排查，既有目录不覆盖，下一次重新运行创建新目录。

运行结束后回传最后几行（含输出目录和验证 MAE/RMSE）。不要将合成测试里打印的分数写入正式基线表。

## 4. 结果怎么比较

- 与 `data/experiments/erco_168_24/naive_validation_v1/metrics.json` 比较，先核对观测 SHA256、准备清单 SHA256、目标列和验证范围。
- 逐变量比较 MAE、RMSE、MSE，分别使用 MWh 或 kg CO₂/MWh；MSE 单位相应平方。
- 不把各物理单位的原始误差混合平均，不把这些分数和 ETTh1/Electricity 的论文分数直接相减。
- 不裁剪负预测、不用未来真实发电量或排放量辅助预测，不依据测试集调整配置。
- 本轮是单种子验证，不代表统计显著性，也不代表工业园实测或实时部署效果。最终测试集另行明确启动。

`validation_predictions.npz` 不需要 pickle，结构为 `prediction/actual=[1441,24,10]`、`targets=[10]`、`origin_utc=[1441]`、`target_utc=[1441,24]`。预测和真实值均为原单位，可用于独立复算和后续画图。

## 5. 文件边界

- 作者网络：独立源码目录里的 `models/SparseTSF.py`，没有改动。
- 本项目加载器：`energy_forecast/sparsetsf.py`，校验来源、直接实例化原 Model、映射目标。
- 本项目训练器：`energy_forecast/training.py`，负责优化和产物，不定义神经网络。
- 此训练入口：`reproductions/01_SparseTSF_TPAMI2026/train_erco.py`。

这些运行产物仍被 Git 忽略。推送只包含实现、配置、测试和说明，不上传 checkpoint、预测、处理表或新数据。
