# ERCO 168→24 小时预测：数据准备与基线入口

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-17
- Verification Status: UNVERIFIED（论文效果与实时可用性未验证；代码测试和本地运行范围见当天进度）
- Version Label: erco_forecast_preparation_v1

本说明落实已经确认的数据协议，不提出新的实验问题。本阶段只建立样本边界和两个朴素验证基线；不是 SparseTSF 课题训练，也不是完整论文复现。

## 1. 三个步骤不能混在一起

| 入口 | 作用 | 是否训练 |
|---|---|---|
| `scripts/build_erco_observations.py` | 检查四份原始文件、对齐时间、核算历史碳因子，生成 8760 小时观测表 | 否 |
| `prepare_forecast_data.py` | 固定字段与 UTC 时间划分、只用训练段拟合标准化参数、登记滑窗起点 | 否 |
| `run_naive_baselines.py` | 在验证集上计算最近值和 24 小时周期基线的误差 | 否；没有可学习权重 |

所有脚本无需 GPU，也不导入 PyTorch。可以继续使用已有 `sparsetsf-repro` 环境，不要为这一步升级或重装训练环境。

## 2. 输入、目标和时间边界

配置：[configs/erco_2023.json](../configs/erco_2023.json)。输入为过去 168 小时的负荷、九类净发电量和综合发电侧 CO₂ 因子，共 11 列；输出为未来 24 小时的九类净发电量和综合因子，共 10 列。负荷不列为预测目标。

| 集合 | 目标时间范围（UTC，左闭右开） | 原始小时数 | 有效窗口数 |
|---|---|---:|---:|
| 训练 | 2023-01-01 06:00 至 2023-09-01 00:00 | 5826 | 5635 |
| 验证 | 2023-09-01 00:00 至 2023-11-01 00:00 | 1464 | 1441 |
| 测试 | 2023-11-01 00:00 至 2024-01-01 06:00 | 1470 | 1447 |

每个样本的第一待预测行为 `j`：历史 `X=[j-168,j)`，标签 `y=[j,j+24)`。完整标签窗口不得跨集合；验证/测试可以利用前一段已经发生的历史。它是每小时滚动预测，不是只在每天零点预测。

标准化均值和标准差仅来自训练时段；验证/测试不重新拟合。保存的 `manifest.json` 固定字段顺序、划分、原文件 SHA256、标准化参数及代码/环境信息。窗口按需切片，不另外保存庞大的三维数组。

非正发电量下的单能源因子 NaN 不属于本轮选入字段，不填零也不整行删除。未来实际发电量、排放、参考因子和未来质量标记不能当作预测时已知输入。OGE 的实际发布时间仍未验证，本任务保持离线回溯试验边界。

## 3. 服务器命令

先确认观测构建脚本已经打印 PASS，且 `data/processed/erco_2023_v1/observations.csv` 存在。本地 Windows 运行成功不能代替服务器验收。

```bash
conda activate sparsetsf-repro &&
cd "/home/gdp/桌面/xc/energy-carbon-repro" &&
git -c http.version=HTTP/1.1 pull --ff-only &&
python -m unittest discover -s tests -v &&
python prepare_forecast_data.py &&
python run_naive_baselines.py
```

准备成功后应显示：

```text
Windows: train=5635 val=1441 test=1447
X: (168, 11); y: (24, 10)
PASS: forecasting data prepared; no training or test-set evaluation.
```

这里 `Windows` 指“滑动窗口数量”，不是操作系统。基线成功后显示：

```text
PASS: validation baselines saved; no training or test-set evaluation.
```

输出路径：

```text
data/processed/erco_168_24_v1/manifest.json
data/experiments/erco_168_24/naive_validation_v1/
├── metrics.json                 # 两种方法、逐变量及逐步长指标、来源和配置
├── metrics.csv                  # 2 个方法 × 10 个目标，共 20 行
└── per_horizon_metrics.csv      # 2 个方法 × 24 个步长 × 10 个目标，共 480 行
```

上述产物被 Git 忽略，不随脚本发布。所有输出目录都必须是新目录，已有目录不会被覆盖。需要独立重跑时：

```bash
python prepare_forecast_data.py --output data/processed/erco_168_24_check2 &&
python run_naive_baselines.py --prepared data/processed/erco_168_24_check2 --output data/experiments/erco_168_24/naive_validation_check2
```

清单记录生成机器的绝对观测路径，不应直接将 Windows 清单搬到 Linux 使用；在服务器用相同观测表重新执行准备入口。

## 4. 两个基线表示什么

- `persistence`：未来每一步都用预测前最后一小时的值。
- `seasonal24`：重复预测前已知的最后 24 小时。预测超过 24 步时循环这段历史，不读取未来实际值。

基线共用同一批验证起点与十个目标。保存原单位的 MSE、MAE、RMSE；MSE 单位是目标单位的平方。不同变量的误差不混合平均，含零发电序列也不以 MAPE/“准确率百分比”作为主指标。重叠窗口有时间依赖，1441 个起点不代表 1441 个独立统计样本。

可以用这些值检查后续模型是否学到比简单历史重复更有价值的模式；它们不能证明任何论文模型已经复现成功，也不能仅凭某次误差下降认定研究创新。

## 5. 后续接入 SparseTSF 的边界

本轮尚未增加 ERCO 的 SparseTSF 训练入口。下一步将固定作者源码版本，单独适配当前窗口与十个目标、统一标准化和指标、仅按验证集选 checkpoint，再另行执行最终测试评估。

尤其要记录：作者 SparseTSF 的 channel-independent 结构分别处理各变量历史，不会因为把负荷列放进输入就自动利用负荷影响风电或碳因子。接入时必须明确负荷通道的处理与损失范围，不能把新增跨变量模块仍称为未改动的原始 SparseTSF。
