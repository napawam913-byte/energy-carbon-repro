# 能源—碳因子数据目录

本目录管理课题数据及其来源，不执行清洗、训练或模型评估。原始文件与处理产物分开存放。

## 本次迁移范围（2026-09-17）

本次仅复制本地已找到的两份 OGE 原始 CSV，保留原位置文件。文件内容未改动；两份 SHA256 与此前用户回传的云端观测表来源记录一致。这说明原文件版本一致，不代表已重新验证全部数据质量或预测流程。

| 文件 | 字节数 | 内容与用途 |
|---|---:|---|
| [power_sector_data/ERCO.csv](raw/oge/v0.8.0/2023/power_sector_data/ERCO.csv) | 24,656,092 | ERCO 按能源类别记录的发电量、排放量和发电侧因子；后续重建能源—碳因子观测表的来源 |
| [carbon_accounting/ERCO.csv](raw/oge/v0.8.0/2023/carbon_accounting/ERCO.csv) | 1,163,006 | 区域消费侧排放因子；核算边界不同，不直接替代发电侧预测标签 |

两份文件同名，但分属不同子目录，不能互相覆盖。详见 [来源清单](raw/oge/v0.8.0/2023/manifest.json) 和 [SHA256 校验和](raw/oge/v0.8.0/2023/SHA256SUMS)。

```text
data/
├── README.md
└── raw/oge/v0.8.0/2023/
    ├── manifest.json
    ├── SHA256SUMS
    ├── power_sector_data/ERCO.csv
    └── carbon_accounting/ERCO.csv
```

## 来源、署名与许可

数据提供者：Singularity Energy / Open Grid Emissions Initiative。版本 `v0.8.0`，数据年 `2023`，区域 `ERCO`，小时粒度、公制单位。

- [官方数据入口](https://singularity.energy/open-grid-emissions)
- [官方 OGE 介绍及 CC-BY-4.0 数据许可说明](https://singularity.energy/tools)
- [CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)
- [发电侧原始下载地址](https://open-grid-emissions.s3.amazonaws.com/open_grid_emissions_data/v0.8.0/results/2023/power_sector_data/hourly/metric_units/ERCO.csv)
- [消费侧原始下载地址](https://open-grid-emissions.s3.amazonaws.com/open_grid_emissions_data/v0.8.0/results/2023/carbon_accounting/hourly/metric_units/ERCO.csv)

修改说明：未修改 CSV 内容，仅调整本项目内存放路径、增加清单和校验和。研究引用时应保留提供者、版本、来源及数据许可。OGE 数据许可与处理工具的代码许可不是同一回事。

## 云端获取与校验

用户于 2026-09-17 明确批准将上述两份公开 CSV 推送到公共仓库 `napawam913-byte/energy-carbon-repro`。它们是原“数据不入库”约定的定向例外，后续新增数据不自动获得公开授权。本次版本将 CSV 连同来源清单、校验和纳入 Git；云端获取后仍需执行下面的校验，不能仅凭说明文件存在就认定 CSV 已下载完整。

当前 L40 已有辅助仓库时，在仓库内快进更新，不需要重复克隆：

```bash
cd "/home/gdp/桌面/xc/energy-carbon-repro" &&
git -c http.version=HTTP/1.1 pull --ff-only &&
sha256sum -c data/raw/oge/v0.8.0/2023/SHA256SUMS
```

换到新服务器且目标目录尚不存在时：

```bash
git -c http.version=HTTP/1.1 clone https://github.com/napawam913-byte/energy-carbon-repro.git &&
cd energy-carbon-repro &&
sha256sum -c data/raw/oge/v0.8.0/2023/SHA256SUMS
```

两份文件均显示 `OK` 才表示本次传输校验通过。整个过程只使用 Git 和系统校验工具，无需 GPU、Python 或重新安装环境。如果 `pull` 提示存在冲突或本地改动，请保留文件并检查；不要强制重置仓库。

## 迁移时的缺项与后续补齐

两份 OGE CSV 最初迁移时，下列文件尚未在已检查的 Windows 位置找到，未随该次迁移生成或下载：

- `data/raw/eia/EIA930_BALANCE_2023_Jan_Jun.csv`
- `data/raw/eia/EIA930_BALANCE_2023_Jul_Dec.csv`
- `data/processed/erco_2023_v1/` 下的整理后观测表与 `metadata.json`

2026-09-17 后续进展：用户已在当前 L40 下载两份 EIA 文件并回传匹配的 SHA256；本地也从 EIA 官方下载同版本文件。新增脚本已在 Windows 本地成功重建 8760 小时观测表；服务器上的脚本运行尚待用户执行。公式、边界及检查结果见 [当天进度](../docs/progress/2026-09-17.md)。

本次不搬迁 ETTh1/Electricity 基准数据、不搬迁模型权重、不创建训练样本，也不运行训练。后续数据和运行产物默认仍不公开，不能使用 `git add -f` 批量绕过忽略规则。

## 重建小时观测表（数据处理，不是训练）

执行入口：[build_erco_observations.py](../scripts/build_erco_observations.py)。脚本检查原文件版本、统一小时键、对齐 EIA/OGE，计算分能源与综合发电侧 CO₂ 因子并写出审计记录。不修改原始数据，不自动插补，不划分数据集，不训练模型。

需要 Python 3.10+、NumPy、pandas。先使用已有 `sparsetsf-repro` 环境，无需 GPU；已本地验证 Python 3.10/3.13，Linux 上仍须实际执行测试。

确认四份原文件均位于上列路径后，在仓库根目录执行：

```bash
python -m unittest discover -s tests -p test_erco_observations.py -v &&
python scripts/build_erco_observations.py
```

默认生成 `data/processed/erco_2023_v1/observations.csv` 和 `metadata.json`，成功后显示 `PASS: observations built and re-read; no training performed.`。输出目录必须尚不存在；需另行重跑时可指定 `--output-dir data/processed/erco_2023_check2`，不会覆盖已有产物。

四份来源的 SHA256 均在脚本内固定。原文件缺失或哈希不符时请检查下载和版本，不绕过校验。EIA 数据不随仓库发布，换服务器后可从官方重新下载：

- [EIA930 2023 上半年](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv)
- [EIA930 2023 下半年](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jul_Dec.csv)

观测表还不是可以直接喂给预测模型的全部输入。后续仍要固定训练/验证/测试时间边界、只用训练集拟合预处理、限制预测时可用变量，并建立 168→24 小时样本。
