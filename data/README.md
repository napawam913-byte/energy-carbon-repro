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

## 仍缺少的课题数据

截至本次迁移，下列文件尚未在已检查的 Windows 位置找到，也没有随本次迁移生成或下载：

- `data/raw/eia/EIA930_BALANCE_2023_Jan_Jun.csv`
- `data/raw/eia/EIA930_BALANCE_2023_Jul_Dec.csv`
- `data/processed/erco_2023_v1/` 下的整理后观测表与 `metadata.json`

旧云端曾生成上述处理表，但尚未核实已迁到当前 L40。不能把本次两份原始 CSV 的迁移说成完整训练数据集迁移；还需找回或重建 EIA/OGE 对齐观测表，核验时间划分、输入可用性和训练集内预处理，才能运行已约定的 168→24 小时课题基线。

本次不搬迁 ETTh1/Electricity 基准数据、不搬迁模型权重、不创建训练样本，也不运行训练。后续数据和运行产物默认仍不公开，不能使用 `git add -f` 批量绕过忽略规则。
