# TEFN：时间证据融合网络

[返回总清单](../../论文复现清单.md)

## 论文与本地状态

IEEE Transactions on Pattern Analysis and Machine Intelligence，47(12): 11220–11233，2025 年 12 月。

[正式论文 / DOI](https://doi.org/10.1109/TPAMI.2025.3596905)。本目录只完成资料准备；实际源码/PDF获取和核验状态见 [source-lock.json](source-lock.json)。没有安装依赖、执行模型导入、启动训练或验证论文指标。

## 目录用途

- [paper/](paper/README.md)：论文与版本说明。
- [code/](code/)：作者固定提交的源码快照，不包含 Git 历史；原始文件保持不改写。
- [runs/](runs/README.md)：后续云端实验回传记录，不使用作者自带结果冒充本项目实测结果。

## 固定源码来源

- 仓库：[ztxtech/Time-Evidence-Fusion-Network](https://github.com/ztxtech/Time-Evidence-Fusion-Network.git)
- 分支：`master`
- 提交：`4577c3397b4f095d49c885b72432c74919b39dca`
- 获取方式：GitHub 官方 codeload 源码快照；完整性核验结果见版本记录。
- 许可证：MIT。

## 作者入口索引（仅供查看，未执行）

- [作者说明](code/README.md)
- [模型目录](code/models)
- [运行入口](code/run.py)
- [配置入口](code/run_config.py)
- [训练与评估](code/exp/exp_long_term_forecasting.py)
- [实验配置](code/configs)
- [依赖声明](code/requirements.txt)

## 复现前必须注意

- arXiv `2405.06419v4`（2025-08-06）标注 TPAMI 接收并关联期刊 DOI；应称为作者公开版本，不是 IEEE 排版版。
- 仓库 README 仍以 2024 年预印本引用，需把论文表格、模型定义和具体配置逐项对应后再启动复现。
- “多源”指时间/通道维度的信息融合，不自动等于能源系统的物理耦合或碳守恒。
- `run_config_dir.py` 等批量入口可能运行许多任务；本轮没有运行任何作者脚本。

## 云端获取相同源码

以下为以后在云端该论文目录、且 `code/` 尚不存在时使用的命令。本轮没有执行这些训练环境操作，也没有创建服务器连接。

```bash
git clone --single-branch --branch master https://github.com/ztxtech/Time-Evidence-Fusion-Network.git code
git -C code checkout --detach 4577c3397b4f095d49c885b72432c74919b39dca
git -C code rev-parse HEAD
```

只在新克隆目录执行上述检出。若已经有代码或改动，先查看状态，不覆盖。固定提交用于还原本次快照，不代表它一定就是论文发表时的实验版本。

