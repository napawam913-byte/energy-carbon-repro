# CAISO：风电、光伏、负荷日前联合预测

[返回总清单](../../论文复现清单.md)

## 论文与本地状态

Nature Communications，17: 3307，2026 年。

[正式论文 / DOI](https://doi.org/10.1038/s41467-026-69015-w)。本目录只完成资料准备；实际源码/PDF获取和核验状态见 [source-lock.json](source-lock.json)。没有安装依赖、执行模型导入、启动训练或验证论文指标。

## 目录用途

- [paper/](paper/README.md)：论文与版本说明。
- [code/](code/)：作者固定提交的源码快照，不包含 Git 历史；原始文件保持不改写。
- [runs/](runs/README.md)：后续云端实验回传记录，不使用作者自带结果冒充本项目实测结果。

## 固定源码来源

- 仓库：[gterren/caiso_power](https://github.com/gterren/caiso_power.git)
- 分支：`POD`
- 提交：`89c41c10e6c8f7d2313957822d376f16e043f7fd`
- 获取方式：GitHub 官方 codeload 源码快照；完整性核验结果见版本记录。
- 许可证：Apache-2.0。

## 作者入口索引（仅供查看，未执行）

- [作者说明](code/README.md)
- [主要代码目录](code/software/shallow_learning)
- [处理后数据与结果](https://doi.org/10.5281/zenodo.16729434)
- [论文代码归档](https://doi.org/10.5281/zenodo.18156677)

## 复现前必须注意

- 固定 `POD` 分支，不能误用其他研究分支。仓库 README 仍使用投稿/修订阶段旧标题，不代表期刊尚未发表。
- 作者说明涉及 MPI/SLURM、高斯过程模型及外部 `Cool_MTGP` 依赖；本轮没有获取依赖仓库、安装环境或提交集群任务。
- 天气特征、数据预处理路径及硬件运行方式需要单独适配，不能直接承诺单卡 4090 的耗时或性能。
- 论文目标是 CAISO 风电、光伏、负荷联合概率预测，不直接输出碳因子。迁移 ERCO 是后续另一个实验。
- 源码中的 `drive.sh`、`.job` 文件可能批量提交 HPC 任务，不能未经检查直接执行。

## 云端获取相同源码

以下为以后在云端该论文目录、且 `code/` 尚不存在时使用的命令。本轮没有执行这些训练环境操作，也没有创建服务器连接。

```bash
git clone --single-branch --branch POD https://github.com/gterren/caiso_power.git code
git -C code checkout --detach 89c41c10e6c8f7d2313957822d376f16e043f7fd
git -C code rev-parse HEAD
```

只在新克隆目录执行上述检出。若已经有代码或改动，先查看状态，不覆盖。固定提交用于还原本次快照，不代表它一定就是论文发表时的实验版本。

