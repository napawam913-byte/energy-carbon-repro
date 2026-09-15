# HEFTCom：风光联合概率预测

[返回总清单](../../论文复现清单.md)

## 论文与本地状态

International Journal of Forecasting，42(3): 736–751，2026 年。

[正式论文 / DOI](https://doi.org/10.1016/j.ijforecast.2025.11.008)。本目录只完成资料准备；实际源码/PDF获取和核验状态见 [source-lock.json](source-lock.json)。没有安装依赖、执行模型导入、启动训练或验证论文指标。

## 目录用途

- [paper/](paper/README.md)：论文与版本说明。
- [code/](code/)：作者固定提交的源码快照，不包含 Git 历史；原始文件保持不改写。
- [runs/](runs/README.md)：后续云端实验回传记录，不使用作者自带结果冒充本项目实测结果。

## 固定源码来源

- 仓库：[BigdogManLuo/HEFTcom24](https://github.com/BigdogManLuo/HEFTcom24.git)
- 分支：`main`
- 提交：`a93dfb574a06fdc25c902de0cf93cd0bca75b658`
- 获取方式：GitHub 官方 codeload 源码快照；完整性核验结果见版本记录。
- 许可证：GitHub 元数据未识别；需进一步确认作者许可。

## 作者入口索引（仅供查看，未执行）

- [作者说明](code/readme.md)
- [预处理目录](code/pre-process)
- [训练目录](code/train)
- [测试目录](code/test)
- [Conda 环境声明](code/conda-spec.txt)
- [pip 依赖声明](code/requirements-pip.txt)

## 复现前必须注意

- 本阶段只整理风电、光伏及其合计发电量预测相关资料，不执行能源交易代码。
- 需要 HEFTCom 竞赛数据及数值天气预报输入，不能直接拿现有 ERCO 发电表当成作者原始数据。
- [代码归档](https://doi.org/10.5281/zenodo.15351009)可作进一步版本对照；本地快照提交不自动等同于论文归档版本。
- 仓库公开可读，但本次 GitHub 元数据未识别到许可证；公开可获取不等于可任意再发布或商用，后续改编/转载先核查作者授权。
- 作者仓库中自带的数据、结果或模型文件只按上游资料保存，本轮没有运行它们，也没有把它们算作本项目结果。

## 云端获取相同源码

以下为以后在云端该论文目录、且 `code/` 尚不存在时使用的命令。本轮没有执行这些训练环境操作，也没有创建服务器连接。

```bash
git clone --single-branch --branch main https://github.com/BigdogManLuo/HEFTcom24.git code
git -C code checkout --detach a93dfb574a06fdc25c902de0cf93cd0bca75b658
git -C code rev-parse HEAD
```

只在新克隆目录执行上述检出。若已经有代码或改动，先查看状态，不覆盖。固定提交用于还原本次快照，不代表它一定就是论文发表时的实验版本。

