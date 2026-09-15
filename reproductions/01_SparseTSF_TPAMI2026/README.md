# SparseTSF：TPAMI 期刊扩展版

[返回总清单](../../论文复现清单.md)

## 论文与本地状态

IEEE Transactions on Pattern Analysis and Machine Intelligence，48(1): 170–183，2026 年 1 月（2025 年在线发表）。

2026-09-15：已新增 [服务器运行说明](SERVER_GUIDE.md)、`scripts/` 准备脚本和 `tests/` 测试。
另已新增 [train_etth1_linear.py](train_etth1_linear.py)，可在服务器 VS Code 中直接点击运行。
这些是本项目的辅助文件，不是作者源码改版。用户已回传首轮服务器训练/测试结束的截图，
但尚未完成与期刊论文目标表格的指标对照；本地辅助测试不等于独立复跑验证。

[正式论文 / DOI](https://doi.org/10.1109/TPAMI.2025.3602445)。源码/PDF最初获取和核验状态见 [source-lock.json](source-lock.json)，该记录保留获取时状态；后续准备脚本的运行方法见服务器说明。本地辅助脚本测试与服务器 GPU 验证分开记录。

## 目录用途

- [paper/](paper/README.md)：论文与版本说明。
- [code/](code/)：作者固定提交的源码快照，不包含 Git 历史；原始文件保持不改写。
- [runs/](runs/README.md)：后续云端实验回传记录，不使用作者自带结果冒充本项目实测结果。
- `scripts/`：本项目的环境、GPU 和 ETTh1 准备脚本，不启动训练。
- `tests/`：准备脚本的无 GPU 自动测试。
- [train_etth1_linear.py](train_etth1_linear.py)：封装已经运行过的单项训练命令，默认会开始新训练，输出单独归档。

## 固定源码来源

- 仓库：[lss-1138/SparseTSF](https://github.com/lss-1138/SparseTSF.git)
- 分支：`main`
- 提交：`b8c2740eecc84d8095ffce49ba5acafe68e53bb8`
- 获取方式：GitHub 官方 codeload 源码快照；完整性核验结果见版本记录。
- 许可证：Apache-2.0。

## 作者入口索引（仅供查看，未执行）

- [作者说明](code/README.md)
- [模型](code/models/SparseTSF.py)
- [实验入口](code/run_longExp.py)
- [训练与评估](code/exp/exp_main.py)
- [依赖声明](code/requirements.txt)
- [原始实验脚本目录](code/scripts/SparseTSF)

## 复现前必须注意

- 作者仓库同时服务于 ICML 2024 会议版和 TPAMI 2026 期刊扩展版，期刊增加了 SparseTSF/MLP 等内容。
- 本地原始作者代码保留 `linear`、`mlp` 及相应脚本；不能用会议版结果替代期刊扩展实验。
- `run_all.sh` 会启动多组实验，本轮没有运行它。正式训练前先选定单项任务。
- 周期参数需要依据训练段确定，不能用完整测试段挑周期。
- 期刊正式全文尚需补齐时，先阅读已获取的会议版背景材料；不称为期刊全文已获取。

## 云端获取相同源码

以下为以后在云端该论文目录、且 `code/` 尚不存在时使用的命令。本轮没有执行这些训练环境操作，也没有创建服务器连接。

```bash
git -c http.version=HTTP/1.1 clone --single-branch --branch main https://github.com/lss-1138/SparseTSF.git code
git -C code checkout --detach b8c2740eecc84d8095ffce49ba5acafe68e53bb8
git -C code rev-parse HEAD
```

只在新克隆目录执行上述检出。若已经有代码或改动，先查看状态，不覆盖。固定提交用于还原本次快照，不代表它一定就是论文发表时的实验版本。
