# 看见但不记住：可运行研究原型

本目录把项目计划书的第一阶段变成可执行实验。保留根目录原始材料。

已实现：64×64 第一视角软件光线投射场景、取文件夹再到达目标点、30 种外观、S0/S1/S2/S3 身份调度、固定脚本离线数据、A/B/C 目标、配对初始化与批次、按回合分离的世界模型/探针训练/验证/测试数据、线性/MLP/k-NN 探针、像素参考、标签置换、坍塌诊断、数据 SHA-256、实际指标图和离线 HTML 报告。

当前是工程预实验，不是完成了年度研究。没有完整三维人体、PPO、P1/P2/P3 统计实验、18 组多种子主实验、等价性检验或退化前沿。C 已有代码，但默认预实验只跑 A/B。

## 本机启动

在 PowerShell 中执行，Python 环境在项目根目录的 `.venv`，不修改系统默认 Python：

```powershell
cd prototype
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m privacywm pilot --output outputs/my-first-run --steps 300
Start-Process 'outputs/my-first-run/index.html'
```

已有输出时使用新的目录，程序拒绝覆盖数据或已有模型。报告是本地静态文件，打开即可查看，无需运行服务器。

本机已完成 30/90 回合两档的数据生成、Oracle 预检和原始像素探针；摘要见仓库根目录 `docs/INITIAL_STATUS.md`。生成数据与本地输出不入库。A/B/C 模型训练代码尚未完成实际训练验证，当前没有模型比较图或 HTML 结果页。

重新安装依赖（Python 3.12 推荐）：

```powershell
py -3.12 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
..\.venv\Scripts\python.exe -m pip install --no-cache-dir numpy pillow matplotlib scikit-learn pytest
```

上面的命令从 `prototype` 目录执行。CPU 足以执行小实验；正式训练可以按 PyTorch 官方安装说明安装与 GPU 兼容的版本，再使用 `--device cuda`。本机 pip 缓存目录写入曾导致安装长时间阻塞，故加 `--no-cache-dir`。

## 建议的下一轮

```powershell
# 只采数据；后续 A/B/C 共用哈希验证过的数据
..\.venv\Scripts\python.exe -m privacywm collect --output outputs/next --episodes 60
# 同种子下的不同目标、正则强度共用编码器初始化和训练批次
..\.venv\Scripts\python.exe -m privacywm train --output outputs/next --steps 1500 --regularization 0
..\.venv\Scripts\python.exe -m privacywm train --output outputs/next --steps 1500 --regularization 0.1
# 对同一数据增加独立训练种子（数据仍固定；身份与物理场景不重采）
..\.venv\Scripts\python.exe -m privacywm train --output outputs/next --steps 1500 --seed 1
..\.venv\Scripts\python.exe -m privacywm train --output outputs/next --steps 1500 --seed 2
```

每次训练会刷新报告。多种子图仅做描述性平均；尚未实现配对置信区间、功效分析或主假设检验。依赖范围见 `requirements.txt`；本次成功安装的非 PyTorch 环境见 `docs/evidence/installed-packages.txt`。

独立运行工程检查与像素探针：

```powershell
..\.venv\Scripts\python.exe -m privacywm preflight --output outputs/pilot90
..\.venv\Scripts\python.exe -m privacywm pixels --output outputs/pilot90
# 已保存模型只需重新评测时，可用 evaluate 代替 train。
```

研究设计的修正理由与后续顺序见 `RESEARCH_NOTES.md`。

## 与计划书有意保留的差异

1. **环境降级**：本原型是二维网格物理 + 第一视角光线投射 + 几何人物贴图。身份由 6 色 × 5 档体宽/纹理组成；体宽与纹理目前绑定，尚未实现五属性完全独立控制。朝向、位置、持物独立于身份。MiniWorld 能力门槛 G0 尚未验证，不声称通过。
2. **防坍塌**：B 的预测损失存在常数解。默认 A/B/C 都加入相同的方差/协方差项（受 [VICReg](https://arxiv.org/abs/2105.04906) 启发）。这是额外实验因素，可能保留身份，必须用 `--regularization 0` 作对照。stop-gradient 单独不构成防坍塌保证。
3. **等熵**：S0/S3 的初相在每 30 回合内平衡，S2 真正逐步独立均匀采样。有限样本熵有差异，实测写进 manifest；不以打乱的等频序列冒充独立随机。
4. **时间结构指标**：不对身份编号做 Pearson 相关。报告同身份重复率与经验条件熵；S3 在 lag=30 高度相关，低 lag=1 重复率不能概括所有尺度。
5. **B0**：下采样像素 + 仅在探针训练集拟合 PCA + 三族探针，是可运行参考，不是理论上界。
6. **探针**：只在人物物理中心可见且投影高度 ≥3 像素的帧训练/评测，选样规则独立于身份轮廓，身份标签不进入世界模型损失。位置标签是绝对 4×4 网格，部分第一视角状态具有别名；低位置分数可能来自不可观测性。下一轮应增加相对位置与持物任务探针。
7. **配对与统计**：同一 S 档、同种子内 A/B 像素数据完全相同；不同 S 档只共用几何与动作，像素身份重渲染。探针数据与世界模型训练回合不同。选择探针看验证集，测试只报告。帧存在自相关，不能按帧计算研究置信区间。
8. **任务能力**：Oracle 成功只证明任务可解，不能证明像素策略能学会；身份替换不改变物理结果的单元测试也不能代替训练策略上的 P2。下一步先做冻结表征行为克隆或 PPO，再做前提检验和隐私—能力比较。

## 文件结构

- `privacywm/env.py`：环境、渲染、身份调度与 Oracle。
- `privacywm/data.py`：离线采集、回合划分、审计和数据哈希。
- `privacywm/models.py`、`train.py`：模型与训练。
- `privacywm/evaluate.py`：冻结表征与探针评测。
- `privacywm/report.py`：实际指标图、报告、回放展示。
- `tests/test_protocol.py`：身份独立性、遮挡、调度、Oracle 与梯度检查。
- `outputs/pilot/`：实际运行数据、权重和报告。
