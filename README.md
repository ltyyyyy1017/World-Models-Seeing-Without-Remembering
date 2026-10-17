# 面向具身智能世界模型的身份隐私泄露评估与防护研究

**网络空间安全方向 · 具身智能世界模型中的身份信息泄露与训练目标防护**

服务机器人与具身智能体必须通过视觉观察人才能安全完成任务，但通常并不需要记住“这个人是谁”。本项目研究：在任务无身份依赖的仿真环境里，重建、预测和动作约束目标会让模型保留哪些身份信息，以及这种差异是否伴随任务能力下降。

当前阶段已有**项目书初稿、基础实验代码和 RTX 4060 本地 GPU 预实验**。A/B 与前景加权 A 已完成三种身份时间结构、三个训练种子的探索性比较；另有一份新采集 S0 数据的方向复验。研究假设和“近乎免费隐私”尚未验证，因为冻结表征上的策略成功率仍未测量。

## 项目材料

- [项目计划书（Markdown，v2.1）](看见但不记住_项目计划书_v2.md)
- [项目计划书（Word）](看见但不记住_项目计划书_v2.docx)
- [技术路线图](fig_roadmap.png)
- [预期结果示意图](fig_hypothesis.png)（假设示意，不是实测结果）
- [当前进展与预检结果](docs/INITIAL_STATUS.md)
- [RTX 4060 本地预实验与结果边界](docs/GPU_4060_PILOT.md)
- [研究设计待修正项](prototype/RESEARCH_NOTES.md)
- [代码说明](prototype/README.md)

## 已有代码

`prototype/privacywm/` 提供：

- 64×64 第一视角软件渲染、取文件夹后到达目标点的简化任务；
- 30 种程序化外观，身份与位置、朝向、持物及碰撞规则分离；
- S0 稳定、S1 慢变、S2 独立随机、S3 周期身份调度；
- 固定脚本采集、回合级数据划分、配对轨迹校验和 SHA-256 数据校验；
- A 重建型、B 预测型、C 预测加逆动力学型的训练代码；
- 冻结表征上的线性/MLP/k-NN 探针、像素参考、标签置换和坍塌诊断；
- 训练完成后生成指标图与本地 HTML 报告的代码。

环境采用二维网格物理与第一视角投影，是计划书三维环境的工程替代原型。模型训练、PPO、P1/P2/P3 策略消融、多种子推断与退化前沿仍待完成。

## 快速开始

推荐 Python 3.12。以下命令在仓库根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

这台 RTX 4060 Laptop GPU 已验证可使用以下 CUDA 版 PyTorch；其余依赖由项目清单安装：

```bash
python -m pip install --no-cache-dir 'torch==2.9.1+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -r prototype/requirements.txt
cd prototype
python -m pytest -q
```

先做环境与像素探针预检：

```bash
python -m privacywm collect --output outputs/my-first-run --episodes 90 --horizon 60
python -m privacywm preflight --output outputs/my-first-run
python -m privacywm pixels --output outputs/my-first-run
```

再训练模型并生成报告：

```bash
python -m privacywm train --output outputs/my-first-run --steps 300 --models A B --prediction-objective contrastive --regularization 0 --device cuda
```

训练和评测均完成后，打开 `prototype/outputs/my-first-run/index.html`。这是小规模工程运行，不是足以检验研究假设的训练预算。已有数据和模型拒绝覆盖；新的采集实验请更换输出目录。

## 当前验证情况

截至 2026-10-18：环境/协议测试 **8 通过**；真值规则 Oracle 在 100 个场景中成功 100 次，平均 14.74 步。CUDA 训练、冻结表征探针和本地报告生成已跑通。具体训练配置、27 次开发数据运行、独立 S0 复验及限制见 [GPU 预实验记录](docs/GPU_4060_PILOT.md)。

所有身份都是程序化仿真编号，不使用真实人物数据。代码和结果只用于研究身份信息保留，不提供部署级隐私保证。生成数据、模型权重和本地环境不纳入 Git。
