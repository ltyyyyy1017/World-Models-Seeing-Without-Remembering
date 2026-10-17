import html
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def report(root):
    root = Path(root)
    records = []
    for path in sorted(root.glob("runs/*/evaluation.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads(path.with_name("training.json").read_text(encoding="utf-8"))
        records.append({"run": path.parent.name, "kind": config["kind"], "seed": config["seed"],
            "mode": config["mode"], "regularization": config["regularization"],
            "covariance_weight": config.get("covariance_weight", 0.01),
            "prediction_objective": config.get("prediction_objective", "mse"),
            "temperature": config.get("temperature", 0.2),
            "reconstruction_weight": config.get("reconstruction_weight", 50.0),
            "foreground_weight": config.get("foreground_weight", 0.0),
            "identity": evaluation["identity"]["selected_test_top1"],
            "position": evaluation["position"]["selected_test_top1"],
            "pixel": evaluation["B0_pixel_reference"]["selected_test_top1"],
            "permuted": evaluation["identity"]["shuffled_train_labels_test_top1"],
            "std": evaluation["latent_mean_std"], "rank": evaluation["effective_rank"],
            "warnings": len(evaluation["identity"]["convergence_warnings"]),
            "details": evaluation, "config": config})
    if not records:
        raise ValueError("No completed evaluations found")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    colors = {"A": "#e6a453", "B": "#5cbab1", "C": "#a68bdd"}
    modes = sorted(set(r["mode"] for r in records))
    groups = sorted(set((r["kind"], r["regularization"], r["covariance_weight"],
                         r["prediction_objective"], r["temperature"], r["reconstruction_weight"],
                         r["foreground_weight"])
                        for r in records))
    for kind, reg, cov, objective, temp, rec_weight, fg_weight in groups:
        rows = [r for r in records if (r["kind"], r["regularization"], r["covariance_weight"],
                                      r["prediction_objective"], r["temperature"],
                                      r["reconstruction_weight"], r["foreground_weight"]) ==
                (kind, reg, cov, objective, temp, rec_weight, fg_weight)]
        for ax, metric in zip(axes, ("identity", "position", "std")):
            available = [mode for mode in modes if any(r["mode"] == mode for r in rows)]
            means = [np.mean([r[metric] for r in rows if r["mode"] == mode]) for mode in available]
            color = "#d7673f" if kind == "A" and fg_weight else colors[kind]
            style = "D-" if fg_weight else ("o-" if reg else "s--")
            label = f"{kind} {objective} fg={fg_weight:g} reg={reg:g}"
            ax.plot(available, means, style, label=label, color=color)
            # Keep individual seeds visible; descriptive means do not imply a formal CI.
            for row in rows:
                ax.scatter(row["mode"], row[metric], color=color, alpha=0.5)
            ax.grid(alpha=0.2)
    axes[0].axhline(1 / 30, color="gray", linestyle="--", label="identity chance")
    axes[1].axhline(1 / 16, color="gray", linestyle="--", label="position chance")
    for ax, title in zip(axes, ("Identity probe (top-1)", "Position probe (top-1)", "Latent mean std (collapse diagnostic)")):
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.savefig(root / "pilot_results.png", dpi=160)
    plt.close(fig)
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    rows_html, mdrows = [], []
    for r in records:
        rows_html.append(f'<tr><td>{html.escape(r["run"])}</td><td>{r["identity"]:.1%}</td><td>{r["position"]:.1%}</td><td>{r["pixel"]:.1%}</td><td>{r["permuted"]:.1%}</td><td>{r["std"]:.3f}</td><td>{r["rank"]:.1f}</td></tr>')
        mdrows.append(f'| {r["run"]} | {r["identity"]:.1%} | {r["position"]:.1%} | {r["pixel"]:.1%} | {r["permuted"]:.1%} | {r["std"]:.3f} | {r["rank"]:.1f} |')
    audits = []
    for mode in modes:
        audit = manifest["schedule_audit"][f"{mode}_world_train"]
        audits.append(f'{mode}: H={audit["marginal_entropy_bits"]:.3f} bits; repeat={audit["lag1_repeat_rate"]:.3f}; empirical H(next|current)={audit["empirical_conditional_entropy_bits"]:.3f} bits')
    md = """# 第一轮工程预实验

这是实际训练输出，不是预设结论。少量训练种子、小数据、简化场景，仅用于排查实验链路和形成后续检验方案。

| 运行 | 身份 top-1 | 位置 top-1 | 像素参考 B0 | 置换标签 | 潜变量 std | 有效秩 |
| --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(mdrows) + "\n\n" + "\n\n".join(audits) + """

## 解释边界

- A/B 共用编码器初始化、训练批次、物理轨迹和回合划分；时间结构改变时重新渲染身份，不能声称不同 S 档共用相同像素数据。
- 探针仅使用人物可见帧，报告帧级描述性准确率。连续帧相关，不能当成独立统计样本。
- 线性、MLP、k-NN 在验证集选择，独立测试集报告；完整混淆矩阵、top-5、训练/验证差距、收敛警告见各运行 evaluation.json。
- B0 是 16×16 下采样 RGB + 训练集拟合 PCA 的实际参考；不是经过证明的泄漏上界。归一化值可超过 1。
- 防坍塌项为全部模型共享的额外干预，受 VICReg 启发，可能直接影响身份保留；正式结论必须比较 regularization=0 与正值。
- 对比式预测以同批其他下一帧为负样本；本轮没有出现单步 MSE 预测那样的严重坍塌，但同一场景可能产生相似负样本，仍是工程方案。
- 前景加权只用于 A 的探索性对照，按输入像素色彩饱和度加权，不使用身份标签；它也会加权文件夹等彩色物体，不能等同于精准人物分割。
- 本轮未训练冻结表征上的控制策略，位置探针不能代替任务成功率。未完成 P1/P2/P3 统计前提检验。
- 未验证 H1/H2/H3/H4；没有策略成功率与退化前沿时，不能宣称选择性隐私改善或任务能力保持。
- 所有训练种子共用同一数据集；本轮曾用测试结果发现问题并调整损失，因此这些数值属于探索性工程预检，后续须用新数据重新确认。
- 身份边缘分布在总体上均匀；有限样本 S2 不会恰好等频。S0/S3 与 S2 的实测熵见上方，正式实验需预登记容差。
- 不把身份编号当有序数值计算 Pearson 自相关；采用同类重复率。周期 S3 的延迟 30 步相关很高，不能称为所有尺度自相关都低。
"""
    (root / "REPORT.md").write_text(md, encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    cards = "".join(f'<div class="card"><h3>{mode} · 固定脚本回放</h3><img class="replay" src="data/replay_{mode}.gif"><p>{html.escape(audits[i])}</p></div>' for i, mode in enumerate(modes))
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>世界模型身份隐私 · 实验原型</title>
<style>body{margin:0;background:#111a21;color:#dce7ee;font:15px/1.7 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1160px;margin:auto;padding:48px 28px}h1{font-size:40px;margin:12px 0}.eyebrow{color:#67c8bb;letter-spacing:3px}.notice{border-left:3px solid #e6a453;background:#24303a;padding:16px 22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;margin:28px 0}.card{background:#1b2833;padding:22px;border-radius:14px}.replay{width:100%;image-rendering:pixelated;border-radius:8px}.chart{width:100%;background:white;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:13px;border-bottom:1px solid #34424f}a{color:#75d5c8}.scroll{overflow:auto}p{color:#acbdc9}code{background:#263540;padding:3px 6px}</style><main>
<div class="eyebrow">WORLD MODEL / PRIVACY LAB / PILOT 01</div><h1>世界模型身份隐私预实验</h1><p>第一视角仿真 → 配对训练 → 冻结表征 → 身份与位置探针</p><div class="notice">真实小规模训练结果 · 简化软件渲染环境 · 尚未检验研究假设<br>身份可识别率下降可能源于整体退化或坍塌，必须结合表征诊断和后续控制任务判断。</div>
""" + f'<div class="grid">{cards}</div><img class="chart" src="pilot_results.png"><h2>实际探针结果</h2><div class="scroll"><table><tr><th>运行</th><th>身份</th><th>位置</th><th>像素参考</th><th>置换标签</th><th>潜变量 std</th><th>有效秩</th></tr>{"".join(rows_html)}</table></div>' + """
<h2>怎样读这页</h2><p>身份和位置列均为在验证集上选择探针后得到的独立测试准确率。人物不可见帧不参与探针。身份机会水平为 3.3%，位置为 6.25%，实际多数类基线还需结合数据分布。std 与有效秩用于观察坍塌，不能单独证明表征足够好。</p><p>回放来自与模型训练分离的固定测试轨迹。当前没有实时模型条形读数，也没有冻结表征策略成功率，避免把场景演示误当成控制能力验证。</p><p><a href="REPORT.md">完整实验说明</a> · <a href="summary.json">全部指标 JSON</a> · <a href="data/manifest.json">数据哈希与时间结构审计</a></p></main></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
    return records
