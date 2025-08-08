import matplotlib.pyplot as plt

# 方法名称和对应分数
methods = [
    "Log Prob Diff", "L1 Norm", "JS Divergence", "Cos Similarity",
    "L2 Norm", "KL Divergence", "Hellinger"
]
scores = [53.4, 53.1, 53.0, 52.9, 52.8, 52.6, 52.4]
# 渐变绿色颜色列表（从深到浅）
colors = ["#4A2F1B", "#6A3E22", "#8B4E2A", "#AA623B", "#C77A57", "#DEA58A", "#F3CBB7"]
colors = ["#5A0A0A", "#7A1B1B", "#9A2C2C", "#BA3D3D", "#D86B6B", "#E9A0A0", "#F5D0D0"]
colors = ["#4B1C1C", "#682C2C", "#853D3D", "#A14E4E", "#C67A7A", "#E3AAAA", "#F4D5D5"]






# 创建画布
fig, ax = plt.subplots(figsize=(7, 4))

# 绘制横向柱状图
bars = ax.barh(methods, scores, color=colors,height=0.65)

# 图标题和轴标签
# ax.set_title("Weighting Methods", fontsize=12)
ax.set_xlabel("Video Reasoning Benchmark Average Score",fontsize=14)

# 翻转 y 轴（让分数高的在上）
ax.invert_yaxis()
y_pos = range(len(methods))
bars = ax.barh(y=y_pos, width=scores, color=colors, height=0.35)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=11)
ax.set_yticklabels(
    [r"$\bf{" + m + "}$" if m == "Log Prob Diff" else m for m in methods],
    fontsize=11
)

# 设置 x 轴范围
ax.set_xlim(51, 54)

ax.grid(axis='x', linestyle='--', alpha=0.3)

# 给每个柱子标注分数
for bar in bars:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}", va='center', fontsize=10)

for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("distance_ablation.pdf", format="pdf")
plt.show()
