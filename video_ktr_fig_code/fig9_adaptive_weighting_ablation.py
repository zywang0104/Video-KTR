
import matplotlib.pyplot as plt

# 方法名称和对应分数
methods = ["Binary Top-20%", "Softmax","Sigmoid", "Linear","Exponential"]
scores = [53.4, 52.5, 51.9,51.8,51.1]
# 渐变绿色颜色列表（从深到浅）
colors = ["#6A4C3F", "#735447", "#8A685A", "#A27D6E", "#BA9283"]
colors = ["#1e4040", "#2c5b5b", "#3c7777", "#4b9393", "#5eaeae"]
colors = ["#1b3a36", "#24524e", "#2f6b66", "#3c847e", "#4e9e96"]






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
# ax.set_yticklabels(methods, fontsize=11)
from matplotlib.text import Text
# ax.set_yticklabels(
#     [Text(text=m, fontweight='bold') if "Binary" in m else m for m in methods],
#     fontsize=11
# )
yticks = ax.set_yticklabels(methods, fontsize=11)
yticks[methods.index("Binary Top-20%")].set_fontweight("bold")

# 设置 x 轴范围
ax.set_xlim(49, 54)

ax.grid(axis='x', linestyle='--', alpha=0.3)

# 给每个柱子标注分数
for bar in bars:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}", va='center', fontsize=10)

for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)


plt.tight_layout()
plt.savefig("adaptive.pdf", format="pdf")
plt.show()
