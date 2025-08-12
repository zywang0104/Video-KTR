import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['Vanilla GRPO', 'T-GRPO', 'TW-GRPO', 'Video-KTR']
benchmarks = ['Video-Holmes', 'VideoMMMU', 'MMVU', 'Reasoning Avg.']
scores = [
    [38.8, 49.8, 64.8, 51.1],  # Vanilla GRPO
    [39.7, 51.2, 65.8, 52.2],  # T-GRPO
    [39.5,51.1,65.1,51.9], # TW-GRPO
    [41.6, 52.6, 65.9, 53.4]   # Video-KTR
]

# 参数
bar_width = 0.22
x = np.arange(len(benchmarks))

# 配色

colors = ['#6A8EAE', '#8FC1A9',  '#FFD97D','#FF8C87']


# 创建图形
fig, ax = plt.subplots(figsize=(8, 4))

# 绘图
for i in range(len(models)):
    ax.bar(x + i * bar_width, scores[i], width=bar_width, label=models[i], color=colors[i])
    for j in range(len(scores[i])):
        ax.text(x[j] + i * bar_width, scores[i][j] + 0.3, f"{scores[i][j]:.1f}", ha='center', fontsize=9)

# 美化
ax.set_ylabel("Score", fontsize=10)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(benchmarks, fontsize=10)
ax.set_ylim(37, 68.5)
ax.legend(loc='upper left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
ax.grid(True, linestyle='--', axis='y', alpha=0.5)
plt.savefig("data_ablation.pdf", format="pdf", bbox_inches="tight")
plt.show()
