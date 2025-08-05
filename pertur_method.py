import matplotlib.pyplot as plt
import numpy as np

# 数据
methods_visual = ["Mask All Images", "Mask Half Images", "Replace Images"]
scores_visual = [52.5, 52.4, 52.0]

methods_temporal_raw = ["Random Shuffle", "Segmental Shuffle", "Reverse"]
scores_temporal_raw = [52.6, 52.0, 52.4]

# Temporal 从高到低排序
sorted_indices = np.argsort(scores_temporal_raw)[::-1]
methods_temporal = [methods_temporal_raw[i] for i in sorted_indices]
scores_temporal = [scores_temporal_raw[i] for i in sorted_indices]


colors_visual = ['#4C78A8', '#72B7B2', '#F2B701']   # 蓝、青绿、亮黄
colors_temporal = ['#E45756', '#FF9DA7', '#A05195'] # 红、粉、紫

# 横轴位置
x_visual = np.arange(len(scores_visual))
x_temporal = np.arange(len(scores_temporal)) + len(scores_visual) + 1  # 空一格分组

# 创建图
fig, ax = plt.subplots(figsize=(9, 4.5))

# 画柱状图
bars_visual = ax.bar(x_visual, scores_visual, color=colors_visual, width=0.6)
bars_temporal = ax.bar(x_temporal, scores_temporal, color=colors_temporal, width=0.6)

# 设置x轴分组标签
ax.set_xticks([1, 5])
ax.set_xticklabels(['Visual-Aware', 'Temporal-Aware'], fontsize=12)

# 设置y轴
ax.set_ylabel('Average Score\non Video Reasoning Dataset', fontsize=12)
ax.set_ylim(51.5, 53)

# 添加分数标注
for bar in bars_visual + bars_temporal:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', fontsize=10)

# 图例（左上：Visual-Aware Methods）
legend_visual = [plt.Rectangle((0,0),1,1, color=c) for c in colors_visual]
visual_legend = ax.legend(legend_visual, methods_visual, title='Visual-Aware Methods',
                          loc='upper left', fontsize=10, title_fontsize=11)

# 添加图例对象
ax.add_artist(visual_legend)

# 图例（右上：Temporal-Aware Methods）
legend_temporal = [plt.Rectangle((0,0),1,1, color=c) for c in colors_temporal]
ax.legend(legend_temporal, methods_temporal, title='Temporal-Aware Methods',
          loc='upper right', fontsize=10, title_fontsize=11)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
ax.grid(True, linestyle='--', axis='y', alpha=0.5)
plt.savefig("perturbation_ablation.pdf", format="pdf")
plt.show()