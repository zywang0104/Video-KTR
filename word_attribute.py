import matplotlib.pyplot as plt
import numpy as np

# 数据（单位：%）
visual = {'NOUN': 24.83, 'VERB': 16.74, 'ADJ': 8.20, 'ADV': 5.59, 'PRON': 8.42}
entropy = {'NOUN': 20.39, 'VERB': 16.09, 'ADJ': 8.52, 'ADV': 8.82, 'PRON': 9.90}
temporal = {'NOUN': 18.32, 'VERB': 21.16, 'ADJ': 8.18, 'ADV': 7.44, 'PRON': 11.04}

core_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']
x = np.arange(len(core_pos))

visual_vals = [visual[pos] for pos in core_pos]
entropy_vals = [entropy[pos] for pos in core_pos]
temporal_vals = [temporal[pos] for pos in core_pos]

# 设置样式
width = 0.25
colors = ['#7EC8E3', '#FFB477', '#C29ED8']

# 创建图表
plt.figure(figsize=(12, 5))
bars1 = plt.bar(x - width, visual_vals, width=width, label='Visual', color=colors[0])
bars2 = plt.bar(x, entropy_vals, width=width, label='Entropy', color=colors[1])
bars3 = plt.bar(x + width, temporal_vals, width=width, label='Temporal', color=colors[2])

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{height:.2f}%', 
                 ha='center', va='bottom', fontsize=9)

# 设置坐标和标题
plt.xticks(x, core_pos, fontsize=12)
plt.yticks(fontsize=11)
for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)
plt.ylabel("POS Proportion (%)", fontsize=12)
# plt.title("POS Proportion by Token Selection Strategy", fontsize=14, weight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
# 导出矢量图
plt.tight_layout()
plt.savefig("word_attribution_beautified.pdf", format="pdf")  # 或改为 .svg
plt.show()
