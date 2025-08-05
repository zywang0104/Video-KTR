import matplotlib.pyplot as plt
import numpy as np

# 标签和模型名称
labels = ['SR', 'IMC', 'TCI', 'TA', 'MHR', 'PAR', 'CTI']
models = ['Vanilla-GRPO', 'T-GRPO', 'Video-KTR-7B (ours)']

# 每个模型在各个任务上的分数
vanilla = [51.7, 43.8, 27.9, 42.5, 34.5, 34.5, 35.9]
t_grpo = [54.4, 45.7, 24.5, 44.5, 34.6, 37.6, 36.3]
ours = [55.8, 45.3, 27.8, 49.0, 37.0, 35.6, 40.7]

# 数据打包
data = [vanilla, t_grpo, ours]

# 配置画图
x = np.arange(len(labels))
bar_width = 0.25

# 设置颜色
colors = ['#4C78A8', '#72B7B2', '#F58518']  # 深蓝、湖绿、橙色
colors = ['#3E7CB1', '#7ED6A5', '#FF6F61']  # 靛蓝、薄荷绿、珊瑚红
# colors = ['#5B8FF9', '#61DDAA', '#F7666F']  # 清亮蓝、薄荷绿、柔和红

# 绘图
fig, ax = plt.subplots(figsize=(9, 4.5))

for i in range(len(data)):
    ax.bar(x + i * bar_width, data[i], width=bar_width, label=models[i], color=colors[i])
    # 标注数值
    for j in range(len(labels)):
        ax.text(x[j] + i * bar_width, data[i][j] + 0.5, str(data[i][j]),
                ha='center', va='bottom', fontsize=9, color='black')

# 设置坐标轴
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(20, 60)

# 添加图例
ax.legend(loc='upper right', fontsize=11)

# 美化边框和网格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', axis='y', alpha=0.5)

# 布局和保存
plt.tight_layout()
plt.savefig("video_holmes_bar_subtasks.pdf")
plt.show()
