import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置现代配色和样式
colors = ['#6A8EAE', '#8FC1A9', '#FF8C87']
plt.style.use("seaborn-v0_8-darkgrid")

# 定义标签
labels = ['SR', 'IMC', 'TCI', 'TA', 'MHR', 'PAR', 'CTI']
num_vars = len(labels)

# 角度计算
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 定义模型数据
values = {
    "Vanilla-GRPO": [51.7, 43.8, 27.9, 42.5, 34.5, 34.5, 35.9],
    "T-GRPO": [54.4, 45.7, 24.5, 44.5, 34.6, 37.6, 36.3],
    "Video-KTR": [55.8, 45.3, 27.8, 49.0, 37.0, 35.6, 40.7]
}
for key in values:
    values[key] += values[key][:1]

# 创建图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 将极坐标轴的圆形轮廓改为多边形
ax.set_frame_on(False)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 手动绘制多边形网格线
y_ticks = [30, 40, 50, 60]
for y in y_ticks:
    coords = [(angle, y) for angle in angles]
    ax.plot([c[0] for c in coords], [c[1] for c in coords],
            color="lightgray", linewidth=1.2)

# 绘图
for i, (label, data) in enumerate(values.items()):
    ax.fill(angles, data, color=colors[i], alpha=0.3, label=label)
    ax.plot(angles, data, color=colors[i], linewidth=2, linestyle='solid', marker='o')

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color="black")
ax.set_yticks(y_ticks)
ax.set_yticklabels(map(str, y_ticks), fontsize=11, color="darkgray")

# 美化图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=12)
plt.tight_layout()
plt.show()
