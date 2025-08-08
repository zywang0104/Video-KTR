import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置现代配色和样式
colors = ['#6A8EAE', '#8FC1A9', '#FF8C87']
plt.style.use("seaborn-v0_8-darkgrid")

# 定义标签
labels = ['PAR','CTI','SR','IMC', 'TCI','TA','MHR' ]
num_vars = len(labels)

# 角度计算
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 定义模型数据
values = {
    "Vanilla-GRPO": [34.5,35.9,51.7, 43.8, 27.9,42.5,34.5],
    "T-GRPO": [ 37.6,36.3,54.4, 45.7, 24.5,44.5,34.6],
    "Video-KTR": [35.6,40.7,55.8, 45.3, 27.8,49,37]
}
for key in values:
    values[key] += values[key][:1]

# 创建图
fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(polar=True))

# 将极坐标轴的圆形轮廓改为多边形
ax.set_frame_on(False)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 手动绘制多边形网格线
y_ticks = [30, 40, 50, 60]
ax.yaxis.grid(False)  # 不画圆圈线
ax.spines['polar'].set_visible(False)  # 不显示极轴圆圈边框
coords = [(angle, y_ticks[-1]) for angle in angles]
ax.fill([c[0] for c in coords], [c[1] for c in coords],
        color='thistle', alpha=0.2, zorder=0)
# 清空默认的 ytick labels

# 手动添加 r 轴数字标签（更可控、更漂亮）
for y in y_ticks:
    coords = [(angle, y) for angle in angles]
    ax.plot([c[0] for c in coords], [c[1] for c in coords],
            color="lightgray", linewidth=1.2)
for i, (label, data) in enumerate(values.items()):
    ax.fill(angles, data, color=colors[i], alpha=0.3, label=label)
    ax.plot(angles, data, color=colors[i], linewidth=2, linestyle='solid', marker='o')

    # 只给 Video-KTR 添加数值标注
    if label == "Video-KTR":
        for angle, value in zip(angles, data):
            if value==55.8:
                print(angle)
                ax.text(angle-0.1, value + 4.2, f"{value:.1f}%",
                    ha='center', va='center', fontsize=11, color=colors[i], fontweight='bold')
            elif value==49:
                print(angle)
                ax.text(angle+0.1, value + 5.2, f"{value:.1f}%",
                    ha='center', va='center', fontsize=11, color=colors[i], fontweight='bold')
            else:
                ax.text(angle, value + 5.3, f"{value:.1f}%",
                    ha='center', va='center', fontsize=11, color=colors[i], fontweight='bold')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color="black")
ax.set_yticks(y_ticks)
ax.set_yticklabels([])
# 美化图例
plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.1), frameon=True, fontsize=12)
plt.tight_layout()
plt.savefig("video_holmes_radar_subtasks.pdf")
plt.show()
