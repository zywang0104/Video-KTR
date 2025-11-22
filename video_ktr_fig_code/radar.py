
# import numpy as np
# import matplotlib.pyplot as plt

# # 定义类别
# labels = ['VQAv2', 'POPE_F1', 'Seed-Img', 'mmbench_en', 'OCRBench_Acc', 'DocVQA']
# num_vars = len(labels)


# # 计算角度
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]  # 让图形闭合


# # 定义多个模型的数值
# values_1 = [79.5,85.4,73.7,75.2,44.8,30.41]  # Baseline
# values_2 = [80.09,87.2,74.09,77.06,50.8,43.91]  # S2
# values_3 = [80.16,86.52,73.81,76.03,50,43.73]  # STC
# values_4 = [80.16,86.52,73.81,76.03,47,41.28]  # TK
# values_5 = [80.52,86.87,74.4,75.85,60.4,70.3]  # AnyRes

# values_1 += values_1[:1]
# values_2 += values_2[:1]
# values_3 += values_3[:1]
# values_4 += values_4[:1]
# values_5 += values_5[:1]
# print(values_1)
# # 绘制多个模型
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
# ax.fill(angles, values_1, color='b', alpha=0.25, label='Baseline')
# ax.plot(angles, values_1, color='b', linewidth=1.5)
# ax.fill(angles, values_2, color='r', alpha=0.25, label='S-Square')
# ax.plot(angles, values_2, color='r', linewidth=1.5)
# ax.fill(angles, values_3, alpha=0.25, label='STC')
# ax.plot(angles, values_3, linewidth=1.5)
# ax.fill(angles, values_4,  alpha=0.25, label='TokenLearner')
# ax.plot(angles, values_4,  linewidth=1.5)
# ax.fill(angles, values_5,  alpha=0.25, label='AnyRes')
# ax.plot(angles, values_5,  linewidth=1.5)
# # 设置标签
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(labels)
# plt.legend(loc='upper right')
# plt.title('LLM Comparison Radar Chart')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 现代化配色
colors = ['#6A8EAE', '#8FC1A9', '#FF8C87']  # 柔和蓝、绿色、珊瑚红
plt.style.use("seaborn-v0_8-darkgrid")  # 现代风格

# 定义类别
labels = ['SR', 'IMC', 'TCI', 'TA', 'MHR', 'PAR', 'CTI']
num_vars = len(labels)

# 计算角度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 让图形闭合

# 定义多个模型的数值
values = {
    "Vanilla-GRPO": [51.7, 43.8, 27.9, 42.5, 34.5, 34.5, 35.9],
    "T-GRPO": [54.4, 45.7, 24.5, 44.5, 34.6, 37.6, 36.3],
    "Video-KTR": [55.8, 45.3, 27.8, 49.0, 37.0, 35.6, 40.7]
}

# 使每个模型的数据闭合
for key in values:
    values[key] += values[key][:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制不同模型的雷达图
for i, (label, data) in enumerate(values.items()):
    ax.fill(angles, data, color=colors[i], alpha=0.3, label=label)  # 半透明填充
    ax.plot(angles, data, color=colors[i], linewidth=2, linestyle='solid', marker='o')  # 轮廓线

# **增强背景圆圈线**
ax.yaxis.grid(True, linestyle="solid", linewidth=1.2, color="lightgray", alpha=0.8)  # 加粗背景网格线

# **确保 labels 在最上层**
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color="black", zorder=10)

# **标注坐标刻度**
y_ticks = [30, 40, 50, 60]
ax.set_yticks(y_ticks)
ax.set_yticklabels(map(str, y_ticks), fontsize=11, color="darkgray")

# **优化网格线**
ax.grid(True, linestyle="solid", alpha=0.6)  # 实线网格增强视觉层次感

# **增强视觉效果**
# plt.title('Comparison on Public Dataset', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=12)

# **确保 labels 文字不被遮挡**
for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    label.set_horizontalalignment("center")  # 确保文字对齐
    label.set_fontweight("bold")

# 显示图表
plt.tight_layout()
plt.show()
