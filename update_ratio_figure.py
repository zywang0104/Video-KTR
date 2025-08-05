import matplotlib.pyplot as plt

# 数据
x = [10, 20, 30, 40, 50]
avg_scores = [52.8, 53.4, 52.0, 53.0, 52.3]
vanilla_score = 51.1

# 创建图形
plt.figure(figsize=(7, 3.5))
plt.plot(x, avg_scores, marker='o', linestyle='-', color='purple', linewidth=2, label='Video-KTR')
plt.axhline(y=vanilla_score, color='gray', linestyle='--', linewidth=1.2, label='Vanilla GRPO')

# 标注每个点的数值
for xi, yi in zip(x, avg_scores):
    plt.text(xi, yi + 0.1, f'{yi:.1f}', ha='center', va='bottom', fontsize=9)

# 美化坐标轴
plt.xticks(x, [f"{i}%" for i in x], fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Token Selecting Ratio', fontsize=11)
plt.ylabel('Average Score on\nVideo Reasoning Dataset', fontsize=11)  # 换行
plt.ylim(50.5, 54)
plt.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
plt.legend(fontsize=9)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("update_ratio_ablation.pdf", format="pdf")
plt.show()
