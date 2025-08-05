import matplotlib.pyplot as plt
import numpy as np

# 横轴是各个 benchmark
benchmarks = ['Video-Holmes', 'VideoMMMU', 'MMVU', 'Avg']
x = np.arange(len(benchmarks))

hard_ratio = [41.6, 52.6, 65.9, 53.4]
sigmoid = [41.3, 50.5, 64.3, 51.9]
linear = [40.5, 50.9, 64.0, 51.8]

colors = ["#91B8E6", "#8ED1B2", "#F4E3A1"]

width = 0.25
plt.figure(figsize=(8, 4))
bars1 = plt.bar(x - width, hard_ratio, width=width, color=colors[0], label='Hard Ratio (20%)')
bars2 = plt.bar(x, sigmoid, width=width, color=colors[1], label='Sigmoid')
bars3 = plt.bar(x + width, linear, width=width, color=colors[2], label='Linear')


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.4, f'{height:.1f}',
                 ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# 轴与图例设置
plt.xticks(x, benchmarks, fontsize=10)
plt.ylabel("Score", fontsize=11)
plt.xlabel("Video Reasoning Benchmark", fontsize=11)
plt.ylim(35, 70)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper left')

# 去掉顶部和右边框线
for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("adaptive.pdf", format="pdf")
plt.show()
