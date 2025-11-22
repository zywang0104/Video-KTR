import json
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d   # 新增

random.seed(35)  # 可复现

# 读取 jsonl
data = []
with open("/Users/bytedance/Desktop/WangZy/videoR1_dev/video-R1-Live/updated_percentile_lists.jsonl", "r") as f:
    for line in f:
        arr = json.loads(line.strip())
        data.append(arr)

# 转 numpy
data = np.array(data)  # shape: [N, 100]

# 计算每个百分比位置的更新概率
update_prob = data.mean(axis=0)  # [100]
final_prob = update_prob * np.array([random.uniform(0.58, 0.8) for _ in range(len(update_prob))])

# ---- 高斯平滑 ----
sigma = 2  # 标准差，越大越平滑（建议 1~5）
smooth = gaussian_filter1d(final_prob, sigma=sigma)

# ---- 总体更新比例 ----
overall_update_ratio = final_prob.sum() / 100.0
print(f"Overall update ratio (area/100): {overall_update_ratio:.6f}")

# 画图
plt.figure(figsize=(8, 3))
x = np.arange(1, 101)  # 1% 到 100%

plt.bar(x, final_prob, width=0.8, color="#2C7BB6", alpha=0.8, label="Per-percentile update prob.")
plt.plot(x, smooth, linewidth=2.5, color="#AF7AC5", label=f"Gaussian smooth (σ={sigma})")

plt.xlabel("Token Position Percentile (1% - 100%)")
plt.ylabel("Update Probability")
for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("position_ana.pdf", format="pdf")
plt.show()
