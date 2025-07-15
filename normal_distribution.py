import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
# 随机生成数据（例如均匀分布 [0, 1]）
data = torch.rand(1024)
data[0:200] = float('-inf')


# 分段正态映射函数
def piecewise_normal_mapping(p):
    ranks = np.argsort(np.argsort(data))
    percentiles = ranks / (len(data) - 1)
    cdf_neg2 = norm.cdf(-2)
    cdf_neg1 = norm.cdf(-1)
    cdf_pos1 = norm.cdf(1)
    cdf_pos2 = norm.cdf(2)

    result = np.zeros_like(p)

    mask1 = p <= 0.2
    result[mask1] = norm.ppf(cdf_neg2 + (cdf_neg1 - cdf_neg2) * (p[mask1] / 0.2))

    mask2 = (p > 0.2) & (p <= 0.8)
    result[mask2] = norm.ppf(cdf_neg1 + (cdf_pos1 - cdf_neg1) * ((p[mask2] - 0.2) / 0.6))

    mask3 = p > 0.8
    result[mask3] = norm.ppf(cdf_pos1 + (cdf_pos2 - cdf_pos1) * ((p[mask3] - 0.8) / 0.2))

    return result

# 映射结果
mapped_values = piecewise_normal_mapping(data)
print(mapped_values[190:220])

# 可视化
plt.figure(figsize=(10, 4))

# 原始数据 vs 映射值
plt.subplot(1, 2, 1)
plt.scatter(data, mapped_values, alpha=0.3, s=5)
plt.title("Random Uniform Data → Mapped Values")
plt.xlabel("Original value")
plt.ylabel("Mapped value")
plt.grid(True)

# 直方图
plt.subplot(1, 2, 2)
plt.hist(mapped_values, bins=30, edgecolor='black')
plt.axvline(-1, color='red', linestyle='--', label='-1')
plt.axvline(1, color='red', linestyle='--', label='1')
plt.axvline(-2, color='gray', linestyle=':')
plt.axvline(2, color='gray', linestyle=':')
plt.title("Histogram of Mapped Values")
plt.xlabel("Mapped value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
