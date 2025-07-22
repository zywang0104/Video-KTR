import torch

def compute_rank_weights_batch(scores, k=20, gamma=1.5):
    batch_size, seq_len = scores.shape
    flat_scores = scores.view(-1)
    sorted_indices = torch.argsort(flat_scores, descending=True)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(flat_scores), device=scores.device)
    normalized_ranks = ranks.float() / (len(flat_scores) - 1)
    weights = 1 + gamma * (torch.sigmoid(k * (0.5 - normalized_ranks)) - 0.5)
    weights = weights.view(batch_size, seq_len)
    return weights

import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

# ----------- 映射函数定义 -----------
def piecewise_normal_mapping(scores):
    batch_size, seq_len = scores.shape
    flat_scores = scores.contiguous().view(-1).float()

    # 计算 percentiles（排序位置除以 N-1）
    sorted_idx = flat_scores.argsort()
    ranks = torch.zeros_like(sorted_idx, dtype=torch.float32, device=flat_scores.device)
    ranks[sorted_idx] = torch.arange(len(flat_scores), device=flat_scores.device, dtype=torch.float32)
    p = ranks / (len(flat_scores) - 1)

    # 正态分布常数
    dist = Normal(0, 1)
    cdf_neg2 = dist.cdf(torch.tensor(-2.0, device=flat_scores.device))
    cdf_neg1 = dist.cdf(torch.tensor(-1.0, device=flat_scores.device))
    cdf_pos1 = dist.cdf(torch.tensor(1.0, device=flat_scores.device))
    cdf_pos2 = dist.cdf(torch.tensor(2.0, device=flat_scores.device))

    result = torch.zeros_like(p, dtype=torch.float32, device=flat_scores.device)

    # 区间1: 0~0.2 -> [-2, -1]
    mask1 = p <= 0.2
    if mask1.any():
        p1 = p[mask1] / 0.2
        result[mask1] = dist.icdf(cdf_neg2 + (cdf_neg1 - cdf_neg2) * p1)

    # 区间2: 0.2~0.8 -> [-1, 1]
    mask2 = (p > 0.2) & (p <= 0.8)
    if mask2.any():
        p2 = (p[mask2] - 0.2) / 0.6
        result[mask2] = dist.icdf(cdf_neg1 + (cdf_pos1 - cdf_neg1) * p2)

    # 区间3: 0.8~1.0 -> [1, 2]
    mask3 = p > 0.8
    if mask3.any():
        p3 = (p[mask3] - 0.8) / 0.2
        result[mask3] = dist.icdf(cdf_pos1 + (cdf_pos2 - cdf_pos1) * p3)

    # reshape 回原始形状
    result = result.view(batch_size, seq_len)

    # 替换 nan 和负数为 0
    result[torch.isnan(result)] = 0
    result[result < 0] = 0

    return result

scores = torch.arange(1, 101).view(1,100)
weights20_15 = [float(i) for i in list(compute_rank_weights_batch(scores, k=20, gamma=1.5).view(100))]
weights20_2 = [float(i) for i in list(compute_rank_weights_batch(scores, k=20, gamma=2).view(100))]
weights20_10 = [float(i) for i in list(compute_rank_weights_batch(scores, k=20, gamma=1).view(100))]
weights10_15 = [float(i) for i in list(compute_rank_weights_batch(scores, k=10, gamma=1.5).view(100))]
weights10_10 = [float(i) for i in list(compute_rank_weights_batch(scores, k=10, gamma=1).view(100))]
weights5_10 = [float(i) for i in list(compute_rank_weights_batch(scores, k=5, gamma=1).view(100))]
weights5_15 = [float(i) for i in list(compute_rank_weights_batch(scores, k=5, gamma=1.5).view(100))]
weights5_75 = [float(i) for i in list(compute_rank_weights_batch(scores, k=5, gamma=0.75).view(100))]
normal_dist = [float(i) for i in list(piecewise_normal_mapping(scores).view(100))]

import matplotlib.pyplot as plt

x = list(range(0, 100))
y = [0]*80
y.extend([1]*20)

plt.figure(figsize=(10, 5))
plt.plot(x, y, color="#1f77b4", linewidth=2, label="20%/80%")  # 蓝色
plt.plot(x, weights20_15, color="#ff7f0e", linewidth=2, label="k=20 gamma=1.5")  # 橙色
plt.plot(x, weights20_2, color="#2ca02c", linewidth=2, label="k=20 gamma=2")  # 绿色
plt.plot(x, weights20_10, color="#d62728", linewidth=2, label="k=20 gamma=1")  # 红色
plt.plot(x, weights10_15, color="#9467bd", linewidth=2, label="k=10 gamma=1.5")  # 紫色
plt.plot(x, weights10_10, color="#8c564b", linewidth=2, label="k=10 gamma=1")  # 棕色
plt.plot(x, weights5_10, color="#e377c2", linewidth=2, label="k=5 gamma=1")  # 粉色
plt.plot(x, weights5_15, color="#7f7f7f", linewidth=2, label="k=5 gamma=1.5")  # 灰色
plt.plot(x, weights5_75, color="black", linewidth=2, label="k=5 gamma=0.75")  # 灰色
plt.plot(x, normal_dist, color="gold", linewidth=2, label="Normal Dist")  # 灰色

plt.title("Soft Adaptive v.s. 20%/80%")
plt.xlabel("rank")
plt.ylabel("weights")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()