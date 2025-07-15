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

scores = torch.arange(1, 101).view(1,100)
weights20_15 = [float(i) for i in list(compute_rank_weights_batch(scores, k=20, gamma=1.5).view(100))]
weights20_2 = [float(i) for i in list(compute_rank_weights_batch(scores, k=20, gamma=2).view(100))]
weights20_10 = [float(i) for i in list(compute_rank_weights_batch(scores, k=20, gamma=1).view(100))]
weights10_15 = [float(i) for i in list(compute_rank_weights_batch(scores, k=10, gamma=1.5).view(100))]
weights10_10 = [float(i) for i in list(compute_rank_weights_batch(scores, k=10, gamma=1).view(100))]
weights5_10 = [float(i) for i in list(compute_rank_weights_batch(scores, k=5, gamma=1).view(100))]
weights5_15 = [float(i) for i in list(compute_rank_weights_batch(scores, k=5, gamma=1.5).view(100))]
weights5_75 = [float(i) for i in list(compute_rank_weights_batch(scores, k=5, gamma=0.75).view(100))]

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
plt.plot(x, weights5_75, color="black", linewidth=2, label="k=5 gamma=0.75 (Best Model)")  # 灰色

plt.title("Soft Adaptive v.s. 20%/80%")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()