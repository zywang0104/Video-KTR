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

# ----------- 验证代码 -----------
# 随机生成数据（可以换成 normal 或别的）
torch.manual_seed(42)
data = torch.rand(8, 1024, device='cpu')  # 使用 CUDA（或换成 CPU）

# 应用映射
mapped = piecewise_normal_mapping(data)

# 转到 CPU 后绘图
mapped_cpu = mapped.detach().cpu().numpy()

# 画直方图
plt.figure(figsize=(8, 4))
plt.hist(mapped_cpu.flatten(), bins=50, edgecolor='black')
plt.axvline(1, color='red', linestyle='--', label='x=1')
plt.axvline(-1, color='red', linestyle='--', label='x=-1')
plt.axvline(2, color='gray', linestyle=':')
plt.axvline(0, color='gray', linestyle=':')
plt.title("Histogram of Mapped Tensor Values")
plt.xlabel("Mapped Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

data_cpu = data.detach().cpu().numpy().flatten()
mapped_cpu = mapped.detach().cpu().numpy().flatten()

# 画原始值 vs 映射值的散点图
plt.figure(figsize=(6, 5))
plt.scatter(data_cpu, mapped_cpu, alpha=0.3, s=5)
plt.title("Original Value vs Mapped Value")
plt.xlabel("Original Value")
plt.ylabel("Mapped Value")
plt.grid(True)
plt.tight_layout()
plt.show()