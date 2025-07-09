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


# 示例数据
scores = torch.tensor([[0.1, 0.5, 0.3], [0.4, 0.2, 0.6]])
weights = compute_rank_weights_batch(scores)
print("Scores:", scores)
print("Weights:", weights)
