import torch

# 模拟数据
prompt_completion_ids = torch.tensor([
    [0, 151655, 10, 11, 151656, 5],
    [151656, 1, 2, 151655, 4, 151656]
])
attention_mask = torch.ones_like(prompt_completion_ids)

img_id = 151655
vid_id = 151656

# 找出所有视觉 token 的位置
vis_pos = (prompt_completion_ids == img_id) | (prompt_completion_ids == vid_id)
vis_indices = vis_pos.nonzero(as_tuple=False)

# 随机 mask 一半视觉 token
num_to_mask = vis_indices.size(0) // 2
selected = vis_indices[torch.randperm(vis_indices.size(0))[:num_to_mask]]

# 构造与原格式一致的布尔 mask
partial_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
partial_mask[selected[:, 0], selected[:, 1]] = True

# 执行 mask
attention_mask[partial_mask] = 0
print(attention_mask)