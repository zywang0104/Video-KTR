    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # ——— 1. 准备 prompt，生成 completions —— 
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        input_copy = self.remove_none_from_data(input_copy)
        # process image/video paths...
        image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        # truncate to max_prompt_length...
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # Generate
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        
        # build completion_mask up to first EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # pop prompt_ids/mask, prepare prompt_inputs for log-prob
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs.pop("second_per_grid_ts", None)
        
        # ——— 2. 备份一份 masked_inputs，用于“无图像”forward —— 
        prompt_inputs_masked = {
            k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in prompt_inputs.items()
        }
        # 定位所有 <image_pad> 位置
        img_id = model.config.image_token_id
        vis_positions = prompt_completion_ids == img_id  # (B, seq_len)
        # 屏蔽视觉特征
        if "pixel_values" in prompt_inputs_masked:
            prompt_inputs_masked["pixel_values"] = torch.zeros_like(prompt_inputs_masked["pixel_values"])
        if "pixel_values_videos" in prompt_inputs_masked:
            prompt_inputs_masked["pixel_values_videos"] = torch.zeros_like(prompt_inputs_masked["pixel_values_videos"])
        # 屏蔽 attention_mask 对应位置
        if "attention_mask" in prompt_inputs_masked:
            prompt_inputs_masked["attention_mask"][vis_positions] = 0
        
        # ——— 3. Forward #1：含图像 per-token log-probs —— 
        per_token_logps_orig, _ = self._get_per_token_logps(
            model, prompt_completion_ids, **prompt_inputs
        )
        per_token_logps_orig = per_token_logps_orig[:, prompt_length - 1 :]  # (B, T)
        
        # ——— 4. Forward #2：无图像 per-token log-probs —— 
        per_token_logps_masked, _ = self._get_per_token_logps(
            model, prompt_completion_ids, **prompt_inputs_masked
        )
        per_token_logps_masked = per_token_logps_masked[:, prompt_length - 1 :]  # (B, T)
        
        # ——— 5. 计算图像依赖度分数 —— 
        dependency_scores = per_token_logps_orig - per_token_logps_masked  # (B, T)
        
        # ——— 6. Top-20% 筛选，重写 completion_mask —— 
        batch_size, seq_len = dependency_scores.size()
        dependency_mask = torch.zeros_like(dependency_scores, dtype=completion_mask.dtype)
        for i in range(batch_size):
            valid = completion_mask[i].bool()
            scores_i = dependency_scores[i].clone()
            scores_i[~valid] = float("-inf")
            k = max(1, int(valid.sum().item() * 0.2))
            topk_idx = torch.topk(scores_i, k=k).indices
            dependency_mask[i, topk_idx] = 1
        completion_mask = dependency_mask
        
        # 清理缓存（可选）
        gc.collect()
        torch.cuda.empty_cache()
        
        # ——— 7. 继续原有的 ref_model log-prob、KL 计算等，然后用新的 completion_mask 计算最终 loss —— 
        # （此处保持你原来的实现不变）
        # ...
        return loss
