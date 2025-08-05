# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random
import numpy as np 

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from torch.distributions.normal import Normal
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url


import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.frame_num = script_args.frame_num
        self.soft_k = script_args.soft_k
        self.soft_gamma = script_args.soft_gamma
        self.exp_type = script_args.exp_type
        self.entropy_ratio = script_args.entropy_ratio
        self.dep_ratio = script_args.dep_ratio
        self.temp_dep_ratio = script_args.temp_dep_ratio
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["torch_dtype"] = torch.bfloat16

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                # model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            model = model.to(device)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        #self.ref_model = None
        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal = script_args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def compute_token_entropy(self, logits):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # [bsz, seq_len] or [nnz]
        return entropy, log_probs

    def _get_per_token_logps(self, model, input_ids, need_entropy=True, **kwargs):
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        per_token_logps = []
        if need_entropy:
            token_entropies,output_logits = self.compute_token_entropy(logits)
        else:
            token_entropies = None
            output_logits = None
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps), token_entropies, output_logits
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        input_copy = copy.deepcopy(inputs[0]['prompt'])            
        input_copy = self.remove_none_from_data(input_copy)
        
        if inputs[0]['data_type'] == 'image':
            input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
        elif inputs[0]['data_type'] == 'video':
            input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
        
        if self.frame_num == 32:
            from qwen_vl_utils import process_vision_info_32frames as process_vision_info
        else:
            from qwen_vl_utils import train_process_vision_info as process_vision_info
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
                
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

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]


        if (self.temporal or 'temp_dep' in self.exp_type) and video_inputs:
            if 'reverse' in self.exp_type:
                indices = torch.arange(video_inputs[0].size(0) - 1, -1, -1)
                shuffled_video_inputs = [video_inputs[0][indices]]
            elif 'partial_shuffle' in self.exp_type:
                T = video_inputs[0].size(0)  # 总帧数
                seg_len = 4  # 每段长度（你可调节）
                num_segs = T // seg_len
                video = video_inputs[0][:num_segs * seg_len]
                segments = video.view(num_segs, seg_len, *video.shape[1:])
                perm = torch.randperm(num_segs)
                shuffled_segments = segments[perm]
                shuffled_video = shuffled_segments.view(-1, *video.shape[1:])
                if T % seg_len != 0:
                    tail = video_inputs[0][num_segs * seg_len:]
                    shuffled_video = torch.cat([shuffled_video, tail], dim=0)
                shuffled_video_inputs = [shuffled_video]
            else:
                indices = torch.randperm(video_inputs[0].size(0))
                shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]
        
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            full_attention = torch.cat([prompt_mask, torch.ones_like(completion_ids, device=prompt_mask.device)], dim=1)
            
            if self.temporal:
                if video_inputs:
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**shuffled_prompt_inputs, generation_config=self.shuffled_generation_config)
                    shuffled_prompt_length = shuffled_prompt_ids.size(1)
                    shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                    shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
                    shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)
                else:
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.dummy_generation_config)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)

        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
                # prompt_inputs["second_per_grid_ts"] = torch.tensor(prompt_inputs["second_per_grid_ts"]).repeat(len(prompt_completion_ids), 1)
        
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        if 'img_dependent' in self.exp_type:
            masked_inputs = copy.deepcopy(prompt_inputs)
            masked_inputs['input_ids'] = prompt_completion_ids
            masked_inputs['attention_mask'] = full_attention
            masked_inputs.pop("input_ids")
            
            img_id = model.config.get('image_token_id', 151655)
            vid_id = model.config.get('image_token_id', 151656)

            vis_pos = (prompt_completion_ids == img_id) | (prompt_completion_ids == vid_id)
            if 'partial' in self.exp_type:
                vis_indices = vis_pos.nonzero(as_tuple=False)  # [N, 2]
                num_to_mask = vis_indices.size(0) // 2
                selected = vis_indices[torch.randperm(vis_indices.size(0))[:num_to_mask]]
                partial_mask = torch.zeros_like(masked_inputs["attention_mask"], dtype=torch.bool)
                partial_mask[selected[:, 0], selected[:, 1]] = True  # 只对选中的位置赋值为 True
                masked_inputs["attention_mask"][partial_mask] = 0
            else:
                masked_inputs["attention_mask"][vis_pos] = 0
        
        if 'temp_dep' in self.exp_type and inputs[0]['data_type'] == 'video':
            shuffled_prompt_inputs["pixel_values_videos"] = shuffled_prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1).to(device)
            shuffled_prompt_inputs["video_grid_thw"] = shuffled_prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1).to(device)
            if 'second_per_grid_ts' in shuffled_prompt_inputs:
                del shuffled_prompt_inputs["second_per_grid_ts"]
            shuffled_prompt_inputs.pop("input_ids")
            shuffled_prompt_inputs.pop('attention_mask')
        
        try:
            per_token_logps, token_entropies, per_token_logits = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
            token_entropies = token_entropies[:, prompt_length - 1 :]
            per_token_logits = per_token_logits[:, prompt_length - 1 :]
            with torch.no_grad():
                if 'img_dependent' in self.exp_type:
                    masked_logps, _, masked_logits = self._get_per_token_logps(model, prompt_completion_ids, **masked_inputs)
                    masked_logps = masked_logps[:, prompt_length - 1:]
                    masked_logits = masked_logits[:, prompt_length - 1:]
                    dep_scores = per_token_logps - masked_logps
                else:
                    # dummy call to ensure NCCL sync
                    _, _, _ =  self._get_per_token_logps(model, prompt_completion_ids, need_entropy=False, **prompt_inputs)

                if 'temp_dep' in self.exp_type:
                    if inputs[0]['data_type'] == 'video':
                        temp_dep_logps, _, temp_dep_logits = self._get_per_token_logps(model, prompt_completion_ids, **shuffled_prompt_inputs)
                        temp_dep_logps = temp_dep_logps[:, prompt_length - 1:]
                        temp_dep_logits = temp_dep_logits[:, prompt_length - 1:]
                        temp_dep_scores = per_token_logps - temp_dep_logps
                    else:
                         _, _, _ =  self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False, **prompt_inputs)
                else:
                     _, _, _ =  self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False, **prompt_inputs)

        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            per_token_logps, token_entropies, per_token_logits = self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False)
            with torch.no_grad():
                if 'img_dependent' in self.exp_type:
                    masked_logps, _, masked_logits = self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False)
                    masked_logps = masked_logps[:, prompt_length - 1:]
                    masked_logits = masked_logits[:, prompt_length - 1:]
                    dep_scores = per_token_logps - masked_logps
                else:
                    # dummy call to ensure NCCL sync
                    _, _, _ =  self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False)

                if 'temp_dep' in self.exp_type:
                    if inputs[0]['data_type'] == 'video':
                        temp_dep_logps, _, temp_dep_logits = self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False)
                        temp_dep_logps = temp_dep_logps[:, prompt_length - 1:]
                        temp_dep_logits = temp_dep_logits[:, prompt_length - 1:]
                        temp_dep_scores = per_token_logps - temp_dep_logps
                    else:
                         _, _, _ =  self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False)
                else:
                     _, _, _ =  self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False)

        valid_mask = copy.deepcopy(completion_mask)
        entropy_valid_mask = copy.deepcopy(completion_mask)
        temp_dep_valid_mask = copy.deepcopy(completion_mask)
        dep_completion_mask, entropy_completion_mask, temp_dep_completion_mask = None, None, None
        dep_weight, entropy_weight, temp_dep_weight = None, None, None

        def compute_rank_weights(scores, k=20, gamma=1.5):
            batch_size, seq_len = scores.shape
            flat_scores = scores.contiguous().view(-1).float()
            sorted_indices = torch.argsort(flat_scores, descending=True)
            ranks = torch.empty_like(sorted_indices)
            ranks[sorted_indices] = torch.arange(len(flat_scores), device=scores.device)
            normalized_ranks = ranks.float() / (len(flat_scores) - 1)
            weights = 1 + gamma * (torch.sigmoid(k * (0.5 - normalized_ranks)) - 0.5)
            weights = weights.view(batch_size, seq_len)
            return weights

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
            
        if 'img_dependent' in self.exp_type:
            if 'soft' in self.exp_type:
                dep_scores[~(valid_mask.bool())] = float('-inf')
                dep_weight = compute_rank_weights(dep_scores, k=self.soft_k, gamma=self.soft_gamma)
            elif 'dist' in self.exp_type:
                dep_scores[~(valid_mask.bool())] = float('-inf')
                dep_weight = piecewise_normal_mapping(dep_scores)
            elif 'kl' in self.exp_type:
                with torch.no_grad():
                    # Convert to probabilities
                    v_p = per_token_logits.exp()  # [B, L, V]
                    v_kl_div = (v_p * (per_token_logits - masked_logits)).sum(dim=-1)  # [B, L]
                    valid_v_kl_div = v_kl_div[valid_mask.bool()]
                    valid_v_kl_div = valid_v_kl_div.contiguous().view(-1).float()
                    valid_v_kl_div = valid_v_kl_div[torch.isfinite(valid_v_kl_div)]
                    threshold = torch.quantile(valid_v_kl_div, 1 - self.dep_ratio)
                    v_kl_div[~(valid_mask.bool())] = float('-inf')
                    dep_mask = (v_kl_div >= threshold).float()  # (bs, seq_len)
                    dep_completion_mask = valid_mask * dep_mask.bool()
            else:
                valid_dep_scores = dep_scores[valid_mask.bool()] 
                flat_dep_scores = valid_dep_scores.contiguous().view(-1).float()
                flat_dep_scores = flat_dep_scores[torch.isfinite(flat_dep_scores)]  # 过滤掉 nan 和 inf
                threshold = torch.quantile(flat_dep_scores, 1 - self.dep_ratio)
                dep_scores[~(valid_mask.bool())] = float('-inf')
                dep_mask = (dep_scores >= threshold).float()  # (bs, seq_len)
                dep_completion_mask = valid_mask * dep_mask.bool()
            # print(f"""Dep threshold: {threshold} 
            #         original data shape: {dep_scores.shape},
            #         dep selected data shape: {(dep_mask == 1).sum(dim=1)},
            #         dep selected token num: {(dep_mask == 1).sum()},
            #         current completion_mask: {dep_completion_mask.sum(dim=1)}""")

        if 'entropy' in self.exp_type:
            if 'soft' in self.exp_type:
                token_entropies[~(entropy_valid_mask.bool())] = float('-inf')
                entropy_weight = compute_rank_weights(token_entropies, k=self.soft_k, gamma=self.soft_gamma)
            elif 'dist' in self.exp_type:
                token_entropies[~(valid_mask.bool())] = float('-inf')
                entropy_weight = piecewise_normal_mapping(token_entropies)
            else:
                valid_entropy = token_entropies[entropy_valid_mask.bool()] 
                if valid_entropy.numel() == 0:
                    raise ValueError("All entries in valid_entropy are NaN or Inf!")

                flat_entropy = valid_entropy.contiguous().view(-1).float()
                flat_entropy = flat_entropy[torch.isfinite(flat_entropy)]   # 过滤掉 nan 和 inf
                threshold = torch.quantile(flat_entropy, 1 - self.entropy_ratio)

                token_entropies[~(entropy_valid_mask.bool())] = float('-inf')
                entropy_mask = (token_entropies >= threshold).float()  # (bs, seq_len)
                entropy_completion_mask = (entropy_valid_mask * entropy_mask).bool()
                
            # print(f"""Entropy threshold: {threshold} 
            #         original data shape: {token_entropies.shape},
            #         entropy selected data shape: {(entropy_mask == 1).sum(dim=1)},
            #         entropy selected token num: {(entropy_mask == 1).sum()},
            #         entropy completion_mask: {entropy_completion_mask.sum(dim=1)}""")
            
            # batch_tokens = completion_ids * completion_mask
            # batch_tokens = [row[row != 0] for row in batch_tokens]
            # high_dep_tokens = self.processing_class.batch_decode(
            #     batch_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            # )
            # print("High-Entropy:", high_dep_tokens)
            # # 以追加模式打开，encoding 根据需要指定
            # with open('/mnt/bn/tns-live-mllm/private/wangzy/Video-R1/high_entropy_tokens_batch_level.txt', "a", encoding="utf-8") as f:
            #     for item in high_dep_tokens:
            #         f.write(f"{item}\n")
        if 'temp_dep' in self.exp_type and inputs[0]['data_type'] == 'video':
            if 'soft' in self.exp_type:
                temp_dep_scores[~(temp_dep_valid_mask.bool())] = float('-inf')
                temp_dep_weight = compute_rank_weights(temp_dep_scores, k=self.soft_k, gamma=self.soft_gamma)
            elif 'dist' in self.exp_type:
                temp_dep_scores[~(valid_mask.bool())] = float('-inf')
                temp_dep_weight = piecewise_normal_mapping(temp_dep_scores)
            elif 'kl' in self.exp_type:
                with torch.no_grad():
                    # Convert to probabilities
                    t_p = per_token_logits.exp()  # [B, L, V]
                    t_kl_div = (t_p * (per_token_logits - temp_dep_logits)).sum(dim=-1)  # [B, L]
                    valid_t_kl_div = t_kl_div[temp_dep_valid_mask.bool()]
                    valid_t_kl_div = valid_t_kl_div.contiguous().view(-1).float()
                    valid_t_kl_div = valid_t_kl_div[torch.isfinite(valid_t_kl_div)]
                    threshold = torch.quantile(valid_t_kl_div, 1 - self.temp_dep_ratio)
                    t_kl_div[~(temp_dep_valid_mask.bool())] = float('-inf')
                    temp_dep_mask = (t_kl_div >= threshold).float()  # (bs, seq_len)
                    temp_dep_completion_mask = temp_dep_valid_mask * temp_dep_mask.bool()
            else:
                valid_temp_dep_scores = temp_dep_scores[temp_dep_valid_mask.bool()] 
                flat_temp_dep_scores = valid_temp_dep_scores.contiguous().view(-1).float()
                flat_temp_dep_scores = flat_temp_dep_scores[torch.isfinite(flat_temp_dep_scores)]
                threshold = torch.quantile(flat_temp_dep_scores, 1 - self.temp_dep_ratio)
                temp_dep_scores[~(temp_dep_valid_mask.bool())] = float('-inf')
                temp_dep_mask = (temp_dep_scores >= threshold).float()  # (bs, seq_len)
                temp_dep_completion_mask = temp_dep_valid_mask * temp_dep_mask.bool()
                
            # print(f"""Temp dep threshold: {threshold} 
            #         original data shape: {temp_dep_scores.shape},
            #         temp dep selected data shape: {(temp_dep_mask == 1).sum(dim=1)},
            #         temp dep selected token num: {(temp_dep_mask == 1).sum()},
            #         current completion_mask: {temp_dep_completion_mask.sum(dim=1)}""")

            # batch_tokens = completion_ids * temp_dep_completion_mask
            # batch_tokens = [row[row != 0] for row in batch_tokens]
            # high_dep_tokens = self.processing_class.batch_decode(
            #     batch_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            # )
            # with open('/mnt/bn/tns-live-mllm/private/wangzy/Video-R1/display_tokens/high_temp_dep_tokens.txt', "a", encoding="utf-8") as f:
            #     for item in high_dep_tokens:
            #         f.write(f"{item}\n")

        
        # compute the final completion mask
        final_weights = None
        if ('soft' in self.exp_type or 'dist' in self.exp_type) and 'max' not in self.exp_type:
            valid_weights = [t for t in [dep_weight, entropy_weight, temp_dep_weight] if t is not None]
            if valid_weights:
                final_weights = valid_weights[0]
                for t in valid_weights[1:]:
                    final_weights = final_weights + t
                final_weights = final_weights / len(valid_weights)
        if ('soft' in self.exp_type or 'dist' in self.exp_type) and 'max' in self.exp_type:
            valid_weights = [t for t in [dep_weight, entropy_weight, temp_dep_weight] if t is not None]
            if len(valid_weights) == 1:
                final_weights = valid_weights[0]
            elif len(valid_weights) == 0:
                final_weights = None
            else:
                valid_weights = torch.stack(valid_weights)
                final_weights, _ = torch.max(valid_weights, dim=0)

        else:
            valid_masks = [t for t in [dep_completion_mask, entropy_completion_mask, temp_dep_completion_mask] if t is not None]
            if valid_masks:
                completion_mask = valid_masks[0]
                for t in valid_masks[1:]:
                    completion_mask = torch.logical_or(completion_mask, t)
        

        print(f"ORGINAL completion_mask: {valid_mask.sum(dim=1)}, ORIGINAL valid tokens: {(valid_mask == 1).sum()}")
        print(f"FINAL completion_mask: {completion_mask.sum(dim=1)}, FINAL updated tokens: {(completion_mask == 1).sum()}, UPDATE RATIO = {(completion_mask == 1).sum()/(valid_mask == 1).sum()}")
        selected_tokens = completion_ids * completion_mask
        unselected_tokens = completion_ids[~completion_mask]
        selected_tokens = [row[row != 0] for row in selected_tokens]
        unselected_tokens = [row[row != 0] for row in unselected_tokens]
        selected_tokens = self.processing_class.batch_decode(
            selected_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        unselected_tokens = self.processing_class.batch_decode(
            unselected_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print("Selected Tokens:", selected_tokens)
        # print("Unselected Tokens:", unselected_tokens)
        # # 以追加模式打开，encoding 根据需要指定
        # with open('/mnt/bn/tns-live-mllm/private/wangzy/Video-R1/display_tokens/selected_tokens.txt', "a", encoding="utf-8") as f:
        #     for item in selected_tokens:
        #         f.write(f"{item}\n")
        # with open('/mnt/bn/tns-live-mllm/private/wangzy/Video-R1/display_tokens/unselected_tokens.txt', "a", encoding="utf-8") as f:
        #     for item in unselected_tokens:
        #         f.write(f"{item}\n")

        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps,_, _ = self._get_per_token_logps(self.ref_model, prompt_completion_ids, need_entropy=False, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps,_, _ = self._get_per_token_logps(model, prompt_completion_ids, need_entropy=False, **prompt_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                # ref_per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device)
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps,_, _ = self._get_per_token_logps(model, prompt_completion_ids,need_entropy=False)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)  # 限制 x 的范围
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        if self.temporal and video_inputs:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]
                
            # Compute the rewards
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        if self.temporal and video_inputs:
            temporal_rewards_per_func = rewards_per_func.clone()
            
            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask, 0] = temporal_rewards_per_func[mask, 0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards =  torch.tensor([0.5]).to('cuda')
        
        # Sum the rewards from all reward functions
        if self.temporal and video_inputs:
            rewards = temporal_rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func.sum(dim=1)
    
        if self.len_control:
            mem_rewards = [0] * self.num_generations
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
            #             if len(selected_indices) > 1 and len(selected_indices) < self.num_generations:
            # if len(selected_indices) > 1:
            #     selected_items = [(i, lenth_list[i]) for i in selected_indices]
            #     sorted_items = sorted(selected_items, key=lambda x: x[1], reverse=True)
            #     N = len(sorted_items)
            #     for rank, (idx, length) in enumerate(sorted_items):
            #         reward = 0.2 - 0.2 * (rank / N)
            #         rewards[idx] += reward
            #         mem_rewards[idx] = reward
            # for idx in range(len(lenth_list)):
            #     if lenth_list[idx] >= 512:
            #         rewards[idx] -= 0.5
                    
            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 512:
                        rewards[idx] += 0.2
        
        print(f"rewards: {rewards}")

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # per_token_loss = -per_token_loss
        if final_weights is not None:
            per_token_loss = per_token_loss * final_weights

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        num_devices = gathered_rewards.size(0) // self.num_generations 
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        if self.temporal:
            temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)
            self._metrics["temporal_rewards"].append(self.accelerator.gather_for_metrics(temporal_rewards_list).mean().item())
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))