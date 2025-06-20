cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
PORTS=$ARNOLD_WORKER_0_PORT 
PORT=(${PORTS//,/ }) 

# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.

# Qwen/Qwen2.5-VL-7B-Instruct

torchrun --nproc_per_node="$ARNOLD_WORKER_GPU" \
    --nnodes="$ARNOLD_WORKER_NUM" \
    --node_rank="$ARNOLD_ID" \
    --master_addr="$ARNOLD_WORKER_0_HOST" \
    --master_port="$PORT" \
    src/open_r1/grpo.py \
    --output_dir "baseline_original_rft" \
    --model_name_or_path 'Qwen2.5-VL-7B-COT-SFT' \
    --dataset_name "./Video-R1-data/Video-R1-260k.json" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 true\
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal false \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Video-R1-Baseline \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
