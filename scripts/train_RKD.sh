#!/bin/bash
# 1. Train fast
# ./scripts/train_RKD.sh
# # 2. Evaluate fast (after training finishes)
# ./scripts/run_eval.sh training/RKD/checkpoint-final eval_outputs/RKD ImageNet_1K


# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=8

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="main.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 2 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "training/RKD" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 0.3 \
    --kd_loss_type "contrastive_rkd" \
    --image_resolution "low" 