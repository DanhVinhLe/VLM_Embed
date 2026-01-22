#!/bin/bash

# --- Configuration ---
# Define the range of learning rates you want to test
LEARNING_RATES=("1e-5" "2e-5" "3e-5" "4e-5" "5e-5" "1e-4")

# Base directories
EXP_NAME="holo_tune_lr"
BASE_TRAIN_DIR="training/$EXP_NAME"
BASE_EVAL_DIR="eval_outputs/$EXP_NAME"

# Ensure output directories exist
mkdir -p $BASE_TRAIN_DIR
mkdir -p $BASE_EVAL_DIR

echo "-------------------------------------------------------"
echo "Starting Learning Rate Search..."
echo "Testing: ${LEARNING_RATES[*]}"
echo "-------------------------------------------------------"

for LR in "${LEARNING_RATES[@]}"; do
    EXP_NAME="HOLO_LR_${LR}"
    CURRENT_MODEL_DIR="$BASE_TRAIN_DIR/$EXP_NAME"
    CURRENT_EVAL_DIR="$BASE_EVAL_DIR/$EXP_NAME"
    
    echo "[$(date +%T)] Testing LR: $LR"

    # 1. RUN TRAINING
    # We override --learning_rate and --output_dir dynamically
    torchrun --nproc_per_node=8 main.py \
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
        --output_dir "$CURRENT_MODEL_DIR" \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate $LR \
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
        --kd_loss_type "holo" \
        --image_resolution "low" \
        --projector_config_path "./config/projector_config.json" \
        --projector_lr 5e-5 \
        --report_to "none"

    # 2. RUN EVALUATION
    # Points to the checkpoint created in the step above
    accelerate launch --multi_gpu --num_processes=8 eval_mmeb.py \
        --model_name "$CURRENT_MODEL_DIR/checkpoint-final" \
        --encode_output_path "$CURRENT_EVAL_DIR" \
        --dataset_name "TIGER-Lab/MMEB-eval" \
        --subset_name "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
        --dataset_split "test" \
        --per_device_eval_batch_size 128 \
        --image_dir "eval_images/" \
        --pooling "eos" \
        --model_backbone "llava_qwen2" \
        --normalize True \
        --bf16 \
        --tgt_prefix_mod
done