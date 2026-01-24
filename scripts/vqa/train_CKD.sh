#!/bin/bash

# GPU per node
NUM_GPUS_PER_NODE=8

# Configs
TRAIN_SCRIPT="main.py"
EXP_NAME="CKD_DocVQA_bs32"
USE_FULLSET=false

# 1. Define Training Subsets
if [ "$USE_FULLSET" = true ]; then
    SUBSETS=("OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W")
    echo "Running with FULL dataset set."
else
    SUBSETS=("DocVQA")
    echo "Running with SINGLE dataset (DocVQA)."
fi

# 2. [FIX] Define Evaluation Subsets
# These are the datasets you want to evaluate and see in the WandB table at the end.
# You can make this the same as SUBSETS or a different list.
EVAL_SUBSETS=("DocVQA" "InfographicsVQA") 

# =========================================================================
# Run with torchrun
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
    --eval_dataset_name "TIGER-Lab/MMEB-eval" \
    --subset_name "${SUBSETS[@]}" \
    --eval_subset_name "${EVAL_SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --eval_image_dir "eval_images/" \
    --output_dir "training/$EXP_NAME" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
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
    --kd_loss_type "ckd" \
    --ckd_temperature 0.07 \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5 \
    --report_to "wandb" \
    --run_name "$EXP_NAME"