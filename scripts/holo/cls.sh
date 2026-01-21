#!/bin/bash
NUM_GPUS_PER_NODE=8
TRAIN_SCRIPT="main.py"
OUTPUT_DIR="training/cls/holo/ImageNet_1K"
# --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
# --subset_name "ImageNet_1K" \


torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --lora_r 2 \
    --teacher_lora True \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" \
    --dataset_split "original" \
    --model_backbone "llava_qwen2" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --image_resolution low \
    --kd_loss_type "holo" \
    --kd_weight 1.0 \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-4
