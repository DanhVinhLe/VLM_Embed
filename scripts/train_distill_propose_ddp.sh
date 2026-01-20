#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_distill_ddp.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "MSCOCO" "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" "VisDial" "CIRR" "VisualNews_i2t" "VisualNews_t2i" "MSCOCO_i2t" "MSCOCO_t2i" "NIGHTS" "WebQA" "VOC2007" "ImageNet_1K" "N24News" "SUN397" "HatefulMemes" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "training/propose_all" \
    --percent_data 1.0 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 0.3 \
    --kd_loss_type "proposal_dtw" \
    --image_resolution "mid" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5 \