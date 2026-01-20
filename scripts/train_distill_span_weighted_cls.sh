#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_distill_ddp.py"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

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
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "/workspace/ComfyUI/models/photomaker/VLM_Embed/vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/meta_span_weighted_cls_v4" \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
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
    --kd_weight 2.5 \
    --w_cross_modal_loss 2.5 \
    --kd_loss_type "span_propose_attn" \
    --image_resolution "low" \
    --teacher_layer_mapping 0 22 25 28 \
    --student_layer_mapping 0 18 21 24 \
    --split_layer_mapping 0 1 4 4 4 \
    --projector_lr 5e-4