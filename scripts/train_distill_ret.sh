#!/bin/bash

# Số lượng GPU bạn muốn sử dụng
NUM_GPUS=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_distillation.py"

# Đường dẫn tới file config DeepSpeed bạn vừa tạo
DS_CONFIG="config/ds_config_stage2.json"

# =========================================================================
# Cách 1: Dùng launcher của DeepSpeed (Khuyên dùng)
# =========================================================================
deepspeed --num_gpus=$NUM_GPUS $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_target_modules "qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj" \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "VisDial" "CIRR" "VisualNews_i2t" "VisualNews_t2i" "MSCOCO_i2t" "MSCOCO_t2i" "NIGHTS" "WebQA" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 0.5 \
    --output_dir "training/deepspeed_projector_ret" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --deepspeed_config $DS_CONFIG \
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
    --image_resolution "low" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5