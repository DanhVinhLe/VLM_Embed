#!/bin/bash

# =========================================================================
# Hyperparameter Tuning Script for SSA (Training + Eval + WandB Sync)
# =========================================================================

# --- Configuration ---
NUM_GPUS_PER_NODE=8
TRAIN_SCRIPT="main.py"
EVAL_SCRIPT="eval_mmeb.py"
BASE_EXP_NAME="SSA_Tune_DocVQA"
WANDB_PROJECT="vlm_distillation" # Ensure this matches what main.py uses

# 1. Dataset Configuration
USE_FULLSET=false

if [ "$USE_FULLSET" = true ]; then
    SUBSETS=("OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W")
    echo "Running with FULL dataset set."
else
    SUBSETS=("OK-VQA")
    echo "Running with SINGLE dataset (DocVQA) for tuning efficiency."
fi

# Evaluation Subsets (Datasets to appear in WandB Table)
# Note: Ensure these are space-separated for the eval script
# EVAL_SUBSETS_ARR=("OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W")
EVAL_SUBSETS_ARR=("OK-VQA")

EVAL_SUBSETS_STR="${EVAL_SUBSETS_ARR[*]}" # Join array to string

# =========================================================================
# Hyperparameter Grid
# =========================================================================

KD_WEIGHTS=(0.3 0.5 1.0)
VAR_THRESHOLDS=(0.90 0.95 0.99)
GAP_WEIGHTS=(0.1 1.0)
MATRYOSHKA_OPTS=("64" "64 128") 

# =========================================================================
# Tuning Loop
# =========================================================================

for kd_w in "${KD_WEIGHTS[@]}"; do
  for var_t in "${VAR_THRESHOLDS[@]}"; do
    for gap_w in "${GAP_WEIGHTS[@]}"; do
      for mat_dim in "${MATRYOSHKA_OPTS[@]}"; do

        # 1. Generate Experiment Name
        MAT_SUFFIX=""
        if [ -n "$mat_dim" ]; then
            MAT_SUFFIX="_mat${mat_dim// /_}" # Replace space with underscore for filename
        fi
        
        CURRENT_EXP_NAME="${BASE_EXP_NAME}_kd${kd_w}_var${var_t}_gap${gap_w}${MAT_SUFFIX}"
        OUTPUT_DIR="training/$CURRENT_EXP_NAME"
        
        # 2. Generate and Export Unique WandB ID
        # This ensures Training and Eval write to the exact same run
        export WANDB_RUN_ID="SSA_${CURRENT_EXP_NAME}_$(date +%s)"
        export WANDB_PROJECT="$WANDB_PROJECT"

        echo "================================================================"
        echo "Starting Experiment: $CURRENT_EXP_NAME"
        echo "Run ID: $WANDB_RUN_ID"
        echo "================================================================"

        # 3. Prepare Arguments
        MAT_ARG=""
        if [ -n "$mat_dim" ]; then
            MAT_ARG="--ssa_matryoshka_dims $mat_dim"
        fi

        # 4. Run Training
        # Note: main.py will pick up WANDB_RUN_ID from environment
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
            --eval_subset_name "${EVAL_SUBSETS_ARR[@]}" \
            --dataset_split "original" \
            --image_dir "vlm2vec_train/MMEB-train" \
            --eval_image_dir "eval_images/" \
            --output_dir "$OUTPUT_DIR" \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-5 \
            --num_train_epochs 1 \
            --bf16 \
            --save_total_limit 1 \
            --logging_steps 1 \
            --save_strategy "epoch" \
            --seed 42 \
            --weight_decay 0.01 \
            --normalize True \
            --teacher_normalize True \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.03 \
            --projector_config_path "./config/projector_config.json" \
            --projector_lr 5e-5 \
            --image_resolution "low" \
            --report_to "wandb" \
            --run_name "$CURRENT_EXP_NAME" \
            --kd_loss_type "ssa" \
            --kd_weight "$kd_w" \
            --spectral_variance_threshold "$var_t" \
            --modality_gap_weight "$gap_w" \
            $MAT_ARG

        # 5. Run Evaluation (On checkpoint-final)
        CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint-final"
        
        echo "--> Training finished. Starting Evaluation on: $CHECKPOINT_PATH"
        
        # Using accelerate launch for multi-GPU eval
        accelerate launch --multi_gpu --num_processes="$NUM_GPUS_PER_NODE" $EVAL_SCRIPT \
            --model_name "$CHECKPOINT_PATH" \
            --encode_output_path "$OUTPUT_DIR" \
            --dataset_name "TIGER-Lab/MMEB-eval" \
            --subset_name "${EVAL_SUBSETS_ARR[@]}" \
            --dataset_split "test" \
            --per_device_eval_batch_size 128 \
            --image_dir "eval_images/" \
            --pooling "eos" \
            --model_backbone "llava_qwen2" \
            --normalize True \
            --bf16 \
            --tgt_prefix_mod 

        # 6. Log Eval Results to WandB
        echo "--> Evaluation finished. Pushing results to WandB..."
        # We pass the ID and Project to the python script to append the table
        python3 scripts/log_eval_results.py "$WANDB_RUN_ID" "$WANDB_PROJECT" "$OUTPUT_DIR"

        echo "Finished Cycle: $CURRENT_EXP_NAME"
        
        # 7. Cleanup (Optional: Remove checkpoints to save space, keep logs/jsons)
        # echo "Cleaning up checkpoints..."
        # rm -rf "$OUTPUT_DIR/checkpoint-*"

      done
    done
  done
done

# Cleanup temporary python script
rm log_eval_results.py
echo "All tuning experiments completed."