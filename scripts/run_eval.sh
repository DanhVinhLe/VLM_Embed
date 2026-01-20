#!/bin/bash

# =========================================================================
# Professional Evaluation Script
# Usage: ./scripts/run_eval.sh [MODEL_PATH] [OUTPUT_DIR] [SUBSET_NAME]
# Example: ./scripts/run_eval.sh training/RKD/checkpoint-final eval_outputs/RKD ImageNet_1K
# =========================================================================

# MODEL_PATH=${1:-"training/RKD/checkpoint-final"}
# OUTPUT_DIR=${2:-"eval_outputs/RKD"}
# SUBSET=${3:-"ImageNet-1K"}  # Default to ImageNet_1K, or pass "all" or list
MODEL_PATH="apple/FastVLM-0.5B"
OUTPUT_DIR=eval_outputs/student-base/full
SUBSET="ImageNet-1K N24News HatefulMemes VOC2007 SUN397"
# Absolute paths
REPO_ROOT=$(pwd)
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

echo "========================================================="
echo "Starting Evaluation"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Subset: $SUBSET"
echo "========================================================="

# Determine subsets
if [ "$SUBSET" == "all" ]; then
    # Full list from eval_all.sh
    SUBSETS=("Wiki-SS-NQ" "VisDial" "CIRR" "VisualNews_t2i" "VisualNews_i2t" "MSCOCO_t2i" "MSCOCO_i2t" "NIGHTS" "WebQA" "OVEN" "FashionIQ" "EDIS" "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" "ScienceQA" "GQA" "TextVQA" "VizWiz" "ImageNet-1K" "HatefulMemes" "SUN397" "N24News" "VOC2007" "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211" "MSCOCO" "RefCOCO" "RefCOCO-Matching" "Visual7W-Pointing")
else
    # Space separated string to array
    IFS=' ' read -r -a SUBSETS <<< "$SUBSET"
fi

# Detect number of GPUs
NUM_GPUS=8
echo "Detected $NUM_GPUS GPU(s)"

# Run Evaluation with Accelerate for multi-GPU support
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using multi-GPU mode with accelerate"
    accelerate launch --multi_gpu --num_processes="$NUM_GPUS" eval_mmeb.py \
        --model_name "$MODEL_PATH" \
        --encode_output_path "$OUTPUT_DIR" \
        --dataset_name "TIGER-Lab/MMEB-eval" \
        --subset_name "${SUBSETS[@]}" \
        --dataset_split "test" \
        --per_device_eval_batch_size 128 \
        --image_dir "eval_images/" \
        --pooling "eos" \
        --model_backbone "llava_qwen2" \
        --normalize True \
        --bf16 \
        --tgt_prefix_mod 
        # --lora \
        # --lora_r 64 \
        # --lora_alpha 64
else
    echo "Using single GPU mode"
    python3 eval_mmeb.py \
        --model_name "$MODEL_PATH" \
        --encode_output_path "$OUTPUT_DIR" \
        --dataset_name "TIGER-Lab/MMEB-eval" \
        --subset_name "${SUBSETS[@]}" \
        --dataset_split "test" \
        --per_device_eval_batch_size 128 \
        --image_dir "eval_images/" \
        --pooling "eos" \
        --model_backbone "llava_qwen2" \
        --normalize True \
        --bf16 \
        --tgt_prefix_mod 
        # --lora \
        # --lora_r 64 \
        # --lora_alpha 64
fi

echo "========================================================="
echo "Evaluation Completed"
echo "Results saved in $OUTPUT_DIR"
echo "========================================================="
