#!/bin/bash

echo "Testing Distiller Model Loading..."
echo "=================================="

# Test loading models with the specific configuration
echo "Student: llava-hf/llava-onevision-qwen2-0.5b-ov-hf (no LoRA)"
echo "Teacher: raghavlite/B3_Qwen2_2B (with LoRA)"
echo ""

python -m src.distiller \
    --student_model_path "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
    --teacher_model_path "raghavlite/B3_Qwen2_2B" \
    --teacher_lora \
    --pooling "last" \
    --temperature 0.02 \
    --device "cuda" 

echo ""
echo "Test completed!"