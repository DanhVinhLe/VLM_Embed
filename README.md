# VLM-Embed: Visual Language Model Embedding

VLM-Embed is a project focused on creating powerful and efficient embeddings from Visual Language Models (VLMs) using distillation techniques. This repository provides the tools to train, evaluate, and distill VLMs for various tasks.

## 🚀 Getting Started

### 1. Environment Setup

First, create and activate a Python virtual environment.

```bash
python -m venv vlm
source vlm/bin/activate
```

### 2. Install Dependencies

Install the required Python packages using pip.

```bash
pip install -r requirements.txt
```

### 3. Apply Library Fix

This project requires a small modification to the `transformers` library to address a specific issue. A script is provided to apply this fix automatically.

```bash
python fix_lib.py
```
This will comment out a few lines in `transformers/models/qwen2_vl/image_processing_qwen2_vl.py` that can cause issues.

### 4. Download Datasets

The training and evaluation datasets can be downloaded using the provided Python script. This script uses `huggingface-hub` for efficient downloading and unzips the files into the correct directory (`./vlm2vec_train/MMEB-train/images`).

The script will download approximately 20 datasets, which can take a significant amount of time and disk space.

```bash
python download.py
```

You can also optionally download the evaluation image files separately if needed:

```bash
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
rm images.zip
```

## 🚂 Training

The `scripts` directory contains various shell scripts for running different training configurations. These scripts are the primary way to initiate training.

For example, to train a model using Relational Knowledge Distillation (RKD), you can run:

```bash
bash scripts/train_RKD.sh
```

Another example for a different distillation setup:

```bash
bash scripts/train_distill_propose_V.sh
```

Please inspect the scripts in the `scripts` directory for more training options and configurations.

## 📊 Inference & Evaluation

To evaluate a trained model on the MMEB (Multi-Modal Evaluation Benchmark), you can use the `eval.sh` script. Make sure your model checkpoint and evaluation data are correctly configured.

```bash
bash scripts/run_eval.sh
```

## 🙏 Acknowledgements

This project builds upon the work of several other open-source projects. We are grateful for their contributions.

- [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec)
- [B3](https://github.com/raghavlite/B3)