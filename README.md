# VLMEmbed
## Set up
```
pip install -r requirement.txt
pip install uv 
uv pip install flash-attn==2.7.3 --no-build-isolation
```
Because the error of code in **Transformers library**, run the following script to find the error and comment some lines: 
```bash 
python  eval_mmeb.py  --model_name raghavlite/B3_Qwen2_2B --encode_output_path  ./MMEB-evaloutputs/B2_Qwen2_2B/  --pooling  eos  --normalize  True  --lora  --lora_r  8  --bf16  --dataset_name  TIGER-Lab/MMEB-eval  --subset_name  MSCOCO_i2t  --dataset_split  test  --per_device_eval_batch_size  4  --image_dir  eval_images/  --tgt_prefix_mod
```

## Inference & Evaluation

Download the image file zip from huggingface
```bash
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```

1. To evaluate our model on an MMEB dataset (e.g., MSCOCO_i2t), run:
```bash 
python  eval_mmeb.py  --model_name raghavlite/B3_Qwen2_2B --encode_output_path  ./MMEB-evaloutputs/B2_Qwen2_2B/  --pooling  eos  --normalize  True  --lora  --lora_r  8  --bf16  --dataset_name  TIGER-Lab/MMEB-eval  --subset_name  MSCOCO_i2t  --dataset_split  test  --per_device_eval_batch_size  4  --image_dir  eval_images/  --tgt_prefix_mod
```

## Running on your Data

To run models on your dataset, just repurpose the **eval_mmeb.py** code. It is quick and simple to do. Lines 120-126 create a query dataset. Lines 127-138 create a target dataset. Ensure your data is in the same format as a reference query and target dataset (Eg. MSCOCO_i2t or MSCOCO_t2i). Lines 159-160 extract query embeddings and Lines 172-172 extract target embeddings. We will soon release a much simpler script for general purpose usage.

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
