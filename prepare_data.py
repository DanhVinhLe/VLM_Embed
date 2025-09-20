import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
import time
from PIL import Image

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig

from src.model.model import MMEBModel
from src.data.dataset.mmeb_dataset import EvalDataset
from src.data.collator.eval_collator import EvalCollator
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset, concatenate_datasets
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor, COLPALI, PHI3V, VLM_IMAGE_TOKENS
from torch.nn.utils.rnn import pad_sequence
import shutil 

from transformers import ProcessorMixin

def batch_to_device(batch, device):
    _batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            _batch[k] = v.to(device)
        else:
            _batch[k] = v
    return _batch

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

class TextImageDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name, subset,
                split=data_args.dataset_split,
            )
            train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)
    
    def __len__(self):
        return len(self.train_data)
    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        
    def __getitem__(self, data_idx):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>get image called, {data_idx}", flush=True)
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
        _qry_texts, _qry_images, _pos_texts, _pos_images = [], [], [], []
        backbone = self.model_args.model_backbone
        for qry_text, qry_image_path, pos_text, pos_image_path in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if backbone != PHI3V:
                qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone])
                pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone])
                neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone]) if neg_text else None
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                print("empty inputs")
                continue
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)

        return {"query_text": _qry_texts, "query_image": _qry_images,
                "pos_text": _pos_texts, "pos_image": _pos_images}
    

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)
    
    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )
    
    for idx, subset in enumerate(data_args.subset_name):
        
        encode_output_path = os.path.join(data_args.encode_output_path, f"{subset}_{data_args.dataset_split}_encoded.pkl")
        if os.path.exists(encode_output_path):
            print_rank(f"Found existing encoded file: {encode_output_path}, skipping...")
            continue
        
        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset_name=subset,
            text_field="qry",
            image_path_field="qry_image_path",
        )
        
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset_name=subset,
            text_field="pos_text",
            image_path_field="pos_image_path",
        )
        
        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        
        
        if not os.path.exists(encode_output_path):
            encoded_qry_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc=f"Encoding {subset} query set"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(qry=batch)
                    encoded_qry_tensor.append(output['qry_embeds'].cpu().detach().float())
                encoded_qry_tensor = np.concatenate(encoded_qry_tensor)
            print(f"encoded_qry_tensor shape: {encoded_qry_tensor.shape}")    
            
            encoded_tgt_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc=f"Encoding {subset} target set"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(tgt=batch)
                    encoded_tgt_tensor.append(output['tgt_embeds'].cpu().detach().float())
                encoded_tgt_tensor = np.concatenate(encoded_tgt_tensor)
            print(f"encoded_tgt_tensor shape: {encoded_tgt_tensor.shape}")
            assert len(eval_qry_dataset) == len(encoded_qry_tensor), f"len(eval_qry_dataset)={len(eval_qry_dataset)} vs len(encoded_qry_tensor)={len(encoded_qry_tensor)}"
            assert len(eval_tgt_dataset) == len(encoded_tgt_tensor), f"len(eval_tgt_dataset)={len(eval_tgt_dataset)} vs len(encoded_tgt_tensor)={len(encoded_tgt_tensor)}"
            with open(encode_output_path, "wb") as f:
                pickle.dump({
                    'qry_embeddings': encoded_qry_tensor,
                    'tgt_embeddings': encoded_tgt_tensor,
                    'index': list(range(len(eval_qry_dataset))),
                })
            print_rank(f"Encoded file saved to {encode_output_path}")
        else:
            print_rank(f"Encoded file already exists: {encode_output_path}, skipping...")
            
if __name__ == "__main__":
    main()