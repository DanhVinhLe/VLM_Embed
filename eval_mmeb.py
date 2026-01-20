import json
import logging
import os
import pickle
import shutil
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, HfArgumentParser

from evaluation.mmeb_baselines.eval_utils import get_pred
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.data.collator.eval_collator import EvalCollator
from src.data.dataset.mmeb_dataset import EvalDataset
from src.model.model import MMEBModel
from src.model.processor import COLPALI, get_backbone_name, load_processor
from src.utils import print_rank

# Module-level logger for use before accelerator initialization
module_logger = logging.getLogger(__name__)


def delete_pycache(root="."):
    """Recursively deletes __pycache__ directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting: {full_path}")
                try:
                    shutil.rmtree(full_path)
                except Exception as e:
                    print(f"Could not delete {full_path}: {e}")


POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
    "ImageNet-1K": POS_MOD_CLASS_LABEL,
    "HatefulMemes": POS_MOD_CLASS_LABEL,
    "SUN397": POS_MOD_CLASS_LABEL,
    "N24News": POS_MOD_CLASS_LABEL,
    "VOC2007": POS_MOD_CLASS_LABEL,
    "Place365": POS_MOD_CLASS_LABEL,
    "ImageNet-A": POS_MOD_CLASS_LABEL,
    "ImageNet-R": POS_MOD_CLASS_LABEL,
    "ObjectNet": POS_MOD_CLASS_LABEL,
    "Country211": POS_MOD_CLASS_LABEL,
    "OK-VQA": POS_MOD_ANSWER,
    "A-OKVQA": POS_MOD_ANSWER,
    "DocVQA": POS_MOD_ANSWER,
    "InfographicsVQA": POS_MOD_ANSWER,
    "ChartQA": POS_MOD_ANSWER,
    "Visual7W": POS_MOD_ANSWER,
    "ScienceQA": POS_MOD_ANSWER,
    "GQA": POS_MOD_ANSWER,
    "TextVQA": POS_MOD_ANSWER,
    "VizWiz": POS_MOD_ANSWER,
    "MSCOCO_i2t": POS_MOD_IMAGE_CAPTION,
    "VisualNews_i2t": POS_MOD_IMAGE_CAPTION,
}


@contextmanager
def time_block(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"[Timer] {name}: {elapsed:.4f}s")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Setup logging (must be after accelerator initialization)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=False)
    
    # Clean pycache after accelerator is initialized
    if accelerator.is_main_process:
        delete_pycache()

    os.makedirs(data_args.encode_output_path, exist_ok=True)

    hf_config = AutoConfig.from_pretrained(
        model_args.model_name, trust_remote_code=True
    )
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(
            hf_config=hf_config, model_type=model_args.model_type
        )
        setattr(model_args, "model_backbone", model_backbone)
        setattr(training_args, "model_backbone", model_backbone)

    logger.info(f"Model backbone: {model_args.model_backbone}")
    
    # Load processor
    processor = load_processor(model_args, data_args)
    
    # Load model
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        
        # Check if score exists (only main process needs to check, effectively)
        # But we need all processes to agree on flow.
        # Simplest: check existence. If exists, skip.
        if os.path.exists(score_path):
            if accelerator.is_main_process:
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    logger.info(f"Found previous eval score for {subset}. Skipping.")
                    logger.info(score_dict)
                except Exception:
                    pass
            # Sync before continuing to next subset
            accelerator.wait_for_everyone()
            continue

        if accelerator.is_main_process:
            logger.info(f"Processing {idx+1}/{len(data_args.subset_name)}: {subset}")

        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        
        # Check if both embeddings exist
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
             # Sync and continue to scoring
             accelerator.wait_for_everyone()
             # We can continue to scoring logic below, which checks for score_path again?
             # Actually the original code had a continue here, meaning it skips re-encoding.
             pass 
        else:
            # --- Encode Query ---
            if not os.path.exists(encode_qry_path):
                eval_qry_dataset = EvalDataset(
                    data_args=data_args,
                    model_args=model_args,
                    subset=subset,
                    text_field="qry_text",
                    img_path_field="qry_img_path",
                )
                
                # Note: larger num_workers recommended for speed
                eval_qry_loader = DataLoader(
                    eval_qry_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    collate_fn=eval_collator,
                    shuffle=False,
                    drop_last=False,
                    num_workers=4, 
                    pin_memory=True
                )
                
                eval_qry_loader = accelerator.prepare(eval_qry_loader)
                
                encoded_qry = []
                # Disable progress bar on non-main processes to avoid clutter
                disable_tqdm = not accelerator.is_main_process
                
                with torch.no_grad():
                    for batch in tqdm(eval_qry_loader, desc=f"Encode query - {subset}", disable=disable_tqdm):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = model(qry=batch)
                        
                        # Gather results
                        reps = accelerator.gather_for_metrics(output["qry_reps"])
                        encoded_qry.append(reps.cpu())
                
                if accelerator.is_main_process:
                    encoded_tensor = torch.cat(encoded_qry).float().numpy()
                    # Truncate to exact dataset length in case of padding
                    if len(encoded_tensor) > len(eval_qry_dataset):
                        encoded_tensor = encoded_tensor[:len(eval_qry_dataset)]
                        
                    with open(encode_qry_path, "wb") as f:
                        pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)
            
            # Barrier to ensure Rank 0 is done writing before anyone proceeds (or checks existence later)
            accelerator.wait_for_everyone()

            # --- Encode Target ---
            if not os.path.exists(encode_tgt_path):
                eval_tgt_dataset = EvalDataset(
                    data_args=data_args,
                    model_args=model_args,
                    subset=subset,
                    text_field="tgt_text",
                    img_path_field="tgt_img_path",
                    mod_instruction=POS_MOD_DICT.get(subset, None)
                    if data_args.tgt_prefix_mod
                    else None,
                )

                eval_tgt_loader = DataLoader(
                    eval_tgt_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    collate_fn=eval_collator,
                    shuffle=False,
                    drop_last=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                eval_tgt_loader = accelerator.prepare(eval_tgt_loader)
                
                encoded_tgt = []
                disable_tqdm = not accelerator.is_main_process

                with torch.no_grad():
                    for batch in tqdm(eval_tgt_loader, desc=f"Encode target - {subset}", disable=disable_tqdm):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = model(tgt=batch)
                        reps = accelerator.gather_for_metrics(output["tgt_reps"])
                        encoded_tgt.append(reps.cpu())

                if accelerator.is_main_process:
                    encoded_tensor = torch.cat(encoded_tgt).float().numpy()
                    if len(encoded_tensor) > len(eval_tgt_dataset):
                        encoded_tensor = encoded_tensor[:len(eval_tgt_dataset)]
                        
                    with open(encode_tgt_path, "wb") as f:
                        pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

            accelerator.wait_for_everyone()

    # --- Calculating Scores ---
    # Only main process needs to calculate scores
    if accelerator.is_main_process:
        for subset in tqdm(data_args.subset_name, desc="Calculating scores"):
            score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
            if os.path.exists(score_path):
                continue
                
            logger.info(f"{subset}: Calculating score now!")
            
            encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
            encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
            
            logger.info("Loading cached tensors...")
            try:
                with open(encode_qry_path, "rb") as f:
                    qry_tensor, qry_index = pickle.load(f)
                with open(encode_tgt_path, "rb") as f:
                    tgt_tensor, tgt_index = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load pickle for {subset}: {e}")
                continue

            logger.info("Loading eval dataset for metadata...")
            eval_data = load_dataset(
                data_args.dataset_name,
                subset,
                split=data_args.dataset_split,
            )
            
            # Post-processing map
            if (subset == "WebQA" or subset == "EDIS") and "qry_text" in eval_data.column_names and model_args.model_backbone == "llava_qwen2":
                eval_data = eval_data.map(
                    lambda x: {"qry_text": x["qry_text"].replace("<|image_1|>", "").strip()}
                )

            # ColPali specific processing
            if model_args.model_backbone == COLPALI:
                # Flatten handling
                if len(qry_tensor) != len(qry_index):
                    qry_tensor = [t for l in qry_tensor for t in l.tolist()]
                if len(tgt_tensor) != len(tgt_index):
                    tgt_tensor = [t for l in tgt_tensor for t in l.tolist()]

                # Build mappings
                tgtkey_to_rowid = {}
                rowid_to_tgtkey = {}
                for rowid, tt in enumerate(tgt_index):
                    key = (tt["text"], tt["img_path"])
                    tgtkey_to_rowid[key] = rowid
                    rowid_to_tgtkey[rowid] = key

                # Convert to tensors and pad
                tgt_tensor = [torch.from_numpy(np.array(t)) for t in tgt_tensor]
                tgt_tensor = pad_sequence(tgt_tensor, batch_first=True)
                tgt_tensor = tgt_tensor.to(accelerator.device)

            # Build maps for dedup
            qry_key2emb, tgt_key2emb = OrderedDict(), OrderedDict()
            for qry_t, tt in zip(qry_tensor, qry_index):
                qry_key2emb[(tt["text"], tt["img_path"])] = qry_t
            for tgt_t, tt in zip(tgt_tensor, tgt_index):
                tgt_key2emb[(tt["text"], tt["img_path"])] = tgt_t
                
            logger.info(f"Unique targets: {len(tgt_key2emb)}")

            n_correct = 0
            all_pred = []
            
            # Score loop
            for row in tqdm(eval_data, desc=f"Scoring {subset}"):
                if model_args.model_backbone == COLPALI:
                    qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]
                    qry_t = torch.from_numpy(np.array([qry_t])).to(accelerator.device)
                    
                    pos_tkey = (row["tgt_text"][0], row["tgt_img_path"][0])
                    # (Optional debug prints removed, stick to logic)

                    scores = processor.score(qry_t, tgt_tensor, batch_size=1024)
                    pred_id = torch.argmax(scores, dim=1).cpu().numpy().tolist()[0]
                    pred_tkey = rowid_to_tgtkey[pred_id]
                    
                    if pred_tkey == pos_tkey:
                        n_correct += 1
                    all_pred.append(pred_tkey)
                else:
                    try:
                        qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]
                        tgt_t_list, all_candidates = [], []
                        
                        for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                            tgt_t_list.append(tgt_key2emb[tt])
                            all_candidates.append(tt)
                            
                        qry_t_arr = torch.from_numpy(np.array(qry_t))
                        tgt_t_arr = np.stack(tgt_t_list, axis=0)
                        
                        scores, pred = get_pred(qry_t_arr, tgt_t_arr, normalization=model_args.normalize)
                        if pred == 0:
                            n_correct += 1
                        all_pred.append(all_candidates[pred])
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                        # Fallback or strict fail? Original code raised ipdb. 
                        # We'll just continue or default to wrong.
                        import traceback
                        traceback.print_exc()

            accuracy = n_correct / len(eval_data)
            logger.info(f"{subset} accuracy: {accuracy:.4f}")
            
            score_dict = {
                "acc": accuracy,
                "num_correct": n_correct,
                "num_pred": len(all_pred),
                "num_data": len(eval_data)
            }
            logger.info(score_dict)
            
            with open(score_path, "w") as f:
                json.dump(score_dict, f, indent=4)
                
            with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
                for item in all_pred:
                    f.write(f"{item}\n")

    # Final barrier
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
