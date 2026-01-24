import os
import json
import pickle
import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from collections import OrderedDict
from datasets import load_dataset

from src.data.dataset.mmeb_dataset import EvalDataset
from src.data.collator.eval_collator import EvalCollator
from src.model.processor import COLPALI
from src.utils import print_rank
from src.evaluation.mmeb_baselines.eval_utils import get_pred

logger = logging.getLogger(__name__)

POS_MOD_DICT = {
    "ImageNet-1K": "Represent the class label: ",
    "HatefulMemes": "Represent the class label: ",
    "SUN397": "Represent the class label: ",
    "N24News": "Represent the class label: ",
    "VOC2007": "Represent the class label: ",
    "Place365": "Represent the class label: ",
    "ImageNet-A": "Represent the class label: ",
    "ImageNet-R": "Represent the class label: ",
    "ObjectNet": "Represent the class label: ",
    "Country211": "Represent the class label: ",
    "OK-VQA": "Represent the answer: ",
    "A-OKVQA": "Represent the answer: ",
    "DocVQA": "Represent the answer: ",
    "InfographicsVQA": "Represent the answer: ",
    "ChartQA": "Represent the answer: ",
    "Visual7W": "Represent the answer: ",
    "ScienceQA": "Represent the answer: ",
    "GQA": "Represent the answer: ",
    "TextVQA": "Represent the answer: ",
    "VizWiz": "Represent the answer: ",
    "MSCOCO_i2t": "Represent the image caption: ",
    "VisualNews_i2t": "Represent the image caption: ",
}

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def gather_tensors(tensor):
    """
    Gathers tensors from all GPUs and concatenates them on the first dimension.
    """
    if not dist.is_initialized():
        return tensor

    # Get world size
    world_size = dist.get_world_size()
    
    # We need to know the size of tensors on other processes to pad if necessary (simplification here assumes equal batch size usually)
    # Ideally, we gather all.
    with torch.no_grad():
        # 1. Gather sizes
        local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max([s.item() for s in all_sizes])

        # 2. Pad if necessary (simple padding, we assume user truncates later based on dataset length)
        if tensor.shape[0] < max_size:
            padding = torch.zeros((max_size - tensor.shape[0], *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=0)

        # 3. Gather tensors
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        
        # 4. Concatenate and remove padding from other ranks (tricky without keeping track, 
        # but usually we slice by exact dataset length later in the main logic)
        final_tensor = torch.cat(gathered_tensors, dim=0)
        
    return final_tensor

def encode_and_save(model, loader, output_path, dataset_len, desc, device):
    """
    Encodes data using the model and saves to disk.
    """
    encoded_list = []
    
    # Only show progress bar on main process
    disable_tqdm = not is_main_process()
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, disable=disable_tqdm):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Detect if it's query or target based on keys or passed arg
                if "input_ids" in batch: # Basic check, adjust based on collator
                    if "pixel_values" in batch and batch["pixel_values"] is not None:
                         # Likely multimodal
                         pass
                
                # We assume the model handles 'qry' or 'tgt' args or we call forward directly
                # Based on MMEBModel structure, it usually takes kwargs.
                # The collator usually returns specific keys. 
                # Let's check keys to decide call.
                if "pixel_values" in batch: # It is MMEB compatible batch
                    # HACK: The collator logic in EvalCollator returns generic keys. 
                    # We need to know if this is query or target loader.
                    # MMEBModel forward signature: forward(qry=None, tgt=None)
                    # We will rely on the caller to name the loader/context.
                    if "qry" in desc.lower():
                        output = model(qry=batch)
                        reps = output["qry_reps"]
                    else:
                        output = model(tgt=batch)
                        reps = output["tgt_reps"]
            
            # Gather from all GPUs
            gathered_reps = gather_tensors(reps)
            encoded_list.append(gathered_reps.cpu())

    if is_main_process():
        encoded_tensor = torch.cat(encoded_list).float().numpy()
        # Truncate to exact dataset length (removes DDP padding)
        if len(encoded_tensor) > dataset_len:
            encoded_tensor = encoded_tensor[:dataset_len]
        return encoded_tensor
    return None

def run_mmeb_evaluation(model, processor, model_args, data_args, training_args, device):
    """
    Main entry point to run MMEB evaluation from main.py
    """
    # Ensure model is in eval mode
    model.eval()
    
    results_dict = {}
    
    # Use eval specific args if provided, else fallback to subset_name (warning: usually different)
    # We expect data_args.eval_subset_name to be populated
    eval_subsets = data_args.eval_subset_name if data_args.eval_subset_name else []
    
    if not eval_subsets:
        logger.warning("No eval_subset_name provided in DataArguments. Skipping MMEB evaluation.")
        return {}

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    # Make output dir
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    for idx, subset in enumerate(eval_subsets):
        if is_main_process():
            logger.info(f"--- MMEB Eval: Processing {subset} ({idx+1}/{len(eval_subsets)}) ---")

        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")

        # 1. ENCODE QUERY
        # Check if already encoded
        if not (os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path)):
            # Load Query Dataset
            eval_qry_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="qry_text",
                img_path_field="qry_img_path",
            )
            
            sampler = DistributedSampler(eval_qry_dataset, shuffle=False) if dist.is_initialized() else SequentialSampler(eval_qry_dataset)
            
            loader = DataLoader(
                eval_qry_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                sampler=sampler,
                collate_fn=eval_collator,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=True
            )
            
            if not os.path.exists(encode_qry_path):
                encoded_qry = encode_and_save(model, loader, encode_qry_path, len(eval_qry_dataset), f"Enc Qry {subset}", device)
                
                if is_main_process():
                    with open(encode_qry_path, "wb") as f:
                        pickle.dump((encoded_qry, eval_qry_dataset.paired_data), f)
            
            if dist.is_initialized(): dist.barrier()

            # 2. ENCODE TARGET
            eval_tgt_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="tgt_text",
                img_path_field="tgt_img_path",
                mod_instruction=POS_MOD_DICT.get(subset, None) if data_args.tgt_prefix_mod else None,
            )

            sampler = DistributedSampler(eval_tgt_dataset, shuffle=False) if dist.is_initialized() else SequentialSampler(eval_tgt_dataset)

            loader = DataLoader(
                eval_tgt_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                sampler=sampler,
                collate_fn=eval_collator,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=True
            )

            if not os.path.exists(encode_tgt_path):
                encoded_tgt = encode_and_save(model, loader, encode_tgt_path, len(eval_tgt_dataset), f"Enc Tgt {subset}", device)

                if is_main_process():
                    with open(encode_tgt_path, "wb") as f:
                        pickle.dump((encoded_tgt, eval_tgt_dataset.paired_data), f)
            
            if dist.is_initialized(): dist.barrier()

        # 3. CALCULATE SCORES (Rank 0 only)
        if is_main_process():
            if os.path.exists(score_path):
                 with open(score_path, "r") as f:
                    res = json.load(f)
                    results_dict[subset] = res['acc']
                    logger.info(f"Loaded cached score for {subset}: {res['acc']}")
            else:
                try:
                    score_dict = calculate_score_for_subset(subset, data_args, model_args, processor, encode_qry_path, encode_tgt_path, device)
                    results_dict[subset] = score_dict['acc']
                    
                    with open(score_path, "w") as f:
                        json.dump(score_dict, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to score {subset}: {e}")
                    import traceback
                    traceback.print_exc()
                    results_dict[subset] = 0.0

        if dist.is_initialized(): dist.barrier()

    return results_dict

def calculate_score_for_subset(subset, data_args, model_args, processor, qry_path, tgt_path, device):
    """
    Logic extracted from eval_mmeb.py for scoring
    """
    with open(qry_path, "rb") as f:
        qry_tensor, qry_index = pickle.load(f)
    with open(tgt_path, "rb") as f:
        tgt_tensor, tgt_index = pickle.load(f)

    # Load ground truth
    eval_data = load_dataset(data_args.eval_dataset_name, subset, split=data_args.dataset_split)
    
    # Specific fix for WebQA/EDIS per original script
    if (subset == "WebQA" or subset == "EDIS") and "qry_text" in eval_data.column_names and model_args.model_backbone == "llava_qwen2":
        eval_data = eval_data.map(lambda x: {"qry_text": x["qry_text"].replace("<|image_1|>", "").strip()})

    # Processing for ColPali vs Others
    if model_args.model_backbone == COLPALI:
        # ColPali specific flattening and padding logic
        if len(qry_tensor) != len(qry_index):
             qry_tensor = [t for l in qry_tensor for t in l.tolist()]
        if len(tgt_tensor) != len(tgt_index):
             tgt_tensor = [t for l in tgt_tensor for t in l.tolist()]
        
        tgtkey_to_rowid = {}
        rowid_to_tgtkey = {}
        for rowid, tt in enumerate(tgt_index):
            key = (tt["text"], tt["img_path"])
            tgtkey_to_rowid[key] = rowid
            rowid_to_tgtkey[rowid] = key

        # Convert to tensor on device for matrix ops
        tgt_tensor_gpu = [torch.from_numpy(np.array(t)) for t in tgt_tensor]
        tgt_tensor_gpu = pad_sequence(tgt_tensor_gpu, batch_first=True).to(device)
    else:
        # Standard dense retrieval
        pass

    # Build Dictionaries
    qry_key2emb, tgt_key2emb = OrderedDict(), OrderedDict()
    for qry_t, tt in zip(qry_tensor, qry_index):
        qry_key2emb[(tt["text"], tt["img_path"])] = qry_t
    for tgt_t, tt in zip(tgt_tensor, tgt_index):
        tgt_key2emb[(tt["text"], tt["img_path"])] = tgt_t

    n_correct = 0
    all_pred = []

    for row in tqdm(eval_data, desc=f"Scoring {subset}"):
        if model_args.model_backbone == COLPALI:
            qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]
            qry_t = torch.from_numpy(np.array([qry_t])).to(device)
            
            pos_tkey = (row["tgt_text"][0], row["tgt_img_path"][0])
            
            # Score
            scores = processor.score(qry_t, tgt_tensor_gpu, batch_size=1024)
            pred_id = torch.argmax(scores, dim=1).cpu().numpy().tolist()[0]
            pred_tkey = rowid_to_tgtkey[pred_id]
            
            if pred_tkey == pos_tkey:
                n_correct += 1
            all_pred.append(pred_tkey)
        else:
            # Standard Dense
            qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]
            tgt_t_list = []
            all_candidates = []
            
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t_list.append(tgt_key2emb[tt])
                all_candidates.append(tt)
            
            qry_t_arr = torch.from_numpy(np.array(qry_t))
            tgt_t_arr = np.stack(tgt_t_list, axis=0)
            
            scores, pred = get_pred(qry_t_arr, tgt_t_arr, normalization=model_args.normalize)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])

    accuracy = n_correct / len(eval_data)
    
    # Save predictions
    with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
         for item in all_pred:
             f.write(f"{item}\n")

    return {
        "acc": accuracy,
        "num_correct": n_correct,
        "num_data": len(eval_data)
    }