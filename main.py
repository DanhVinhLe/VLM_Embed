import os
import sys
import argparse
import logging
import math
import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

# --- FIX 1: Disable Tokenizer Parallelism/OMP to prevent Deadlock ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, random_split, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import wandb
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
)

from src.distiller import Distiller, DistillationCollator, DistillationDataset
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.utils import print_rank, print_master
from src.criterions import build_criterion

logger = logging.getLogger(__name__)

# ... [Keep setup_logging, ddp_setup, cleanup_ddp, is_main_process, to_device, download_artifacts unchanged] ...
def setup_logging(training_args: TrainingArguments) -> None:
    """Configures logging for the training process."""
    log_level = logging.INFO
    logging.basicConfig(
        format=f"[Rank {training_args.local_rank}] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(log_level)
    import transformers
    if training_args.local_rank in [-1, 0]:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def ddp_setup() -> None:
    if not dist.is_initialized() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def to_device(obj, device):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    elif hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj

def download_artifacts(model_args: ModelArguments):
    logger.info("  [Pre-load] Starting artifact download...")
    if model_args.model_name:
        try:
            AutoConfig.from_pretrained(model_args.model_name)
            AutoTokenizer.from_pretrained(model_args.model_name)
            AutoProcessor.from_pretrained(model_args.model_name)
        except Exception as e:
            logger.warning(f"  [Pre-load] Warning: Some artifacts failed to pre-download: {e}")
    logger.info("  [Pre-load] Finished.")

# --- CHANGED: Updated save_checkpoint to handle specific folder names (for best model) ---
def save_checkpoint(
    output_dir: str,
    epoch: int,
    distiller: nn.Module,
    model_args: ModelArguments,
    step: Optional[int] = None,
    folder_name: Optional[str] = None,
) -> None:
    """Saves model checkpoint. Supports custom folder name for best model."""
    if not is_main_process():
        return

    if folder_name:
        ckpt_dir = os.path.join(output_dir, folder_name)
    elif step is not None:
         ckpt_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
    else:
        ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
         
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Saving checkpoint to {ckpt_dir}...")

    # Unwrap DDP model
    model_to_save = distiller.module if hasattr(distiller, "module") else distiller
    student = model_to_save.student
    
    # Save encoder/adapter
    if hasattr(student, "peft_config"):
        student.save_pretrained(ckpt_dir)
        logger.info("Saved LoRA adapter model.")
    else:
        if hasattr(student, "encoder"):
            student.encoder.save_pretrained(ckpt_dir)
        else:
            try:
                student.save_pretrained(ckpt_dir)
            except:
                torch.save(student.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        logger.info("Saved student model.")
    
    # Save Projector
    projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
    try:
        projector_weights = None
        if hasattr(student, "encoder") and hasattr(student.encoder, "model"):
             if hasattr(student.encoder.model, "multi_modal_projector"):
                 projector_weights = student.encoder.model.multi_modal_projector.state_dict()
             elif hasattr(student.encoder.model, "model") and hasattr(student.encoder.model.model, "mm_projector"):
                 projector_weights = student.encoder.model.model.mm_projector.state_dict()
        
        if projector_weights is not None:
            torch.save(projector_weights, projector_dir)
    except AttributeError:
        pass

    # Save tokenizer and config
    try:
        if model_args.model_name:
            AutoTokenizer.from_pretrained(model_args.model_name).save_pretrained(ckpt_dir)
            AutoConfig.from_pretrained(model_args.model_name).save_pretrained(ckpt_dir)
            try:
                AutoProcessor.from_pretrained(model_args.model_name).save_pretrained(ckpt_dir)
            except Exception:
                pass
    except Exception:
        pass

# --- NEW FUNCTION: Evaluate Loss ---
def evaluate_loss(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """Computes average loss on the validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process()):
            batch = to_device(batch, device)
            outputs = model(criterion, batch)
            
            # Extract loss
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            total_loss += loss.item()
            num_batches += 1
            
    # Calculate average on this device
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # If distributed, average across all GPUs
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / dist.get_world_size()
        
    model.train() # Switch back to train mode
    return avg_loss

def main():
    ddp_setup()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args)
    
    if is_main_process() and "wandb" in training_args.report_to:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "vlm_distillation"),
            name=training_args.run_name or f"run-{int(time.time())}",
            config={
                "model_args": vars(model_args),
                "data_args": vars(data_args),
                "training_args": vars(training_args),
            }
        )

    # Artifact Sync
    logger.info("Handling artifact downloading...")
    if dist.is_initialized():
        if is_main_process():
            download_artifacts(model_args)
        dist.barrier()
    else:
        download_artifacts(model_args)

    # --- CHANGED: Dataset Splitting Logic ---
    logger.info("Preparing dataset...")
    full_dataset = DistillationDataset(data_args, model_args)
    
    train_dataset = full_dataset
    eval_dataset = None
    
    if data_args.val_split_ratio > 0:
        val_size = int(len(full_dataset) * data_args.val_split_ratio)
        train_size = len(full_dataset) - val_size
        logger.info(f"Splitting dataset: {train_size} training, {val_size} validation.")
        
        # Use a fixed generator for reproducibility across ranks
        generator = torch.Generator().manual_seed(42)
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    else:
        logger.info("No validation split ratio provided. Using full dataset for training.")

    # Samplers
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        # For eval, we don't necessarily need to shuffle, but we need DistributedSampler 
        # to split data across GPUs to speed up eval
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if eval_dataset else None
    else:
        train_sampler = None
        eval_sampler = SequentialSampler(eval_dataset) if eval_dataset else None

    # Load Model
    logger.info("Loading Distiller model...")
    distiller = Distiller(model_args, training_args)

    # Collator
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collator,
        drop_last=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size or training_args.per_device_train_batch_size,
            sampler=eval_sampler,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True
        )

    # Optimizer setup
    logger.info("Setting up optimizer...")
    trainable_params = []
    
    model_to_optimize = distiller.module if hasattr(distiller, "module") else distiller
    for n, p in model_to_optimize.student.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
             p.requires_grad = True
        if "lm_head" in n:
             p.requires_grad = False
        if p.requires_grad:
            if training_args.bf16:
                 p.data = p.data.to(torch.bfloat16)
            trainable_params.append(p)

    optimizer = AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    if model_args.projector_config_path is not None:
         if hasattr(distiller, "add_optimizer_param_group"):
            optimizer = distiller.add_optimizer_param_group(optimizer)

    # Move to device and wrap DDP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dist.is_initialized():
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    
    distiller = distiller.to(device)
    if dist.is_initialized():
        distiller = DDP(distiller, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_ratio * max_train_steps,
        num_training_steps=max_train_steps,
    )

    criterion = build_criterion(training_args).to(device)

    # Training Stats
    if is_main_process():
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Eval examples = {len(eval_dataset) if eval_dataset else 0}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        logger.info(f"  Eval step = {training_args.eval_steps}")

    global_step = 0
    best_val_loss = float('inf') # Track best loss

    for epoch in range(int(training_args.num_train_epochs)):
        if dist.is_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        
        distiller.train()
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
        
        for step, batch in enumerate(epoch_iterator):
            batch = to_device(batch, device)
            
            outputs = distiller(criterion, batch)
            loss = outputs["loss"] / training_args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                     torch.nn.utils.clip_grad_norm_(distiller.parameters(), training_args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if is_main_process() and global_step % training_args.logging_steps == 0:
                    metrics = {
                        "train/loss": outputs["loss"].item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(train_dataloader),
                    }
                    for k, v in outputs.items():
                        if k != "loss" and isinstance(v, torch.Tensor):
                            metrics[f"train/{k}"] = v.item()

                    if "wandb" in training_args.report_to:
                        wandb.log(metrics, step=global_step)
                    epoch_iterator.set_postfix(**{k.replace("train/", ""): f"{v:.4f}" for k, v in metrics.items() if "loss" in k})

                # --- CHANGED: Periodic Evaluation & Best Model Saving ---
                if eval_dataloader is not None and training_args.eval_steps > 0 and global_step % training_args.eval_steps == 0:
                    logger.info(f"Step {global_step}: Running evaluation...")
                    val_loss = evaluate_loss(distiller, eval_dataloader, criterion, device)
                    
                    if is_main_process():
                        logger.info(f"Step {global_step} | Validation Loss: {val_loss:.4f}")
                        if "wandb" in training_args.report_to:
                            wandb.log({"eval/loss": val_loss}, step=global_step)
                        
                        # Save Best Model
                        if val_loss < best_val_loss:
                            logger.info(f"New best model found! (Loss: {val_loss:.4f} < {best_val_loss:.4f})")
                            best_val_loss = val_loss
                            save_checkpoint(
                                training_args.output_dir, 
                                epoch=epoch+1, 
                                distiller=distiller, 
                                model_args=model_args, 
                                step=global_step, 
                                folder_name="checkpoint-best"
                            )

        # End of epoch Saving
        save_checkpoint(training_args.output_dir, epoch + 1, distiller, model_args)
        
        if dist.is_initialized():
            dist.barrier()

    logger.info("Training completed.")
    if is_main_process() and eval_dataloader:
        logger.info(f"Best Validation Loss: {best_val_loss:.4f}")

    # Final Save
    save_checkpoint(training_args.output_dir, int(training_args.num_train_epochs), distiller, model_args, folder_name="checkpoint-final")
    
    if is_main_process() and "wandb" in training_args.report_to:
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    main()