import os
import sys
import argparse
import logging
import math
import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
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

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(training_args: TrainingArguments) -> None:
    """Configures logging for the training process."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.local_rank not in [-1, 0]:
        logger.setLevel(logging.WARN)
    else:
        logger.setLevel(logging.INFO)
    
    # Also set transformers verbosity
    if training_args.local_rank in [-1, 0]:
        import transformers
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

def ddp_setup() -> None:
    """Initializes distributed data parallel group."""
    if not dist.is_initialized() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

def cleanup_ddp() -> None:
    """Cleans up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    """Checks if the current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def to_device(obj, device):
    """Recursively moves tensors in obj to the specified device."""
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

def save_checkpoint(
    output_dir: str,
    epoch: int,
    distiller: nn.Module,
    model_args: ModelArguments,
    step: Optional[int] = None,
    is_final: bool = False,
) -> None:
    """Saves model checkpoint."""
    if not is_main_process():
        return

    ckpt_dir = os.path.join(output_dir, "checkpoint-final" if is_final else f"checkpoint-epoch-{epoch}")
    if step is not None and not is_final:
         ckpt_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
         
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Saving checkpoint to {ckpt_dir}...")

    # Unwrap DDP model if necessary
    model_to_save = distiller.module if hasattr(distiller, "module") else distiller
    student = model_to_save.student
    
    # Save encoder/adapter
    if hasattr(student, "peft_config"):
        student.save_pretrained(ckpt_dir)
        logger.info("Saved LoRA adapter model.")
    else:
        student.encoder.save_pretrained(ckpt_dir)
        logger.info("Saved full student model.")
    
    # Save Projector
    projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
    try:
        if model_args.model_backbone in ["llava_onevision", "llava_two_vision"]:
            torch.save(student.encoder.model.multi_modal_projector.state_dict(), projector_dir)
        else:
            torch.save(student.encoder.model.model.mm_projector.state_dict(), projector_dir)
        logger.info(f"Saved projector weights to {projector_dir}")
    except AttributeError:
        logger.warning("Could not find mm_projector to save separate weights, skipping.")

    # Save tokenizer and config
    try:
        if model_args.model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
            tokenizer.save_pretrained(ckpt_dir)
            
            config = AutoConfig.from_pretrained(model_args.model_name)
            config.save_pretrained(ckpt_dir)
            
            try:
                processor = AutoProcessor.from_pretrained(model_args.model_name)
                processor.save_pretrained(ckpt_dir)
            except Exception:
                logger.warning("Could not save processor.")
    except Exception as e:
        logger.warning(f"Error saving config/tokenizer: {e}")

def main():
    # Setup DDP first, before anything else
    ddp_setup()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logging(training_args)
    
    logger.info(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
                f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Initialize WandB
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

    # Prepare Data
    logger.info("Preparing dataset...")
    train_dataset = DistillationDataset(data_args, model_args)
    
    if dist.is_initialized():
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None

    # Load Model (Distiller)
    logger.info("Loading Distiller model...")
    distiller = Distiller(model_args, training_args)

    # Prepare Collator
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        drop_last=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
    )

    # Optimizer
    logger.info("Setting up optimizer...")
    
    # Freeze/Unfreeze parameters
    trainable_params = []
    num_trainable = 0
    for n, p in distiller.student.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
             p.requires_grad = True
        
        # Example logic from previous script: lock lm_head, etc. 
        # Ideally this logic should be inside Distiller or specific model wrapper, 
        # but porting here for consistency with prev scripts.
        if "lm_head" in n:
             p.requires_grad = False
             
        if p.requires_grad:
            if training_args.bf16:
                 p.data = p.data.to(torch.bfloat16)
            trainable_params.append(p)
            num_trainable += p.numel()
            
    logger.info(f"Number of trainable parameters: {num_trainable}")

    optimizer = AdamW(
        [p for p in distiller.student.parameters() if p.requires_grad],
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    if model_args.projector_config_path is not None:
         # If special handling for projector optimizer groups is needed
         optimizer = distiller.add_optimizer_param_group(optimizer)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dist.is_initialized():
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    
    distiller = distiller.to(device)
    
    if dist.is_initialized():
        distiller = DDP(distiller, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True) # find_unused might be needed

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_ratio * max_train_steps,
        num_training_steps=max_train_steps,
    )

    # Criterion
    criterion = build_criterion(training_args)
    criterion = criterion.to(device)

    # Training Loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    start_time = time.time()

    for epoch in range(int(training_args.num_train_epochs)):
        if dist.is_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        
        distiller.train()
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
        
        accumulated_loss = 0.0
        
        for step, batch in enumerate(epoch_iterator):
            batch = to_device(batch, device)
            
            # Forward pass
            outputs = distiller(criterion, batch)
            
            loss = outputs["loss"] / training_args.gradient_accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()

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
                    # Add detailed losses if available
                    for k, v in outputs.items():
                        if k != "loss" and isinstance(v, torch.Tensor):
                            metrics[f"train/{k}"] = v.item()

                    if "wandb" in training_args.report_to:
                        wandb.log(metrics, step=global_step)
                    
                    epoch_iterator.set_postfix(**{k.replace("train/", ""): f"{v:.4f}" for k, v in metrics.items() if "loss" in k})

        # Save checkpoint at end of epoch
        save_checkpoint(training_args.output_dir, epoch + 1, distiller, model_args)
        
        if dist.is_initialized():
            dist.barrier()

    logger.info("Training completed.")
    
    # Final Save
    save_checkpoint(training_args.output_dir, training_args.num_train_epochs, distiller, model_args, is_final=True)
    
    if is_main_process() and "wandb" in training_args.report_to:
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    main()
