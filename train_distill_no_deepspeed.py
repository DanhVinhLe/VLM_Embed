import json
from src.distiller import Distiller, DistillationCollator, DistillationDataset
from src.arguments import DataArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
from src.criterions import build_criterion
import time 
import os
import sys
from tqdm import tqdm 
import math

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

# Xóa 'import deepspeed'
import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser

def get_optimizer_params(model, training_args):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters

def get_optimizer(model, training_args):
    while isinstance(model, DDP):
        model = model.module
    optimizer_grouped_parameters = get_optimizer_params(model, training_args)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    return optimizer

def prepare_dataset(data_args, model_args):
    dataset = DistillationDataset(data_args, model_args)
    return dataset

def push_to_hub(repo_name=None, token=None, commit_message="Upload model", 
                local_dir="./temp_model", private=False):
    try:
        if not repo_name:
            raise ValueError("must specify a repo name to push to hub")
        
        if not os.path.exists(local_dir):
            raise ValueError(f"local_dir {local_dir} does not exist")
        
        print_rank(f"Pushing model to the hub at {repo_name}...")
        api = HfApi()
        create_repo(repo_name, token=token, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name, 
            token=token, 
            commit_message=commit_message
        )

        print_rank(f"Model has been pushed to the hub at: {repo_name}")
        return True
        
    except Exception as e:
        print_rank(f"Error pushing to hub: {str(e)}")
        return False

def batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch

# Hàm finetune được viết lại hoàn toàn
def finetune(
    model_args: ModelArguments, 
    data_args: DataArguments,
    training_args: TrainingArguments,
    distiller: Distiller, 
    train_dataset: DistillationDataset,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    collator: DistillationCollator,
    criterion: nn.Module,
    local_rank: int,
    world_size: int,
):
    print_rank("Start finetuning...")
    start_time = time.time()
    
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        sampler=sampler, 
        drop_last=True,
        pin_memory=True,
        num_workers=training_args.dataloader_num_workers,
    )
    
    # Chuyển model lên đúng device và bọc bằng DDP
    distiller.to(local_rank)
    model = DDP(distiller, device_ids=[local_rank], find_unused_parameters=False) # find_unused_parameters có thể cần tùy chỉnh
    
    # Khởi tạo GradScaler cho mixed precision (bf16/fp16)
    scaler = GradScaler(enabled=training_args.bf16 or training_args.fp16)

    # --- KẾT THÚC THIẾT LẬP ---

    print_rank(f"model device: {next(model.parameters()).device}")
    model.train()
    
    if "wandb" in training_args.report_to and local_rank == 0:
        print("Initialized wandb")
        wandb.init(
            project="vlm_distillation", 
            name=model_args.model_backbone if model_args.model_backbone else "distillation_experiment", 
            config={
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            }
        )

    step = 0
    # main epoch loop
    for epoch in range(training_args.num_train_epochs):
        print_rank("Start iteration of epoch {}".format(epoch + 1))
        model.train()
        sampler.set_epoch(epoch)
        
        grad_accum = training_args.gradient_accumulation_steps
        total_steps = math.ceil(len(train_dataloader) / grad_accum)
        
        progress_bar = tqdm(
            total=total_steps,
            desc=f"Epoch {epoch+1}",
            disable=(local_rank != 0)
        )
        print_rank(f"[INFO] Batches per epoch: {len(train_dataloader)}, GradAccum: {grad_accum}, Total steps this epoch: {total_steps}")
        epoch_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_kd_loss = 0.0
        losses, contrastive_losses, kd_losses = [], [], []
        
        for batch_idx, batch in enumerate(train_dataloader):
            device_batch = batch_to_device(batch, local_rank)
            
            # --- QUẢN LÝ MIXED PRECISION VỚI AUTOCAST ---
            with autocast(dtype=torch.bfloat16 if training_args.bf16 else torch.float16):
                loss_dict = model(criterion, device_batch)
                loss = loss_dict['loss']
                loss = loss / grad_accum
                losses.append(loss_dict['loss'].detach().item())
                contrastive_losses.append(loss_dict.get('contrastive_loss').detach().item())
                kd_losses.append(loss_dict.get('kd_loss').detach().item())
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_dataloader):
                # Gradient Clipping
                if training_args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Scheduler step
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                step += 1
                if local_rank == 0 and step % training_args.logging_steps == 0:
                    current_lr = None
                    try: 
                        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else None
                    except Exception:
                        print_rank("Cannot get learning rate from lr_scheduler")
                        current_lr = None
                    batch_loss = sum(losses) / len(losses) if len(losses) > 0 else 0
                    batch_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses) if len(contrastive_losses) > 0 else 0
                    batch_kd_loss = sum(kd_losses) / len(kd_losses) if len(kd_losses) > 0 else 0
                    epoch_loss += sum(losses)
                    epoch_contrastive_loss += sum(contrastive_losses)
                    epoch_kd_loss += sum(kd_losses)
                    progress_bar.set_postfix({
                        "loss": f"{batch_loss:.4f}",
                        "contrastive_loss": f"{batch_contrastive_loss:.4f}",
                        "kd_loss": f"{batch_kd_loss:.4f}",
                        "lr": f"{current_lr:.6f}" if current_lr is not None else "N/A",
                    })

            progress_bar.update(1)
            
        progress_bar.close()
        
        if local_rank == 0 and training_args.save_strategy == "epoch":
            ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Unwrap model từ DDP để lưu
            unwarpped_model = model.module
            unwarpped_student = getattr(unwarpped_model, "student", None)
            
            if unwarpped_student is not None:
                raise ValueError("Cannot find the student model in the model engine.")
            else: 
                if hasattr(unwarpped_student, 'peft_config'):
                    try: 
                        unwarpped_student.peft_config.save_pretrained(ckpt_dir)
                        print_rank("Saved LoRA adapter model.")
                    except Exception as e: 
                        print_rank(f"Warning: cannot save peft_config: {e}")
                    try: 
                        unwarpped_student.encoder.save_pretrained(ckpt_dir)
                    except Exception as e:
                        print_rank(f"Warning: cannot save encoder: {e}")
                    print_rank("Saved LoRA adapter model.")
                else:
                    try:
                        unwarpped_student.encoder.save_pretrained(ckpt_dir)
                        print_rank("Saved full student model")
                    except Exception as e:
                        print_rank(f"Warning: cannot save encoder: {e}")
                    print_rank("Saved full student model.")
                
                print_rank(f"Checkpoint saved at {ckpt_dir}")
            
            student_config = AutoConfig.from_pretrained(model_args.model_name) if model_args.model_name else None
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name) if model_args.model_name else None
            processor = AutoProcessor.from_pretrained(model_args.model_name) if model_args.model_name else None
            student_config.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
        print_rank(f"Checkpoint epoch {epoch + 1} saved at {ckpt_dir}")

    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save final model
    if local_rank == 0 and training_args.save_strategy == "epoch":
        final_ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-final")
        os.makedirs(final_ckpt_dir, exist_ok=True)

        unwarpped_student = getattr(getattr(model, "module", model), "student", None)
        if unwarpped_student is not None:
            try:
                unwarpped_student.encoder.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning saving final encoder: {e}")

            if hasattr(unwarpped_student, 'peft_config'):
                try:
                    unwarpped_student.peft_config.save_pretrained(final_ckpt_dir)
                    print_rank("Saved LoRA adapter model.")
                except Exception as e:
                    print_rank(f"Warning saving final peft_config: {e}")
            else:
                print_rank("Saved full student model.")
        else:
            print_rank("Warning: cannot find student model to save in final checkpoint.")

        print_rank(f"Final model saved at {final_ckpt_dir}")

def main():
    # --- BẮT BUỘC: KHỞI TẠO MÔI TRƯỜNG DISTRIBUTED ---
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    # ----------------------------------------------------

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    train_dataset = prepare_dataset(data_args, model_args)
    print_rank(f"Number of training samples: {len(train_dataset)}")
    
    # device không còn cần thiết ở đây nữa, model sẽ được chuyển device sau
    distiller = Distiller(model_args, training_args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Number of parameters in student model: {sum(p.numel() for p in distiller.student.parameters())}")
    print(f"Number of parameters in teacher model: {sum(p.numel() for p in distiller.teacher.parameters())}")
    print(f"Number of trainable parameters in student model: {sum(p.numel() for p in distiller.student.parameters() if p.requires_grad)}")
    print(f"Number of trainable parameters in teacher model: {sum(p.numel() for p in distiller.teacher.parameters() if p.requires_grad)}")
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    optimizer = get_optimizer(distiller, training_args)
    print_rank(f"Number of optimizer parameters: {sum(p.numel() for group in optimizer.param_groups for p in group['params'])}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print("World size: ", world_size)
    # Initialize learning rate scheduler
    steps_per_epoch = len(train_dataset) // (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    )
    print(f"Steps per epoch: {steps_per_epoch}")
    total_steps = steps_per_epoch * training_args.num_train_epochs
    print(f"Total training steps: {total_steps}")
    print(f"Num warmup steps: {training_args.warmup_ratio * total_steps}")
        
    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps ,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    else:
        # Default constant learning rate
        from transformers import get_constant_schedule
        lr_scheduler = get_constant_schedule(optimizer)
        
    criterion = build_criterion(training_args)
    
    # Truyền thêm local_rank và world_size vào finetune
    logging_output = finetune(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        distiller=distiller,
        train_dataset=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        collator=collator,
        criterion=criterion,
        local_rank=local_rank,
        world_size=world_size
    )
    
    print_rank("Training completed successfully!")
    dist.destroy_process_group()
    return logging_output

if __name__ == "__main__":
    main()