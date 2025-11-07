import json
from src.distiller import Distiller, DistillationCollator, DistillationDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
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
from torch.optim import AdamW

import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
# Todo
wandb.login(key="f5a118efa8813fb4edc7f6b8a7ab5c9c5f9e1ece")

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

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

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
):
    print_rank("Start finetuning...")
    start_time = time.time()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if "wandb" in training_args.report_to else None,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        shuffle=True, 
        drop_last=True,
    )
    distiller, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        distiller, optimizer, train_dataloader, lr_scheduler
    )
    for n, p in distiller.student.named_parameters():
        if p.requires_grad:  # thường chỉ là LoRA
            p.data = p.data.to(torch.bfloat16)
            # print(f"Cast {n} to bf16")

    # cast_lora_to_bf16(distiller.student)
    num_trainable_params = sum(p.numel() for p in distiller.student.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in distiller.student.parameters())
    num_trainable_params_teacher = sum(p.numel() for p in distiller.teacher.parameters() if p.requires_grad)
    num_total_params_teacher = sum(p.numel() for p in distiller.teacher.parameters())
    print_rank(f"Number of trainable parameters: {num_trainable_params} / {num_total_params}")
    print_rank(f"Number of trainable parameters (teacher): {num_trainable_params_teacher} / {num_total_params_teacher}")
    distiller.student.train()
    
    if "wandb" in training_args.report_to and accelerator.is_main_process:
        print("Initialized wandb")
        wandb.init(
            project=training_args.output_dir.split("/")[-1], 
            name=model_args.model_backbone, 
            config={
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            }
        )
    
    step = 0
    logging_output = {
        'epoch': 0, 
        'global_step': 0, 
        'loss': [],
        'contrastive_loss': [],
        'kd_loss': [],
        'micro_step_time': [],
        'step_time': []
    }
    
    for epoch in range(training_args.num_train_epochs):
        logging_output['epoch'] = epoch + 1
        print_rank("Start iteration of epoch {}".format(epoch + 1))
        end_epoch = False
        epoch_step = 0
        epoch_loss, epoch_contrastive_loss, epoch_kd_loss = 0, 0, 0
        print(f"Device: {training_args.device}")
        
        progress_bar = tqdm(
            total=len(train_dataloader)//training_args.gradient_accumulation_steps,
            desc=f"Epoch {epoch+1}",
            disable=not accelerator.is_main_process
        )
        
        train_iter = iter(train_dataloader)
        
        while True:
            global_batch = []
            global_st_time = time.time() 
            losses, contrastive_losses, kd_losses = [], [], []
            kd_rkd_losses, ot_losses, kd_dtw_losses = [], [], []
            for i in range(training_args.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                    global_batch.append(batch)
                except StopIteration:
                    end_epoch = True
                    break
            
            if end_epoch:
                break
            
            for batch in global_batch:
                st_time = time.time()
                # print(f"Teacher_qry_reps dtype: {teacher_qry_reps.dtype}, device: {teacher_qry_reps.device}")
                with accelerator.accumulate(distiller):
                    loss_dict = distiller(criterion, batch)
                
                    loss = loss_dict['loss'] / training_args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    contrastive_loss = loss_dict['contrastive_loss']
                    kd_loss = loss_dict['kd_loss']
                    kd_loss_rkd = loss_dict.get('kd_loss_rkd', torch.tensor(0.0))
                    kd_dtw_loss = loss_dict.get('kd_loss_dtw', torch.tensor(0.0))
                    ot_loss = loss_dict.get('ot_loss', torch.tensor(0.0))

                    losses.append(loss_dict['loss'].detach().item())
                    contrastive_losses.append(contrastive_loss.detach().item())
                    kd_losses.append(kd_loss.detach().item())
                    kd_rkd_losses.append(kd_loss_rkd.detach().item())
                    kd_dtw_losses.append(kd_dtw_loss.detach().item())
                    ot_losses.append(ot_loss.detach().item())
                    logging_output['micro_step_time'].append(time.time() - st_time)
                
            if accelerator.sync_gradients:
                if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(distiller.student.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            step += 1
            epoch_step += 1
            logging_output['global_step'] = step
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            batch_loss = sum(losses)/len(losses)
            batch_contrastive_loss = sum(contrastive_losses)/len(contrastive_losses)
            batch_kd_loss = sum(kd_losses)/len(kd_losses)
            batch_kd_rkd_loss = sum(kd_rkd_losses)/len(kd_rkd_losses)
            batch_kd_dtw_loss = sum(kd_dtw_losses)/len(kd_dtw_losses)
            batch_ot_loss = sum(ot_losses)/len(ot_losses)

            epoch_loss += sum(losses)
            epoch_contrastive_loss += sum(contrastive_losses)
            epoch_kd_loss += sum(kd_losses)
            
            if accelerator.is_main_process and step % training_args.logging_steps == 0:
                progress_bar.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "contrastive_loss": f"{batch_contrastive_loss:.4f}",
                    "kd_loss": f"{batch_kd_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                    "kd_loss_rkd": f"{batch_kd_rkd_loss:.4f}",
                    "kd_loss_dtw": f"{batch_kd_dtw_loss:.4f}",
                    "ot_loss": f"{batch_ot_loss:.4f}",
                })
                progress_bar.update(1)
                if "wandb" in training_args.report_to:
                    wandb.log({
                        "train/loss": batch_loss,
                        "train/contrastive_loss": batch_contrastive_loss,
                        "train/kd_loss": batch_kd_loss,
                        "train/kd_loss_rkd": batch_kd_rkd_loss,
                        "train/kd_loss_dtw": batch_kd_dtw_loss,
                        "train/ot_loss": batch_ot_loss,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/epoch": epoch + 1,
                        "train/global_step": step,
                    })
                    
                    logging_output['micro_step_time'] = []
                    logging_output['step_time'] = []
        # End of epoch
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / max(1, epoch_step)
            avg_contrastive_loss = epoch_contrastive_loss / max(1, epoch_step)
            avg_kd_loss = epoch_kd_loss / max(1, epoch_step)
            
            print_rank(
                f"Epoch {epoch + 1} completed. Avg Loss: {avg_epoch_loss:.4f} | "
                f"Avg Contrastive Loss: {avg_contrastive_loss:.4f} | Avg KD Loss: {avg_kd_loss:.4f} | "
            )
            
            if "wandb" in training_args.report_to:
                wandb.log({
                    "epoch/avg_loss": avg_epoch_loss,
                    "epoch/avg_contrastive_loss": avg_contrastive_loss,
                    "epoch/avg_kd_loss": avg_kd_loss,
                    "epoch/epoch": epoch + 1,
                })
            # Save checkpoint
            if training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unwarpped_student = accelerator.unwrap_model(distiller.student)
                
                if hasattr(unwarpped_student, 'peft_config'):
                    unwarpped_student.peft_config.save_pretrained(ckpt_dir)
                    unwarpped_student.save_pretrained(ckpt_dir)
                    print_rank("Saved LoRA adapter model.")
                else:
                    unwarpped_student.encoder.save_pretrained(ckpt_dir)
                    print_rank("Saved full student model.")
                
                accelerator.save_state(ckpt_dir)
                print_rank(f"Checkpoint saved at {ckpt_dir}")
                
                student_config = AutoConfig.from_pretrained(model_args.model_name) if model_args.model_name else None
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_name) if model_args.model_name else None

                student_config.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                try:   
                    processor = AutoProcessor.from_pretrained(model_args.model_name) if model_args.model_name else None
                    processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Error saving processor: {e}. No processor saved.")
        print_rank(f"Epoch {epoch + 1} finished.")
    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save final model
    if accelerator.is_main_process and training_args.save_strategy == "epoch":
        final_ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-final")
        os.makedirs(final_ckpt_dir, exist_ok=True)

        unwarpped_student = accelerator.unwrap_model(distiller.student)
        unwarpped_student.encoder.save_pretrained(final_ckpt_dir)
        if hasattr(unwarpped_student, 'peft_config'):
            unwarpped_student.peft_config.save_pretrained(final_ckpt_dir)
            print_rank("Saved LoRA adapter model.")
        else:
            print_rank("Saved full student model.")
        
        print_rank(f"Final model saved at {final_ckpt_dir}")
        
        # Push final model to hub
        if model_args.model_name:
            student_config = AutoConfig.from_pretrained(model_args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
            try:
                processor = AutoProcessor.from_pretrained(model_args.model_name)
                processor.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Error saving processor: {e}. No processor saved.")
            student_config.save_pretrained(final_ckpt_dir)
            tokenizer.save_pretrained(final_ckpt_dir)

        if "wandb" in training_args.report_to:
            wandb.finish()

    return logging_output

def main():
    for arg in sys.argv:
        if arg.startswith("--local_rank"):
            local_rank = int(arg.split("=")[-1])
            sys.argv.remove(arg)
            sys.argv.append(f"--local_rank")
            sys.argv.append(f"{local_rank}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    train_dataset = prepare_dataset(data_args, model_args)
    distiller = Distiller(model_args, training_args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    optimizer = AdamW(
        distiller.student.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    
    # Initialize learning rate scheduler
    total_steps = len(train_dataset) // (
        training_args.per_device_train_batch_size * 
        training_args.gradient_accumulation_steps
    ) * training_args.num_train_epochs
    
    if model_args.projector_config_path is not None:
        optimizer = distiller.add_optimizer_param_group(optimizer)
    
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
    
    # Start finetuning
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
    )
    
    print_rank("Training completed successfully!")
    return logging_output

if __name__ == "__main__":
    main()