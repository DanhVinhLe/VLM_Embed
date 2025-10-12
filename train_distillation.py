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
from torch.optim import AdamW

import deepspeed
import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
# Todo

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
        # Nếu là dictionary, gọi đệ quy cho từng value
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        # Nếu là list, gọi đệ quy cho từng phần tử
        return [batch_to_device(v, device) for v in batch]
    elif isinstance(batch, torch.Tensor):
        # Nếu là tensor, chuyển nó lên device
        return batch.to(device)
    else:
        # Giữ nguyên các kiểu dữ liệu khác (string, int, ...)
        return batch

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
    
    is_distributed = dist.is_initialized()
    if is_distributed:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=collator,
            sampler=sampler, 
            drop_last=True,
            pin_memory=True,
        )   
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=collator,
            shuffle=True, 
            drop_last=True,
            pin_memory=True,
        )
    
    ds_config = {}
    ds_config_path = getattr(training_args, "deepspeed_config", None)
    if ds_config_path:
        # If it's a dict already, use it; if it's a file path, load it.
        if isinstance(ds_config_path, dict):
            ds_config = ds_config_path
        elif isinstance(ds_config_path, str) and os.path.exists(ds_config_path):
            with open(ds_config_path, "r") as f:
                ds_config = json.load(f)
        else:
            # path provided but not found: fall back to default params and warn
            print_rank(f"Warning: deepspeed config path {ds_config_path} not found. Using default config_params.")
            ds_config = {}
    
    ds_config["gradient_accumulation_steps"] = training_args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = training_args.per_device_train_batch_size
    ds_config["gradient_clipping"] = training_args.max_grad_norm
    ds_config["train_batch_size"] = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * (dist.get_world_size() if is_distributed else 1)
        
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config,
    )
    print(model_engine.module is distiller)  # phải là True
    print(sum(p.numel() for p in model_engine.module.parameters()))
    
    for n, p in model_engine.named_parameters():
        if p.requires_grad:
            try:
                p.data = p.data.to(dtype=torch.bfloat16)
            except Exception as e:
                # If bf16 not supported, ignore and continue
                print_rank(f"Warning: cannot cast param {n} to bfloat16: {e}")
                pass

    print_rank(f"model device: {next(model_engine.parameters()).device}")
    model_engine.train()
    
    if "wandb" in training_args.report_to and getattr(model_engine, "global_rank", 0) == 0:
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
    logging_output = {
        'epoch': 0, 
        'global_step': 0, 
        'loss': [],
        'contrastive_loss': [],
        'kd_loss': [],
        'micro_step_time': [],
        'step_time': []
    }

    # main epoch loop
    for epoch in range(training_args.num_train_epochs):
        logging_output['epoch'] = epoch + 1
        print_rank("Start iteration of epoch {}".format(epoch + 1))
        end_epoch = False
        epoch_step = 0
        epoch_loss, epoch_contrastive_loss, epoch_kd_loss = 0, 0, 0
        model_engine.train()
        
        if is_distributed and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        train_iter = iter(train_dataloader)
        grad_accum = int(ds_config.get("gradient_accumulation_steps", 1))
        total_steps = math.ceil(len(train_dataloader) / grad_accum)
        print_rank(f"[INFO] Batches per epoch: {len(train_dataloader)}, GradAccum: {grad_accum}, Total steps this epoch: {total_steps}")
        
        progress_bar = tqdm(
            total=total_steps,
            desc=f"Epoch {epoch+1}",
            disable=(getattr(model_engine, "global_rank", 0) != 0)
        )
        
        while True:
            global_batch = []
            losses, contrastive_losses, kd_losses = [], [], []
            for i in range(grad_accum):
                try:
                    batch = next(train_iter)
                    global_batch.append(batch)
                except StopIteration:
                    end_epoch = True
                    break
            
            if end_epoch and len(global_batch) == 0:
                break
            
            for batch in global_batch:
                # print(f"Batch device before to device: {batch['student_inputs']['qry']['input_ids'].device}")
                device_batch = batch_to_device(batch, model_engine.device)
                # print(f"Model device: {next(model_engine.parameters()).device}")
                # print(f"Batch device after to device: {device_batch['student_inputs']['qry']['input_ids'].device}")
                st_time = time.time()
                
                # print(f"Teacher_qry_reps dtype: {teacher_qry_reps.dtype}, device: {teacher_qry_reps.device}")
                loss_dict = model_engine(criterion, device_batch)

                loss = loss_dict['loss']
                model_engine.backward(loss)
                
                contrastive_loss = loss_dict['contrastive_loss']
                kd_loss = loss_dict['kd_loss']
                
                losses.append(loss_dict['loss'].detach().item())
                contrastive_losses.append(contrastive_loss.detach().item())
                kd_losses.append(kd_loss.detach().item())
                logging_output['micro_step_time'].append(time.time() - st_time)
            model_engine.step()
            
            try: 
                if lr_scheduler is not None:
                    lr_scheduler.step()
            except Exception: 
                pass 
            step += 1
            epoch_step += 1
            logging_output['global_step'] = step
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            batch_loss = sum(losses)/len(losses)
            batch_contrastive_loss = sum(contrastive_losses)/len(contrastive_losses)
            batch_kd_loss = sum(kd_losses)/len(kd_losses)
            
            epoch_loss += sum(losses)
            epoch_contrastive_loss += sum(contrastive_losses)
            epoch_kd_loss += sum(kd_losses)
            
            if getattr(model_engine, "global_rank", 0) == 0 and step % training_args.logging_steps == 0:
                current_lr = None
                try: 
                    current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else None
                except Exception:
                    print_rank("Cannot get learning rate from lr_scheduler")
                    current_lr = None
                progress_bar.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "contrastive_loss": f"{batch_contrastive_loss:.4f}",
                    "kd_loss": f"{batch_kd_loss:.4f}",
                    "lr": f"{current_lr:.6f}" if current_lr is not None else "N/A",
                })
                progress_bar.update(1)
                
                if "wandb" in training_args.report_to:
                    wandb.log({
                        "train/loss": batch_loss,
                        "train/contrastive_loss": batch_contrastive_loss,
                        "train/kd_loss": batch_kd_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch + 1,
                        "train/global_step": step,
                    })
                    
                    logging_output['micro_step_time'] = []
                    logging_output['step_time'] = []
                    
            if end_epoch:
                continue
            
        # End of epoch
        if getattr(model_engine, "global_rank", 0) == 0:
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
                unwarpped_student = getattr(getattr(model_engine, "module", model_engine), "student", None)
                
                if unwarpped_student is None:
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

        print_rank(f"Epoch {epoch + 1} finished.")
    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save final model
    if getattr(model_engine, "global_rank", 0) == 0 and training_args.save_strategy == "epoch":
        final_ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-final")
        os.makedirs(final_ckpt_dir, exist_ok=True)

        unwarpped_student = getattr(getattr(model_engine, "module", model_engine), "student", None)
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
        
        if model_args.model_name:
            student_config = AutoConfig.from_pretrained(model_args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
            processor = AutoProcessor.from_pretrained(model_args.model_name)
            student_config.save_pretrained(final_ckpt_dir)
            tokenizer.save_pretrained(final_ckpt_dir)
            processor.save_pretrained(final_ckpt_dir)

        if "wandb" in training_args.report_to:
            try:
                wandb.finish()
            except Exception as e:
                print_rank(f"Warning: cannot finalize wandb run: {e}")

    return logging_output

def main():
    # for arg in sys.argv:
    #     if arg.startswith("--local_rank"):
    #         local_rank = int(arg.split("=")[-1])
    #         sys.argv.remove(arg)
    #         sys.argv.append(f"--local_rank")
    #         sys.argv.append(f"{local_rank}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    train_dataset = prepare_dataset(data_args, model_args)
    print_rank(f"Number of training samples: {len(train_dataset)}")
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
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # Initialize learning rate scheduler
    steps_per_epoch = len(train_dataset) // (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    )
    total_steps = steps_per_epoch * training_args.num_train_epochs
        
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