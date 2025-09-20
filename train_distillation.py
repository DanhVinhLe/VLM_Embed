import json
from src.distiller import Distiller, DistillationCollator, DistillationDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
import time 
import os
import sys
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.optim import AdamW

from safetensors.torch import save_file, load_file 
import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
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

def finetune(
    model_args: ModelArguments, 
    data_args: DataArguments,
    training_args: TrainingArguments,
    distiller: Distiller, 
    train_dataset: DistillationDataset,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    collator: DistillationCollator,
):
    print_rank("Start finetuning...")
    start_time = time.time()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if training_args.report_to == "wandb" else None,
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
    # distiller.teacher.to(accelerator.device)
    for param in distiller.teacher.parameters():
        param.requires_grad = False
    distiller.teacher.eval()
    distiller.student.train()
    print(f"Number of parameters in student model: {sum(p.numel() for p in distiller.student.parameters() if p.requires_grad)}")
    print(f"Number of parameters in teacher model: {sum(p.numel() for p in distiller.teacher.parameters() if p.requires_grad)}")
    
    if training_args.report_to == "wandb" and accelerator.is_main_process:
        wandb.init(
            project="vlm_distillation", 
            name=training_args.student_backbone, 
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
                student_inputs = batch['student_inputs']
                teacher_inputs = batch['teacher_inputs']
                with accelerator.accumulate(distiller.student):
                    loss_dict = distiller(student_inputs['qry'], student_inputs['pos'], teacher_inputs['qry'], teacher_inputs['pos'])
                
                    loss = loss_dict['loss'] / training_args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    contrastive_loss = loss_dict['contrastive_loss']
                    kd_loss = loss_dict['kd_loss']
                    
                    losses.append(loss_dict['loss'].item())
                    contrastive_losses.append(contrastive_loss.item())
                    kd_losses.append(kd_loss.item())
                    logging_output['micro_step_time'].append(time.time() - st_time)
                
            if accelerator.sync_gradients:
                if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(distiller.student.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
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
            
            # logging_output['loss'].append(batch_loss)
            # logging_output['contrastive_loss'].append(batch_contrastive_loss)
            # logging_output['kd_loss'].append(batch_kd_loss)
            # logging_output['step_time'].append(time.time() - global_st_time)
            
            if accelerator.is_main_process and step % training_args.logging_steps == 0:
                # avg_micro_step_time = sum(logging_output['micro_step_time'])/len(logging_output['micro_step_time'])
                # avg_step_time = sum(logging_output['step_time'])/len(logging_output['step_time'])
                
                # total_step = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
                # remaining_steps = total_step - step
                # eta_seconds = remaining_steps * avg_step_time
                # eta_hours = eta_seconds / 3600
                # print_master(
                #     f"Epoch {epoch + 1} | Step {step} | "
                #     f"Loss: {batch_loss:.4f} | Contrastive Loss: {batch_contrastive_loss:.4f} | KD Loss: {batch_kd_loss:.4f} | "
                #     f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                #     f"Micro Step Time: {avg_micro_step_time:.4f}s | "
                #     f"Step Time: {avg_step_time:.4f}s | "
                #     f"ETA: {eta_hours:.2f}h"
                # )
                progress_bar.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "contrastive_loss": f"{batch_contrastive_loss:.4f}",
                    "kd_loss": f"{batch_kd_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })
                progress_bar.update(1)
                if training_args.report_to == "wandb":
                    wandb.log({
                        "train/loss": batch_loss,
                        "train/contrastive_loss": batch_contrastive_loss,
                        "train/kd_loss": batch_kd_loss,
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
            
            if training_args.report_to == "wandb":
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
                    # save_file(
                    #     unwarpped_student.state_dict(),
                    #     os.path.join(ckpt_dir, "adapter_model.safetensors")
                    # )
                    print_rank("Saved LoRA adapter model.")
                else:
                    # save_file(
                    #     unwarpped_student.state_dict(),
                    #     os.path.join(ckpt_dir, "model.safetensors")
                    # )
                    unwarpped_student.encoder.save_pretrained(ckpt_dir)
                    print_rank("Saved full student model.")
                # training_args_dict = {
                #     "num_train_epochs": training_args.num_train_epochs,
                #     "learning_rate": training_args.learning_rate,
                #     "per_device_train_batch_size": data_args.per_device_train_batch_size,
                #     "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                #     "temperature": model_args.temperature,
                #     "pooling": model_args.pooling,
                # }
                # with open(os.path.join(ckpt_dir, "training_args.json"), "w") as f:
                #     json.dump(training_args_dict, f, indent=2)
                
                accelerator.save_state(ckpt_dir)
                print_rank(f"Checkpoint saved at {ckpt_dir}")
                
                # if training_args.push_to_hub and training_args.hub_model_id:
                #     hub_repo_name = f"{training_args.hub_model_id}-epoch{epoch + 1}"
                student_config = AutoConfig.from_pretrained(model_args.student_model_path) if model_args.student_model_path else None
                tokenizer = AutoTokenizer.from_pretrained(model_args.student_model_path) if model_args.student_model_path else None
                processor = AutoProcessor.from_pretrained(model_args.student_model_path) if model_args.student_model_path else None
                student_config.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                    # push_to_hub(
                    #     repo_name=hub_repo_name,
                    #     token=training_args.hub_token,
                    #     commit_message=f"Checkpoint at epoch {epoch + 1}",
                    #     local_dir=ckpt_dir
                    # )
        print_rank(f"Epoch {epoch + 1} finished.")
    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save final model
    if accelerator.is_main_process and training_args.save_strategy == "epoch":
        final_ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-final")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        # save_file(
        #     accelerator.unwrap_model(distiller.student).state_dict(),
        #     os.path.join(final_ckpt_dir, "student_model.safetensors")
        # )
        unwarpped_student = accelerator.unwrap_model(distiller.student)
        unwarpped_student.encoder.save_pretrained(final_ckpt_dir)
        if hasattr(unwarpped_student, 'peft_config'):
            unwarpped_student.peft_config.save_pretrained(final_ckpt_dir)
            print_rank("Saved LoRA adapter model.")
        else:
            print_rank("Saved full student model.")
        
        print_rank(f"Final model saved at {final_ckpt_dir}")
        
        # Push final model to hub
        # if training_args.push_to_hub and training_args.hub_model_id:
        final_hub_repo_name = f"{training_args.hub_model_id}-final"
        if model_args.student_model_path:
            student_config = AutoConfig.from_pretrained(model_args.student_model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_args.student_model_path)
            processor = AutoProcessor.from_pretrained(model_args.student_model_path)
            student_config.save_pretrained(final_ckpt_dir)
            tokenizer.save_pretrained(final_ckpt_dir)
            processor.save_pretrained(final_ckpt_dir)
        # hub_repo_name = f"{training_args.hub_model_id}-final"
        # push_to_hub(
        #     repo_name=final_hub_repo_name,
        #     token=training_args.hub_token,
        #     commit_message="Final model",
        #     local_dir=final_ckpt_dir
        # )

        if training_args.report_to == "wandb":
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
    
    # Initialize learning rate scheduler
    total_steps = len(train_dataset) // (
        training_args.per_device_train_batch_size * 
        training_args.gradient_accumulation_steps
    ) * training_args.num_train_epochs
    
    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        # Default constant learning rate
        from transformers import get_constant_schedule
        lr_scheduler = get_constant_schedule(optimizer)
    
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
    )
    
    print_rank("Training completed successfully!")
    return logging_output

if __name__ == "__main__":
    main()