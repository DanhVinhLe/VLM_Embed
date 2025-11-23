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
from datetime import timedelta

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import deepspeed
# import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
from deepspeed.runtime.zero import GatheredParameters
# Todo
def get_optimizer_params(model, training_args):
    # Unwrap DDP
    original_model = model
    while isinstance(model, DDP):
        model = model.module
    
    # Lấy encoder (lora_model)
    if hasattr(model, 'encoder'):
        target_model = model.encoder
    else:
        target_model = model
    
    # Collect ALL trainable parameters (không dùng named_parameters)
    trainable_params = []
    for name, param in target_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            # print_master(f"Added to optimizer: {name} ({param.numel()} params)")
    
    print_master(f"Total trainable params: {sum(p.numel() for p in trainable_params)}")
    
    optimizer_grouped_parameters = [
        {'params': trainable_params},
    ]

    return optimizer_grouped_parameters

def get_optimizer(model, training_args):
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
    else:
        if hasattr(obj, 'to') and callable(obj.to):
            return obj.to(device)
        return obj

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
    device=None,
):
    print_rank("Start finetuning...")
    start_time = time.time()
    
    is_distributed = dist.is_initialized()
    dp_world_size = dist.get_world_size() if is_distributed else 1
    dp_rank = dist.get_rank() if is_distributed else 0
    
    sampler = DistributedSampler(train_dataset, 
                                 shuffle=True,
                                 drop_last=True,
                                 rank=dp_rank,
                                 num_replicas=dp_world_size
                                 )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        sampler=sampler, 
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
    print(sum(p[1].numel() for p in model_engine.named_parameters() if p[1].requires_grad))
    total_trainable = 0
    for n, p in model_engine.named_parameters():
        if p.requires_grad:
            total_trainable += p.numel()
            try:
                p.data = p.data.to(dtype=torch.bfloat16)
            except Exception as e:
                # If bf16 not supported, ignore and continue
                print_rank(f"Warning: cannot cast param {n} to bfloat16: {e}")
                pass
    print_rank(f"Total trainable parameters: {total_trainable}")
    print_rank(f"model device: {next(model_engine.parameters()).device}")
    model_engine.train()

    # if "wandb" in training_args.report_to and dist.get_rank() == 0:
    #     print("Initialized wandb")
    #     wandb.init(
    #         project="vlm_distillation_projector", 
    #         name=model_args.model_backbone if model_args.model_backbone else "distillation_experiment", 
    #         config={
    #             "learning_rate": training_args.learning_rate,
    #             "batch_size": training_args.per_device_train_batch_size,
    #             "epochs": training_args.num_train_epochs,
    #             "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    #         }
    #     )
    
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
        losses, contrastive_losses, kd_losses = [], [], []
        kd_rkd_losses, ot_losses, kd_dtw_losses = [], [], []
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
            for _ in range(grad_accum):
                try:
                    batch = next(train_iter)
                    batch = to_device(batch, device)
                    global_batch.append(batch)
                except StopIteration:
                    end_epoch = True
                    break
            
            if end_epoch:
                break
            for batch in global_batch:
                
            # print(f"Teacher_qry_reps dtype: {teacher_qry_reps.dtype}, device: {teacher_qry_reps.device}")
                loss_dict = model_engine(criterion, batch)

                loss = loss_dict['loss']
                model_engine.backward(loss)
                
                kd_loss = loss_dict.get('kd_loss', torch.tensor(0.0))
                contrastive_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0))
                kd_rkd_loss = loss_dict.get('kd_loss_rkd', torch.tensor(0.0))
                ot_loss = loss_dict.get('ot_loss', torch.tensor(0.0))
                kd_dtw_loss = loss_dict.get('kd_loss_dtw', torch.tensor(0.0))

                losses.append(loss.detach().item() * training_args.gradient_accumulation_steps)
                contrastive_losses.append(contrastive_loss.detach().item())
                kd_losses.append(kd_loss.detach().item())
                kd_rkd_losses.append(kd_rkd_loss.detach().item())
                ot_losses.append(ot_loss.detach().item())
                kd_dtw_losses.append(kd_dtw_loss.detach().item())
                
                model_engine.step()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step += 1
            if dist.get_rank() == 0 and step % training_args.logging_steps == 0:
                current_lr = None
                try: 
                    current_lr = optimizer.param_groups[0]['lr']
                except Exception:
                    print_rank("Cannot get learning rate from optimizer")
                    current_lr = None
                batch_loss = sum(losses) / len(losses)
                batch_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
                batch_kd_loss = sum(kd_losses) / len(kd_losses)
                batch_kd_rkd_loss = sum(kd_rkd_losses) / len(kd_rkd_losses)
                batch_ot_loss = sum(ot_losses) / len(ot_losses)
                batch_kd_dtw_loss = sum(kd_dtw_losses) / len(kd_dtw_losses)
                epoch_loss += sum(losses)
                epoch_contrastive_loss += sum(contrastive_losses)
                epoch_kd_loss += sum(kd_losses)
                progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'kd_loss': f"{batch_kd_loss:.4f}",
                    'contrastive_loss': f"{batch_contrastive_loss:.4f}",
                    'kd_rkd_loss': f"{batch_kd_rkd_loss:.4f}",
                    'ot_loss': f"{batch_ot_loss:.4f}",
                    'kd_dtw_loss': f"{batch_kd_dtw_loss:.4f}",
                    "lr": f"{current_lr:.6f}" if current_lr is not None else "N/A",
                })
                progress_bar.update(1)
                
                # if "wandb" in training_args.report_to:
                #     wandb.log({
                #         "train/loss": batch_loss,
                #         "train/contrastive_loss": batch_contrastive_loss,
                #         "train/kd_loss": batch_kd_loss,
                #         "train/kd_rkd_loss": batch_kd_rkd_loss,
                #         "train/ot_loss": batch_ot_loss,
                #         "train/kd_dtw_loss": batch_kd_dtw_loss,
                #         "train/lr": current_lr,
                #         "train/epoch": epoch + 1,
                #         "train/global_step": step,
                #     })
                    
                #     logging_output['micro_step_time'] = []
                #     logging_output['step_time'] = []
                
                epoch_step += 1

            
        # End of epoch
        if dist.get_rank() == 0:
            avg_epoch_loss = epoch_loss / max(1, epoch_step)
            avg_contrastive_loss = epoch_contrastive_loss / max(1, epoch_step)
            avg_kd_loss = epoch_kd_loss / max(1, epoch_step)
            
            print_rank(
                f"Epoch {epoch + 1} completed. Avg Loss: {avg_epoch_loss:.4f} | "
                f"Avg Contrastive Loss: {avg_contrastive_loss:.4f} | Avg KD Loss: {avg_kd_loss:.4f} | "
            )
            
            # if "wandb" in training_args.report_to:
            #     wandb.log({
            #         "epoch/avg_loss": avg_epoch_loss,
            #         "epoch/avg_contrastive_loss": avg_contrastive_loss,
            #         "epoch/avg_kd_loss": avg_kd_loss,
            #         "epoch/epoch": epoch + 1,
            #     })
            # Save checkpoint
            if training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                with GatheredParameters(model_engine.module.student.parameters(), modifier_rank=0):
                    model_engine.module.student.encoder.save_pretrained(ckpt_dir)
                    
                try:
                    student_config = AutoConfig.from_pretrained(model_args.model_name) if model_args.model_name else None
                    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name) if model_args.model_name else None
                    if student_config is not None:
                        student_config.save_pretrained(ckpt_dir)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning saving tokenizer/config: {e}")
                try: 
                    processor = AutoProcessor.from_pretrained(model_args.model_name) if model_args.model_name else None
                    if processor is not None:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning saving processor: {e}")
        dist.barrier()

        print_rank(f"Epoch {epoch + 1} finished.")
    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save final model
    if dist.get_rank() == 0 and training_args.save_strategy == "epoch":
        final_ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-final")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        with GatheredParameters(model_engine.module.student.parameters(), modifier_rank=0):
            model_engine.module.student.encoder.save_pretrained(final_ckpt_dir)

        print_rank(f"Final model saved at {final_ckpt_dir}")
        
        if model_args.model_name:
            try:
                student_config = AutoConfig.from_pretrained(model_args.model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
                if student_config is not None:
                    student_config.save_pretrained(final_ckpt_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning saving final tokenizer/config: {e}")
            try: 
                processor = AutoProcessor.from_pretrained(model_args.model_name)
                if processor is not None:
                    processor.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning saving final processor: {e}")
        # if "wandb" in training_args.report_to:
        #     try:
        #         wandb.finish()
        #     except Exception as e:
        #         print_rank(f"Warning: cannot finalize wandb run: {e}")
        
    dist.barrier()

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
    
    torch.backends.cudnn.enabled = False
    device =  torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    deepspeed.init_distributed(timeout=timedelta(minutes=1))
    
    train_dataset = prepare_dataset(data_args, model_args)
    print_rank(f"Number of training samples: {len(train_dataset)}")
    distiller = Distiller(model_args, training_args)
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
    optimizer = get_optimizer(distiller.student, training_args)
    # optimizer = AdamW(
    #     distiller.student.parameters(),
    #     lr=training_args.learning_rate,
    #     weight_decay=training_args.weight_decay,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    # )
    if model_args.projector_config_path is not None:
        optimizer = distiller.add_optimizer_param_group(optimizer)
    
    print_rank(f"Number of optimizer parameters: {sum(p.numel() for group in optimizer.param_groups for p in group['params'])}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print("World size: ", world_size)
    # Initialize learning rate scheduler
    steps_per_epoch = len(train_dataset) // (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    )
    # print(f"Steps per epoch: {steps_per_epoch}")
    total_steps = steps_per_epoch * training_args.num_train_epochs
    # print(f"Total training steps: {total_steps}")
    # print(f"Num warmup steps: {training_args.warmup_ratio * total_steps}")
        
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
        device=device,
    )
    
    print_rank("Training completed successfully!")
    return logging_output

if __name__ == "__main__":
    main()