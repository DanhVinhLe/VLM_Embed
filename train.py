import logging 
import os.path 
import sys 

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import torch 
import wandb 
import yaml 
from tqdm import tqdm

from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import AutoConfig
from src.data.dataset.mmeb_dataset import TrainTextImageDataset
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.data.loader.concat_dataset import init_concat_dataset
from src.model.model import MMEBModel
from src.trainer import MMEBTrainer
from src.utils import print_master, print_rank, find_latest_checkpoint
from src.model.processor import load_processor, get_backbone_name, process_vlm_inputs_fns


def main(): 
    for arg in sys.argv: 
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[-1]
            sys.argv.remove(arg)
            sys.argv.append(f"--local_rank")
            sys.argv.append(f"{rank}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    
    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")
            
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("No checkpoint found. Starting fresh training.")

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)
            
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model = MMEBModel.build(model_args)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, "model_backbone", model_backbone)
        setattr(training_args, "model_backbone", model_backbone)
    print_rank(f"Model backbone: {model_args.model_backbone}")
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)
    
    with open(data_args.data_config, 'r') as f: 
        data_config = yaml.safe_load(f)
        train_dataset = init_mixed_dataset(data_config, model_args, data_args, training_args)
    
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)
    
    trainer = MMEBTrainer(
        model=model, 
        processing_class=processor,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=data_args.max_len,
    )
    
    train_dataset.trainer = trainer 
    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload
    
    if trainer.is_world_process_zero(): 
        processor.save_pretrained(training_args.output_dir)
        
if __name__ == "__main__":
    main()