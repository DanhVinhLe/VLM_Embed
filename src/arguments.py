"""
Arguments configuration for the project.

This module defines the dataclasses for Model, Data, and Training arguments,
inheriting from and extending HuggingFace's configuration classes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name: str = field(
        metadata={"help": "HuggingFace model name or path"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type, typically included in config file, but sometimes needs manually add"}
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "Processor name, HuggingFace model name or path"}
    )
    model_backbone: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace model type backbone"}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "A local model path, could be a LoRA version"}
    )
    pooling: str = field(
        default='last',
        metadata={"help": "Pooling method for encoder (e.g., 'last', 'mean')"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "Normalize query and passage representations"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "Temperature for softmax"}
    )
    lora: bool = field(
        default=False,
        metadata={"help": "Do parameter-efficient fine-tuning with LoRA"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA Alpha value"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA Dropout rate"}
    )
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "Comma-separated list of LoRA target modules"}
    )
    num_crops: int = field(
        default=16,
        metadata={"help": "Number of crops used in image encoder"}
    )
    uigraph_use: bool = field(
        default=False,
        metadata={"help": "Enable UI graph for token selection"}
    )
    uigraph_diff: int = field(
        default=1,
        metadata={"help": "Pixel difference used for constructing UI graph for token selection"}
    )
    uigraph_rand: bool = field(
        default=False,
        metadata={"help": "Enable random graph construction for token selection"}
    )
    uimask_ratio: float = field(
        default=0.5,
        metadata={"help": "Percentage of patch tokens to skip per component for token selection"}
    )
    uimask_rand: bool = field(
        default=False,
        metadata={"help": "Enable random token selection instead of uniform selection"}
    )
    lm_skip_layer: str = field(
        default='[1,28,0]',
        metadata={"help": "Specify the layers of the language model to skip for token selection"}
    )
    vis_skip_layer: str = field(
        default='[1,32,0]',
        metadata={"help": "Specify the layers of the vision model to skip for token selection"}
    )

    # New arguments
    init_lora_model: bool = field(
        default=False,
        metadata={"help": "Initializing with LoRA model"}
    )

    # Distiller arguments
    teacher_backbone: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model backbone"}
    )
    teacher_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model name or path"}
    )
    teacher_lora: bool = field(
        default=False,
        metadata={"help": "Whether teacher uses LoRA"}
    )
    teacher_lora_r: int = field(
        default=16,
        metadata={"help": "Teacher LoRA R value"}
    )
    teacher_lora_alpha: int = field(
        default=64,
        metadata={"help": "Teacher LoRA Alpha value"}
    )
    teacher_lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Teacher LoRA Dropout rate"}
    )
    teacher_lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "Teacher LoRA target modules"}
    )
    teacher_pooling: str = field(
        default='last',
        metadata={"help": "Pooling method for teacher encoder"}
    )
    teacher_normalize: bool = field(
        default=False,
        metadata={"help": "Normalize query and passage representations for teacher"}
    )
    projector_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Projector config path (JSON). If None, no projector will be used"}
    )
    projector_path: Optional[str] = field(
        default=None,
        metadata={"help": "Projector model path. If None, no projector will be used"}
    )
    projector_lr: float = field(
        default=1e-4,
        metadata={"help": "Projector learning rate"}
    )
    student_hidden_dim: int = field(
        default=896,
        metadata={"help": "Student hidden dimension"}
    )
    teacher_hidden_dim: int = field(
        default=1536,
        metadata={"help": "Teacher hidden dimension"}
    )
    load_pretrained_lora: bool = field(
        default=False,
        metadata={"help": "Load pretrained LoRA model for student"}
    )
    # New args for span loss



@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file with dataset configuration"}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name"}
    )
    subset_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Subset names. Useful for datasets with subsets"}
    )
    dataset_split: str = field(
        default='train',
        metadata={"help": "Dataset split (e.g., train, validation)"}
    )
    # --- NEW ARGUMENT HERE ---
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of the training set to be used for validation (e.g., 0.1 for 10%)"}
    )
    # -------------------------
    num_sample_per_subset: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training samples per subset"}
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Image directory path"}
    )
    encode_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save encoded output"}
    )
    max_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum total input sequence length after tokenization."}
    )
    embedding_type: str = field(
        default="",
        metadata={"help": "Embedding type"}
    )
    image_resolution: Optional[str] = field(
        default=None,
        metadata={"help": "Resolution for manual resize (high/mid/low)."}
    )
    resize_use_processor: bool = field(
        default=False,
        metadata={"help": "Resize visual inputs inside processor."}
    )
    resize_min_pixels: int = field(
        default=28*28*4,
        metadata={"help": "Min pixels for processor resize."}
    )
    resize_max_pixels: int = field(
        default=28*28*1280,
        metadata={"help": "Max pixels for processor resize."}
    )
    image_decay_factor: Optional[float] = field(
        default=None,
        metadata={"help": "Image decay factor for resizing temporal images"}
    )
    num_hardneg: int = field(
        default=0,
        metadata={"help": "Number of hard negatives"}
    )
    sdibn: bool = field(
        default=False,
        metadata={"help": "Enable SDIBN"}
    )
    odibn: bool = field(
        default=False,
        metadata={"help": "Enable ODIBN"}
    )
    rdibn: bool = field(
        default=False,
        metadata={"help": "Enable RDIBN"}
    )
    tgt_prefix_mod: bool = field(
        default=False,
        metadata={"help": "Modify the pos_prefix"}
    )
    chunk_size: int = field(
        default=32,
        metadata={"help": "Cluster sizes in metis. Only used in odibn"}
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation dataset name"}
    )
    eval_subset_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Evaluation subset name"}
    )
    eval_image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation Image directory path"}
    )
    pos_only: bool = field(
        default=False,
        metadata={"help": "Only use positives"}
    )
    percent_data: float = field(
        default=1.0,
        metadata={"help": "Percentage of data used for distillation training"}
    )



@dataclass
class TrainingArguments(HFTrainingArguments):
    """
    Training arguments specific to this project, extending HuggingFace's TrainingArguments.
    """
    image_encoder_freeze: bool = field(
        default=False,
        metadata={"help": "Freeze the image encoder during training"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for saving trained models"}
    )
    resume_from: str = field(
        default="none",
        metadata={"help": "'auto' will detect if any previous checkpoints should be resumed, "
                          "or specify specific step of the checkpoint."}
    )
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project name"}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Number of update steps between two logs"}
    )
    num_train_epochs: float = field(
        default=1.0,  # Changed to float as it's often fractional
        metadata={"help": "Total number of training epochs to perform"}
    )
    grad_cache: bool = field(
        default=False,
        metadata={"help": "Use gradient cache update"}
    )
    gc_q_chunk_size: int = field(
        default=128,
        metadata={"help": "Gradient cache: query side subset size. Should be power of 2"}
    )
    gc_p_chunk_size: int = field(
        default=128,
        metadata={"help": "Gradient cache: target side subset size. Should be power of 2"}
    )
    interleave_stopping_strategy: str = field(
        default="all_exhausted",
        metadata={"help": "Strategy for stopping when interleaving datasets: 'all_exhausted' or 'first_exhausted'"}
    )
    interleave_batch_size: float = field(
        default=0,
        metadata={"help": "Specify mini-batch size to interleave data from multi-sources. "
                          "0/None means random sampling by examples, 1 means full batch."}
    )
    # New args
    gc_dynamic_limit: int = field(
        default=125,
        metadata={"help": "gc_chunk default limit. E.g., for Qwen2b (128, 125) works. "
                          "gc_dynamic_limit would be 125 and gc_p|q_chunk_size would be 128"}
    )
    # New kd loss weight
    kd_weight: float = field(
        default=0.01,
        metadata={"help": "Weight of KD loss in total loss"}
    )
    rkd_distance_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of distance loss in total KD loss"}
    )
    rkd_angle_weight: float = field(
        default=2.0,
        metadata={"help": "Weight of angle loss in total KD loss"}
    )
    kd_loss_type: str = field(
        default="contrastive_rkd",
        metadata={"help": "Type of KD loss. Currently only supports 'contrastive_rkd'"}
    )
    ds_config: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed config json file path"}
    )
    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed config json file path (duplicate arg for compatibility)"}
    )
    # New args for span loss
    teacher_layer_mapping: List[int] = field(
        default_factory=list,
        metadata={"help": "List of teacher layers used for distillation; number of elements equals number of projectors"}
    )
    student_layer_mapping: List[int] = field(
        default_factory=list,
        metadata={"help": "List of student layers used for distillation; number of elements equals number of projectors"}
    )
    split_layer_mapping: List[int] = field(
        default_factory=list,
        metadata={"help": "List of split layers for student; number of elements equals number of projectors"}
    )
    w_cross_modal_loss: float = field(
        default=1.0,
        metadata={"help": "Weight for cross modal loss"}
    )
    # eval_steps: int field(
    #     default=100,
    #     metadata={"help" : "Evaluation each eval_steps"}
    # )


@dataclass
class MTEBArguments:
    """
    Arguments for MTEB evaluation.
    """
    device: str = field(
        default="cuda",
        metadata={"help": "Device to use (e.g., 'cuda', 'cpu'). If multiple GPUs, DP is used automatically"}
    )
    batch_size_per_device: int = field(
        default=16,
        metadata={"help": "Batch size per device"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    eval_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for saving evaluation results"}
    )
    task_types: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of task types to evaluate"}
    )
    tasks: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of specific tasks to evaluate"}
    )
    prompt_family: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Prompt family for evaluation"}
    )
