import os
from typing import Dict, Tuple, Optional
import time
import json
import torch
import torch.nn as nn
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name, process_vlm_inputs_fns, backbone2model, \
    LLAVA_NEXT, QWEN2_VL, LLAVA_ONEVISION, QWEN2_5_VL_TOKENSELECTION, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION
from src.data.collator.train_collator import MultimodalDataCollator, TrainTextImageDataCollator
from src.data.dataset.mmeb_dataset import TrainTextImageDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from src.utils import print_rank, print_master
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel 

def add_distiller_arguments(parser):
    """Thêm arguments cho Distiller"""
    # Student model arguments
    parser.add_argument('--student_model_path', type=str, required=True,
                       help='Path to student model')
    parser.add_argument('--student_checkpoint_path', type=str, default=None,
                       help='Path to student checkpoint')
    parser.add_argument('--student_lora', action='store_true',
                       help='Whether student uses LoRA')
    
    # Teacher model arguments  
    parser.add_argument('--teacher_model_path', type=str, required=True,
                       help='Path to teacher model')
    parser.add_argument('--teacher_checkpoint_path', type=str, default=None,
                       help='Path to teacher checkpoint')
    parser.add_argument('--teacher_lora', action='store_true',
                       help='Whether teacher uses LoRA')
    
    # Common model arguments
    parser.add_argument('--pooling', type=str, default='last',
                       help='Pooling strategy')
    parser.add_argument('--normalize', action='store_true',
                       help='Whether to normalize embeddings')
    parser.add_argument('--temperature', type=float, default=0.02,
                       help='Temperature for similarity')
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (e.g., "cuda", "cpu")')
    
    return parser

class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student = self._load_student()
        self.teacher = self._load_teacher()
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.temperature = args.temperature
    
    def _create_model_args(self, model_type='student'):
        """Tạo ModelArguments từ args hiện tại"""
        if model_type == 'student':
            model_args = ModelArguments(
                model_name=self.args.student_model_path,
                checkpoint_path=getattr(self.args, 'student_checkpoint_path', None),
                lora=self.args.student_lora,
                pooling=self.args.pooling,
                normalize=self.args.normalize,
                temperature=self.args.temperature,
                model_type=getattr(self.args, 'model_type', None)
            )
        else:  # teacher
            model_args = ModelArguments(
                model_name=self.args.teacher_model_path,
                checkpoint_path=getattr(self.args, 'teacher_checkpoint_path', None),
                lora=self.args.teacher_lora,
                pooling=self.args.pooling,
                normalize=self.args.normalize,
                temperature=self.args.temperature,
                model_type=getattr(self.args, 'model_type', None)
            )
        return model_args
    
    def _load_teacher(self):
        model_args = self._create_model_args('teacher')
        teacher = MMEBModel.load(model_args, is_trainable=False)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        return teacher
    
    def _load_student(self):
        model_args = self._create_model_args('student')
        student = MMEBModel.load(model_args, is_trainable=True)
        return student 
    
    def student_forward(self, qry: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]= None, *args, **kwargs):
        qry_reps = self.student.encode_input(qry) if qry is not None else None
        tgt_reps = self.student.encode_input(tgt) if tgt is not None else None
        
        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}
        
        scores = self.student.compute_similarity(qry_reps, tgt_reps)
        scores = scores.view(qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (qry_reps.size(0) // tgt_reps.size(0))
        loss = nn.CrossEntropyLoss()(scores / self.temperature, target)
        
        return {"contrastive_loss": loss, "stu_qry_reps": qry_reps, "stu_tgt_reps": tgt_reps}
    
    def teacher_forward(self, qry: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]= None, *args, **kwargs):
        with torch.no_grad():
            qry_reps = self.teacher.encode_input(qry) if qry is not None else None
            tgt_reps = self.teacher.encode_input(tgt) if tgt is not None else None
            
            if qry_reps is None or tgt_reps is None:
                return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}
        
        return {"tea_qry_reps": qry_reps, "tea_tgt_reps": tgt_reps}
    
    
    