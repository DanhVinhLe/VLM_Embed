import os
import io
from typing import Dict, Tuple, Optional
import time
import json
from datasets import load_dataset, concatenate_datasets
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
from src.model.processor import VLM_IMAGE_TOKENS, load_processor, get_backbone_name, process_vlm_inputs_fns, backbone2model, \
    LLAVA_NEXT, QWEN2_VL, LLAVA_ONEVISION, QWEN2_5_VL_TOKENSELECTION, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, PHI3V
from src.data.collator.train_collator import MultimodalDataCollator, TrainTextImageDataCollator
from src.data.dataset.mmeb_dataset import TrainTextImageDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from src.utils import print_rank, print_master
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel 
from transformers import ProcessorMixin
from qwen_vl_utils import smart_resize
from PIL import Image

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

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
        
        return {"tea_qry_reps": qry_reps, "tea_tgt_reps": tgt_reps}
    
    def compute_loss(self, qry: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]= None, *args, **kwargs):
        student_outputs = self.student_forward(qry, tgt, *args, **kwargs)
        with torch.no_grad():
            teacher_outputs = self.teacher_forward(qry, tgt, *args, **kwargs)
        
        loss = student_outputs["contrastive_loss"]
        kd_loss_1 = nn.MSELoss()(torch.matmul(student_outputs["study_reps"], student_outputs["stu_tgt_reps"].t()),
                                 torch.matmul(teacher_outputs["tea_qry_reps"], teacher_outputs["tea_tgt_reps"].t()))
        kd_loss_2 = nn.MSELoss()(torch.matmul(student_outputs["stu_tgt_reps"], student_outputs["stu_qry_reps"].t()),
                                 torch.matmul(teacher_outputs["tea_tgt_reps"], teacher_outputs["tea_qry_reps"].t()))
        kd_loss = 0.1 * (kd_loss_1 + kd_loss_2)
        loss += kd_loss
        
        return loss

class DistillationCollator:
    student_processor: ProcessorMixin
    teacher_processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None
    
    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = ' ', None
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                if type(raw_images) == dict: 
                    visual_input = []
                    assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                    num_images = len(raw_images['resolutions'])
                    for image_idx in range(num_images):
                        bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                        path = raw_images['path'][image_idx] if 'path' in raw_images else None
                        image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                        if bytes is None and path is None:
                            image = None
                        elif bytes is not None: 
                            image = Image.open(io.BytesIO(bytes))
                        elif path is not None:
                            image = Image.open(path).convert('RGB')
                        else: 
                            print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                        if not self.data_args.resize_min_pixels and image is not None and image_resolution: 
                            image = image.resize(image_resolution) 
                        if image is not None and (self.data_args.image_decay_factor is not None and image_resolution is None): 
                            assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                            assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                            # TODO: this is a hacky way to decay image resolution, need to be refactored 
                            max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_min_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                            width, height = image.size
                            resized_width, resized_height = smart_resize(width, height, min_pixels=self.data_args.resize_min_pixels, max_pixels=max_pixels)
                            image = image.resize((resized_width, resized_height))
                        visual_input.append(image)
                else:
                    visual_input = None
            texts.append(text)
            visual_inputs.append(visual_input)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs
    
    def __call__(self, examples):
        student_qry_inputs = self._get_batch_inputs(examples, "student_query_text", "student_query_image")
        student_pos_inputs = self._get_batch_inputs(examples, "student_pos_text", "student_pos_image")
        student_neg_inputs = self._get_batch_inputs(examples, "student_neg_text", "student_neg_image")
        teacher_qry_inputs = self._get_batch_inputs(examples, "teacher_query_text", "teacher_query_image")
        teacher_pos_inputs = self._get_batch_inputs(examples, "teacher_pos_text", "teacher_pos_image")
        teacher_neg_inputs = self._get_batch_inputs(examples, "teacher_neg_text", "teacher_neg_image")

        bs = len(student_qry_inputs['text'])
        assert bs > 0, 'An empty batch is detected!'
        
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, but got {bs}.")
        
        process_student_fn = process_vlm_inputs_fns(self.model_args.student_backbone)
        process_teacher_fn = process_vlm_inputs_fns(self.model_args.teacher_backbone)
        processed_student_qry_inputs = process_student_fn(student_qry_inputs, processor=self.student_processor, max_length=self.data_args.max_len)
        processed_student_pos_inputs = process_student_fn(student_pos_inputs, processor=self.student_processor, max_length=self.data_args.max_len)
        processed_teacher_qry_inputs = process_teacher_fn(teacher_qry_inputs, processor=self.teacher_processor, max_length=self.data_args.max_len)
        processed_teacher_pos_inputs = process_teacher_fn(teacher_pos_inputs, processor=self.teacher_processor, max_length=self.data_args.max_len)

        processed_student_qry_inputs['text'] = [e['query_text'] for e in examples]
        processed_student_pos_inputs['text'] = [e['positive_text'] for e in examples]
        processed_student_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        processed_student_pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        processed_teacher_qry_inputs['text'] = [e['query_text'] for e in examples]
        processed_teacher_pos_inputs['text'] = [e['positive_text'] for e in examples]
        processed_teacher_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        processed_teacher_pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        
        return {
            'student_inputs': [processed_student_qry_inputs, processed_student_pos_inputs],
            'teacher_inputs': [processed_teacher_qry_inputs, processed_teacher_pos_inputs]
        }
        
class DistillationDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name, 
                subset,
                split=f"{self.data_args.dataset_split}"
            )
            train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)
        print(f"Loaded {len(self.train_data)} samples from {self.data_args.dataset_name} with subsets {self.data_args.subset_name}")
    
    def __len__(self):
        return len(self.train_data)
    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        
    def __getitem__(self, data_idx):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>get image called, {data_idx}", flush=True)
        
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )
        if 'neg_text' in self.train_data.column_names:
            neg_texts, neg_image_paths = self.train_data[data_idx]["neg_text"], self.train_data[data_idx]["neg_image_path"]
        else:
            neg_texts, neg_image_paths = [''] * len(data_idx), [] * len(data_idx)
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            neg_texts = [neg_texts]
            neg_image_paths = [neg_image_paths]
        student_qry_texts, student_qry_images, student_pos_texts, student_pos_images, student_neg_texts, student_neg_images = [], [], [], [], [], []
        teacher_qry_texts, teacher_qry_images, teacher_pos_texts, teacher_pos_images, teacher_neg_texts, teacher_neg_images = [], [], [], [], [], []
        
        student_backbone = self.model_args.student_backbone
        teacher_backbone = self.model_args.teacher_backbone

        for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if student_backbone != PHI3V:
                stu_qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
                stu_pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
                stu_neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone]) if neg_text else None
            stu_qry_image = self._get_image(qry_image_path)
            stu_pos_image = self._get_image(pos_image_path)
            stu_neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not stu_qry_text and not stu_qry_image) or (not stu_pos_text and not stu_pos_image):
                print("empty inputs")
                continue
            student_qry_texts.append(stu_qry_text)
            student_qry_images.append(stu_qry_image)
            student_pos_texts.append(stu_pos_text)
            student_pos_images.append(stu_pos_image)
            student_neg_texts.append(stu_neg_text)
            student_neg_images.append(stu_neg_image)
            
            if teacher_backbone != PHI3V:
                tea_qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[teacher_backbone])
                tea_pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[teacher_backbone])
                tea_neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[teacher_backbone]) if neg_text else None
            tea_qry_image = self._get_image(qry_image_path)
            tea_pos_image = self._get_image(pos_image_path)
            tea_neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not tea_qry_text and not tea_qry_image) or (not tea_pos_text and not tea_pos_image):
                print("empty inputs")
                continue
            teacher_qry_texts.append(tea_qry_text)
            teacher_qry_images.append(tea_qry_image)
            teacher_pos_texts.append(tea_pos_text)
            teacher_pos_images.append(tea_pos_image)
            teacher_neg_texts.append(tea_neg_text)
            teacher_neg_images.append(tea_neg_image)
            
        return {
            "student_query_text": student_qry_texts,
            "student_query_image": student_qry_images,
            "student_pos_text": student_pos_texts,
            "student_pos_image": student_pos_images,
            "student_neg_text": student_neg_texts,
            "student_neg_image": student_neg_images,
            "teacher_query_text": teacher_qry_texts,
            "teacher_query_image": teacher_qry_images,
            "teacher_pos_text": teacher_pos_texts,
            "teacher_pos_image": teacher_pos_images,
            "teacher_neg_text": teacher_neg_texts,
            "teacher_neg_image": teacher_neg_images,
        }