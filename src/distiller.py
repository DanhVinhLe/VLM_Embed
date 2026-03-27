import os
import io
from typing import Dict, Tuple, Optional
import time
import json
import pickle
import math
from datasets import load_dataset, concatenate_datasets
import torch
import torch.nn as nn
import PIL
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
from transformers import AutoTokenizer

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
                "ImageNet_1K": POS_MOD_CLASS_LABEL,"HatefulMemes":POS_MOD_CLASS_LABEL,"SUN397":POS_MOD_CLASS_LABEL,"N24News":POS_MOD_CLASS_LABEL,"VOC2007":POS_MOD_CLASS_LABEL, "Place365":POS_MOD_CLASS_LABEL,"ImageNet-A":POS_MOD_CLASS_LABEL,"ImageNet-R":POS_MOD_CLASS_LABEL,"ObjectNet":POS_MOD_CLASS_LABEL,"Country211":POS_MOD_CLASS_LABEL,
                
                "OK-VQA":POS_MOD_ANSWER, "A-OKVQA":POS_MOD_ANSWER, "DocVQA":POS_MOD_ANSWER, "InfographicsVQA":POS_MOD_ANSWER, "ChartQA":POS_MOD_ANSWER, "Visual7W":POS_MOD_ANSWER,"ScienceQA":POS_MOD_ANSWER, "GQA":POS_MOD_ANSWER, "TextVQA":POS_MOD_ANSWER, "VizWiz":POS_MOD_ANSWER,
                
                "MSCOCO_i2t":POS_MOD_IMAGE_CAPTION, "VisualNews_i2t":POS_MOD_IMAGE_CAPTION,
                }

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None

    width, height = image.size
    max_side = max(width, height)

    if resolution == "high":
        target_max = 1344
    elif resolution == "mid":
        target_max = 672
    elif resolution == "low":
        target_max = 448
    else:
        target_max = max_dim

    # resize if larger than target_max
    if max_side > target_max:
        image = image.resize((target_max, target_max))

    return image

def create_semi_orthogonal_matrix(tensor):
    rows, cols = tensor.shape
    if rows >= cols:
        # QR trực tiếp
        a = torch.randn(rows, cols, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode='reduced')
        tensor.data[:] = q[:, :cols]
    else:
        # QR trên ma trận transpose để đảm bảo W W^T = I
        a = torch.randn(cols, rows, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode='reduced')
        tensor.data[:] = q.T[:rows, :]
    return tensor

class Distiller(nn.Module):
    def __init__(self, model_args, training_args):
        super(Distiller, self).__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.student = self._load_student()
        self.teacher = self._load_teacher()
        self.student_hidden_dim = self.model_args.student_hidden_dim
        self.teacher_hidden_dim = self.model_args.teacher_hidden_dim
        self.temperature = model_args.temperature
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.teacher_model_name)
        # if self.model_args.projector_config_path is not None:
        self.set_projector()
        print("Projectors set.")
    
    def _create_model_args(self, model_type='teacher'):
        if model_type == 'teacher': 
            model_args = ModelArguments(
                model_name=self.model_args.teacher_model_name, 
                checkpoint_path=getattr(self.model_args, 'teacher_checkpoint_path', None),
                lora=self.model_args.teacher_lora,
                lora_r=self.model_args.teacher_lora_r,
                lora_alpha=self.model_args.teacher_lora_alpha,
                lora_dropout=self.model_args.teacher_lora_dropout,
                lora_target_modules=self.model_args.teacher_lora_target_modules,
                pooling=self.model_args.teacher_pooling,
                normalize=self.model_args.teacher_normalize,
                model_backbone=self.model_args.teacher_backbone,
            )
        else:
            print_rank("Not implemented student model args creation.")
            raise NotImplementedError
        return model_args
    
    def _load_student(self):
        print("Load student with lora rank:", self.model_args.lora_r)
        print("Student use lora:", self.model_args.lora)
        student = MMEBModel.build(self.model_args)
        print("Student model built.")
        return student 
    
    def _load_teacher(self):
        model_args = self._create_model_args('teacher')
        print("Load teacher with lora rank:", model_args.lora_r)
        print("Teacher use lora:", model_args.lora)
        teacher = MMEBModel.load(model_args, is_trainable=False)
        for param in teacher.parameters():
            param.requires_grad = False
        print("Teacher model loaded.")
        return teacher
    
    def get_student_processor(self):
        processor = load_processor(self.model_args, None)
        print("Student processor loaded.")
        return processor

    def get_teacher_processor(self):
        model_args = self._create_model_args('teacher')
        processor = load_processor(model_args, None)
        print("Teacher processor loaded.")
        return processor
    
    def forward(self, criterion, batch):
        if self.training_args.kd_loss_type in ['span_propose_attn', 'span_propose', 'span_propose_attn_only_phrase' ]:
            loss = criterion(self, batch, tokenizer = self.tokenizer)
        else: 
            loss = criterion(self, batch)
        return loss
    
    def set_projector(self):
        """
        Create a list of linear projectors mapping
        student_hidden_dim -> teacher_hidden_dim
        One projector per teacher layer mapping.
        """
        projector_list = nn.ModuleList()
        
        if self.model_args.projector_config_path is not None:
            self.projectors = nn.ModuleDict()
            projector_config = json.load(open(self.model_args.projector_config_path, 'r'))
            
            name_dict = {
                "s": self.student_hidden_dim,
                "t": self.teacher_hidden_dim,
                "relu": nn.ReLU()
            }
            
            for name, cfg in projector_config.items():
                if not cfg.get("enabled", False):
                    continue
                seq = nn.Sequential()
                parts = cfg["structure"].split("-")
                parsed = []
                
                for p in parts:
                    if p == "relu":
                        parsed.append("relu")
                    else:
                        coef = int(p[:-1]) if len(p) > 1 and p[:-1].isdigit() else 1
                        parsed.append(coef * name_dict[p[-1]])
                for i in range(len(parsed) -1):
                    a, b = parsed[i], parsed[i+1]
                    if isinstance(a, int) and isinstance(b, int):
                        layer = nn.Linear(a, b)
                        create_semi_orthogonal_matrix(layer.weight)
                        layer = layer.to(dtype=torch.bfloat16)
                        seq.append(layer)
                    elif b == "relu":
                        seq.append(name_dict[b])
                    elif a =="relu" and isinstance(b, int):
                        prev_out = parsed[i-1] if isinstance(parsed[i-1], int) else None
                        layer = nn.Linear(prev_out, b)
                        create_semi_orthogonal_matrix(layer.weight)
                        layer = layer.to(dtype=torch.bfloat16)
                        seq.append(layer)
                self.projectors[name] = seq
        else:
            for _ in range(len(self.training_args.teacher_layer_mapping)):
                projector = nn.Linear(
                    self.student_hidden_dim,
                    self.teacher_hidden_dim,
                    dtype=torch.bfloat16
                )
                projector_list.append(projector)

            self.projectors = projector_list
        print(f"Created {len(self.projectors)} linear projectors.")
    
    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, 'projectors') and self.projectors is not None:
            lr = getattr(self.training_args, "projector_lr", None) or self.training_args.learning_rate
            optimizer.add_param_group({
                "params": self.projectors.parameters(),
                "lr": lr
            })
        print("Projector parameters added to optimizer.")
        return optimizer
class DistillationCollator:
    def __init__(self, student_processor: ProcessorMixin, teacher_processor: ProcessorMixin,
                 model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments,
                 batch_size: Optional[int] = None):
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.batch_size = batch_size
    
    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        # print("Processing batch for keys:", text_keyname, image_keyname)
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = ' ', None
                texts.append(text)
                visual_inputs.append(visual_input)
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                if not isinstance(text, list):
                    text = [text]
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                if not text and not raw_images:
                    text, visual_input = ' ', None
                    texts.append(text)
                    visual_inputs.append(visual_input)
                else:
                    for t, img in zip(text, raw_images):
                        if not t and img is None:
                            t, img = ' ', None
                        texts.append(t)
                        visual_inputs.append(img)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs
    
    def __call__(self, examples):
        student_qry_inputs = self._get_batch_inputs(examples, "student_query_text", "student_query_image")
        student_pos_inputs = self._get_batch_inputs(examples, "student_pos_text", "student_pos_image")

        teacher_qry_inputs = self._get_batch_inputs(examples, "teacher_query_text", "teacher_query_image")
        teacher_pos_inputs = self._get_batch_inputs(examples, "teacher_pos_text", "teacher_pos_image")

        bs = len(student_qry_inputs['text'])
        assert bs > 0, 'An empty batch is detected!'
        
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, but got {bs}.")
        
        process_student_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        process_teacher_fn = process_vlm_inputs_fns[self.model_args.teacher_backbone]
        
        processed_student_qry_inputs = process_student_fn(student_qry_inputs, processor=self.student_processor, max_length=self.data_args.max_len)
        processed_student_pos_inputs = process_student_fn(student_pos_inputs, processor=self.student_processor, max_length=self.data_args.max_len)
        processed_teacher_qry_inputs = process_teacher_fn(teacher_qry_inputs, processor=self.teacher_processor, max_length=self.data_args.max_len)
        processed_teacher_pos_inputs = process_teacher_fn(teacher_pos_inputs, processor=self.teacher_processor, max_length=self.data_args.max_len)
        
        return {
            'student_inputs':{
                'qry': processed_student_qry_inputs,
                'pos': processed_student_pos_inputs
            },
            'teacher_inputs':{
                'qry': processed_teacher_qry_inputs,
                'pos': processed_teacher_pos_inputs
            }
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
            if subset == "WebQA" and "qry" in subset_data.column_names:
                subset_data = subset_data.map(
                    lambda x: {"qry": x["qry"].replace("<|image_1|>", "").strip()}
                )
                print_rank("Preprocessed WebQA to remove <image_1> tokens in queries.")
            total_samples = len(subset_data)
            num_samples_to_keep = math.ceil(total_samples * self.data_args.percent_data)
            subset_data = subset_data.select(range(num_samples_to_keep))
            subset_data = subset_data.add_column("pos_text_instruction", [POS_MOD_DICT.get(subset, "") + text for text in subset_data['pos_text']])
            subset_data = subset_data.remove_columns(set(['neg_text', 'neg_image_path']) & set(subset_data.column_names))
            subset_data = subset_data.remove_columns(set(subset_data.column_names) - set(['qry', 'qry_image_path', 'pos_image_path', 'pos_text_instruction']))
            subset_data = subset_data.rename_column("pos_text_instruction", "pos_text")
            train_data.append(subset_data)
            
        self.train_data = concatenate_datasets(train_data)
        print_rank(f"Loaded {len(self.train_data)} samples from {self.data_args.dataset_name} with subsets {self.data_args.subset_name}")
    
    def __len__(self):
        return len(self.train_data)
    def _get_image(self, img_path, backbone):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        image = image.convert("RGB")
        width, height = image.size
        MIN_SIZE = 16
        if width < MIN_SIZE or height < MIN_SIZE:
            new_width = max(width, MIN_SIZE)
            new_height = max(height, MIN_SIZE)
            result = Image.new(image.mode, (new_width, new_height), (0,0,0))
            x_offset = (new_width - width) // 2
            y_offset = (new_height - height) // 2
            result.paste(image, (x_offset, y_offset))
            image = result
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        
    def __getitem__(self, data_idx):
        # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>get image called, {data_idx}", flush=True)
        
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )

        if not isinstance(qry_texts, list):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            
        student_qry_texts, student_qry_images, student_pos_texts, student_pos_images = [], [], [], []
        teacher_qry_texts, teacher_qry_images, teacher_pos_texts, teacher_pos_images = [], [], [], []
                
        student_backbone = self.model_args.model_backbone
        teacher_backbone = self.model_args.teacher_backbone

        for qry_text, qry_image_path, pos_text, pos_image_path in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2, qwenvl
            
            stu_qry_text, stu_pos_text = qry_text, pos_text
            if student_backbone != PHI3V:
                stu_qry_text = stu_qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
                stu_pos_text = stu_pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
            stu_qry_image = self._get_image(qry_image_path, student_backbone)
            stu_pos_image = self._get_image(pos_image_path, student_backbone)

            if (not stu_qry_text and not stu_qry_image) or (not stu_pos_text and not stu_pos_image):
                print("empty inputs")
                continue
            
            student_qry_texts.append(stu_qry_text)
            student_qry_images.append(stu_qry_image)
            student_pos_texts.append(stu_pos_text)
            student_pos_images.append(stu_pos_image)

            teacher_qry_text, teacher_pos_text = qry_text, pos_text
            if teacher_backbone != PHI3V:
                teacher_qry_text = teacher_qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[teacher_backbone])
                teacher_pos_text = teacher_pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[teacher_backbone])
            teacher_qry_image = self._get_image(qry_image_path, teacher_backbone)
            teacher_pos_image = self._get_image(pos_image_path, teacher_backbone)

            if (not teacher_qry_text and not teacher_qry_image) or (not teacher_pos_text and not teacher_pos_image):
                print("empty inputs")
                continue
            teacher_qry_texts.append(teacher_qry_text)
            teacher_qry_images.append(teacher_qry_image)
            teacher_pos_texts.append(teacher_pos_text)
            teacher_pos_images.append(teacher_pos_image)
            
        return {
            "student_query_text": student_qry_texts,
            "student_query_image": student_qry_images,
            "student_pos_text": student_pos_texts,
            "student_pos_image": student_pos_images,
            "teacher_query_text": teacher_qry_texts,
            "teacher_query_image": teacher_qry_images,
            "teacher_pos_text": teacher_pos_texts,
            "teacher_pos_image": teacher_pos_images,
        }