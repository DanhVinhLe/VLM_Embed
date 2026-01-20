"""
This module implements Knowledge Distillation (KD) for Vision-Language Models (VLMs).

It contains the `Distiller` model wrapper for teacher-student training, `DistillationCollator` for batch processing,
and `DistillationDataset` for loading and preprocessing multimodal datasets.

The general flow is:
1. `DistillationDataset` loads image/text pairs and instructions.
2. `DistillationCollator` processes these into batch tensors using model-specific processors.
3. `Distiller` wraps the student and teacher models, computing projections to align their hidden states
   and passing the data to a loss function (criterion).
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import (
    PHI3V,
    VLM_IMAGE_TOKENS,
    load_processor,
    process_vlm_inputs_fns,
)
from src.utils import print_rank

logger = logging.getLogger(__name__)

# Global constants defining the instruction prompts for different tasks.
# These prefixes are prepended to the positive text samples to guide the model's generation
# or embedding representation during training (Instruction Tuning).

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

# Mapping from dataset names to their corresponding instruction prompts.
# This ensures that each dataset (like ImageNet, OK-VQA, MSCOCO) gets the correct
# task-specific instruction.
POS_MOD_DICT = {
    "ImageNet_1K": POS_MOD_CLASS_LABEL,
    "HatefulMemes": POS_MOD_CLASS_LABEL,
    "SUN397": POS_MOD_CLASS_LABEL,
    "N24News": POS_MOD_CLASS_LABEL,
    "VOC2007": POS_MOD_CLASS_LABEL,
    "Place365": POS_MOD_CLASS_LABEL,
    "ImageNet-A": POS_MOD_CLASS_LABEL,
    "ImageNet-R": POS_MOD_CLASS_LABEL,
    "ObjectNet": POS_MOD_CLASS_LABEL,
    "Country211": POS_MOD_CLASS_LABEL,
    "OK-VQA": POS_MOD_ANSWER,
    "A-OKVQA": POS_MOD_ANSWER,
    "DocVQA": POS_MOD_ANSWER,
    "InfographicsVQA": POS_MOD_ANSWER,
    "ChartQA": POS_MOD_ANSWER,
    "Visual7W": POS_MOD_ANSWER,
    "ScienceQA": POS_MOD_ANSWER,
    "GQA": POS_MOD_ANSWER,
    "TextVQA": POS_MOD_ANSWER,
    "VizWiz": POS_MOD_ANSWER,
    "MSCOCO_i2t": POS_MOD_IMAGE_CAPTION,
    "VisualNews_i2t": POS_MOD_IMAGE_CAPTION,
}


def process_image(image: Image.Image, resolution: str, max_dim: int = 1344) -> Optional[Image.Image]:
    """
    Resizes an image based on the specified resolution setting.
    Maintains aspect ratio by resizing the largest side to match the target dimension.

    Args:
        image: The input PIL Image.
        resolution: resolution strategy ('high', 'mid', 'low') or user-defined.
        max_dim: Maximum dimension for custom resizing.

    Returns:
        The resized PIL Image, or None if input is None.
    """
    if image is None:
        return None

    width, height = image.size
    max_side = max(width, height)

    if resolution == "high":
        target_max = 1344
    elif resolution == "mid":
        target_max = 672
    elif resolution == "low":
        target_max = 128
    else:
        target_max = max_dim

    # Resize if larger than target_max to avoid unnecessary downscaling if already small
    if max_side > target_max:
        image = image.resize((target_max, target_max))

    return image


def create_semi_orthogonal_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """
    Initializes a tensor with a semi-orthogonal matrix using QR decomposition.

    This technique ensures that the initial projection weights preserve the norm of vectors
    and their relative angles as much as possible, which stabilizes training for
    high-dimensional feature alignment tasks.

    Args:
        tensor: The input tensor to initialize (typically nn.Linear.weight).

    Returns:
        The initialized tensor.
    """
    rows, cols = tensor.shape
    if rows >= cols:
        # Direct QR decomposition
        # We generate a random Gaussian matrix and orthogonalize its columns.
        a = torch.randn(rows, cols, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode='reduced')
        tensor.data[:] = q[:, :cols]
    else:
        # QR on transposed matrix to ensure W W^T = I
        # If output dim < input dim, we want rows to be orthogonal.
        a = torch.randn(cols, rows, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode='reduced')
        tensor.data[:] = q.T[:rows, :]
    return tensor


class Distiller(nn.Module):
    """
    A wrapper class for Knowledge Distillation that manages student and teacher models.
    It handles the lifecycle of both models and the projection layers used to align them.
    """

    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments):
        """
        Initializes the Distiller.

        Args:
            model_args: Arguments for model configuration (e.g., model names, LoRA settings).
            training_args: Arguments for training configuration (e.g., loss type, learning rates).
        """
        super(Distiller, self).__init__()
        self.model_args = model_args
        self.training_args = training_args

        # Load models
        self.student = self._load_student()
        self.teacher = self._load_teacher()

        self.student_hidden_dim: int = self.model_args.student_hidden_dim
        self.teacher_hidden_dim: int = self.model_args.teacher_hidden_dim
        self.temperature: float = model_args.temperature

        # Tokenizer is needed for certain loss types that might involve text decoding/matching
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.model_args.teacher_model_name)

        # Setup projection layers to map student space to teacher space
        self.set_projector()
        logger.info("Projectors set.")

    def _create_model_args(self, model_type: str = 'teacher') -> ModelArguments:
        """
        Creates model arguments specifically for the teacher model.
        The teacher might have different LoRA or pooling settings than the student.

        Args:
            model_type: Only 'teacher' is currently supported.

        Returns:
            ModelArguments object configured for the teacher.

        Raises:
            NotImplementedError: If model_type is not 'teacher'.
        """
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

    def _load_student(self) -> nn.Module:
        """Loads the student model (trainable)."""
        logger.info(f"Load student with lora rank: {self.model_args.lora_r}")
        logger.info(f"Student use lora: {self.model_args.lora}")
        student = MMEBModel.build(self.model_args)
        logger.info("Student model built.")
        return student

    def _load_teacher(self) -> nn.Module:
        """Loads the teacher model and freezes its parameters (inference only)."""
        model_args = self._create_model_args('teacher')
        logger.info(f"Load teacher with lora rank: {model_args.lora_r}")
        logger.info(f"Teacher use lora: {model_args.lora}")
        teacher = MMEBModel.load(model_args, is_trainable=False)
        for param in teacher.parameters():
            param.requires_grad = False
        logger.info("Teacher model loaded.")
        return teacher

    def get_student_processor(self) -> ProcessorMixin:
        """
        Gets the processor needed to prepare inputs for the student model.
        Crucial for correct tokenization/image formatting.
        """
        processor = load_processor(self.model_args, None)
        logger.info("Student processor loaded.")
        return processor

    def get_teacher_processor(self) -> ProcessorMixin:
        """Gets the processor needed to prepare inputs for the teacher model."""
        model_args = self._create_model_args('teacher')
        processor = load_processor(model_args, None)
        logger.info("Teacher processor loaded.")
        return processor

    def forward(self, criterion: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass for distillation.

        NOTE: This module delegates the actual loss calculation to the `criterion` object.
        The `criterion` is expected to take `self` (the distiller model) and the `batch`
        to compute the distillation loss (e.g., KL Divergence, MSE).

        Args:
            criterion: The loss function/criterion object (likely a callable).
            batch: The batch of data containing inputs for both student and teacher.

        Returns:
            The calculated scalar loss tensor.
        """
        if self.training_args.kd_loss_type in [
            'span_propose_attn', 'span_propose', 'span_propose_attn_only_phrase'
        ]:
            loss = criterion(self, batch, tokenizer=self.tokenizer)
        else:
            loss = criterion(self, batch)
        return loss

    def set_projector(self) -> None:
        """
        Creates linear (or non-linear) projectors to map:
        Student Hidden States -> Teacher Hidden States.

        This allows us to compare feature vectors even if the models have different dimensions.
        """
        projector_list = nn.ModuleList()

        # Complex projector configuration from JSON file
        if self.model_args.projector_config_path is not None:
            self.projectors = nn.ModuleDict()
            with open(self.model_args.projector_config_path, 'r') as f:
                projector_config = json.load(f)

            # Dictionary to resolve dimension constants in the config string
            name_dict = {
                "s": self.student_hidden_dim,
                "t": self.teacher_hidden_dim,
                "relu": nn.ReLU()
            }

            for name, cfg in projector_config.items():
                if not cfg.get("enabled", False):
                    continue
                seq = nn.Sequential()

                # Config structure format example: "s-4096-relu-t"
                parts = cfg["structure"].split("-")
                parsed = []

                # Parse the structure string
                for p in parts:
                    if p == "relu":
                        parsed.append("relu")
                    else:
                        # Extract coefficient (e.g., "2s" -> 2 * student_dim)
                        coef = int(p[:-1]) if len(p) > 1 and p[:-1].isdigit() else 1
                        parsed.append(coef * name_dict[p[-1]])

                # Build the sequential module from parsed parts
                for i in range(len(parsed) - 1):
                    a, b = parsed[i], parsed[i + 1]
                    if isinstance(a, int) and isinstance(b, int):
                        # Linear Layer definition
                        layer = nn.Linear(a, b)
                        create_semi_orthogonal_matrix(layer.weight)
                        layer = layer.to(dtype=torch.bfloat16)  # FORCE bfloat16
                        seq.append(layer)
                    elif b == "relu":
                        # Activation
                        seq.append(name_dict[b])
                    elif a == "relu" and isinstance(b, int):
                        # Linear layer triggered after a ReLU
                        prev_out = parsed[i - 1] if isinstance(parsed[i - 1], int) else None
                        layer = nn.Linear(prev_out, b)
                        create_semi_orthogonal_matrix(layer.weight)
                        layer = layer.to(dtype=torch.bfloat16)
                        seq.append(layer)
                self.projectors[name] = seq
        else:
            # Default: Simple Linear Projectors for each layer mapping
            for _ in range(len(self.training_args.teacher_layer_mapping)):
                projector = nn.Linear(
                    self.student_hidden_dim,
                    self.teacher_hidden_dim,
                    dtype=torch.bfloat16
                )
                projector_list.append(projector)

            self.projectors = projector_list
        logger.info(f"Created {len(self.projectors)} linear projectors.")

    def add_optimizer_param_group(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """
        Adds projector parameters to the optimizer.
        This is necessary because projectors are initialized *after* the main model
        and might need a different learning rate (projector_lr).

        Args:
            optimizer: The PyTorch optimizer object.

        Returns:
            The updated optimizer with the new parameter group.
        """
        if hasattr(self, 'projectors') and self.projectors is not None:
            lr = getattr(self.training_args, "projector_lr", None) or self.training_args.learning_rate
            optimizer.add_param_group({
                "params": self.projectors.parameters(),
                "lr": lr
            })
        logger.info("Projector parameters added to optimizer.")
        return optimizer


class DistillationCollator:
    """
    Data collator for distillation training.
    Responsible for batching samples and preparing them for BOTH student and teacher models.
    """

    def __init__(self,
                 student_processor: ProcessorMixin,
                 teacher_processor: ProcessorMixin,
                 model_args: ModelArguments,
                 data_args: DataArguments,
                 training_args: TrainingArguments,
                 batch_size: Optional[int] = None):
        """
        Initializes the DistillationCollator.

        Args:
            student_processor: Processor for the student model (e.g. Qwen2-VL processor).
            teacher_processor: Processor for the teacher model (e.g. GPT-4o proxy logic).
            model_args: Model configuration.
            data_args: Data configuration.
            training_args: Training configuration.
            batch_size: Optional fixed batch size for validation.
        """
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.batch_size = batch_size

    def _get_batch_inputs(self, batch: List[Dict], text_keyname: str, image_keyname: str) -> Dict[str, List]:
        """
        Helper to extract specific keys from a list of examples.
        Handles missing data, single items vs lists, and ensures safe returns.

        Args:
            batch: The raw list of examples from the dataset.
            text_keyname: The key corresponding to text logic (e.g., 'student_query_text').
            image_keyname: The key corresponding to image logic (e.g., 'student_query_image').

        Returns:
            A dict with 'text' (List[str]) and 'images' (List[PIL.Image or None]).
        """
        texts: List[str] = []
        visual_inputs: List[Optional[Image.Image]] = []

        for example in batch:
            if example is None or not example:
                # Handle corrupted/empty examples
                text, visual_input = ' ', None
                texts.append(text)
                visual_inputs.append(visual_input)
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                # Normalize to lists
                if not isinstance(text, list):
                    text = [text]
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]

                # Check for empty content
                if not text and not raw_images:
                    text, visual_input = ' ', None
                    texts.append(text)
                    visual_inputs.append(visual_input)
                else:
                    # Flatten lists
                    for t, img in zip(text, raw_images):
                        if not t and img is None:
                            t, img = ' ', None
                        texts.append(t)
                        visual_inputs.append(img)

        inputs = {'text': texts, 'images': visual_inputs}
        return inputs

    def __call__(self, examples: List[Any]) -> Dict[str, Any]:
        """
        Collates a list of examples into a batch.

        This involves:
        1. Extracting student-specific inputs (Query/Pos).
        2. Extracting teacher-specific inputs (Query/Pos).
        3. Running `process_vlm_inputs_fns` to tokenize text and process images.

        Args:
            examples: List of input examples from the dataset.

        Returns:
            A nested dictionary containing processed 'student_inputs' and 'teacher_inputs'.
        """
        student_qry_inputs = self._get_batch_inputs(examples, "student_query_text", "student_query_image")
        student_pos_inputs = self._get_batch_inputs(examples, "student_pos_text", "student_pos_image")

        teacher_qry_inputs = self._get_batch_inputs(examples, "teacher_query_text", "teacher_query_image")
        teacher_pos_inputs = self._get_batch_inputs(examples, "teacher_pos_text", "teacher_pos_image")

        bs = len(student_qry_inputs['text'])
        assert bs > 0, 'An empty batch is detected!'

        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, but got {bs}.")

        # Retrieve backbone-specific processing functions
        process_student_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        process_teacher_fn = process_vlm_inputs_fns[self.model_args.teacher_backbone]

        # Process inputs (Tokenization + Image Preprocessing)
        processed_student_qry_inputs = process_student_fn(
            student_qry_inputs, processor=self.student_processor, max_length=self.data_args.max_len
        )
        processed_student_pos_inputs = process_student_fn(
            student_pos_inputs, processor=self.student_processor, max_length=self.data_args.max_len
        )
        processed_teacher_qry_inputs = process_teacher_fn(
            teacher_qry_inputs, processor=self.teacher_processor, max_length=self.data_args.max_len
        )
        processed_teacher_pos_inputs = process_teacher_fn(
            teacher_pos_inputs, processor=self.teacher_processor, max_length=self.data_args.max_len
        )

        return {
            'student_inputs': {
                'qry': processed_student_qry_inputs,
                'pos': processed_student_pos_inputs
            },
            'teacher_inputs': {
                'qry': processed_teacher_qry_inputs,
                'pos': processed_teacher_pos_inputs
            }
        }


class DistillationDataset(Dataset):
    """
    Dataset class for loading and preprocessing data for distillation.
    Handles multiple subsets, instruction formatting, and image loading.
    """

    def __init__(self, data_args: DataArguments, model_args: ModelArguments):
        """
        Initializes the DistillationDataset.

        Args:
            data_args: Data configuration (paths, subsets, split).
            model_args: Model configuration (backbone types).
        """
        self.data_args = data_args
        self.model_args = model_args
        train_data = []

        # Load each subset defined in data_args
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=f"{self.data_args.dataset_split}"
            )

            # Special preprocessing for WebQA
            if subset == "WebQA" and "qry" in subset_data.column_names:
                subset_data = subset_data.map(
                    lambda x: {"qry": x["qry"].replace("<|image_1|>", "").strip()}
                )
                print_rank("Preprocessed WebQA to remove <image_1> tokens in queries.")

            # Data sub-sampling
            total_samples = len(subset_data)
            num_samples_to_keep = math.ceil(total_samples * self.data_args.percent_data)
            subset_data = subset_data.select(range(num_samples_to_keep))

            # Add instruction prefixes (Instruction Tuning)
            subset_data = subset_data.add_column(
                "pos_text_instruction",
                [POS_MOD_DICT.get(subset, "") + text for text in subset_data['pos_text']]
            )

            # Column cleanup
            subset_data = subset_data.remove_columns(
                set(['neg_text', 'neg_image_path']) & set(subset_data.column_names)
            )
            # Keep only necessary columns
            subset_data = subset_data.remove_columns(
                set(subset_data.column_names) - set(
                    ['qry', 'qry_image_path', 'pos_image_path', 'pos_text_instruction']
                )
            )
            subset_data = subset_data.rename_column("pos_text_instruction", "pos_text")
            train_data.append(subset_data)

        self.train_data = concatenate_datasets(train_data)
        print_rank(
            f"Loaded {len(self.train_data)} samples from {self.data_args.dataset_name} "
            f"with subsets {self.data_args.subset_name}"
        )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.train_data)

    def _get_image(self, img_path: str, backbone: str) -> Optional[Image.Image]:
        """
        Loads and processes an image from the given path.
        Includes padding for small images and resizing.

        Args:
            img_path: Relative path to the image file.
            backbone: Model backbone name (used to determine processing logic).

        Returns:
            The processed PIL Image or None if loading fails.
        """
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        try:
            image = Image.open(full_img_path)
        except Exception as e:
            # Important: We silently fail here to avoid crashing the whole training job
            # but this might lead to empty batches if not handled downstream.
            logger.error(f"Error opening image {full_img_path}: {e}")
            return None

        image = image.convert("RGB")
        width, height = image.size

        # Pad small images to at least 16x16
        MIN_SIZE = 16
        if width < MIN_SIZE or height < MIN_SIZE:
            new_width = max(width, MIN_SIZE)
            new_height = max(height, MIN_SIZE)
            result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
            x_offset = (new_width - width) // 2
            y_offset = (new_height - height) // 2
            result.paste(image, (x_offset, y_offset))
            image = result

        # Resize if the backbone is not Phi-3 (Phi-3 has its own resizing logic inside processor)
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx: int) -> Dict[str, List[Any]]:
        """
        Retrieves an item at the specified index.
        Prepares distinct inputs for student and teacher models, handling special token replacements.

        Args:
            data_idx: The index of the item.

        Returns:
            A dictionary containing lists of query/pos texts and images for student and teacher.
        """
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )

        # Normalize to lists to handle single vs multiple inputs consistently
        if not isinstance(qry_texts, list):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]

        student_qry_texts, student_qry_images, student_pos_texts, student_pos_images = [], [], [], []
        teacher_qry_texts, teacher_qry_images, teacher_pos_texts, teacher_pos_images = [], [], [], []

        student_backbone = self.model_args.model_backbone
        teacher_backbone = self.model_args.teacher_backbone

        for qry_text, qry_image_path, pos_text, pos_image_path in zip(
                qry_texts, qry_image_paths, pos_texts, pos_image_paths):

            # Prepare Student Inputs
            stu_qry_text, stu_pos_text = qry_text, pos_text

            # TOKEN REPLACEMENT LOGIC:
            # The dataset might contain PHI3V specific image tokens. We need to swap them
            # if the current student/teacher model uses a different token (e.g. <image> vs <|image|>).
            if student_backbone != PHI3V:
                stu_qry_text = stu_qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
                stu_pos_text = stu_pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])

            stu_qry_image = self._get_image(qry_image_path, student_backbone)
            stu_pos_image = self._get_image(pos_image_path, student_backbone)

            # Skip empty pairs (safeguard against bad data)
            if (not stu_qry_text and not stu_qry_image) or (not stu_pos_text and not stu_pos_image):
                logger.warning("Empty inputs detected in student processing.")
                continue

            student_qry_texts.append(stu_qry_text)
            student_qry_images.append(stu_qry_image)
            student_pos_texts.append(stu_pos_text)
            student_pos_images.append(stu_pos_image)

            # Prepare Teacher Inputs (Same logic, different backbone)
            teacher_qry_text, teacher_pos_text = qry_text, pos_text
            if teacher_backbone != PHI3V:
                teacher_qry_text = teacher_qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[teacher_backbone])
                teacher_pos_text = teacher_pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[teacher_backbone])

            teacher_qry_image = self._get_image(qry_image_path, teacher_backbone)
            teacher_pos_image = self._get_image(pos_image_path, teacher_backbone)

            if (not teacher_qry_text and not teacher_qry_image) or (not teacher_pos_text and not teacher_pos_image):
                logger.warning("Empty inputs detected in teacher processing.")
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