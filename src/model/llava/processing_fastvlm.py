"""
Processor class for FastVLM.
"""

import math 
from collections.abc import Iterable
from typing import Optional, Union, Optional

from PIL import Image
import numpy as np
import torch

from transformers import CLIPImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

from src.model.llava.mm_utils import tokenizer_image_token
from src.model.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

logger = logging.get_logger(__name__)

def expand2square(pil_img, background_color):
    pil_img = pil_img.convert("RGB")
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class FastVLMProcessor(ProcessorMixin):
    r"""
    Constructs a FastVLM processor which wraps a FastVLM image processor and a tokenizer into a single processor.
    [`FastVLMProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`PreTrainedTokenizer`]. See the
    [`~FastVLMProcessor.__call__`] and [`~FastVLMProcessor.decode`] for more information.
    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    """
    
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = []
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        super().__init__(image_processor, tokenizer, **kwargs)
        
    def __call__(
        self, 
        images: Optional[ImageInput] = None,
        texts: Optional[Union[TextInput, PreTokenizedInput, Iterable[TextInput], Iterable[PreTokenizedInput]]] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        input_ids = [tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for text in texts]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        image_tensors = []
        for image in images:
            if image is not None: 
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_tensors.append(image)
        
        if len(image_tensors) > 0: 
            image_tensors = torch.stack(image_tensors, dim=0)
        else:
            image_tensors = None
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if image_tensors is not None:
            data["images"] = image_tensors

        return BatchFeature(data=data, tensor_type="pt")
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    
    def post_process_image_text_to_text(self, generated_outputs):
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

__all__ = ["FastVLMProcessor"]