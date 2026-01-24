"""
This module defines the Exact Matching Knowledge Distillation (EMKD) loss,
specifically adapted for the LLaVA One-Vision (OV) model.
"""
import logging
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from transformers import PreTrainedModel

from ..arguments import EMKDArguments

logger = logging.getLogger(__name__)

# Token IDs from a specific tokenizer's vocabulary.
BOS_TOKEN_ID = 151643
VISION_END_TOKEN_ID = 151656

class EMKDLLavaLoss(nn.Module):
    """
    Implements the Exact Matching Knowledge Distillation (EMKD) loss tailored for
    the LLaVA One-Vision architecture.
    
    This loss is fundamentally similar to the standard EMKD loss but includes specific
    adaptations for the LLaVA-OV model, such as padding to handle potential dimension
    mismatches in vocabulary logits and different indexing for vision tokens which are 
    located at the end of the sequence for both student and teacher.
    
    The loss combines:
    1.  **Contrastive Loss:** Standard InfoNCE loss on the final embeddings.
    2.  **Global MSE Loss:** MSE loss on the projected global representations.
    3.  **Vision Semantic Distillation (VSD):** MSE loss on vocabulary logits of matched vision tokens.
    4.  **Vision-Language Alignment Distillation (VLAD):** Smooth L1 loss on the cosine affinity
        between matched vision tokens and text tokens.
    """

    def __init__(self, args: Type[EMKDArguments]):
        super(EMKDLLavaLoss, self).__init__()
        self.args = args
        self.kd_loss_weight: float = self.args.kd_weight
        self.vsd_loss_weight: float = self.args.vsd_loss_weight
        self.vlad_loss_weight: float = self.args.vlad_loss_weight
        self.contrastive_loss_weight: float = self.args.contrastive_loss_weight
        self.mse_loss_weight: float = self.args.mse_loss_weight
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        """Gathers tensors from all processes in a distributed setting."""
        if self.world_size == 1: return t
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, Tensor]:
        """Computes the combined EMKD loss for LLaVA-OV."""
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher

        # --- 1. Get Student and Teacher Representations ---
        with torch.no_grad():
            teacher_model.eval()
            t_qry_out = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos_out = teacher_model.encode_input(input_data['teacher_inputs']['pos'])
        
        s_qry_out = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos_out = student_model.encode_input(input_data['student_inputs']['pos'])

        s_qry_reps, s_qry_img_feats, _, s_qry_hiddens = s_qry_out
        s_pos_reps, s_pos_img_feats, _, s_pos_hiddens = s_pos_out
        t_qry_reps, t_qry_img_feats, _, t_qry_hiddens = t_qry_out
        t_pos_reps, t_pos_img_feats, _, t_pos_hiddens = t_pos_out
        
        # --- 2. Compute Contrastive and Global MSE Loss ---
        all_s_qry_reps = self._dist_gather_tensor(s_qry_reps)
        all_s_pos_reps = self._dist_gather_tensor(s_pos_reps)
        scores = student_model.compute_similarity(all_s_qry_reps, all_s_pos_reps)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_s_pos_reps.size(0) // all_s_qry_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores.view(all_s_qry_reps.size(0), -1) / distiller.temperature, target)

        mse_loss = 0.5 * (
            nn.MSELoss()(s_qry_reps, distiller.projectors["t2s"](t_qry_reps)) +
            nn.MSELoss()(s_pos_reps, distiller.projectors["t2s"](t_pos_reps))
        )
        
        # --- 3. Compute VSD and VLAD Distillation Losses ---
        batch_size = s_qry_reps.size(0)
        total_vsd_loss, total_vlad_loss = 0.0, 0.0

        for i in range(batch_size):
            vsd_qry, vlad_qry = self._compute_emkd_for_side(
                student_model, teacher_model, s_qry_img_feats, t_qry_img_feats,
                s_qry_hiddens[-1][i], t_qry_hiddens[-1][i],
                input_data['teacher_inputs']['qry']['input_ids'][i]
            )
            total_vsd_loss += vsd_qry
            total_vlad_loss += vlad_qry

            vsd_pos, vlad_pos = self._compute_emkd_for_side(
                student_model, teacher_model, s_pos_img_feats, t_pos_img_feats,
                s_pos_hiddens[-1][i], t_pos_hiddens[-1][i],
                input_data['teacher_inputs']['pos']['input_ids'][i]
            )
            total_vsd_loss += vsd_pos
            total_vlad_loss += vlad_pos
            
        distill_loss = (self.vsd_loss_weight * total_vsd_loss + self.vlad_loss_weight * total_vlad_loss) / batch_size
        
        # --- 4. Combine All Losses ---
        loss = (self.contrastive_loss_weight * contrastive_loss + 
                self.mse_loss_weight * mse_loss + 
                self.kd_loss_weight * distill_loss)
        
        return {'loss': loss, 'contrastive_loss': contrastive_loss, 'kd_loss': distill_loss}

    def _compute_emkd_for_side(
        self, student_model: nn.Module, teacher_model: nn.Module,
        s_img_feats: Optional[Tensor], t_img_feats: Optional[Tensor],
        s_hidden: Tensor, t_hidden: Tensor, t_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Computes the VSD and VLAD losses for a single side (query or positive)."""
        if s_img_feats is None or t_img_feats is None:
            return torch.tensor(0.0, device=s_hidden.device), torch.tensor(0.0, device=s_hidden.device)

        num_s_vision = s_img_feats.size(0)
        num_t_vision = t_img_feats.size(0)
        num_t_text = ((t_ids < BOS_TOKEN_ID) | (t_ids > VISION_END_TOKEN_ID)).sum()

        # For LLaVA-OV, vision tokens are at the end of the sequence for both student and teacher.
        s_vision_hidden = s_hidden[-(num_s_vision + num_t_text):-num_t_text, :]
        t_vision_hidden = t_hidden[-(num_t_vision + num_t_text):-num_t_text, :]
        s_text_hidden = s_hidden[-num_t_text:, :]
        t_text_hidden = t_hidden[-num_t_text:, :]

        s_vision_logits = student_model.encoder.lm_head(s_vision_hidden)
        t_vision_logits = teacher_model.encoder.lm_head(t_vision_hidden)
        
        # Pad logits to handle vocabulary size mismatches before matching
        if s_vision_logits.size(1) != t_vision_logits.size(1):
            diff = s_vision_logits.size(1) - t_vision_logits.size(1)
            if diff > 0:
                t_vision_logits = F.pad(t_vision_logits, (0, diff), 'constant', 0)
            else:
                s_vision_logits = F.pad(s_vision_logits, (0, -diff), 'constant', 0)

        # Match student and teacher vision tokens using the Hungarian algorithm
        cost_matrix = torch.cdist(t_vision_logits, s_vision_logits, p=1).float().detach().cpu().numpy()
        t_indices, s_indices = linear_sum_assignment(cost_matrix)
        
        vsd_loss = nn.MSELoss()(s_vision_logits[s_indices], t_vision_logits[t_indices])

        # Compute vision-language alignment distillation loss
        s_affinity = F.cosine_similarity(s_vision_hidden[s_indices].unsqueeze(1), s_text_hidden.unsqueeze(0), dim=-1)
        t_affinity = F.cosine_similarity(t_vision_hidden[t_indices].unsqueeze(1), t_text_hidden.unsqueeze(0), dim=-1)
        vlad_loss = F.smooth_l1_loss(s_affinity, t_affinity)
        
        return vsd_loss, vlad_loss
