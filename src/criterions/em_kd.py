"""
This module defines the Exact Matching Knowledge Distillation (EMKD) loss,
which aligns student and teacher models by matching vision and text tokens.
"""
import logging
from typing import Any, Dict, Tuple, Type

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
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151656
BOS_TOKEN_ID = 151643

class EMKDLoss(nn.Module):
    """
    Implements Exact Matching Knowledge Distillation (EMKD) loss.
    This loss combines several components to distill knowledge from a teacher to a student:
    1.  **Contrastive Loss:** A standard InfoNCE loss on the final embeddings.
    2.  **MSE Loss:** A simple MSE loss on the projected global representations.
    3.  **Vision Semantic Distillation (VSD):** An MSE loss on the vocabulary logits of matched
        vision tokens between student and teacher. The matching is done via the Hungarian
        algorithm (linear sum assignment).
    4.  **Vision-Language Alignment Distillation (VLAD):** A smooth L1 loss on the cosine
        similarity (affinity) between matched vision tokens and all text tokens. This ensures
        that the student learns the teacher's vision-language alignment.
    """

    def __init__(self, args: Type[EMKDArguments]):
        super(EMKDLoss, self).__init__()
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
        """Computes the combined EMKD loss."""
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
            # Query-side distillation
            vsd_qry, vlad_qry = self._compute_emkd_for_side(
                student_model, teacher_model,
                s_qry_img_feats[i] if s_qry_img_feats else None,
                t_qry_img_feats[i] if t_qry_img_feats else None,
                s_qry_hiddens[-1][i], t_qry_hiddens[-1][i],
                input_data['teacher_inputs']['qry']['input_ids'][i]
            )
            total_vsd_loss += vsd_qry
            total_vlad_loss += vlad_qry

            # Positive-side distillation
            vsd_pos, vlad_pos = self._compute_emkd_for_side(
                student_model, teacher_model,
                s_pos_img_feats[i] if s_pos_img_feats else None,
                t_pos_img_feats[i] if t_pos_img_feats else None,
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
        s_img_feats: Tensor, t_img_feats: Tensor,
        s_hidden: Tensor, t_hidden: Tensor, t_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Computes the VSD and VLAD losses for a single side (query or positive)."""
        if s_img_feats is None or t_img_feats is None:
            return torch.tensor(0.0, device=s_hidden.device), torch.tensor(0.0, device=s_hidden.device)

        num_s_vision = s_img_feats.size(0)
        num_t_vision = t_img_feats.size(0)
        num_t_text = ((t_ids < BOS_TOKEN_ID) | (t_ids > VISION_END_TOKEN_ID)).sum()

        # Extract relevant hidden states for vision and text
        s_vision_hidden = s_hidden[:num_s_vision, :]
        t_vision_hidden = t_hidden[-(num_t_vision + num_t_text):-num_t_text, :]
        s_text_hidden = s_hidden[num_s_vision:num_s_vision + num_t_text, :]
        t_text_hidden = t_hidden[-num_t_text:, :]

        # Get vocabulary logits for vision tokens
        s_vision_logits = student_model.encoder.lm_head(s_vision_hidden)
        t_vision_logits = teacher_model.encoder.lm_head(t_vision_hidden)
        
        # Match student and teacher vision tokens using the Hungarian algorithm
        cost_matrix = torch.cdist(t_vision_logits, s_vision_logits, p=1).float().detach().cpu().numpy()
        t_indices, s_indices = linear_sum_assignment(cost_matrix)
        
        s_logits_matched = s_vision_logits[s_indices]
        t_logits_matched = t_vision_logits[t_indices]
        vsd_loss = nn.MSELoss()(s_logits_matched, t_logits_matched)

        # Compute vision-language alignment distillation loss
        s_vision_matched = s_vision_hidden[s_indices]
        t_vision_matched = t_vision_hidden[t_indices]

        s_affinity = F.cosine_similarity(s_vision_matched.unsqueeze(1), s_text_hidden.unsqueeze(0), dim=-1)
        t_affinity = F.cosine_similarity(t_vision_matched.unsqueeze(1), t_text_hidden.unsqueeze(0), dim=-1)
        vlad_loss = F.smooth_l1_loss(s_affinity, t_affinity)
        
        return vsd_loss, vlad_loss