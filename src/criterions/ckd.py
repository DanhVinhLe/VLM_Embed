"""
This module defines the Contrastive Knowledge Distillation (CKD) loss.
"""
import logging
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..arguments import CKDArguments

logger = logging.getLogger(__name__)

class CKDLoss(nn.Module):
    """
    Implements the Contrastive Knowledge Distillation (CKD) loss.
    This loss combines a standard contrastive loss with a distillation loss that aims to
    preserve the pairwise relationships between embeddings within a batch.
    
    The total loss is a weighted sum of:
    1.  **Contrastive Loss (InfoNCE):** Applied to the student's embeddings to learn
        discriminative representations.
    2.  **Pairwise MSE Loss:** A distillation loss calculated as the Mean Squared Error
        between the pairwise differences of the student's embeddings and the teacher's
        projected embeddings. This encourages the student to learn the relational
        structure of the teacher's embedding space.
    """
    def __init__(self, args: Type[CKDArguments]):
        """
        Initializes the CKDLoss module.
        Args:
            args: Configuration arguments, expected to contain `kd_weight`.
        """
        super(CKDLoss, self).__init__()
        self.args = args
        self.kd_loss_weight: float = self.args.kd_weight
        
    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Computes the combined CKD loss."""
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher
        
        # --- 1. Get Student and Teacher Representations ---
        with torch.no_grad():
            teacher_model.eval()
            t_qry_reps, _, _, _ = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos_reps, _, _, _ = teacher_model.encode_input(input_data['teacher_inputs']['pos'])

        s_qry_reps, _, _, _ = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos_reps, _, _, _ = student_model.encode_input(input_data['student_inputs']['pos'])

        # --- 2. Compute Contrastive Loss ---
        scores = student_model.compute_similarity(s_qry_reps, s_pos_reps)
        scores = scores.view(s_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (s_pos_reps.size(0) // s_qry_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)
        
        # --- 3. Compute Distillation Loss ---
        distillation_loss = self._compute_pairwise_mse_loss(s_qry_reps, t_qry_reps, distiller)
        
        # --- 4. Combine Losses ---
        total_loss = contrastive_loss + self.kd_loss_weight * distillation_loss

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": distillation_loss,
        }
    
    def _compute_pairwise_mse_loss(
        self, student_embs: torch.Tensor, teacher_embs: torch.Tensor, distiller: nn.Module
    ) -> torch.Tensor:
        """
        Computes the MSE on pairwise differences between student and projected teacher embeddings.
        """
        num_samples = student_embs.size(0)
        if num_samples <= 1:
            return 0.0

        # Project teacher embeddings to match student's dimension
        teacher_embs_proj = distiller.projectors["t2s"](teacher_embs)
        
        # Calculate pairwise differences for student and teacher embeddings
        student_diffs = student_embs.unsqueeze(1) - student_embs.unsqueeze(0)
        teacher_diffs = teacher_embs_proj.unsqueeze(1) - teacher_embs_proj.unsqueeze(0)

        # Calculate MSE of the differences
        per_pair_mse = ((student_diffs - teacher_diffs) ** 2).mean(dim=2)

        # Exclude self-comparisons from the loss
        mask = ~torch.eye(num_samples, dtype=torch.bool, device=student_embs.device)
        
        return per_pair_mse.masked_select(mask).mean()
