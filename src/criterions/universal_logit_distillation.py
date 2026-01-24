"""
This module defines the Universal Logit Distillation (ULD) criterion.
"""
import logging
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..arguments import DistillationArguments

logger = logging.getLogger(__name__)

class UniversalLogitDistillation(nn.Module):
    """
    Implements Universal Logit Distillation (ULD) loss, which combines a standard contrastive
    loss with a distillation loss based on the mean squared error (MSE) between student and
    teacher embeddings.

    The "universal" aspect refers to its ability to handle mismatches in embedding dimensions
    between the student and teacher models by padding the smaller embedding to match the
    larger one.

    The total loss is a weighted sum of:
    1.  **Contrastive Loss (InfoNCE):** Ensures that the student model learns to produce
        discriminative embeddings.
    2.  **ULD Loss:** An MSE-based loss that aligns the student's embeddings with the teacher's.
        It computes the average MSE across all four pairs of student/teacher query/positive
        embeddings (student_qry-teacher_qry, student_pos-teacher_pos, student_qry-teacher_pos,
        and student_pos-teacher_qry). This encourages a broad alignment of the embedding spaces.
    """

    def __init__(self, args: Type[DistillationArguments]):
        """
        Initializes the UniversalLogitDistillation module.

        Args:
            args: A configuration object containing the `kd_weight` for the distillation loss.
        """
        super(UniversalLogitDistillation, self).__init__()
        self.args = args
        self.kd_loss_weight: float = self.args.kd_weight
        
    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes the total ULD loss.

        Args:
            distiller: The distillation wrapper module with student and teacher models.
            input_data: A dictionary with 'student_inputs' and 'teacher_inputs'.

        Returns:
            A dictionary containing the "loss", "contrastive_loss", and "kd_loss".
        """
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher
        
        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']
        
        teacher_input_qry = input_data['teacher_inputs']['qry']
        teacher_input_pos = input_data['teacher_inputs']['pos']
        
        # Get teacher representations
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_reps, _, _, _ = teacher_model.encode_input(teacher_input_qry)
            teacher_pos_reps, _, _, _ = teacher_model.encode_input(teacher_input_pos)

        # Get student representations
        student_qry_reps, _, _, _ = student_model.encode_input(student_input_qry)
        student_pos_reps, _, _, _ = student_model.encode_input(student_input_pos)

        # 1. Compute standard contrastive loss for the student
        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_pos_reps.size(0) // student_qry_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # 2. Compute the Universal Logit Distillation loss
        uld_loss = self.compute_universal_logit_loss(
            student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps
        )
        kd_loss = self.kd_loss_weight * uld_loss
        
        # 3. Combine losses
        total_loss = contrastive_loss + kd_loss
        
        return {
            "loss": total_loss, 
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }

    def compute_universal_logit_loss(
        self, 
        student_qry_reps: torch.Tensor, 
        student_pos_reps: torch.Tensor, 
        teacher_qry_reps: torch.Tensor, 
        teacher_pos_reps: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the MSE between student and teacher representations, handling dimension
        mismatches by padding with zeros.

        The loss is the average MSE over all four combinations of query and positive embeddings
        between the student and the teacher.

        Args:
            student_qry_reps: Student query embeddings.
            student_pos_reps: Student positive embeddings.
            teacher_qry_reps: Teacher query embeddings.
            teacher_pos_reps: Teacher positive embeddings.

        Returns:
            The calculated ULD loss as a scalar tensor.
        """
        s_dim = student_qry_reps.shape[-1]
        t_dim = teacher_qry_reps.shape[-1]
        
        # Pad embeddings to match dimensions if they differ
        if s_dim != t_dim:
            if s_dim > t_dim:
                # Pad teacher embeddings
                padding = torch.zeros_like(teacher_qry_reps[:, :(s_dim - t_dim)])
                teacher_qry_reps = torch.cat([teacher_qry_reps, padding], dim=-1)
                teacher_pos_reps = torch.cat([teacher_pos_reps, padding], dim=-1)
            else:
                # Pad student embeddings
                padding = torch.zeros_like(student_qry_reps[:, :(t_dim - s_dim)])
                student_qry_reps = torch.cat([student_qry_reps, padding], dim=-1)
                student_pos_reps = torch.cat([student_pos_reps, padding], dim=-1)

        # Calculate MSE loss for all four pairs
        loss_qq = F.mse_loss(student_qry_reps, teacher_qry_reps)
        loss_pp = F.mse_loss(student_pos_reps, teacher_pos_reps)
        loss_qp = F.mse_loss(student_qry_reps, teacher_pos_reps)
        loss_pq = F.mse_loss(student_pos_reps, teacher_qry_reps)

        # Average the four losses
        uld_loss = (loss_qq + loss_pp + loss_qp + loss_pq) / 4.0
        return uld_loss
