import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class UniversalLogitDistillation(nn.Module):
    """
    Implements Universal Logit Distillation (ULD) loss.
    """
    def __init__(self, args: Any):
        super(UniversalLogitDistillation, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        
    def forward(self, distiller: Any, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes the ULD loss.
        """
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']
        
        teacher_input_qry = input_data['teacher_inputs']['qry']
        teacher_input_pos = input_data['teacher_inputs']['pos']
        
        with torch.no_grad():
            teacher_qry_reps, _, _ = teacher_model.encode_input(teacher_input_qry)
            teacher_pos_reps, _, _ = teacher_model.encode_input(teacher_input_pos)

        student_qry_reps, _, _ = student_model.encode_input(student_input_qry)
        student_pos_reps, _, _ = student_model.encode_input(student_input_pos)

        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_qry_reps.size(0) // student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)
        uld_loss = self.compute_universal_logit_loss(student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps)
        kd_loss = self.kd_loss_weight * uld_loss
        
        total_loss = contrastive_loss + kd_loss
        return {
            "loss": total_loss, 
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }

    def compute_universal_logit_loss(self, student_qry_reps: torch.Tensor, student_pos_reps: torch.Tensor, teacher_qry_reps: torch.Tensor, teacher_pos_reps: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mean Squared Error between student and teacher representations, handling dimension mismatches by padding.
        """
        size_gap = student_qry_reps.shape[-1] - teacher_qry_reps.shape[-1]
        if size_gap > 0:
            teacher_qry_reps = torch.cat(
                [teacher_qry_reps, torch.zeros_like(teacher_qry_reps[:, :size_gap])], dim=-1
            )
            teacher_pos_reps = torch.cat(
                [teacher_pos_reps, torch.zeros_like(teacher_pos_reps[:, :size_gap])], dim=-1
            )
        elif size_gap < 0:
            student_qry_reps = torch.cat(
                [student_qry_reps, torch.zeros_like(student_qry_reps[:, :(-size_gap)])], dim=-1
            )
            student_pos_reps = torch.cat(
                [student_pos_reps, torch.zeros_like(student_pos_reps[:, :(-size_gap)])], dim=-1
            )

        uld_loss = (F.mse_loss(student_qry_reps, teacher_qry_reps) + F.mse_loss(student_pos_reps, teacher_pos_reps) + F.mse_loss(student_qry_reps, teacher_pos_reps) + F.mse_loss(student_pos_reps, teacher_qry_reps)) / 4.0
        return uld_loss