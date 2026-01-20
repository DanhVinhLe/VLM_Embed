import logging
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ContrastiveLossWithRKD(nn.Module):
    """
    Computes Contrastive Loss combined with Relational Knowledge Distillation (RKD) Loss.
    
    The loss consists of:
    1. Contrastive Loss (InfoNCE) on student representations.
    2. RKD Distance Loss: Penalizes differences in pairwise distances between student and teacher.
    3. RKD Angle Loss: Penalizes differences in angles formed by triplets between student and teacher.
    """

    def __init__(self, args: Any):
        """
        Initializes the ContrastiveLossWithRKD module.

        Args:
            args: Configuration arguments containing weights for KD, distance, and angle losses.
        """
        super(ContrastiveLossWithRKD, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        self.distance_weight = self.args.rkd_distance_weight
        self.angle_weight = self.args.rkd_angle_weight

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the total loss.

        Args:
            distiller: The distillation wrapper containing student and teacher models.
            input_data: Dictionary containing 'student_inputs' and 'teacher_inputs'.

        Returns:
            A dictionary containing 'loss' (total), 'contrastive_loss', and 'kd_loss'.
        """
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']

        teacher_input_qry = input_data['teacher_inputs']['qry']
        teacher_input_pos = input_data['teacher_inputs']['pos']

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_reps, _, _, output_hidden_states = teacher_model.encode_input(teacher_input_qry)
            teacher_pos_reps, _, _, output_hidden_states = teacher_model.encode_input(teacher_input_pos)

        student_qry_reps, _, _, output_hidden_states = student_model.encode_input(student_input_qry)
        student_pos_reps, _, _, output_hidden_states = student_model.encode_input(student_input_pos)

        # Compute Contrastive Loss
        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_qry_reps.size(0) // student_pos_reps.size(0))
        
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)

        # Compute RKD Loss
        distance_loss = self.compute_distance_loss(student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps)
        angle_loss = self.compute_angle_loss(student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps)

        kd_loss = (0.5 * distance_loss + 0.5 * angle_loss)

        total_loss = contrastive_loss + self.kd_loss_weight * kd_loss
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }

    def pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise Euclidean distances between rows of x.

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            Distance matrix of shape (N, N).
        """
        norm = (x**2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2.0 * torch.mm(x, x.t())
        return dist

    def compute_distance_loss(self, 
                              student_qry: torch.Tensor, 
                              student_pos: torch.Tensor, 
                              teacher_qry: torch.Tensor, 
                              teacher_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the RKD Distance Loss.

        Args:
            student_qry: Student query representations.
            student_pos: Student positive representations.
            teacher_qry: Teacher query representations.
            teacher_pos: Teacher positive representations.

        Returns:
            Scalar distance loss.
        """
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)

        dist_student = self.pairwise_distance(student_repr)
        dist_teacher = self.pairwise_distance(teacher_repr)

        # Use upper triangle only to avoid redundancy and self-distance (0)
        mask = torch.triu(torch.ones_like(dist_student), diagonal=1).bool()
        dist_student = dist_student[mask]
        dist_teacher = dist_teacher[mask]

        # Normalize distances by mean
        mean_td = dist_teacher.mean().detach() + 1e-8
        mean_sd = dist_student.mean().detach() + 1e-8

        dist_student = dist_student / mean_sd
        dist_teacher = dist_teacher / mean_td

        # Huber Loss-like calculation
        diff = dist_student - dist_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5

        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        return loss.mean()

    def angle_potentials(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between difference vectors (angles).

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            Tensor of cosine angles of shape (N, N, N).
        """
        # diffs[i, j, :] = x[i] - x[j]
        diffs = x.unsqueeze(0) - x.unsqueeze(1)
        norms = torch.norm(diffs, dim=-1, keepdim=True) + 1e-8
        e = diffs / norms

        # cos_angles[i, j, k] = e[i, j] . e[i, k]
        # This represents the angle at vertex i formed by vectors to j and k
        cos_angles = torch.einsum('ijd,kjd->ijk', e, e)
        return cos_angles

    def compute_angle_loss(self, 
                           student_qry: torch.Tensor, 
                           student_pos: torch.Tensor, 
                           teacher_qry: torch.Tensor, 
                           teacher_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the RKD Angle Loss.

        Args:
            student_qry: Student query representations.
            student_pos: Student positive representations.
            teacher_qry: Teacher query representations.
            teacher_pos: Teacher positive representations.

        Returns:
            Scalar angle loss.
        """
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)

        psi_student = self.angle_potentials(student_repr)
        psi_teacher = self.angle_potentials(teacher_repr)

        n = psi_student.size(0)
        
        # Mask out cases where i, j, k are not distinct
        mask = torch.ones((n, n, n), dtype=torch.bool, device=psi_student.device)
        idx = torch.arange(n, device=psi_student.device)
        mask[idx, idx, :] = 0
        mask[idx, :, idx] = 0
        mask[:, idx, idx] = 0

        psi_teacher = psi_teacher[mask]
        psi_student = psi_student[mask]

        diff = psi_student - psi_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        
        return loss.mean()

        