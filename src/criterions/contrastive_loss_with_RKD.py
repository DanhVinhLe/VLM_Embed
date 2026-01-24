"This module defines a criterion that combines Contrastive Loss with Relational Knowledge Distillation (RKD)."
import logging
from typing import Dict, Any, Type

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..arguments import RKDArguments

logger = logging.getLogger(__name__)


class ContrastiveLossWithRKD(nn.Module):
    """
    Computes Contrastive Loss combined with Relational Knowledge Distillation (RKD) Loss.
    
    This loss is designed for knowledge distillation in embedding models. It aligns the student's
    embedding space with the teacher's by using both a standard contrastive loss and RKD, which
    preserves the relational structure of the embeddings.

    The total loss is a weighted sum of:
    1.  **Contrastive Loss (InfoNCE):** Applied to the student's output embeddings to ensure that
        semantically similar inputs (query-positive pairs) are pulled together in the embedding space,
        while dissimilar ones are pushed apart.
    2.  **RKD Distance Loss:** Penalizes the difference in pairwise Euclidean distances between embeddings
        from the student and teacher models. This encourages the student to learn the relative distances
        between data points that the teacher model has learned.
    3.  **RKD Angle Loss:** Penalizes the difference in angles formed by triplets of embeddings from the
        student and teacher. This helps the student to capture the angular relationships and preserve
        the geometric structure of the embedding space.
    """

    def __init__(self, args: Type[RKDArguments]):
        """
        Initializes the ContrastiveLossWithRKD module.

        Args:
            args: A configuration object with arguments for the loss function. It must contain:
                  - kd_weight: The weight for the overall RKD loss.
                  - rkd_distance_weight: The weight for the RKD distance loss component.
                  - rkd_angle_weight: The weight for the RKD angle loss component.
        """
        super(ContrastiveLossWithRKD, self).__init__()
        self.args = args
        self.kd_loss_weight: float = self.args.kd_weight
        self.distance_weight: float = self.args.rkd_distance_weight
        self.angle_weight: float = self.args.rkd_angle_weight

    def forward(
        self, distiller: nn.Module, input_data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the total distillation loss.

        Args:
            distiller: The distillation wrapper module containing the student and teacher models.
            input_data: A dictionary containing the inputs for both student and teacher.
                        Expected keys: 'student_inputs' and 'teacher_inputs', each with 'qry' and 'pos' data.

        Returns:
            A dictionary containing the computed losses:
            - "loss": The total combined loss.
            - "contrastive_loss": The InfoNCE contrastive loss on student embeddings.
            - "kd_loss": The combined RKD (distance and angle) loss.
        """
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher

        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']

        teacher_input_qry = input_data['teacher_inputs']['qry']
        teacher_input_pos = input_data['teacher_inputs']['pos']

        # Get teacher representations in no_grad context
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_reps, _, _, _ = teacher_model.encode_input(teacher_input_qry)
            teacher_pos_reps, _, _, _ = teacher_model.encode_input(teacher_input_pos)

        # Get student representations
        student_qry_reps, _, _, _ = student_model.encode_input(student_input_qry)
        student_pos_reps, _, _, _ = student_model.encode_input(student_input_pos)

        # 1. Compute Contrastive Loss on student embeddings
        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        # Adjust target for cases where there are multiple positives per query
        target = target * (student_pos_reps.size(0) // student_qry_reps.size(0))
        
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # 2. Compute RKD Loss
        distance_loss = self.compute_distance_loss(
            student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps
        )
        angle_loss = self.compute_angle_loss(
            student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps
        )

        # Combine RKD losses with their respective weights
        kd_loss = (self.distance_weight * distance_loss + self.angle_weight * angle_loss)

        # 3. Combine contrastive loss and KD loss
        total_loss = contrastive_loss + self.kd_loss_weight * kd_loss
        
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }

    def _pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix of pairwise Euclidean distances between rows of a tensor.

        Args:
            x: Input tensor of shape (N, D), where N is the number of vectors.

        Returns:
            A symmetric distance matrix of shape (N, N).
        """
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
        return torch.sqrt(torch.clamp(dist, min=0.0)) # Clamp for numerical stability

    def compute_distance_loss(
        self, 
        student_qry: torch.Tensor, 
        student_pos: torch.Tensor, 
        teacher_qry: torch.Tensor, 
        teacher_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the RKD Distance-wise Loss. This loss encourages the student to preserve the
        pairwise distances of the teacher's embeddings.

        Args:
            student_qry: Student's query embeddings.
            student_pos: Student's positive embeddings.
            teacher_qry: Teacher's query embeddings.
            teacher_pos: Teacher's positive embeddings.

        Returns:
            A scalar tensor representing the mean distance loss.
        """
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)

        dist_student = self._pairwise_distance(student_repr)
        dist_teacher = self._pairwise_distance(teacher_repr)

        # We only need one triangle of the distance matrix (excluding the diagonal)
        mask = torch.triu(torch.ones_like(dist_student), diagonal=1).bool()
        dist_student = dist_student[mask]
        dist_teacher = dist_teacher[mask]

        # Normalize distances to be scale-invariant
        dist_teacher_mean = dist_teacher.mean().detach() + 1e-8
        dist_student_mean = dist_student.mean().detach() + 1e-8

        normalized_dist_teacher = dist_teacher / dist_teacher_mean
        normalized_dist_student = dist_student / dist_student_mean

        # Use Huber Loss (Smooth L1 Loss)
        return nn.functional.smooth_l1_loss(normalized_dist_student, normalized_dist_teacher)

    def _angle_potentials(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity of angles formed by triplets of vectors.
        For each triplet (i, j, k), it computes the cosine of the angle at vertex i.

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            A tensor of cosine angles of shape (N, N, N).
        """
        # Create vectors between all pairs of points: diffs[i, j, :] = x[i] - x[j]
        diffs = x.unsqueeze(1) - x.unsqueeze(0)
        # Normalize these vectors to get unit vectors
        norms = torch.norm(diffs, p=2, dim=-1, keepdim=True) + 1e-8
        unit_vectors = diffs / norms

        # Compute cosine similarity (dot product of unit vectors)
        # cos_angles[i, j, k] = unit_vectors[i, j] . unit_vectors[i, k]
        cos_angles = torch.einsum('ijd,ikd->ijk', unit_vectors, unit_vectors)
        return cos_angles

    def compute_angle_loss(
        self, 
        student_qry: torch.Tensor, 
        student_pos: torch.Tensor, 
        teacher_qry: torch.Tensor, 
        teacher_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the RKD Angle-wise Loss. This loss encourages the student to preserve the
        angles between triplets of the teacher's embeddings.

        Args:
            student_qry: Student's query embeddings.
            student_pos: Student's positive embeddings.
            teacher_qry: Teacher's query embeddings.
            teacher_pos: Teacher's positive embeddings.

        Returns:
            A scalar tensor representing the mean angle loss.
        """
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)

        psi_student = self._angle_potentials(student_repr)
        psi_teacher = self._angle_potentials(teacher_repr)

        n = psi_student.size(0)
        
        # Create a mask to select only triplets with distinct indices (i, j, k)
        mask = 1.0 - torch.eye(n, device=psi_student.device)
        mask = mask.unsqueeze(0).expand(n, -1, -1) * \
               mask.unsqueeze(1).expand(-1, n, -1) * \
               mask.unsqueeze(2).expand(-1, -1, n)
        
        psi_student_masked = torch.masked_select(psi_student, mask.bool())
        psi_teacher_masked = torch.masked_select(psi_teacher, mask.bool())

        # Use Huber Loss (Smooth L1 Loss)
        return nn.functional.smooth_l1_loss(psi_student_masked, psi_teacher_masked)