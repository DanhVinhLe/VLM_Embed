"""
This module defines the HoloDistillLoss, a criterion for hybrid geometry distillation
using spectral decoupling, designed for Matryoshka-style models.
"""
import logging
from typing import Any, Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..arguments import HoloDistillArguments

logger = logging.getLogger(__name__)

class HoloDistillLoss(nn.Module):
    """
    Implements HoloDistill, a hybrid geometry distillation method optimized for models
    with Matryoshka Representation Learning. It decouples global and local structure
    distillation across different embedding dimensions (slices).

    This loss addresses the "capacity conflict" between global (e.g., ImageNet classification)
    and local (e.g., dense object detection) tasks by applying different distillation
    strategies to different spectral slices of the embeddings:

    1.  **Global RKD (Relational Knowledge Distillation):** Applied to ALL embedding slices
        to enforce consistent global semantic relationships. It uses projected features at
        the full dimension to allow the student's raw embeddings to remain flexible.
        A spectral multiplier gives more weight to the core (low-dimension) slice to
        ensure a stable global structure.

    2.  **Local Holo (Fused Gromov-Wasserstein):** Applied ONLY to high-dimension slices
        to refine the dense, local topological structure without corrupting the global
        indexing capabilities of the core slice.
    """

    def __init__(self, args: Type[HoloDistillArguments]):
        super(HoloDistillLoss, self).__init__()
        self.args = args
        self.temperature: float = args.temperature
        self.alpha: float = args.holo_alpha
        self.ot_epsilon: float = args.ot_epsilon
        self.ot_iters: int = args.ot_iters
        self.holo_weight: float = args.holo_weight
        self.rkd_distance_weight: float = args.rkd_distance_weight
        self.rkd_angle_weight: float = args.rkd_angle_weight
        self.global_rkd_weight: float = args.global_rkd_weight
        self.matryoshka_dims: List[int] = args.matryoshka_dims
        self.matryoshka_weights: List[float] = args.matryoshka_weights

        # Internal spectral multipliers to emphasize the core slice for RKD.
        self.spectral_multipliers: List[float] = [2.0, 1.0]

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Computes the combined HoloDistill loss across all Matryoshka dimensions."""
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher
        
        # --- 1. Get Teacher and Student Representations ---
        with torch.no_grad():
            teacher_model.eval()
            t_q_reps, _, _, t_q_states = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_p_reps, _, _, t_p_states = teacher_model.encode_input(input_data['teacher_inputs']['pos'])
            if isinstance(t_q_states, (tuple, list)): t_q_states, t_p_states = t_q_states[-1], t_p_states[-1]

        s_q_reps, _, _, s_q_states = student_model.encode_input(input_data['student_inputs']['qry'])
        s_p_reps, _, _, s_p_states = student_model.encode_input(input_data['student_inputs']['pos'])
        if isinstance(s_q_states, (tuple, list)): s_q_states, s_p_states = s_q_states[-1], s_p_states[-1]

        contrastive_loss, holo_loss, rkd_loss = 0.0, 0.0, 0.0
        full_dim = s_q_reps.shape[-1]
        
        # --- 2. Matryoshka Loop for Spectral Decoupling ---
        for i, dim in enumerate(self.matryoshka_dims):
            dim_weight = self.matryoshka_weights[i] if i < len(self.matryoshka_weights) else 1.0
            spec_w = self.spectral_multipliers[i] if i < len(self.spectral_multipliers) else 1.0
            current_dim = dim if dim is not None else full_dim
            
            # Slice and normalize embeddings for the current dimension
            s_q_slice = F.normalize(s_q_reps[:, :current_dim], p=2, dim=-1)
            s_p_slice = F.normalize(s_p_reps[:, :current_dim], p=2, dim=-1)
            t_q_slice = F.normalize(t_q_reps[:, :current_dim], p=2, dim=-1)
            t_p_slice = F.normalize(t_p_reps[:, :current_dim], p=2, dim=-1)

            # --- 2a. Contrastive Loss (Uniform Spectrum) ---
            scores = torch.mm(s_q_slice, s_p_slice.t()) / self.temperature
            labels = torch.arange(scores.size(0), device=scores.device)
            contrastive_loss += dim_weight * nn.CrossEntropyLoss()(scores, labels)

            # --- 2b. Global RKD (Applied to all slices) ---
            projector = getattr(distiller, 'holo_projector', None)
            s_batch_for_rkd = torch.cat([s_q_slice, s_p_slice], dim=0)
            if projector and current_dim == full_dim: # Use projected RKD only at full dimension
                 s_q_proj = F.normalize(projector(s_q_reps), p=2, dim=-1)
                 s_p_proj = F.normalize(projector(s_p_reps), p=2, dim=-1)
                 s_batch_for_rkd = torch.cat([s_q_proj, s_p_proj], dim=0)

            t_batch = torch.cat([t_q_slice, t_p_slice], dim=0)
            rkd_dist = self._compute_rkd_distance_loss(s_batch_for_rkd, t_batch)
            rkd_ang = self._compute_rkd_angle_loss(s_batch_for_rkd, t_batch)
            rkd_val = self.rkd_distance_weight * rkd_dist + self.rkd_angle_weight * rkd_ang
            rkd_loss += (self.global_rkd_weight * spec_w) * rkd_val 

            # --- 2c. Local Holo/FGW (Applied to high-dim slices) ---
            if projector and (current_dim is None or current_dim > 128):
                fgw_q = self._fused_gromov_wasserstein(s_q_states, t_q_states, projector)
                fgw_p = self._fused_gromov_wasserstein(s_p_states, t_p_states, projector)
                holo_loss += (self.holo_weight * spec_w) * (fgw_q + fgw_p) / 2

        total_loss = contrastive_loss + holo_loss + rkd_loss
        return {"loss": total_loss, "contrastive_loss": contrastive_loss, "holo_fgw_loss": holo_loss, "global_rkd_loss": rkd_loss}

    def _compute_rkd_distance_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes RKD Distance Loss using pairwise Euclidean distances with Huber-like penalty."""
        dist_s = torch.pdist(s, p=2).pow(2)
        dist_t = torch.pdist(t, p=2).pow(2)
        dist_s = dist_s / (dist_s.mean().detach() + 1e-8)
        dist_t = dist_t / (dist_t.mean().detach() + 1e-8)
        return F.smooth_l1_loss(dist_s, dist_t)

    def _compute_rkd_angle_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes RKD Angle Loss using cosine similarity of triplets."""
        def angle_potentials(x):
            diffs = x.unsqueeze(0) - x.unsqueeze(1)
            e = F.normalize(diffs, p=2, dim=-1)
            return torch.einsum('ijd,kjd->ijk', e, e)
        
        psi_s, psi_t = angle_potentials(s), angle_potentials(t)
        n = psi_s.size(0)
        mask = 1 - torch.eye(n, device=s.device)
        mask = mask.unsqueeze(0) * mask.unsqueeze(1) * mask.unsqueeze(2)
        psi_s, psi_t = psi_s[mask.bool()], psi_t[mask.bool()]
        return F.smooth_l1_loss(psi_s, psi_t)

    def _fused_gromov_wasserstein(self, z_s: torch.Tensor, z_t: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        """Calculates the Fused Gromov-Wasserstein alignment loss."""
        # 1. Saliency and Structure Matrices
        mu = F.softmax(torch.norm(z_s, p=2, dim=-1) / 0.1, dim=-1).unsqueeze(-1)
        nu = F.softmax(torch.norm(z_t, p=2, dim=-1) / 0.1, dim=-1).unsqueeze(-1)
        C_s = 1.0 - torch.bmm(F.normalize(z_s, p=2, dim=-1), F.normalize(z_s, p=2, dim=-1).transpose(1, 2))
        C_t = 1.0 - torch.bmm(F.normalize(z_t, p=2, dim=-1), F.normalize(z_t, p=2, dim=-1).transpose(1, 2))

        # 2. Feature Cost (Ground Metric)
        z_s_proj = projector(z_s)
        M = 1.0 - torch.bmm(F.normalize(z_s_proj, p=2, dim=-1), F.normalize(z_t, p=2, dim=-1).transpose(1, 2))
        M = M / (M.mean() + 1e-8)
        
        # 3. Solve Optimal Transport with Sinkhorn-Knopp
        log_K = -M / self.ot_epsilon
        u, v = torch.zeros_like(mu), torch.zeros_like(nu)
        for _ in range(self.ot_iters):
            u = torch.log(mu + 1e-8) - torch.logsumexp(log_K + v.transpose(1, 2), dim=2, keepdim=True)
            v = torch.log(nu + 1e-8) - torch.logsumexp(log_K.transpose(1, 2) + u.transpose(1, 2), dim=2, keepdim=True)
        Gamma = torch.exp(u + log_K + v.transpose(1, 2))
        
        # 4. Compute Losses
        feat_loss = torch.sum(M * Gamma, dim=(1, 2)).mean()
        Gamma_norm = Gamma / (Gamma.sum(dim=2, keepdim=True) + 1e-8)
        C_t_mapped = torch.bmm(torch.bmm(Gamma_norm, C_t), Gamma_norm.transpose(1, 2))
        struct_loss = torch.sum(((C_s - C_t_mapped) ** 2) * (mu @ mu.transpose(1, 2)), dim=(1, 2)).mean()

        return (1 - self.alpha) * feat_loss + self.alpha * struct_loss
