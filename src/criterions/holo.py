import logging
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class HoloDistillLoss(nn.Module):
    """
    HoloDistill + Global RKD (Hybrid Geometry) via Spectral Decoupling.
    
    This criterion solves the "Capacity Conflict" between Global (ImageNet) and Local (VOC) tasks:
    1. Global RKD: Applied to ALL slices, but uses 'Projected' features for the Full Dimension 
       to prevent stiffening the manifold. Weighted heavier on the Core (Low-Dim) slice.
    2. Local Holo (FGW): Applied ONLY to High-Dimension slices to refine dense structural 
       topology without adding noise to the global index.
    """

    def __init__(self, args: Any):
        super(HoloDistillLoss, self).__init__()
        self.args = args
        self.temperature = getattr(args, "temperature", 0.07)
        
        # --- Holo/FGW Hyperparameters (Local Structure) ---
        # alpha=0.8: Focus 80% on Structure (Geometry), 20% on Features.
        self.alpha = getattr(args, "holo_alpha", 0.8) 
        self.ot_epsilon = getattr(args, "ot_epsilon", 0.1)
        self.ot_iters = getattr(args, "ot_iters", 20)
        # holo_weight=0.5: Moderate weight to prevent noise from overpowering Global RKD.
        self.holo_weight = getattr(args, "holo_weight", 0.5) 
        
        # --- RKD Hyperparameters (Global Semantics) ---
        self.rkd_distance_weight = getattr(args, "rkd_distance_weight", 1.0)
        self.rkd_angle_weight = getattr(args, "rkd_angle_weight", 2.0)
        # global_rkd_weight=2.0: Safe base weight for Full-Dimension RKD.
        self.global_rkd_weight = getattr(args, "global_rkd_weight", 2.0)

        # --- Matryoshka Config ---
        # Default to [128, 768] (Core, Detail) if not provided
        self.matryoshka_dims = getattr(args, "matryoshka_dims", [128, 768]) 
        
        # Argument weights for the overall loss contribution of each slice
        self.matryoshka_weights = getattr(args, "matryoshka_weights", [1.0] * len(self.matryoshka_dims))

        # Internal Spectral Weights: [Core Multiplier, Detail Multiplier]
        # We enforce 2x importance on the Core (128) slice for RKD stability.
        self.spectral_multipliers = [2.0, 1.0]

    # ==========================
    # Part 1: RKD Modules (Global)
    # ==========================
    def pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Computes pairwise Euclidean distances."""
        norm = (x**2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2.0 * torch.mm(x, x.t())
        return dist

    def compute_rkd_distance_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """RKD Distance Loss (Huber-like) for robust global spacing."""
        dist_s = self.pairwise_distance(s)
        dist_t = self.pairwise_distance(t)

        # Upper triangle only
        mask = torch.triu(torch.ones_like(dist_s), diagonal=1).bool()
        dist_s = dist_s[mask]
        dist_t = dist_t[mask]

        # Normalization (Crucial for stability across different dims)
        mean_s = dist_s.mean().detach() + 1e-8
        mean_t = dist_t.mean().detach() + 1e-8
        dist_s = dist_s / mean_s
        dist_t = dist_t / mean_t

        diff = dist_s - dist_t
        abs_diff = torch.abs(diff)
        # Huber loss to handle outliers in noisy batches (e.g., News)
        loss = torch.where(abs_diff < 1.0, 0.5 * (abs_diff ** 2), abs_diff - 0.5)
        return loss.mean()

    def angle_potentials(self, x: torch.Tensor) -> torch.Tensor:
        """Computes cosine angles between triplets."""
        # diffs[i, j, :] = x[i] - x[j]
        diffs = x.unsqueeze(0) - x.unsqueeze(1)
        norms = torch.norm(diffs, dim=-1, keepdim=True) + 1e-8
        e = diffs / norms
        # cos_angles[i, j, k] = e[i, j] . e[i, k]
        cos_angles = torch.einsum('ijd,kjd->ijk', e, e)
        return cos_angles

    def compute_rkd_angle_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """RKD Angle Loss for higher-order geometric consistency."""
        psi_s = self.angle_potentials(s)
        psi_t = self.angle_potentials(t)

        n = psi_s.size(0)
        # Mask distinct i,j,k
        mask = torch.ones((n, n, n), dtype=torch.bool, device=s.device)
        idx = torch.arange(n, device=s.device)
        mask[idx, idx, :] = 0
        mask[idx, :, idx] = 0
        mask[:, idx, idx] = 0

        psi_s = psi_s[mask]
        psi_t = psi_t[mask]

        diff = psi_s - psi_t
        abs_diff = torch.abs(diff)
        loss = torch.where(abs_diff < 1.0, 0.5 * (abs_diff ** 2), abs_diff - 0.5)
        return loss.mean()

    # ==========================
    # Part 2: Holo/FGW Modules (Local)
    # ==========================
    def get_saliency_measure(self, hidden_states: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """Computes importance (probability mass) of each token based on Norm."""
        norms = torch.norm(hidden_states, p=2, dim=-1)
        weights = F.softmax(norms / temp, dim=-1)
        return weights.unsqueeze(-1)

    def compute_structure_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Computes Intra-Relational Structure (Cosine Distance Matrix)."""
        z_norm = F.normalize(z, p=2, dim=-1)
        sim = torch.bmm(z_norm, z_norm.transpose(1, 2))
        return 1.0 - sim

    def sinkhorn_knopp_log_domain(self, C: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Solves Entropic Optimal Transport via Sinkhorn-Knopp."""
        B, N_s, N_t = C.shape
        log_K = -C / self.ot_epsilon
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        for _ in range(self.ot_iters):
            term = log_K + v.transpose(1, 2)
            u = torch.log(mu + 1e-8) - torch.logsumexp(term, dim=2, keepdim=True)
            term = log_K.transpose(1, 2) + u.transpose(1, 2)
            v = torch.log(nu + 1e-8) - torch.logsumexp(term, dim=2, keepdim=True)
            
        return torch.exp(u + log_K + v.transpose(1, 2))

    def fused_gromov_wasserstein(self, z_s: torch.Tensor, z_t: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        """Calculates Fused Gromov-Wasserstein alignment loss."""
        mu = self.get_saliency_measure(z_s)
        nu = self.get_saliency_measure(z_t)
        C_s = self.compute_structure_matrix(z_s)
        C_t = self.compute_structure_matrix(z_t)

        # Feature Cost (Ground Metric): Projected Student <-> Teacher
        z_s_proj = projector(z_s)
        z_s_norm = F.normalize(z_s_proj, p=2, dim=-1)
        z_t_norm = F.normalize(z_t, p=2, dim=-1)
        M = 1.0 - torch.bmm(z_s_norm, z_t_norm.transpose(1, 2))
        
        # Normalize M for numerical stability with C
        M = M / (M.mean() + 1e-8)

        # Solve OT
        Gamma = self.sinkhorn_knopp_log_domain(M, mu, nu)
        
        # Feature Loss (Wasserstein)
        feat_loss = torch.sum(M * Gamma, dim=(1, 2)).mean()
        
        # Structure Loss (Gromov) - Approximated via mapped structure MSE
        Gamma_norm = Gamma / (Gamma.sum(dim=2, keepdim=True) + 1e-8)
        C_t_mapped = torch.bmm(torch.bmm(Gamma_norm, C_t), Gamma_norm.transpose(1, 2))
        diff = (C_s - C_t_mapped) ** 2
        struct_loss = torch.sum(diff * (mu @ mu.transpose(1, 2)), dim=(1, 2)).mean()

        # Weighted Combination
        return (1 - self.alpha) * feat_loss + self.alpha * struct_loss

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        # Unpack inputs
        s_input_qry = input_data['student_inputs']['qry']
        s_input_pos = input_data['student_inputs']['pos']
        t_input_qry = input_data['teacher_inputs']['qry']
        t_input_pos = input_data['teacher_inputs']['pos']

        # --- Teacher Forward (No Grad) ---
        with torch.no_grad():
            teacher_model.eval()
            t_q_reps, _, _, t_q_states = teacher_model.encode_input(t_input_qry)
            t_p_reps, _, _, t_p_states = teacher_model.encode_input(t_input_pos)
            # Handle tuple returns (take last layer)
            if isinstance(t_q_states, (tuple, list)): 
                t_q_states = t_q_states[-1]
                t_p_states = t_p_states[-1]

        # --- Student Forward ---
        s_q_reps, _, _, s_q_states = student_model.encode_input(s_input_qry)
        s_p_reps, _, _, s_p_states = student_model.encode_input(s_input_pos)
        if isinstance(s_q_states, (tuple, list)): 
            s_q_states = s_q_states[-1]
            s_p_states = s_p_states[-1]

        total_loss = 0.0
        contrastive_acc = 0.0
        holo_acc = 0.0
        rkd_acc = 0.0

        full_dim = s_q_reps.shape[-1]
        
        # Handle spectral multipliers (default fallback)
        spec_mults = self.spectral_multipliers
        if len(spec_mults) < len(self.matryoshka_dims):
            spec_mults = [1.0] * len(self.matryoshka_dims)

        # --- Matryoshka Loop (Spectral Decoupling) ---
        for i, dim in enumerate(self.matryoshka_dims):
            # Base weight for this slice (e.g. from args)
            dim_weight = self.matryoshka_weights[i] if i < len(self.matryoshka_weights) else 1.0
            
            # Spectral multiplier (e.g. 2.0 for Core, 1.0 for Detail)
            # We ONLY apply this to the Geometry Losses (RKD/Holo), NOT Contrastive
            # to preserve VOC details (as verified in v3).
            spec_w = spec_mults[i]
            
            current_dim = dim if dim is not None else full_dim
            
            # --- 1. Slice & Normalize ---
            s_q_slice = F.normalize(s_q_reps[:, :current_dim], p=2, dim=-1)
            s_p_slice = F.normalize(s_p_reps[:, :current_dim], p=2, dim=-1)
            
            # --- 2. Contrastive Loss (Uniform Spectrum) ---
            # We do NOT multiply by spec_w here. We want uniform contrastive pressure.
            scores = torch.mm(s_q_slice, s_p_slice.t()) / self.temperature
            labels = torch.arange(scores.size(0), device=scores.device)
            contrastive_acc += dim_weight * nn.CrossEntropyLoss()(scores, labels)

            # --- 3. Hybrid Geometry (RKD + Holo) ---
            projector = getattr(distiller, 'holo_projector', None)

            # A. GLOBAL RKD (ImageNet/News)
            # Applied to ALL slices to ensure consistent class separation.
            # Fix: If Full Dimension + Projector exists, use Projected RKD.
            # This allows the Student to be "Flexible" (in raw space) but "Accurate" (in teacher space).
            
            if projector is not None and current_dim == full_dim:
                 # Projected RKD (Flexible)
                 s_q_proj = F.normalize(projector(s_q_reps), p=2, dim=-1)
                 s_p_proj = F.normalize(projector(s_p_reps), p=2, dim=-1)
                 s_batch = torch.cat([s_q_proj, s_p_proj], dim=0)
            else:
                 # Standard RKD (Rigid) - Necessary for low-dim slices
                 s_batch = torch.cat([s_q_slice, s_p_slice], dim=0)

            t_q_slice = F.normalize(t_q_reps[:, :current_dim], p=2, dim=-1)
            t_p_slice = F.normalize(t_p_reps[:, :current_dim], p=2, dim=-1)
            t_batch = torch.cat([t_q_slice, t_p_slice], dim=0)

            loss_dist = self.compute_rkd_distance_loss(s_batch, t_batch)
            loss_angle = self.compute_rkd_angle_loss(s_batch, t_batch)
            
            rkd_val = (self.rkd_distance_weight * loss_dist) + (self.rkd_angle_weight * loss_angle)
            
            # Apply Spectral Multiplier (2.0x for Core, 1.0x for Detail)
            rkd_acc += (self.global_rkd_weight * spec_w) * rkd_val 

            # B. LOCAL HOLO (VOC)
            # Applied ONLY to High/Full Dimensions.
            # Low dims are too compressed for meaningful Gromov-Wasserstein structure.
            if current_dim is None or current_dim > 128:
                if projector is not None:
                    loss_q = self.fused_gromov_wasserstein(s_q_states, t_q_states, projector)
                    loss_p = self.fused_gromov_wasserstein(s_p_states, t_p_states, projector)
                    
                    fgw_val = 0.5 * (loss_q + loss_p)
                    # Apply Spectral Multiplier (1.0x usually)
                    holo_acc += (self.holo_weight * spec_w) * fgw_val

        # Final Weighted Sum
        total_loss = contrastive_acc + holo_acc + rkd_acc

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_acc,
            "holo_fgw_loss": holo_acc,
            "global_rkd_loss": rkd_acc
        }