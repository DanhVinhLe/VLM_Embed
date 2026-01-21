import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

class HoloDistillLoss(nn.Module):
    """
    HoloDistill: Holographic Knowledge Distillation via Fused Gromov-Wasserstein 
    and Matryoshka Representation Learning.

    Implements the ICML plan:
    1. Treats models as Metric Measure Spaces (mm-spaces).
    2. Aligns structures via Fused Gromov-Wasserstein (FGW) Optimal Transport.
    3. Incorporates Saliency-Aware Probability Measures.
    4. Supports Elastic Saliency Compression (Matryoshka).
    """

    def __init__(self, args: Any):
        super(HoloDistillLoss, self).__init__()
        self.args = args
        self.temperature = getattr(args, "temperature", 0.07)
        
        # Hyperparameters for HoloDistill
        self.alpha = getattr(args, "holo_alpha", 0.5)      # Balance between Feature Cost and Structure Cost
        self.ot_epsilon = getattr(args, "ot_epsilon", 0.1) # Entropy regularization for Sinkhorn
        self.ot_iters = getattr(args, "ot_iters", 20)      # Number of Sinkhorn iterations
        self.kd_weight = getattr(args, "kd_weight", 1.0)
        
        # Matryoshka Representation Learning (MRL) Config
        # We enforce structure at these embedding slices (e.g., [64, 128, 768])
        # If not provided, defaults to just the full dimension.
        self.matryoshka_dims = getattr(args, "matryoshka_dims", [None]) 
        
        # Weights for different matryoshka granularities (usually higher for smaller dims)
        self.matryoshka_weights = getattr(args, "matryoshka_weights", [1.0] * len(self.matryoshka_dims))

    def get_saliency_measure(self, hidden_states: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """
        Computes Saliency-Aware Probability Measures (\mu).
        
        Instead of assuming uniform probability over tokens (1/N), we weight tokens 
        by their information density. We use the L2-norm of the hidden states as a 
        proxy for saliency, which correlates with attention significance in Transformers.
        
        Args:
            hidden_states: (B, N, D)
        Returns:
            mu: (B, N, 1) Normalized probability mass per token.
        """
        # Calculate L2 norm as proxy for significance
        # Shape: (B, N)
        norms = torch.norm(hidden_states, p=2, dim=-1)
        
        # Softmax to create a valid probability distribution
        weights = F.softmax(norms / temp, dim=-1)
        
        return weights.unsqueeze(-1) # (B, N, 1)

    def compute_structure_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the Intra-Relational Holographic Structure Matrix (C).
        C_ij = 1 - CosineSimilarity(z_i, z_j)
        
        Args:
            z: (B, N, D)
        Returns:
            C: (B, N, N) Distance matrix
        """
        # Normalize for cosine similarity
        z_norm = F.normalize(z, p=2, dim=-1)
        
        # Cosine similarity: (B, N, D) @ (B, D, N) -> (B, N, N)
        sim = torch.bmm(z_norm, z_norm.transpose(1, 2))
        
        # Convert to distance (0 to 2)
        C = 1.0 - sim
        return C

    def sinkhorn_knopp_log_domain(self, C: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        Solves the Entropic Optimal Transport problem using the Sinkhorn-Knopp algorithm 
        in the log-domain for numerical stability.
        
        Args:
            C: (B, N_s, N_t) Cost matrix
            mu: (B, N_s, 1) Source marginals (Student saliency)
            nu: (B, N_t, 1) Target marginals (Teacher saliency)
        Returns:
            Gamma: (B, N_s, N_t) Optimal Transport Plan
        """
        B, N_s, N_t = C.shape
        
        # Log-domain initialization
        # K = exp(-C / epsilon)
        log_K = -C / self.ot_epsilon
        
        # Potentials (dual variables)
        u = torch.zeros_like(mu) # (B, N_s, 1)
        v = torch.zeros_like(nu) # (B, N_t, 1)
        
        # Sinkhorn iterations
        for _ in range(self.ot_iters):
            # Update u: u = log(mu) - logsumexp(log_K + v.T)
            # log_K + v.transpose: (B, N_s, N_t) + (B, 1, N_t) -> (B, N_s, N_t)
            term = log_K + v.transpose(1, 2)
            u = torch.log(mu + 1e-8) - torch.logsumexp(term, dim=2, keepdim=True)
            
            # Update v: v = log(nu) - logsumexp(log_K.T + u)
            term = log_K.transpose(1, 2) + u.transpose(1, 2) # (B, N_t, N_s)
            v = torch.log(nu + 1e-8) - torch.logsumexp(term, dim=2, keepdim=True)
            
        # Get optimal plan Gamma = diag(u) * K * diag(v)
        # log_Gamma = u + log_K + v.T
        log_Gamma = u + log_K + v.transpose(1, 2)
        Gamma = torch.exp(log_Gamma)
        
        return Gamma

    def fused_gromov_wasserstein(self, 
                                 z_s: torch.Tensor, 
                                 z_t: torch.Tensor, 
                                 projector: nn.Module) -> torch.Tensor:
        """
        Computes the Fused Gromov-Wasserstein (FGW) loss.
        
        FGW combines:
        1. Feature Cost (M): Cosine distance between Projected Student and Teacher features.
        2. Structure Cost (GW): Discrepancy between intra-relational matrices.
        
        Args:
            z_s: Student hidden states (B, N_s, D_s)
            z_t: Teacher hidden states (B, N_t, D_t)
            projector: Linear layer D_s -> D_t
        """
        B, N_s, D_s = z_s.shape
        _, N_t, D_t = z_t.shape

        # 1. Compute Saliency Measures (Mass)
        mu = self.get_saliency_measure(z_s) # (B, N_s, 1)
        nu = self.get_saliency_measure(z_t) # (B, N_t, 1)

        # 2. Compute Structure Matrices (Geometry)
        C_s = self.compute_structure_matrix(z_s) # (B, N_s, N_s)
        C_t = self.compute_structure_matrix(z_t) # (B, N_t, N_t)

        # 3. Compute Feature Cost Matrix M (Ground Metric)
        # Align feature spaces first using the projector
        z_s_proj = projector(z_s) # (B, N_s, D_t)
        
        # M_ik = 1 - cos(z_s_proj_i, z_t_k)
        z_s_norm = F.normalize(z_s_proj, p=2, dim=-1)
        z_t_norm = F.normalize(z_t, p=2, dim=-1)
        M = 1.0 - torch.bmm(z_s_norm, z_t_norm.transpose(1, 2)) # (B, N_s, N_t)

        # 4. Fused Cost for Sinkhorn
        # Strictly, GW requires solving a Quadratic Assignment Problem (QAP).
        # We use a standard relaxation: Use the Feature Cost M to find the transport plan Gamma,
        # then evaluate the GW cost using that Gamma. This is efficient and stable.
        
        Gamma = self.sinkhorn_knopp_log_domain(M, mu, nu)
        
        # 5. Calculate Final Loss terms
        
        # Term A: Feature Alignment Loss (Wasserstein part)
        # sum(M * Gamma)
        feat_loss = torch.sum(M * Gamma, dim=(1, 2)).mean()
        
        # Term B: Structure Alignment Loss (Gromov part)
        # L_gw = sum_{ijkl} |C_s_ij - C_t_kl|^2 * Gamma_ik * Gamma_jl
        # Efficient calculation via tensor operations:
        # |C_s - C_t|^2 = C_s^2 + C_t^2 - 2 C_s C_t
        # This is expensive (O(N^4)). We approximate via the "Structure-wise MSE"
        # weighted by the transport plan.
        
        # Project Teacher structure to Student space using Gamma: C_t_aligned = Gamma @ C_t @ Gamma.T
        # Note: Since Gamma is non-square (N_s x N_t), we do:
        # C_t_aligned ~ (B, N_s, N_s)
        
        # Normalize Gamma for mapping purposes
        Gamma_norm = Gamma / (Gamma.sum(dim=2, keepdim=True) + 1e-8)
        
        C_t_mapped = torch.bmm(torch.bmm(Gamma_norm, C_t), Gamma_norm.transpose(1, 2))
        
        # Structure Loss: MSE between Student Structure and Mapped Teacher Structure
        # Weighted by student saliency to focus on important tokens
        diff = (C_s - C_t_mapped) ** 2
        struct_loss = torch.sum(diff * (mu @ mu.transpose(1, 2)), dim=(1, 2)).mean()

        # Total Fused Loss
        loss = (1 - self.alpha) * feat_loss + self.alpha * struct_loss
        return loss

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        # Inputs
        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']
        teacher_input_qry = input_data['teacher_inputs']['qry']
        teacher_input_pos = input_data['teacher_inputs']['pos']

        # --- 1. Teacher Forward (No Grad) ---
        with torch.no_grad():
            teacher_model.eval()
            # We assume encode_input returns (reps, ..., hidden_states)
            # hidden_states is usually a tuple of layers. We want the last one.
            t_q_reps, _, _, t_q_states = teacher_model.encode_input(teacher_input_qry)
            t_p_reps, _, _, t_p_states = teacher_model.encode_input(teacher_input_pos)
            
            # Handle if states are tuple (take last layer)
            if isinstance(t_q_states, (tuple, list)):
                t_q_states = t_q_states[-1]
                t_p_states = t_p_states[-1]

        # --- 2. Student Forward (Grad) ---
        s_q_reps, _, _, s_q_states = student_model.encode_input(student_input_qry)
        s_p_reps, _, _, s_p_states = student_model.encode_input(student_input_pos)
        
        if isinstance(s_q_states, (tuple, list)):
            s_q_states = s_q_states[-1]
            s_p_states = s_p_states[-1]

        total_loss = 0.0
        contrastive_loss_acc = 0.0
        holo_loss_acc = 0.0

        # --- 3. Matryoshka Loop (Elasticity) ---
        # We calculate loss at multiple granularities. 
        # For full dimension, slice is the whole tensor.
        
        full_dim = s_q_reps.shape[-1]
        
        for i, dim in enumerate(self.matryoshka_dims):
            dim_weight = self.matryoshka_weights[i]
            current_dim = dim if dim is not None else full_dim
            
            # Slice Representations (for Contrastive)
            # Matryoshka slicing: take first k dimensions
            s_q_rep_slice = F.normalize(s_q_reps[:, :current_dim], p=2, dim=-1)
            s_p_rep_slice = F.normalize(s_p_reps[:, :current_dim], p=2, dim=-1)
            
            # --- A. Contrastive Loss (Global Alignment) ---
            # InfoNCE on the sliced embeddings
            scores = torch.mm(s_q_rep_slice, s_p_rep_slice.t()) / self.temperature
            labels = torch.arange(scores.size(0), device=scores.device)
            mrl_contrastive = nn.CrossEntropyLoss()(scores, labels)
            
            contrastive_loss_acc += dim_weight * mrl_contrastive

            # --- B. HoloDistill Loss (FGW on Hidden States) ---
            # NOTE: We only apply FGW on the full hidden states usually, or we can project
            # the hidden states if specific Matryoshka projectors exist. 
            # For simplicity and efficiency, we apply HoloDistill mainly on the Full Dimension
            # or use the projector defined in Distiller to align dimensions.
            
            if current_dim == full_dim:
                # Use the main projector from Distiller
                # Check if distiller has specific projectors for alignment
                # For this implementation, we use the 'last_layer_projector' or 't2s_align'
                # defined in the Distiller class.
                
                projector = getattr(distiller, 'holo_projector', None) or \
                            getattr(distiller, 't2s_img_align', None)
                
                if projector is None:
                    # Fallback if no specific projector, though distiller.py usually ensures one
                    logger.warning("No projector found for HoloDistill. Skipping FGW.")
                    fgw_val = torch.tensor(0.0, device=scores.device)
                else:
                    # Apply FGW on Query and Pos samples separately and average
                    loss_q = self.fused_gromov_wasserstein(s_q_states, t_q_states, projector)
                    loss_p = self.fused_gromov_wasserstein(s_p_states, t_p_states, projector)
                    fgw_val = 0.5 * (loss_q + loss_p)

                holo_loss_acc += dim_weight * fgw_val

        # Final Weighted Sum
        total_loss = contrastive_loss_acc + (self.kd_weight * holo_loss_acc)

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss_acc,
            "holo_fgw_loss": holo_loss_acc
        }