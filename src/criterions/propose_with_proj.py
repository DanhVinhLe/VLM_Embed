"""
This module defines a proposal-based distillation criterion, which combines contrastive loss,
feature MSE loss with projectors, and an attention-based CKA loss.
"""
import logging
from typing import Any, Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from transformers import PreTrainedModel

from ..arguments import ProposalArguments





logger = logging.getLogger(__name__)



# Token IDs, likely from a specific tokenizer's vocabulary.

# These are hardcoded as they are specific to the model being used.

VISION_START_TOKEN_ID = 151652

VISION_END_TOKEN_ID = 151656

BOS_TOKEN_ID = 151643





class CKALoss(nn.Module):

    """

    Computes the Centered Kernel Alignment (CKA) loss, a similarity metric for representations.

    CKA is invariant to isotropic scaling and orthogonal transformations, making it a robust

    metric for comparing learned representations.

    """

    def __init__(self, eps: float = 1e-8):

        """

        Initializes the CKALoss module.

        Args:

            eps: A small epsilon value for numerical stability.

        """

        super().__init__()

        self.eps = eps



    def forward(self, student_hidden: Tensor, teacher_hidden: Tensor) -> Tensor:

        """

        Computes the CKA loss between student and teacher hidden representations.



        Args:

            student_hidden: Student's representation tensor.

            teacher_hidden: Teacher's representation tensor.



        Returns:

            The CKA loss as a scalar tensor.

        """

        t_dim, s_dim = teacher_hidden.size(-1), student_hidden.size(-1)

        student_hidden = student_hidden.view(-1, s_dim).to(torch.float64)

        teacher_hidden = teacher_hidden.view(-1, t_dim).to(torch.float64)



        # Center the representations

        student_hidden = student_hidden - student_hidden.mean(0, keepdim=True)

        teacher_hidden = teacher_hidden - teacher_hidden.mean(0, keepdim=True)



        # Compute the HSIC (Hilbert-Schmidt Independence Criterion)

        hsic = torch.norm(student_hidden.t().matmul(teacher_hidden), 'fro')



        # Compute the normalization terms

        norm1 = torch.norm(student_hidden.t().matmul(student_hidden), 'fro') + self.eps

        norm2 = torch.norm(teacher_hidden.t().matmul(teacher_hidden), 'fro') + self.eps



        # CKA is the normalized HSIC

        cka_similarity = hsic / torch.sqrt(norm1 * norm2)

        

        # Return 1 - similarity as a loss

        return 1 - cka_similarity



class ProposalLossWithProj(nn.Module):

    """

    Implements a proposal-based distillation loss with projection heads and attention CKA.

    The total loss consists of three main components:

    1.  **Contrastive Loss:** A standard InfoNCE loss on the student's embeddings to ensure

        discriminative representations.

    2.  **MSE Distillation Loss:** An MSE loss between the student's embeddings and the teacher's

        projected embeddings for both text and image modalities. This requires projectors

        to align the feature dimensions.

    3.  **Attention CKA Loss:** A Centered Kernel Alignment (CKA) loss applied to the attention

        maps of the student and teacher models to encourage structural similarity in how they

        attend to different parts of the input.

    """

    def __init__(self, args: Type[ProposalArguments]):

        """

        Initializes the ProposalLossWithProj module.

        Args:

            args: Configuration arguments, expected to contain `kd_weight`.

        """

        super(ProposalLossWithProj, self).__init__()

        self.args = args

        self.kd_loss_weight: float = self.args.kd_weight

        self.attn_loss_weight: float = self.args.attn_loss_weight

        self.k_layer_attention: int = self.args.k_layer_attention

        self.mse_loss = nn.MSELoss(reduction='mean')

        self.cka_loss = CKALoss().to(self.args.device)

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        """Gathers tensors from all processes in a distributed setting."""
        if self.world_size == 1:
            return t
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)
    
    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, Tensor]:
        """Computes the combined distillation loss."""
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher
        
        # --- 1. Get Student and Teacher Representations ---
        with torch.no_grad():
            teacher_model.eval()
            t_qry_reps, t_qry_img_feats, t_qry_attn, _ = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos_reps, t_pos_img_feats, t_pos_attn, _ = teacher_model.encode_input(input_data['teacher_inputs']['pos'])

        s_qry_reps, s_qry_img_feats, s_qry_attn, _ = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos_reps, s_pos_img_feats, s_pos_attn, _ = student_model.encode_input(input_data['student_inputs']['pos'])

        # --- 2. Compute Contrastive Loss ---
        all_s_qry_reps = self._dist_gather_tensor(s_qry_reps)
        all_s_pos_reps = self._dist_gather_tensor(s_pos_reps)
        scores = student_model.compute_similarity(all_s_qry_reps, all_s_pos_reps)
        scores = scores.view(all_s_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_s_pos_reps.size(0) // all_s_qry_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # --- 3. Compute MSE Distillation Loss ---
        kd_loss_mse = self._compute_mse_loss(
            distiller, s_qry_reps, s_pos_reps, t_qry_reps, t_pos_reps,
            s_qry_img_feats, s_pos_img_feats, t_qry_img_feats, t_pos_img_feats
        )

        # --- 4. Compute Attention CKA Loss ---
        attn_loss = self._compute_attention_loss(
            input_data, t_qry_attn, t_pos_attn, s_qry_attn, s_pos_attn
        )
        
        # --- 5. Combine Losses ---
        kd_loss = kd_loss_mse + self.attn_loss_weight * attn_loss
        total_loss = (1 - self.kd_loss_weight) * contrastive_loss + self.kd_loss_weight * kd_loss

        return {
            "loss": total_loss, 
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
            "attn_loss": attn_loss,
            "kd_loss_mse": kd_loss_mse,
        }

    def _compute_mse_loss(
        self, distiller: nn.Module, s_qry_reps: Tensor, s_pos_reps: Tensor,
        t_qry_reps: Tensor, t_pos_reps: Tensor, s_qry_img_feats: List[Tensor],
        s_pos_img_feats: List[Tensor], t_qry_img_feats: List[Tensor], t_pos_img_feats: List[Tensor]
    ) -> Tensor:
        """Computes the MSE loss for both text and image features."""
        # Project teacher text representations and compute MSE loss
        proj_t_qry_reps = F.normalize(distiller.projectors["t2s_txt"](t_qry_reps), p=2, dim=-1)
        proj_t_pos_reps = F.normalize(distiller.projectors["t2s_txt"](t_pos_reps), p=2, dim=-1)
        
        loss_text = 0.25 * (
            self.mse_loss(s_qry_reps, proj_t_qry_reps) + 
            self.mse_loss(s_pos_reps, proj_t_pos_reps) +
            self.mse_loss(s_qry_reps, proj_t_pos_reps) + 
            self.mse_loss(s_pos_reps, proj_t_qry_reps)
        )

        # Compute MSE loss for image features
        loss_img = 0.0
        batch_size = s_qry_reps.size(0)
        for i in range(batch_size):
            if s_qry_img_feats and t_qry_img_feats and s_qry_img_feats[i] is not None and t_qry_img_feats[i] is not None:
                s_feat = F.normalize(s_qry_img_feats[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                t_feat = F.normalize(t_qry_img_feats[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                proj_t_feat = distiller.projectors["t2s_img"](t_feat)
                loss_img += self.mse_loss(s_feat, proj_t_feat)
            
            if s_pos_img_feats and t_pos_img_feats and s_pos_img_feats[i] is not None and t_pos_img_feats[i] is not None:
                s_feat = F.normalize(s_pos_img_feats[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                t_feat = F.normalize(t_pos_img_feats[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                proj_t_feat = distiller.projectors["t2s_img"](t_feat)
                loss_img += self.mse_loss(s_feat, proj_t_feat)
        
        return loss_text + (loss_img / batch_size if batch_size > 0 else 0.0)

    def _compute_attention_loss(
        self, input_data: Dict[str, Any], t_qry_attn: Tuple[Tensor], t_pos_attn: Tuple[Tensor],
        s_qry_attn: Tuple[Tensor], s_pos_attn: Tuple[Tensor]
    ) -> Tensor:
        """Computes the CKA loss on the attention maps."""
        # Extract top-k important text tokens from the teacher's attention
        topk_results = self._extract_top_k_text_tokens(
            input_data['teacher_inputs']['qry']['input_ids'],
            input_data['teacher_inputs']['pos']['input_ids'],
            t_qry_attn, t_pos_attn
        )
        
        # Find the corresponding indices in the student's token sequence
        student_indices = self._extract_student_indices(
            input_data['student_inputs']['qry']['input_ids'],
            input_data['student_inputs']['pos']['input_ids'],
            topk_results
        )
        
        batch_size = len(topk_results)
        total_attn_loss = 0.0

        teacher_layer_num = len(t_qry_attn)
        student_layer_num = len(s_qry_attn)
        layer_per_block = teacher_layer_num // student_layer_num

        # Select last k layers for attention CKA loss
        new_teacher_qry_attns = [t_qry_attn[i * layer_per_block + layer_per_block - 1] for i in range(student_layer_num)]
        new_teacher_pos_attns = [t_pos_attn[i * layer_per_block + layer_per_block - 1] for i in range(student_layer_num)]
        
        teacher_qry_last_k = new_teacher_qry_attns[-self.k_layer_attention:]
        teacher_pos_last_k = new_teacher_pos_attns[-self.k_layer_attention:]
        student_qry_last_k = s_qry_attn[-self.k_layer_attention:]
        student_pos_last_k = s_pos_attn[-self.k_layer_attention:]

        for t_qry_layer_attn, t_pos_layer_attn, s_qry_layer_attn, s_pos_layer_attn in zip(
            teacher_qry_last_k, teacher_pos_last_k, student_qry_last_k, student_pos_last_k
        ):
            for i in range(batch_size):
                t_qry_topk_idx = [idx for idx, _, _ in topk_results[i]['qry_topk']]
                t_pos_topk_idx = [idx for idx, _, _ in topk_results[i]['pos_topk']]
                
                s_qry_topk_idx = [idx for idx in student_indices[i]['qry'] if idx < s_qry_layer_attn.size(2)]
                s_pos_topk_idx = [idx for idx in student_indices[i]['pos'] if idx < s_pos_layer_attn.size(2)]

                if not t_qry_topk_idx or not t_pos_topk_idx or not s_qry_topk_idx or not s_pos_topk_idx:
                    logger.warning(f"Instance {i}: Not enough tokens for attention loss. Skipping.")
                    continue

                # Average attention maps over selected tokens
                tq_mean = t_qry_layer_attn[i, :, t_qry_topk_idx, :].mean(dim=0)
                tp_mean = t_pos_layer_attn[i, :, t_pos_topk_idx, :].mean(dim=0)
                sq_mean = s_qry_layer_attn[i, :, s_qry_topk_idx, :].mean(dim=0)
                sp_mean = s_pos_layer_attn[i, :, s_pos_topk_idx, :].mean(dim=0)

                # Mask out -inf values that can occur from attention masks
                tq_mean.masked_fill_(tq_mean <= -1e2, 0.0)
                sq_mean.masked_fill_(sq_mean <= -1e2, 0.0)
                tp_mean.masked_fill_(tp_mean <= -1e2, 0.0)
                sp_mean.masked_fill_(sp_mean <= -1e2, 0.0)
                
                # Compute CKA loss and accumulate
                attn_loss = self.cka_loss(sq_mean, tq_mean) + self.cka_loss(sp_mean, tp_mean)
                total_attn_loss += attn_loss / 2
        
        return total_attn_loss / batch_size if batch_size > 0 else 0.0

    def _extract_top_k_text_tokens(
        self, t_qry_ids: Tensor, t_pos_ids: Tensor, t_qry_attn: Tuple[Tensor], t_pos_attn: Tuple[Tensor]
    ) -> List[Dict[str, List]]:
        """Extracts top-k most important text tokens based on teacher's attention."""
        batch_size, _ = t_qry_ids.size()
        qry_importance = t_qry_attn[-1].mean(dim=1).sum(dim=1)
        pos_importance = t_pos_attn[-1].mean(dim=1).sum(dim=1)
        results = []

        for i in range(batch_size):
            qry_mask = ((t_qry_ids[i] < VISION_START_TOKEN_ID) | (t_qry_ids[i] > VISION_END_TOKEN_ID)) & (t_qry_ids[i] != BOS_TOKEN_ID)
            pos_mask = ((t_pos_ids[i] < VISION_START_TOKEN_ID) | (t_pos_ids[i] > VISION_END_TOKEN_ID)) & (t_pos_ids[i] != BOS_TOKEN_ID)
            
            qry_imp = qry_importance[i] * qry_mask.float()
            pos_imp = pos_importance[i] * pos_mask.float()
            
            num_text_qry = int(qry_mask.sum().item())
            num_text_pos = int(pos_mask.sum().item())

            qry_topk_indices = torch.topk(qry_imp, min(num_text_qry // 2, num_text_qry)).indices
            pos_topk_indices = torch.topk(pos_imp, min((num_text_pos + 1) // 2, num_text_pos)).indices
            
            results.append({
                "qry_topk": [(int(idx), int(t_qry_ids[i, idx]), float(qry_imp[idx])) for idx in qry_topk_indices],
                "pos_topk": [(int(idx), int(t_pos_ids[i, idx]), float(pos_imp[idx])) for idx in pos_topk_indices]
            })
        return results
    
    def _extract_student_indices(
        self, s_qry_ids: Tensor, s_pos_ids: Tensor, topk_results: List[Dict[str, List]]
    ) -> List[Dict[str, List[int]]]:
        """Maps teacher's top-k token IDs to indices in the student's token sequence."""
        batch_size = len(topk_results)
        student_indices = []

        for i in range(batch_size):
            s_qry_id_list = s_qry_ids[i].tolist()
            s_pos_id_list = s_pos_ids[i].tolist()
            
            s_qry_id_map = {token_id: [j for j, tid in enumerate(s_qry_id_list) if tid == token_id] for token_id in set(s_qry_id_list)}
            s_pos_id_map = {token_id: [j for j, tid in enumerate(s_pos_id_list) if tid == token_id] for token_id in set(s_pos_id_list)}

            qry_student_idx, used_qry_indices = [], set()
            for _, token_id, _ in topk_results[i]['qry_topk']:
                if token_id in s_qry_id_map:
                    for index in s_qry_id_map[token_id]:
                        if index not in used_qry_indices:
                            qry_student_idx.append(index)
                            used_qry_indices.add(index)
                            break 
            
            pos_student_idx, used_pos_indices = [], set()
            for _, token_id, _ in topk_results[i]['pos_topk']:
                if token_id in s_pos_id_map:
                    for index in s_pos_id_map[token_id]:
                        if index not in used_pos_indices:
                            pos_student_idx.append(index)
                            used_pos_indices.add(index)
                            break

            student_indices.append({"qry": qry_student_idx, "pos": pos_student_idx})
        return student_indices