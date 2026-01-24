"""
This module defines the Earth Mover's Distance based Loss (EMO), a distillation criterion
that aligns student and teacher models using Optimal Transport and Attention CKA.
"""
import logging
from typing import Any, Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from transformers import PreTrainedModel

from ..arguments import EMOArguments

logger = logging.getLogger(__name__)

# Token IDs from a specific tokenizer's vocabulary.
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151656
BOS_TOKEN_ID = 151643

class CKALoss(nn.Module):
    """
    Computes the Centered Kernel Alignment (CKA) loss, a similarity metric for representations.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, sh: Tensor, th: Tensor) -> Tensor:
        sh = sh.view(-1, sh.size(-1)).to(torch.float64)
        th = th.view(-1, th.size(-1)).to(torch.float64)
        sh = sh - sh.mean(0, keepdim=True)
        th = th - th.mean(0, keepdim=True)
        hsic = torch.norm(sh.t() @ th, 'fro')
        norm1 = torch.norm(sh.t() @ sh, 'fro') + self.eps
        norm2 = torch.norm(th.t() @ th, 'fro') + self.eps
        return 1 - hsic / torch.sqrt(norm1 * norm2)

class EMOLoss(nn.Module):
    """
    Implements the Earth Mover's distance based distillation loss (EMO).
    The total loss is a combination of:
    1.  **Contrastive Loss:** Standard InfoNCE loss on the student's embeddings.
    2.  **Optimal Transport (OT) Loss:** Uses the Sinkhorn algorithm to align the distribution
        of student's token embeddings with the teacher's projected token embeddings.
        The "mass" for each token is derived from its importance in the teacher's attention.
    3.  **Attention CKA Loss:** Encourages structural similarity between student and teacher
        attention maps using the CKA metric.
    """
    def __init__(self, args: Type[EMOArguments]):
        super(EMOLoss, self).__init__()
        self.args = args
        self.kd_loss_weight: float = self.args.kd_weight
        self.attn_loss_weight: float = self.args.attn_loss_weight
        self.ot_loss_weight: float = self.args.ot_loss_weight
        self.k_layer_attention: int = self.args.k_layer_attention
        self.sinkhorn_alpha: float = self.args.sinkhorn_alpha
        self.ot_max_iter: int = self.args.ot_max_iter
        self.ot_dist_type: str = self.args.ot_dist_type
        self.stop_threshold = 1e-7
        self.epsilon = 1e-9
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        if self.world_size == 1: return t
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)
    
    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, Tensor]:
        """Computes the combined EMO distillation loss."""
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher
        
        # --- 1. Get Student and Teacher Representations ---
        with torch.no_grad():
            teacher_model.eval()
            t_qry_output = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos_output = teacher_model.encode_input(input_data['teacher_inputs']['pos'])
        
        s_qry_output = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos_output = student_model.encode_input(input_data['student_inputs']['pos'])
        
        s_qry_reps, _, _, _ = s_qry_output
        s_pos_reps, _, _, _ = s_pos_output
        t_qry_reps, _, t_qry_attn, _ = t_qry_output
        t_pos_reps, _, t_pos_attn, _ = t_pos_output

        # --- 2. Compute Contrastive Loss ---
        all_s_qry_reps = self._dist_gather_tensor(s_qry_reps)
        all_s_pos_reps = self._dist_gather_tensor(s_pos_reps)
        scores = student_model.compute_similarity(all_s_qry_reps, all_s_pos_reps)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_s_pos_reps.size(0) // all_s_qry_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores.view(all_s_qry_reps.size(0), -1) / distiller.temperature, target)
        
        # --- 3. Compute Distillation Losses ---
        topk_results = self._extract_top_k_text_tokens(input_data, t_qry_attn, t_pos_attn)
        ot_loss = self._compute_ot_loss(s_qry_output, s_pos_output, t_qry_output, t_pos_output, distiller, input_data, topk_results)
        attn_loss = self._compute_attention_loss(s_qry_output, t_qry_output, t_pos_output, distiller, input_data, topk_results)

        # --- 4. Combine Losses ---
        kd_loss = self.ot_loss_weight * ot_loss + self.attn_loss_weight * attn_loss
        total_loss = (1 - self.kd_loss_weight) * contrastive_loss + self.kd_loss_weight * kd_loss
        
        return {'loss': total_loss, 'contrastive_loss': contrastive_loss, 'ot_loss': ot_loss, 'kd_loss': kd_loss}

    def _extract_top_k_text_tokens(self, input_data: Dict[str, Any], t_qry_attn: Tuple[Tensor], t_pos_attn: Tuple[Tensor]) -> List[Dict[str, List]]:
        """Extracts top-k most important text tokens based on teacher's attention."""
        t_qry_ids = input_data['teacher_inputs']['qry']['input_ids']
        t_pos_ids = input_data['teacher_inputs']['pos']['input_ids']
        qry_importance = t_qry_attn[-1].mean(dim=(0, 1))
        pos_importance = t_pos_attn[-1].mean(dim=(0, 1))
        results = []

        for i in range(t_qry_ids.size(0)):
            qry_mask = ((t_qry_ids[i] < VISION_START_TOKEN_ID) | (t_qry_ids[i] > VISION_END_TOKEN_ID)) & (t_qry_ids[i] != BOS_TOKEN_ID)
            pos_mask = ((t_pos_ids[i] < VISION_START_TOKEN_ID) | (t_pos_ids[i] > VISION_END_TOKEN_ID)) & (t_pos_ids[i] != BOS_TOKEN_ID)
            
            qry_imp = qry_importance * qry_mask.float()
            pos_imp = pos_importance * pos_mask.float()

            num_text_qry, num_text_pos = int(qry_mask.sum()), int(pos_mask.sum())
            qry_k, pos_k = min(num_text_qry // 3, num_text_qry), min((num_text_pos + 1) // 3, num_text_pos)

            results.append({
                "qry_topk": [(int(idx), int(t_qry_ids[i, idx]), float(qry_imp[idx])) for idx in torch.topk(qry_imp, qry_k).indices],
                "pos_topk": [(int(idx), int(t_pos_ids[i, idx]), float(pos_imp[idx])) for idx in torch.topk(pos_imp, pos_k).indices]
            })
        return results

    def _get_text_token_indices(self, ids: Tensor) -> List[int]:
        """Gets indices of text tokens, filtering out special/vision tokens."""
        mask = ((ids < VISION_START_TOKEN_ID) | (ids > VISION_END_TOKEN_ID)) & (ids != BOS_TOKEN_ID)
        return torch.where(mask)[0].tolist()

    def _compute_ot_loss(self, s_qry_out, s_pos_out, t_qry_out, t_pos_out, distiller, input_data, topk_results):
        """Computes the Optimal Transport loss between student and teacher token embeddings."""
        _, _, _, s_qry_hiddens = s_qry_out
        _, _, _, s_pos_hiddens = s_pos_out
        _, _, t_qry_attn, t_qry_hiddens = t_qry_out
        _, _, t_pos_attn, t_pos_hiddens = t_pos_out
        
        batch_size = len(topk_results)
        total_ot_loss = 0.0
        
        for i in range(batch_size):
            t_qry_ids = input_data['teacher_inputs']['qry']['input_ids'][i]
            t_pos_ids = input_data['teacher_inputs']['pos']['input_ids'][i]
            s_qry_ids = input_data['student_inputs']['qry']['input_ids'][i]
            s_pos_ids = input_data['student_inputs']['pos']['input_ids'][i]

            t_qry_txt_idx, t_pos_txt_idx = self._get_text_token_indices(t_qry_ids), self._get_text_token_indices(t_pos_ids)
            s_qry_txt_idx, s_pos_txt_idx = self._get_text_token_indices(s_qry_ids), self._get_text_token_indices(s_pos_ids)
            
            if not all([t_qry_txt_idx, t_pos_txt_idx, s_qry_txt_idx, s_pos_txt_idx]):
                logger.warning(f"Instance {i}: Missing text tokens for OT loss. Skipping.")
                continue

            # Compute token importance (mass)
            t_qry_imp = torch.softmax(t_qry_attn[-1][i].mean(dim=0).sum(dim=0)[t_qry_txt_idx], dim=0)
            t_pos_imp = torch.softmax(t_pos_attn[-1][i].mean(dim=0).sum(dim=0)[t_pos_txt_idx], dim=0)
            s_qry_imp = self._project_importance(t_qry_imp, t_qry_ids, s_qry_ids, t_qry_txt_idx, s_qry_txt_idx)
            s_pos_imp = self._project_importance(t_pos_imp, t_pos_ids, s_pos_ids, t_pos_txt_idx, s_pos_txt_idx)

            # Get hidden states and project teacher's
            s_qry_hidden = s_qry_hiddens[-1][i][s_qry_txt_idx, :]
            s_pos_hidden = s_pos_hiddens[-1][i][s_pos_txt_idx, :]
            proj_t_qry_hidden = distiller.projectors["t2s"](t_qry_hiddens[-1][i][t_qry_txt_idx, :])
            proj_t_pos_hidden = distiller.projectors["t2s"](t_pos_hiddens[-1][i][t_pos_txt_idx, :])
            
            # Compute cost matrices and OT loss
            cost_qry = self._pairwise_distance(s_qry_hidden, proj_t_qry_hidden)
            cost_pos = self._pairwise_distance(s_pos_hidden, proj_t_pos_hidden)
            ot_loss_qry, _ = self._sinkhorn(cost_qry, s_qry_imp, t_qry_imp)
            ot_loss_pos, _ = self._sinkhorn(cost_pos, s_pos_imp, t_pos_imp)
            total_ot_loss += (ot_loss_qry + ot_loss_pos) / 2
        
        return total_ot_loss / batch_size if batch_size > 0 else 0.0

    def _project_importance(self, t_imp, t_ids, s_ids, t_txt_idx, s_txt_idx):
        """Projects teacher's token importance to student's tokens."""
        s_imp = torch.zeros(len(s_txt_idx), device=t_imp.device)
        t_id_to_imp = {t_ids[t_idx].item(): t_imp[i].item() for i, t_idx in enumerate(t_txt_idx)}
        
        for i, s_idx in enumerate(s_txt_idx):
            s_id = s_ids[s_idx].item()
            if s_id in t_id_to_imp:
                s_imp[i] = t_id_to_imp[s_id]
        
        min_imp = t_imp.min().item() if len(t_imp) > 0 else 0.0
        s_imp.masked_fill_(s_imp == 0, min_imp)
        return torch.softmax(s_imp, dim=0)

    def _pairwise_distance(self, a: Tensor, b: Tensor) -> Tensor:
        if self.ot_dist_type == 'cosine':
            a_norm = F.normalize(a, p=2, dim=1)
            b_norm = F.normalize(b, p=2, dim=1)
            return 1 - torch.mm(a_norm, b_norm.transpose(0, 1))
        elif self.ot_dist_type == 'euclidean':
            return torch.cdist(a, b, p=2)
        raise ValueError(f"Unsupported OT distance type: {self.ot_dist_type}")

    def _sinkhorn(self, cost: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the Sinkhorn distance."""
        m, n = cost.shape
        if m == 0 or n == 0: return torch.tensor(0.0, device=cost.device), torch.zeros_like(cost)
        
        a, b = a.view(m, 1), b.view(n, 1)
        a, b = a / a.sum(), b / b.sum() # Ensure normalization

        K = torch.exp(-cost / self.sinkhorn_alpha)
        u = torch.ones(m, 1, device=cost.device, dtype=cost.dtype) / m
        
        for _ in range(self.ot_max_iter):
            u_prev = u.clone()
            v = b / (K.t() @ u + self.epsilon)
            u = a / (K @ v + self.epsilon)
            if torch.norm(u - u_prev, p=float('inf')) < self.stop_threshold:
                break
                
        P = u * K * v.t()
        return torch.sum(P * cost), P

    def _compute_attention_loss(self, s_out, t_qry_out, t_pos_out, distiller, input_data, topk_results):
        """Computes the Attention CKA loss."""
        _, _, s_qry_attn, _ = s_out
        _, _, t_qry_attn, _ = t_qry_out
        _, _, t_pos_attn, _ = t_pos_out
        
        batch_size = len(topk_results)
        total_attn_loss = 0.0
        cka_loss_fn = CKALoss().to(self.args.device)

        t_layer_num, s_layer_num = len(t_qry_attn), len(s_qry_attn)
        layer_per_block = t_layer_num // s_layer_num
        
        s_indices = self._extract_student_indices(input_data, topk_results)

        s_qry_last_k, s_pos_last_k = s_qry_attn[-self.k_layer_attention:], s_pos_attn[-self.k_layer_attention:]
        t_qry_mapped = [t_qry_attn[i * layer_per_block + layer_per_block - 1] for i in range(s_layer_num - self.k_layer_attention, s_layer_num)]
        t_pos_mapped = [t_pos_attn[i * layer_per_block + layer_per_block - 1] for i in range(s_layer_num - self.k_layer_attention, s_layer_num)]

        for t_qry, t_pos, s_qry, s_pos in zip(t_qry_mapped, t_pos_mapped, s_qry_last_k, s_pos_last_k):
            for i in range(batch_size):
                t_qry_idx = [idx for idx, _, _ in topk_results[i]['qry_topk']]
                t_pos_idx = [idx for idx, _, _ in topk_results[i]['pos_topk']]
                s_qry_idx = [idx for idx in s_indices[i]['qry'] if idx < s_qry.size(2)]
                s_pos_idx = [idx for idx in s_indices[i]['pos'] if idx < s_pos.size(2)]

                if not all([t_qry_idx, t_pos_idx, s_qry_idx, s_pos_idx]): continue

                tq_mean, tp_mean = t_qry[i, :, t_qry_idx, :].mean(0), t_pos[i, :, t_pos_idx, :].mean(0)
                sq_mean, sp_mean = s_qry[i, :, s_qry_idx, :].mean(0), s_pos[i, :, s_pos_idx, :].mean(0)

                for attn_map in [tq_mean, tp_mean, sq_mean, sp_mean]:
                    attn_map.masked_fill_(attn_map <= -1e2, 0.0)

                total_attn_loss += (cka_loss_fn(sq_mean, tq_mean) + cka_loss_fn(sp_mean, tp_mean)) / 2
        
        return total_attn_loss / batch_size if batch_size > 0 else 0.0
    
    def _extract_student_indices(self, input_data, topk_results):
        s_qry_ids = input_data['student_inputs']['qry']['input_ids']
        s_pos_ids = input_data['student_inputs']['pos']['input_ids']
        student_indices = []

        for i in range(len(topk_results)):
            s_qry_map = {tid: j for j, tid in reversed(list(enumerate(s_qry_ids[i].tolist())))}
            s_pos_map = {tid: j for j, tid in reversed(list(enumerate(s_pos_ids[i].tolist())))}
            
            student_indices.append({
                "qry": [s_qry_map.get(tid) for _, tid, _ in topk_results[i]['qry_topk'] if tid in s_qry_map],
                "pos": [s_pos_map.get(tid) for _, tid, _ in topk_results[i]['pos_topk'] if tid in s_pos_map]
            })
        return student_indices
