import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


class CKALoss(nn.Module):
    """
    Centered Kernel Alignment (CKA) Loss.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, SH: Tensor, TH: Tensor) -> Tensor:
        """
        Computes CKA loss.
        """
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(torch.float64)
        TH = TH.view(-1, dT).to(torch.float64)

        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)

        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps

        return 1 - num / torch.sqrt(den1 * den2)

class ProposalLossWithProj(nn.Module):
    """
    Proposal Loss with extra projection heads and Attention CKA Loss.
    """
    def __init__(self, args: Any):
        """
        Initializes the ProposalLossWithProj module.

        Args:
            args: Configuration arguments.
        """
        super(ProposalLossWithProj, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        self.sinkhorn_alpha = 0.1
        self.stop_threshold = 1e-6
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'attention'
        self.mse_loss = nn.MSELoss(reduction='mean')
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        """
        Gathers tensors from all processes.
        """
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        num_text_qry_tokens = (((teacher_qry_input['input_ids'] < 151652) | (teacher_qry_input['input_ids'] > 151656)) & (teacher_qry_input['input_ids'] != 151643)).sum(dim=1)
        num_text_pos_tokens = (((teacher_pos_input['input_ids'] < 151652) | (teacher_pos_input['input_ids'] > 151656)) & (teacher_pos_input['input_ids'] != 151643)).sum(dim=1)
        batch_size = student_qry_input['input_ids'].size(0)
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, _ = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, _ = teacher_model.encode_input(teacher_pos_input)

        student_qry_reps, student_qry_image_features, student_qry_attention, _ = student_model.encode_input(student_qry_input)
        student_pos_reps, student_pos_image_features, student_pos_attention, _ = student_model.encode_input(student_pos_input)
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)

        projected_teacher_qry_reps = F.normalize(self.distiller.projectors["t2s_txt"](teacher_qry_reps), p=2, dim=-1)
        projected_teacher_pos_reps = F.normalize(self.distiller.projectors["t2s_txt"](teacher_pos_reps), p=2, dim=-1)
        self.kd_mse_loss_text = 0.25 * (self.mse_loss(student_qry_reps, projected_teacher_qry_reps) + self.mse_loss(student_pos_reps, projected_teacher_pos_reps) + \
                                        self.mse_loss(student_qry_reps, projected_teacher_pos_reps) + self.mse_loss(student_pos_reps, projected_teacher_qry_reps))
        self.kd_mse_loss_img = 0.0

        for i in range(batch_size):
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                if student_qry_image_features[i] is not None and teacher_qry_image_features[i] is not None:
                    s_qry_image_features = F.normalize(student_qry_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                    t_qry_image_features = F.normalize(teacher_qry_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                    projected_t_qry_image_features = self.distiller.projectors["t2s_img"](t_qry_image_features)
                    self.kd_mse_loss_img = self.kd_mse_loss_img + self.mse_loss(s_qry_image_features, projected_t_qry_image_features)
            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if student_pos_image_features[i] is not None and teacher_pos_image_features[i] is not None:
                    s_pos_image_features = F.normalize(student_pos_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                    t_pos_image_features = F.normalize(teacher_pos_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1)
                    projected_t_pos_image_features = self.distiller.projectors["t2s_img"](t_pos_image_features)
                    self.kd_mse_loss_img = self.kd_mse_loss_img + self.mse_loss(s_pos_image_features, projected_t_pos_image_features)
        self.kd_mse_loss_img = self.kd_mse_loss_img / batch_size

        self.kd_loss_mse = self.kd_mse_loss_text + self.kd_mse_loss_img
        # self.kd_loss_mse = self.kd_mse_loss_img

        # Attention loss with CKA
        topk_token_text_results = self.extract_top_k_text_token(input_data, teacher_qry_attention, teacher_pos_attention, num_text_qry_tokens, num_text_pos_tokens)
        self.attn_loss = self.compute_attention_loss(teacher_qry_attention, teacher_pos_attention, 
                                                     student_qry_attention, student_pos_attention, 
                                                     input_data, topk_token_text_results, k_layer=3)
        total_loss = (1 - self.kd_loss_weight) * contrastive_loss + self.kd_loss_weight *(self.kd_loss_mse + 0.1 * self.attn_loss)
        # total_loss = contrastive_loss + self.kd_loss_weight *(0.1 * self.attn_loss)
        return {
            "loss": total_loss, 
            "contrastive_loss": contrastive_loss,
            "kd_loss": self.kd_loss_mse + 0.1 * self.attn_loss,
            "attn_loss": self.attn_loss,
            "kd_loss_mse": self.kd_loss_mse,
            # "kd_loss": 0.1 * self.attn_loss,
        }

    def extract_top_k_text_token(self, input_data, teacher_qry_attention, teacher_pos_attention, num_text_qry_tokens, num_text_pos_tokens):
        VISION_START_TOKEN_ID = 151652
        VISION_END_TOKEN_ID = 151656
        BOS_TOKEN_ID = 151643
        teacher_qry_input_ids = input_data['teacher_inputs']['qry']['input_ids']
        teacher_pos_input_ids = input_data['teacher_inputs']['pos']['input_ids']
        batch_size, qry_len = teacher_qry_input_ids.size()
        _, pos_len = teacher_pos_input_ids.size()
        
        qry_atten = teacher_qry_attention[-1].mean(dim=1)
        pos_atten = teacher_pos_attention[-1].mean(dim=1)
        
        qry_importance = qry_atten.sum(dim=1)
        pos_importance = pos_atten.sum(dim=1)
        
        results = []
        for i in range(batch_size):
            qry_ids = teacher_qry_input_ids[i]
            pos_ids = teacher_pos_input_ids[i]
            
            qry_imp = qry_importance[i]
            pos_imp = pos_importance[i]
            
            qry_mask = ((qry_ids < VISION_START_TOKEN_ID) | (qry_ids > VISION_END_TOKEN_ID)) & (qry_ids != BOS_TOKEN_ID)
            pos_mask = ((pos_ids < VISION_START_TOKEN_ID) | (pos_ids > VISION_END_TOKEN_ID)) & (pos_ids != BOS_TOKEN_ID)

            qry_imp = qry_imp * qry_mask.float()
            pos_imp = pos_imp * pos_mask.float()
            # logger.debug(f"Num text tokens - QRY: {num_text_qry_tokens[i]}, POS: {num_text_pos_tokens[i]}")
            qry_topk_idx = torch.topk(qry_imp, min(num_text_qry_tokens[i]//2, int(qry_mask.sum().item()))).indices
            pos_topk_idx = torch.topk(pos_imp, min((num_text_pos_tokens[i]+1)//2, int(pos_mask.sum().item()))).indices

            qry_topk = [(int(idx), int(qry_ids[idx]), float(qry_imp[idx])) for idx in qry_topk_idx if qry_mask[idx]]
            pos_topk = [(int(idx), int(pos_ids[idx]), float(pos_imp[idx])) for idx in pos_topk_idx if pos_mask[idx]]

            results.append({
                "qry_topk": qry_topk,
                "pos_topk": pos_topk
            })
            # logger.debug(f"Instance {i}: QRY top-k tokens (index, id, importance): {qry_topk}")
            # logger.debug(f"Instance {i}: POS top-k tokens (index, id, importance): {pos_topk}")

        return results
    
    def extract_student_indices(self, input_data, topk_results):
        student_qry_input_ids = input_data['student_inputs']['qry']['input_ids']
        student_pos_input_ids = input_data['student_inputs']['pos']['input_ids']
        batch_size = len(topk_results)
        student_indices = []
        
        for i in range(batch_size):
            s_qry_ids = student_qry_input_ids[i].tolist()
            s_pos_ids = student_pos_input_ids[i].tolist()
            
            s_qry_id_to_indices = {}
            for j, token_id in enumerate(s_qry_ids):
                if token_id not in s_qry_id_to_indices:
                    s_qry_id_to_indices[token_id] = []
                s_qry_id_to_indices[token_id].append(j)

            s_pos_id_to_indices = {}
            for j, token_id in enumerate(s_pos_ids):
                if token_id not in s_pos_id_to_indices:
                    s_pos_id_to_indices[token_id] = []
                s_pos_id_to_indices[token_id].append(j)

            qry_topk = topk_results[i]['qry_topk']
            pos_topk = topk_results[i]['pos_topk']
            
            qry_student_idx = []
            used_qry_indices = set()
            for _, token_id, _ in qry_topk:
                if token_id in s_qry_id_to_indices:
                    for index in s_qry_id_to_indices[token_id]:
                        if index not in used_qry_indices:
                            qry_student_idx.append(index)
                            used_qry_indices.add(index)
                            break 

            pos_student_idx = []
            used_pos_indices = set()
            for _, token_id, _ in pos_topk:
                if token_id in s_pos_id_to_indices:
                    for index in s_pos_id_to_indices[token_id]:
                        if index not in used_pos_indices:
                            pos_student_idx.append(index)
                            used_pos_indices.add(index)
                            break
                            
            student_indices.append({
                "qry": qry_student_idx,
                "pos": pos_student_idx
            })

        return student_indices

    def compute_attention_loss(self, teacher_qry_attention, teacher_pos_attention, student_qry_attention, student_pos_attention, input_data, topk_results, k_layer):
        device = input_data['student_inputs']['qry']['input_ids'].device
        # batch_size = input_data['student_inputs']['qry']['input_ids'].size(0)
        batch_size = len(topk_results)
        # logger.debug(f"Batch size for attention loss computation: {batch_size}")
        att_loss_total = 0.0
        
        cka_fn_loss = CKALoss(eps=1e-8).to(device)
        
        teacher_layer_num = len(teacher_qry_attention)
        student_layer_num = len(student_qry_attention)
        layer_per_block = teacher_layer_num // student_layer_num

        # new_teacher_attns = [teacher_qry_attention[i * layer_per_block + layer_per_block - 1] for i in range(student_layer_num)]
        # teacher_qry_attention = new_teacher_attns[-k_layer:]
        # new_teacher_attns = [teacher_pos_attention[i * layer_per_block + layer_per_block - 1] for i in range(student_layer_num)]
        # teacher_pos_attention = new_teacher_attns[-k_layer:]
        # student_last_k_qry = student_qry_attention[-k_layer:]
        # student_last_k_pos = student_pos_attention[-k_layer:]
        teacher_qry_last = teacher_qry_attention[-1]
        teacher_pos_last = teacher_pos_attention[-1]
        student_qry_last = student_qry_attention[-1]
        student_pos_last = student_pos_attention[-1]

        teacher_qry_first = teacher_qry_attention[0]
        teacher_pos_first = teacher_pos_attention[0]
        student_qry_first = student_qry_attention[0]
        student_pos_first = student_pos_attention[0]
        
        student_idx = self.extract_student_indices(input_data, topk_results)
        
        for i in range(batch_size):
            # logger.debug(f"Top k tokens for instance {i}: QRY - {topk_results[i]['qry_topk']}, POS - {topk_results[i]['pos_topk']}")
            qry_topk_idx = [idx for idx, _, _ in topk_results[i]['qry_topk']]
            pos_topk_idx = [idx for idx, _, _ in topk_results[i]['pos_topk']]
            
            if len(qry_topk_idx) == 0 or len(pos_topk_idx) == 0:
                logger.warning(f"Warning: No valid top-k tokens found for instance {i}, skipping attention loss computation.")
                continue
            
            s_qry_topk_idx = [idx for idx in student_idx[i]['qry'] if idx < student_qry_first.size(2)]
            s_pos_topk_idx = [idx for idx in student_idx[i]['pos'] if idx < student_pos_first.size(2)]
            # logger.debug(f"Mapped student indices for instance {i}: QRY - {s_qry_topk_idx}, POS - {s_pos_topk_idx}")
            
            tq_mean = teacher_qry_first[i, :, qry_topk_idx, :].mean(dim=0)
            tp_mean = teacher_pos_first[i, :, pos_topk_idx, :].mean(dim=0)
            sq_mean = student_qry_first[i, :, s_qry_topk_idx, :].mean(dim=0)
            sp_mean = student_pos_first[i, :, s_pos_topk_idx, :].mean(dim=0)
            # print(tq_mean.shape, tp_mean.shape, sq_mean.shape, sp_mean.shape)
            # print(qry_topk_idx, pos_topk_idx, s_qry_topk_idx, s_pos_topk_idx)
            
            # mask -inf
            tq_mean = torch.where(tq_mean <= -1e2, torch.zeros_like(tq_mean), tq_mean)
            sq_mean = torch.where(sq_mean <= -1e2, torch.zeros_like(sq_mean), sq_mean)
            tp_mean = torch.where(tp_mean <= -1e2, torch.zeros_like(tp_mean), tp_mean)
            sp_mean = torch.where(sp_mean <= -1e2, torch.zeros_like(sp_mean), sp_mean)
            # logger.debug(f"Shape of means for instance {i}: tq_mean {tq_mean.shape}, tp_mean {tp_mean.shape}, sq_mean {sq_mean.shape}, sp_mean {sp_mean.shape}")
            # calculate CKA loss
            att_loss = cka_fn_loss(tq_mean, sq_mean) + cka_fn_loss(tp_mean, sp_mean)
            att_loss_total += att_loss / 2
            
            # for teacher_qry_att, teacher_pos_att, student_qry_att, student_pos_att in zip(teacher_qry_attention, teacher_pos_attention, student_last_k_qry, student_last_k_pos):
            #     tq_mean = teacher_qry_att[i, :, qry_topk_idx, :].mean(dim=0)
            #     tp_mean = teacher_pos_att[i, :, pos_topk_idx, :].mean(dim=0)
            #     sq_mean = student_qry_att[i, :, s_qry_topk_idx, :].mean(dim=0)
            #     sp_mean = student_pos_att[i, :, s_pos_topk_idx, :].mean(dim=0)
                
            #     # mask -inf
            #     tq_mean = torch.where(tq_mean <= -1e2, torch.zeros_like(tq_mean), tq_mean)
            #     sq_mean = torch.where(sq_mean <= -1e2, torch.zeros_like(sq_mean), sq_mean)
            #     tp_mean = torch.where(tp_mean <= -1e2, torch.zeros_like(tp_mean), tp_mean)
            #     sp_mean = torch.where(sp_mean <= -1e2, torch.zeros_like(sp_mean), sp_mean)

            #     att_loss = cka_fn_loss(tq_mean, sq_mean) + cka_fn_loss(tp_mean, sp_mean)
            #     att_loss_total += att_loss / 2
            
            tq_mean = teacher_qry_last[i, :, qry_topk_idx, :].mean(dim=0)
            tp_mean = teacher_pos_last[i, :, pos_topk_idx, :].mean(dim=0)
            sq_mean = student_qry_last[i, :, s_qry_topk_idx, :].mean(dim=0)
            sp_mean = student_pos_last[i, :, s_pos_topk_idx, :].mean(dim=0)
            # print(tq_mean.shape, tp_mean.shape, sq_mean.shape, sp_mean.shape)
            # print(qry_topk_idx, pos_topk_idx, s_qry_topk_idx, s_pos_topk_idx)
            # mask -inf
            tq_mean = torch.where(tq_mean <= -1e2, torch.zeros_like(tq_mean), tq_mean)
            sq_mean = torch.where(sq_mean <= -1e2, torch.zeros_like(sq_mean), sq_mean)
            tp_mean = torch.where(tp_mean <= -1e2, torch.zeros_like(tp_mean), tp_mean)
            sp_mean = torch.where(sp_mean <= -1e2, torch.zeros_like(sp_mean), sp_mean)
            att_loss = cka_fn_loss(tq_mean, sq_mean) + cka_fn_loss(tp_mean, sp_mean)
            att_loss_total += att_loss / 2
        # logger.debug(f"Total attention loss before averaging: {att_loss_total.item()}")
        return att_loss_total / batch_size
    
    