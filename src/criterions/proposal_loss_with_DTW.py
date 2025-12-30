import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from .soft_DTW import SoftDTW
import logging
from tslearn.metrics import SoftDTWLossPyTorch

logging.getLogger("numba").setLevel(logging.ERROR)

import wandb

class ProposalLossWithDTW(nn.Module):
    def __init__(self, args):
        super(ProposalLossWithDTW, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        self.sinkhorn_alpha = 0.1
        self.stopThr = 1e-7
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'cosine'
        self.dtw_criterion = SoftDTWLossPyTorch(gamma=0.1, normalize=True)
        self.mse_loss = nn.MSELoss(reduction='mean')
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor):
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
        num_text_qry_tokens = ((teacher_qry_input['input_ids'] < 151652) | (teacher_qry_input['input_ids'] > 151656)).sum(dim=1)
        num_text_pos_tokens = ((teacher_pos_input['input_ids'] < 151652) | (teacher_pos_input['input_ids'] > 151656)).sum(dim=1)
        batch_size = student_qry_input['input_ids'].size(0)
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, _ = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, _ = teacher_pos_output

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, _ = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, _ = student_pos_output

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
        
        # RKD Loss
        distance_loss = self.compute_distance_loss(student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps)
        angle_loss = self.compute_angle_loss(student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps)
        self.kd_loss_rkd = (0.5 * distance_loss + 0.5 * angle_loss)

        # KD loss with DTW
        projected_teacher_qry_reps = self.distiller.projectors["t2s_txt"](teacher_qry_reps)
        projected_teacher_pos_reps = self.distiller.projectors["t2s_txt"](teacher_pos_reps)
        self.kd_loss_mse_seq = 0.25 * (self.mse_loss(student_qry_reps, projected_teacher_qry_reps) + self.mse_loss(student_pos_reps, projected_teacher_pos_reps) + self.mse_loss(student_qry_reps, projected_teacher_pos_reps) + self.mse_loss(student_pos_reps, projected_teacher_qry_reps))
        self.kd_loss_dtw_image = 0.0
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0
        for i in range(batch_size):
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    if student_qry_image_features[cur_idx_qry_img] is not None and teacher_qry_image_features[cur_idx_qry_img] is not None:
                        s_qry_image_features = F.normalize(student_qry_image_features[cur_idx_qry_img], p=2, dim=-1)
                        t_qry_image_features = F.normalize(teacher_qry_image_features[cur_idx_qry_img], p=2, dim=-1)
                        projected_t_qry_image_features = self.distiller.projectors["t2s_img"](t_qry_image_features)
                        s = s_qry_image_features.unsqueeze(0).to(torch.float32)
                        t = projected_t_qry_image_features.unsqueeze(0).to(torch.float32)
                        self.kd_loss_dtw_image = self.kd_loss_dtw_image + self.dtw_criterion(s, t).mean()
                        self.kd_loss_dtw_image = self.kd_loss_dtw_image.to(torch.bfloat16)
                        cur_idx_qry_img += 1
            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    if student_pos_image_features[cur_idx_pos_img] is not None and teacher_pos_image_features[cur_idx_pos_img] is not None:
                        s_pos_image_features = F.normalize(student_pos_image_features[cur_idx_pos_img], p=2, dim=-1)
                        t_pos_image_features = F.normalize(teacher_pos_image_features[cur_idx_pos_img], p=2, dim=-1)
                        projected_t_pos_image_features = self.distiller.projectors["t2s_img"](t_pos_image_features)
                        s = s_pos_image_features.unsqueeze(0).to(torch.float32)
                        t = projected_t_pos_image_features.unsqueeze(0).to(torch.float32)
                        self.kd_loss_dtw_image = self.kd_loss_dtw_image + self.dtw_criterion(s, t).mean()
                        self.kd_loss_dtw_image = self.kd_loss_dtw_image.to(torch.bfloat16)
                        cur_idx_pos_img += 1
        self.kd_loss_dtw_image = self.kd_loss_dtw_image / batch_size

        self.kd_loss_dtw = 0.5 * self.kd_loss_dtw_image

        # OT loss
        # topk_token_text_results = self.extract_top_k_text_token(input_data, teacher_qry_attention, teacher_pos_attention, num_text_qry_tokens, num_text_pos_tokens)
        topk_token_text_results = self.extract_top_k_text_token(input_data, teacher_qry_attention, teacher_pos_attention)
        self.ot_loss = self.compute_ot_loss(student_qry_output, student_pos_output, teacher_qry_output, teacher_pos_output, distiller, input_data, topk_token_text_results)
        total_loss = contrastive_loss + self.kd_loss_weight * (self.kd_loss_rkd + self.kd_loss_mse_seq + 0.5 * self.kd_loss_dtw_image + 0.5 * self.ot_loss)
        # total_loss = contrastive_loss + self.kd_loss_weight *(0.1 * self.attn_loss)
        return {
            "loss": total_loss, 
            "contrastive_loss": contrastive_loss,
            "kd_loss": self.kd_loss_rkd + self.kd_loss_mse_seq + 0.5 * self.kd_loss_dtw_image + 0.5 * self.ot_loss,
            "kd_loss_rkd": self.kd_loss_rkd,
            "kd_loss_dtw": self.kd_loss_dtw,
            "ot_loss": self.ot_loss,
        }

    def extract_top_k_text_token(self, input_data, teacher_qry_attention, teacher_pos_attention, threshold=0.8):
        VISION_START_TOKEN_ID = 151652
        VISION_END_TOKEN_ID = 151656
        BOS_TOKEN_ID = 151643
        
        teacher_qry_input_ids = input_data['teacher_inputs']['qry']['input_ids']
        teacher_pos_input_ids = input_data['teacher_inputs']['pos']['input_ids']
        batch_size, qry_len = teacher_qry_input_ids.size()
        _, pos_len = teacher_pos_input_ids.size()
        
        qry_atten = teacher_qry_attention[-1].mean(dim=1)
        pos_atten = teacher_pos_attention[-1].mean(dim=1)
        
        qry_importance = qry_atten[:, -1, :]
        pos_importance = pos_atten[:, -1, :]
        
        results = []
        for i in range(batch_size):
            qry_ids = teacher_qry_input_ids[i]
            pos_ids = teacher_pos_input_ids[i]
            
            qry_imp = qry_importance[i]
            pos_imp = pos_importance[i]
            
            # Mask để chỉ lấy text tokens (loại bỏ vision tokens và BOS)
            qry_mask = ((qry_ids < VISION_START_TOKEN_ID) | (qry_ids > VISION_END_TOKEN_ID)) & (qry_ids != BOS_TOKEN_ID)
            pos_mask = ((pos_ids < VISION_START_TOKEN_ID) | (pos_ids > VISION_END_TOKEN_ID)) & (pos_ids != BOS_TOKEN_ID)
            
            # Áp dụng mask
            qry_imp_masked = qry_imp * qry_mask.float()
            pos_imp_masked = pos_imp * pos_mask.float()
            
            # Loại bỏ token cuối cùng khỏi việc tính toán (vì sẽ luôn được thêm vào)
            qry_imp_without_last = qry_imp_masked.clone()
            pos_imp_without_last = pos_imp_masked.clone()
            qry_imp_without_last[-1] = 0
            pos_imp_without_last[-1] = 0
            
            qry_total = qry_imp_without_last.sum()
            pos_total = pos_imp_without_last.sum()
            
            # Sort và chọn tokens cho query
            qry_sorted_imp, qry_sorted_idx = torch.sort(qry_imp_without_last, descending=True)
            qry_cumsum = torch.cumsum(qry_sorted_imp, dim=0)
            qry_target = qry_total * threshold
            qry_num_tokens = (qry_cumsum < qry_target).sum() + 1  # +1 để đảm bảo vượt qua threshold
            qry_selected_idx = qry_sorted_idx[:qry_num_tokens]
            
            # Sort và chọn tokens cho positive
            pos_sorted_imp, pos_sorted_idx = torch.sort(pos_imp_without_last, descending=True)
            pos_cumsum = torch.cumsum(pos_sorted_imp, dim=0)
            pos_target = pos_total * threshold
            pos_num_tokens = (pos_cumsum < pos_target).sum() + 1
            pos_selected_idx = pos_sorted_idx[:pos_num_tokens]
            
            
            qry_valid_idx = qry_selected_idx[qry_mask[qry_selected_idx]]
            pos_valid_idx = pos_selected_idx[pos_mask[pos_selected_idx]]
            
            # Thêm token cuối vào kết quả
            qry_final_idx = torch.cat([qry_valid_idx, torch.tensor([qry_len - 1], device=qry_valid_idx.device)])
            pos_final_idx = torch.cat([pos_valid_idx, torch.tensor([pos_len - 1], device=pos_valid_idx.device)])
            
            qry_selected_imp = qry_imp_masked[qry_final_idx]
            pos_selected_imp = pos_imp_masked[pos_final_idx]
            
            qry_imp_sum = qry_selected_imp.sum()
            pos_imp_sum = pos_selected_imp.sum()
            
            # Normalize importance
            qry_imp_normalized = qry_selected_imp / qry_imp_sum if qry_imp_sum > 0 else qry_selected_imp
            pos_imp_normalized = pos_selected_imp / pos_imp_sum if pos_imp_sum > 0 else pos_selected_imp
            
            
            # Tạo kết quả với mask để loại bỏ các token không hợp lệ
            qry_topk = [(int(idx), int(qry_ids[idx]), float(imp)) 
                        for idx, imp in zip(qry_final_idx, qry_imp_normalized)]
            pos_topk = [(int(idx), int(pos_ids[idx]), float(imp)) 
                        for idx, imp in zip(pos_final_idx, pos_imp_normalized)]
            
            results.append({
                "qry_topk": qry_topk,
                "pos_topk": pos_topk,
            })
        
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
    
    # Compute OT loss
    def compute_ot_loss(self, student_qry_output, student_pos_output, teacher_qry_output, teacher_pos_output, distiller, input_data, topk_results):
        student_qry_rep, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_rep, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
        teacher_qry_rep, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
        teacher_pos_rep, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output

        device = input_data['student_inputs']['qry']['input_ids'].device
        batch_size = len(topk_results)
        
        student_idx = self.extract_student_indices(input_data, topk_results)
        total_ot_loss = 0.0
        
        for i in range(batch_size):
            qry_topk_idx = [idx for idx, _, _ in topk_results[i]['qry_topk']]
            pos_topk_idx = [idx for idx, _, _ in topk_results[i]['pos_topk']]
            
            qry_topk_importance = [imp for _, _, imp in topk_results[i]['qry_topk']]
            pos_topk_importance = [imp for _, _, imp in topk_results[i]['pos_topk']]

            if len(qry_topk_idx) == 0 or len(pos_topk_idx) == 0:
                print("Warning: No top-k tokens found for OT loss computation for instance ", i)
                continue
            
            s_qry_topk_idx = [idx for idx in student_idx[i]['qry'] if idx < student_qry_hidden_states[-1][i].size(0)]
            s_pos_topk_idx = [idx for idx in student_idx[i]['pos'] if idx < student_pos_hidden_states[-1][i].size(0)]


            # teacher_qry_attention_matrix = teacher_qry_attention[-1][i]
            # teacher_pos_attention_matrix = teacher_pos_attention[-1][i]
            
            # teacher_qry_topk_attn = teacher_qry_attention_matrix[:, -1, qry_topk_idx]
            # teacher_pos_topk_attn = teacher_pos_attention_matrix[:, -1, pos_topk_idx]

            teacher_qry_topk_importance = torch.tensor(qry_topk_importance, device=device)
            teacher_pos_topk_importance = torch.tensor(pos_topk_importance, device=device)
            
            attn_mask_stu_qry = input_data['student_inputs']['qry']['attention_mask'][i]
            attn_mask_stu_pos = input_data['student_inputs']['pos']['attention_mask'][i]
            
            if attn_mask_stu_qry.dim() > 1:
                attn_mask_stu_qry = attn_mask_stu_qry.view(-1)
            if attn_mask_stu_pos.dim() > 1:
                attn_mask_stu_pos = attn_mask_stu_pos.view(-1)
            num_student_qry_pad_token = int((attn_mask_stu_qry == 0).sum().item())
            num_student_pos_pad_token = int((attn_mask_stu_pos == 0).sum().item())
            
            student_qry_attention_matrix = student_qry_attention[-1][i]
            student_pos_attention_matrix = student_pos_attention[-1][i]
            student_qry_topk_attn = student_qry_attention_matrix[:, -(num_student_qry_pad_token + 1), s_qry_topk_idx]
            student_pos_topk_attn = student_pos_attention_matrix[:, -(num_student_pos_pad_token + 1), s_pos_topk_idx]

            # student_qry_topk_importance = torch.softmax(student_qry_topk_attn.mean(dim=0), dim=0)
            # student_pos_topk_importance = torch.softmax(student_pos_topk_attn.mean(dim=0), dim=0)
            student_qry_topk_attn_mean = student_qry_topk_attn.mean(dim=0)
            student_pos_topk_attn_mean = student_pos_topk_attn.mean(dim=0)
            
            student_qry_topk_importance = student_qry_topk_attn_mean / student_qry_topk_attn_mean.sum() if student_qry_topk_attn_mean.sum() > 0 else student_qry_topk_attn_mean
            student_pos_topk_importance = student_pos_topk_attn_mean / student_pos_topk_attn_mean.sum() if student_pos_topk_attn_mean.sum() > 0 else student_pos_topk_attn_mean
            
            teacher_qry_mass = teacher_qry_topk_importance.view(-1, 1)
            teacher_pos_mass = teacher_pos_topk_importance.view(-1, 1)
            student_qry_mass = student_qry_topk_importance.view(-1, 1)
            student_pos_mass = student_pos_topk_importance.view(-1, 1)
            
            student_qry_topk_hidden = student_qry_hidden_states[-1][i][s_qry_topk_idx, :]
            student_pos_topk_hidden = student_pos_hidden_states[-1][i][s_pos_topk_idx, :]
            projected_teacher_qry_topk_hidden = distiller.projectors["t2s"](teacher_qry_hidden_states[-1][i][qry_topk_idx, :])
            projected_teacher_pos_topk_hidden = distiller.projectors["t2s"](teacher_pos_hidden_states[-1][i][pos_topk_idx, :])

            if self.ot_dist_type == 'cosine':
                cost_matrix_qry = self.pairwise_cosine_distance(student_qry_topk_hidden, projected_teacher_qry_topk_hidden)
                cost_matrix_pos = self.pairwise_cosine_distance(student_pos_topk_hidden, projected_teacher_pos_topk_hidden)
            elif self.ot_dist_type == 'euclidean':
                cost_matrix_qry = self.pairwise_euclidean_distance(student_qry_topk_hidden, projected_teacher_qry_topk_hidden)
                cost_matrix_pos = self.pairwise_euclidean_distance(student_pos_topk_hidden, projected_teacher_pos_topk_hidden)
            else:
                raise ValueError(f"Unsupported OT distance type: {self.ot_dist_type}")
            ot_loss_qry, _ = self.sinkhorn(cost_matrix_qry, student_qry_mass, teacher_qry_mass, num_iters=self.OT_max_iter)
            ot_loss_pos, _ = self.sinkhorn(cost_matrix_pos, student_pos_mass, teacher_pos_mass, num_iters=self.OT_max_iter)
            total_ot_loss = total_ot_loss + 0.5 * (ot_loss_qry + ot_loss_pos)
        
        if not hasattr(distiller, 'projectors') or "t2s" not in distiller.projectors:
            raise AttributeError("Projector 't2s' not found in distiller.projectors for OT loss computation.")
        
        total_ot_loss = total_ot_loss / batch_size
        return total_ot_loss
    
    def pairwise_euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2)
    
    def pairwise_cosine_distance(self, a, b, eps=1e-8):
        """
        Computes pairwise cosine distance with numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=a.dtype))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=b.dtype))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        
        sim_mt = 1 - sim_mt
        return sim_mt

    
    def sinkhorn(self, cost_matrix, a, b, num_iters=None):
        if num_iters is None:
            num_iters = self.OT_max_iter
        
        m, n = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype
        
        if m == 0 or n == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), torch.zeros((m, n), device=device, dtype=dtype)
        
        if a.dim() == 1:
            a = a.view(-1, 1)
        if b.dim() == 1:
            b = b.view(-1, 1)
            
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        
        if a.shape[0] != m:
            a = torch.ones(m, 1, device=device, dtype=dtype) / m
        if b.shape[0] != n:
            b = torch.ones(n, 1, device=device, dtype=dtype) / n
        
        if torch.sum(a) < self.epsilon or torch.sum(b) < self.epsilon:
            a = torch.ones(m, 1, device=device, dtype=dtype) / m
            b = torch.ones(n, 1, device=device, dtype=dtype) / n
        else:
            a = a / torch.sum(a)
            b = b / torch.sum(b)       
        K = torch.exp(-cost_matrix / self.sinkhorn_alpha)
        u = torch.ones(m, 1, device=device, dtype=dtype)
        v = torch.ones(n, 1, device=device, dtype=dtype)
        
        for _ in range(num_iters):
            u_prev = u.clone()  
            KTu = torch.matmul(K.t(), u)
            v = b / (KTu + self.epsilon)           
            Kv = torch.matmul(K, v)
            u = a / (Kv + self.epsilon)        
            err = torch.norm(u - u_prev, p=float('inf'))
            if err < self.stopThr:
                break
        P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())
        ot_loss = torch.sum(P * cost_matrix)
        return ot_loss, P

    # Code for RKD Loss
    def pairwise_distance(self, x):
        norm = (x**2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2.0 * torch.mm(x, x.t())
        return dist
    
    def compute_distance_loss(self, student_qry, student_pos, teacher_qry, teacher_pos):
        
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)
        
        dist_student = self.pairwise_distance(student_repr)
        dist_teacher = self.pairwise_distance(teacher_repr)
        
        mask = torch.triu(torch.ones_like(dist_student), diagonal=1).bool()
        dist_student = dist_student[mask]
        dist_teacher = dist_teacher[mask]
        
        mean_td = dist_teacher.mean().detach() + 1e-8
        mean_sd = dist_student.mean().detach() + 1e-8
        
        dist_student = dist_student / mean_sd
        dist_teacher = dist_teacher / mean_td
        
        diff = dist_student - dist_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        loss = loss.mean()
        return loss
    
    def angle_potentials(self, x):
        n = x.size(0)
        diffs = x.unsqueeze(0) - x.unsqueeze(1)
        norms = torch.norm(diffs, dim=-1, keepdim=True) + 1e-8
        e = diffs / norms
        
        cos_angles = torch.einsum('ijd,kjd->ijk', e, e)
        return cos_angles
    
    def compute_angle_loss(self, student_qry, student_pos, teacher_qry, teacher_pos):
        
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)
        
        psi_student = self.angle_potentials(student_repr)
        psi_teacher = self.angle_potentials(teacher_repr)
        
        n = psi_student.size(0)
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
        loss = loss.mean()
        return loss
    
        