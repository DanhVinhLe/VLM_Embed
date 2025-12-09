import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

class CKALoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, SH, TH):
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

class EMOLoss(nn.Module):
    def __init__(self, args):
        super(EMOLoss, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        self.sinkhorn_alpha = 0.1
        self.stopThr = 1e-7
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'cosine'
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
        topk_token_text_results = self.extract_top_k_text_token(input_data, teacher_qry_attention, teacher_pos_attention, num_text_qry_tokens, num_text_pos_tokens)
        self.ot_loss = self.compute_ot_loss_for_retrieval(
            student_qry_output, student_pos_output,
            teacher_qry_output, teacher_pos_output,
            distiller, input_data, topk_token_text_results
        )
        self.attention_loss = self.compute_attention_loss(
            teacher_qry_attention, teacher_pos_attention,
            student_qry_attention, student_pos_attention,
            input_data, topk_token_text_results,
            k_layer=3
        )
        total_loss = self.kd_loss_weight * contrastive_loss + (1 - self.kd_loss_weight) * (self.ot_loss + self.attention_loss)
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'ot_loss': self.ot_loss,
            'kd_loss': self.attention_loss + self.ot_loss
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
        
        qry_importance = qry_atten.mean(dim=1)
        pos_importance = pos_atten.mean(dim=1)
        
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
            qry_topk_idx = torch.topk(qry_imp, min(num_text_qry_tokens[i]//3, int(qry_mask.sum().item()))).indices
            pos_topk_idx = torch.topk(pos_imp, min((num_text_pos_tokens[i]+1)//3, int(pos_mask.sum().item()))).indices

            qry_topk = [(int(idx), int(qry_ids[idx]), float(qry_imp[idx])) for idx in qry_topk_idx if qry_mask[idx]]
            pos_topk = [(int(idx), int(pos_ids[idx]), float(pos_imp[idx])) for idx in pos_topk_idx if pos_mask[idx]]

            results.append({
                "qry_topk": qry_topk,
                "pos_topk": pos_topk
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
    
    def compute_token_importance(self, attention_weights, num_tokens):
        """
        Compute token importance from attention weights
        Args:
            attention_weights: [num_heads, seq_len, seq_len] or [seq_len, seq_len]
            num_tokens: number of valid tokens
        Returns:
            norm_importance: normalized importance scores [num_tokens]
        """
        device = attention_weights.device
        
        if len(attention_weights.shape) == 3:
            avg_attention = attention_weights.mean(dim=0)
        else:
            avg_attention = attention_weights
        
        seq_len = min(avg_attention.shape[0], num_tokens)
        avg_attention = avg_attention[:seq_len, :seq_len]
        token_importance = avg_attention.sum(dim=0)
        norm_importance = torch.softmax(token_importance, dim=0)
        return norm_importance
    
    def compute_ot_loss_for_retrieval(self, student_qry_output, student_pos_output, 
                                   teacher_qry_output, teacher_pos_output, 
                                   distiller, input_data, topk_results):
        """
        Compute OT loss for retrieval task between teacher and student hidden states
        using full text tokens, with importance mass computed from top-k tokens
        """
        # Unpack outputs
        student_qry_rep, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_rep, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
        teacher_qry_rep, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
        teacher_pos_rep, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
        
        VISION_START_TOKEN_ID = 151652
        VISION_END_TOKEN_ID = 151656
        BOS_TOKEN_ID = 151643
        
        device = input_data['student_inputs']['qry']['input_ids'].device
        batch_size = len(topk_results)
        
        if not hasattr(distiller, 'projectors') or "t2s" not in distiller.projectors:
            raise AttributeError("Projector 't2s' not found in distiller.projectors for OT loss computation.")
        
        student_idx = self.extract_student_indices(input_data, topk_results)
        total_ot_loss = 0.0
        
        for i in range(batch_size):
            # Get top-k token indices (only for computing importance)
            qry_topk_idx = [idx for idx, _, _ in topk_results[i]['qry_topk']]
            pos_topk_idx = [idx for idx, _, _ in topk_results[i]['pos_topk']]
            
            if len(qry_topk_idx) == 0 or len(pos_topk_idx) == 0:
                print(f"Warning: No top-k tokens found for OT loss computation for instance {i}")
                continue
            
            # === Get all text token indices for teacher ===
            teacher_qry_input_ids = input_data['teacher_inputs']['qry']['input_ids'][i]
            teacher_pos_input_ids = input_data['teacher_inputs']['pos']['input_ids'][i]
            
            teacher_qry_text_mask = ((teacher_qry_input_ids < VISION_START_TOKEN_ID) | 
                                    (teacher_qry_input_ids > VISION_END_TOKEN_ID)) & \
                                    (teacher_qry_input_ids != BOS_TOKEN_ID)
            teacher_pos_text_mask = ((teacher_pos_input_ids < VISION_START_TOKEN_ID) | 
                                    (teacher_pos_input_ids > VISION_END_TOKEN_ID)) & \
                                    (teacher_pos_input_ids != BOS_TOKEN_ID)
            
            teacher_qry_text_indices = torch.where(teacher_qry_text_mask)[0].tolist()
            teacher_pos_text_indices = torch.where(teacher_pos_text_mask)[0].tolist()
            
            if len(teacher_qry_text_indices) == 0 or len(teacher_pos_text_indices) == 0:
                print(f"Warning: No text tokens found for instance {i}")
                continue
            
            # === Get all text token indices for student ===
            student_qry_input_ids = input_data['student_inputs']['qry']['input_ids'][i]
            student_pos_input_ids = input_data['student_inputs']['pos']['input_ids'][i]
            
            student_qry_text_mask = ((student_qry_input_ids < VISION_START_TOKEN_ID) | 
                                    (student_qry_input_ids > VISION_END_TOKEN_ID)) & \
                                    (student_qry_input_ids != BOS_TOKEN_ID)
            student_pos_text_mask = ((student_pos_input_ids < VISION_START_TOKEN_ID) | 
                                    (student_pos_input_ids > VISION_END_TOKEN_ID)) & \
                                    (student_pos_input_ids != BOS_TOKEN_ID)
            
            student_qry_text_indices = torch.where(student_qry_text_mask)[0].tolist()
            student_pos_text_indices = torch.where(student_pos_text_mask)[0].tolist()
            
            if len(student_qry_text_indices) == 0 or len(student_pos_text_indices) == 0:
                print(f"Warning: No text tokens found for student instance {i}")
                continue
            
            # === Compute importance mass for Query (using top-k) ===
            teacher_qry_attn = teacher_qry_attention[-1][i]
            teacher_qry_importance_full = self.compute_token_importance(teacher_qry_attn, teacher_qry_input_ids.size(0))
            teacher_qry_importance = teacher_qry_importance_full[teacher_qry_text_indices]
            
            # Project importance to student tokens
            student_qry_importance = torch.zeros(len(student_qry_text_indices), device=device)
            for t_idx, teacher_idx in enumerate(teacher_qry_text_indices):
                teacher_token_id = teacher_qry_input_ids[teacher_idx].item()
                for s_idx, student_idx in enumerate(student_qry_text_indices):
                    student_token_id = student_qry_input_ids[student_idx].item()
                    if teacher_token_id == student_token_id:
                        student_qry_importance[s_idx] = teacher_qry_importance[t_idx]
                        break
            
            min_importance = teacher_qry_importance.min().item() if len(teacher_qry_importance) > 0 else 0.0
            student_qry_importance = torch.where(student_qry_importance == 0,
                                                torch.tensor(min_importance, device=device),
                                                student_qry_importance)
            
            student_qry_importance = torch.softmax(student_qry_importance, dim=0)
            teacher_qry_importance = torch.softmax(teacher_qry_importance, dim=0)
            
            teacher_qry_mass = teacher_qry_importance.view(-1, 1)
            student_qry_mass = student_qry_importance.view(-1, 1)
            
            # === Compute importance mass for Positive (using top-k) ===
            teacher_pos_attn = teacher_pos_attention[-1][i]
            teacher_pos_importance_full = self.compute_token_importance(teacher_pos_attn, teacher_pos_input_ids.size(0))
            teacher_pos_importance = teacher_pos_importance_full[teacher_pos_text_indices]
            
            student_pos_importance = torch.zeros(len(student_pos_text_indices), device=device)
            for t_idx, teacher_idx in enumerate(teacher_pos_text_indices):
                teacher_token_id = teacher_pos_input_ids[teacher_idx].item()
                for s_idx, student_idx in enumerate(student_pos_text_indices):
                    student_token_id = student_pos_input_ids[student_idx].item()
                    if teacher_token_id == student_token_id:
                        student_pos_importance[s_idx] = teacher_pos_importance[t_idx]
                        break
            
            min_importance = teacher_pos_importance.min().item() if len(teacher_pos_importance) > 0 else 0.0
            student_pos_importance = torch.where(student_pos_importance == 0,
                                                torch.tensor(min_importance, device=device),
                                                student_pos_importance)
            
            student_pos_importance = torch.softmax(student_pos_importance, dim=0)
            teacher_pos_importance = torch.softmax(teacher_pos_importance, dim=0)
            
            teacher_pos_mass = teacher_pos_importance.view(-1, 1)
            student_pos_mass = student_pos_importance.view(-1, 1)
            
            # === Extract FULL text token hidden states ===
            student_qry_text_hidden = student_qry_hidden_states[-1][i][student_qry_text_indices, :]
            student_pos_text_hidden = student_pos_hidden_states[-1][i][student_pos_text_indices, :]
            projected_teacher_qry_text_hidden = distiller.projectors["t2s"](teacher_qry_hidden_states[-1][i][teacher_qry_text_indices, :])
            projected_teacher_pos_text_hidden = distiller.projectors["t2s"](teacher_pos_hidden_states[-1][i][teacher_pos_text_indices, :])
            
            # === Compute cost matrices ===
            if self.ot_dist_type == 'cosine':
                cost_matrix_qry = self.pairwise_cosine_distance(student_qry_text_hidden, projected_teacher_qry_text_hidden)
                cost_matrix_pos = self.pairwise_cosine_distance(student_pos_text_hidden, projected_teacher_pos_text_hidden)
            elif self.ot_dist_type == 'euclidean':
                cost_matrix_qry = self.pairwise_euclidean_distance(student_qry_text_hidden, projected_teacher_qry_text_hidden)
                cost_matrix_pos = self.pairwise_euclidean_distance(student_pos_text_hidden, projected_teacher_pos_text_hidden)
            else:
                raise ValueError(f"Unsupported OT distance type: {self.ot_dist_type}")
            
            # === Compute OT loss ===
            ot_loss_qry, _ = self.sinkhorn(cost_matrix_qry, student_qry_mass, teacher_qry_mass, num_iters=self.OT_max_iter)
            ot_loss_pos, _ = self.sinkhorn(cost_matrix_pos, student_pos_mass, teacher_pos_mass, num_iters=self.OT_max_iter)
            total_ot_loss = total_ot_loss + 0.5 * (ot_loss_qry + ot_loss_pos)
        
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
    
    def compute_attention_loss(self, teacher_qry_attention, teacher_pos_attention, student_qry_attention, student_pos_attention, input_data, topk_results, k_layer):
        device = input_data['student_inputs']['qry']['input_ids'].device
        batch_size = len(topk_results)
        att_loss_total = 0.0
        
        cka_fn_loss = CKALoss(eps=1e-8).to(device)
        
        teacher_layer_num = len(teacher_qry_attention)
        student_layer_num = len(student_qry_attention)
        layer_per_block = teacher_layer_num // student_layer_num

        # Lấy k layer cuối cùng của student
        student_last_k_qry = student_qry_attention[-k_layer:]
        student_last_k_pos = student_pos_attention[-k_layer:]
        
        teacher_qry_mapped = []
        teacher_pos_mapped = []
        for i in range(student_layer_num - k_layer, student_layer_num):
            teacher_qry_mapped.append(teacher_qry_attention[i * layer_per_block + layer_per_block - 1])
            teacher_pos_mapped.append(teacher_pos_attention[i * layer_per_block + layer_per_block - 1])
        
        student_idx = self.extract_student_indices(input_data, topk_results)
        
        for i in range(batch_size):
            qry_topk_idx = [idx for idx, _, _ in topk_results[i]['qry_topk']]
            pos_topk_idx = [idx for idx, _, _ in topk_results[i]['pos_topk']]
            
            if len(qry_topk_idx) == 0 or len(pos_topk_idx) == 0:
                print("Warning: No valid top-k tokens found for instance {}, skipping attention loss computation.".format(i))
                continue
            
            s_qry_topk_idx = [idx for idx in student_idx[i]['qry'] if idx < student_last_k_qry[0].size(2)]
            s_pos_topk_idx = [idx for idx in student_idx[i]['pos'] if idx < student_last_k_pos[0].size(2)]
            
            # Tính attention loss cho k layer cuối
            for teacher_qry_att, teacher_pos_att, student_qry_att, student_pos_att in zip(
                teacher_qry_mapped, teacher_pos_mapped, student_last_k_qry, student_last_k_pos
            ):
                tq_mean = teacher_qry_att[i, :, qry_topk_idx, :].mean(dim=0)
                tp_mean = teacher_pos_att[i, :, pos_topk_idx, :].mean(dim=0)
                sq_mean = student_qry_att[i, :, s_qry_topk_idx, :].mean(dim=0)
                sp_mean = student_pos_att[i, :, s_pos_topk_idx, :].mean(dim=0)
                
                # Mask -inf values
                tq_mean = torch.where(tq_mean <= -1e2, torch.zeros_like(tq_mean), tq_mean)
                sq_mean = torch.where(sq_mean <= -1e2, torch.zeros_like(sq_mean), sq_mean)
                tp_mean = torch.where(tp_mean <= -1e2, torch.zeros_like(tp_mean), tp_mean)
                sp_mean = torch.where(sp_mean <= -1e2, torch.zeros_like(sp_mean), sp_mean)

                # Tính CKA loss cho layer hiện tại
                att_loss = cka_fn_loss(tq_mean, sq_mean) + cka_fn_loss(tp_mean, sp_mean)
                att_loss_total += att_loss / 2
        
        # Average over batch_size và k_layer
        return att_loss_total / (batch_size)