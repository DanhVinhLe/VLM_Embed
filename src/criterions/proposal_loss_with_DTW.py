import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .soft_DTW import SoftDTW

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

class ProposalLossWithDTW(nn.Module):
    def __init__(self, args):
        super(ProposalLossWithDTW, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        self.sinkhorn_alpha = 0.1
        self.stop_threshold = 1e-6
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'attention'
        self.dtw_criterion = SoftDTW(use_cuda=True, gamma=0.0001, normalize=False)
    
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
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention = teacher_model.encode_input(teacher_pos_input)
        
        student_qry_reps, student_qry_image_features, student_qry_attention = student_model.encode_input(student_qry_input)
        student_pos_reps, student_pos_image_features, student_pos_attention = student_model.encode_input(student_pos_input)
        
        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_qry_reps.size(0) // student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)

        # KD loss with DTW
        self.kd_loss_dtw_text = self.dtw_criterion(student_qry_reps.unsqueeze(dim=-1), teacher_qry_reps.unsqueeze(dim=-1)).mean() + self.dtw_criterion(student_pos_reps.unsqueeze(dim=-1), teacher_pos_reps.unsqueeze(dim=-1)).mean() + \
                           self.dtw_criterion(student_qry_reps.unsqueeze(dim=-1), teacher_pos_reps.unsqueeze(dim=-1)).mean() + self.dtw_criterion(student_pos_reps.unsqueeze(dim=-1), teacher_qry_reps.unsqueeze(dim=-1)).mean()
        self.kd_loss_dtw_image = 0.0

        for i in range(batch_size):
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                s_qry_image_features = F.normalize(student_qry_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1).unsqueeze(dim=-1)
                t_qry_image_features = F.normalize(teacher_qry_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1).unsqueeze(dim=-1)
                self.kd_loss_dtw_image = self.kd_loss_dtw_image + self.dtw_criterion(s_qry_image_features, t_qry_image_features).mean()
            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                s_pos_image_features = F.normalize(student_pos_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1).unsqueeze(dim=-1)
                t_pos_image_features = F.normalize(teacher_pos_image_features[i].mean(dim=0, keepdim=True), p=2, dim=-1).unsqueeze(dim=-1)
                self.kd_loss_dtw_image = self.kd_loss_dtw_image + self.dtw_criterion(s_pos_image_features, t_pos_image_features).mean()
        self.kd_loss_dtw_image = self.kd_loss_dtw_image / batch_size

        self.kd_loss_dtw = self.kd_loss_dtw_text + self.kd_loss_dtw_image
        # self.kd_loss_dtw = self.kd_loss_dtw_image

        # Attention loss with CKA
        topk_token_text_results = self.extract_top_k_text_token(input_data, teacher_qry_attention, teacher_pos_attention, num_text_qry_tokens, num_text_pos_tokens)
        self.attn_loss = self.compute_attention_loss(teacher_qry_attention, teacher_pos_attention, 
                                                     student_qry_attention, student_pos_attention, 
                                                     input_data, topk_token_text_results, k_layer=1)
        total_loss = contrastive_loss + self.kd_loss_weight *(self.kd_loss_dtw + 0.3 * self.attn_loss)
        # total_loss = contrastive_loss + self.kd_loss_weight *(0.1 * self.attn_loss)
        return {
            "loss": total_loss, 
            "contrastive_loss": contrastive_loss,
            "kd_loss": self.kd_loss_dtw + 0.1 * self.attn_loss,
            # "kd_loss": 0.1 * self.attn_loss,
        }

    def extract_top_k_text_token(self, input_data, teacher_qry_attention, teacher_pos_attention, num_text_qry_tokens, num_text_pos_tokens):
        VISION_START_TOKEN_ID = 151652
        VISION_END_TOKEN_ID = 151656
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
            
            qry_mask = (qry_ids < VISION_START_TOKEN_ID) | (qry_ids > VISION_END_TOKEN_ID)
            pos_mask = (pos_ids < VISION_START_TOKEN_ID) | (pos_ids > VISION_END_TOKEN_ID)

            qry_imp = qry_imp * qry_mask.float()
            pos_imp = pos_imp * pos_mask.float()
            # print(f"Num text tokens - QRY: {num_text_qry_tokens[i]}, POS: {num_text_pos_tokens[i]}")
            qry_topk_idx = torch.topk(qry_imp, min(num_text_qry_tokens[i]//2, int(qry_mask.sum().item()))).indices
            pos_topk_idx = torch.topk(pos_imp, min((num_text_pos_tokens[i]+1)//2, int(pos_mask.sum().item()))).indices

            qry_topk = [(int(idx), int(qry_ids[idx]), float(qry_imp[idx])) for idx in qry_topk_idx if qry_mask[idx]]
            pos_topk = [(int(idx), int(pos_ids[idx]), float(pos_imp[idx])) for idx in pos_topk_idx if pos_mask[idx]]

            results.append({
                "qry_topk": qry_topk,
                "pos_topk": pos_topk
            })
            # print(f"Instance {i}: QRY top-k tokens (index, id, importance): {qry_topk}")
            # print(f"Instance {i}: POS top-k tokens (index, id, importance): {pos_topk}")

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
        # print("Batch size for attention loss computation:", batch_size)
        att_loss_total = 0.0
        
        cka_fn_loss = CKALoss(eps=1e-8).to(device)
        
        teacher_layer_num = len(teacher_qry_attention)
        student_layer_num = len(student_qry_attention)
        layer_per_block = teacher_layer_num // student_layer_num

        # new_teacher_attns = [teacher_qry_attention[i * layer_per_block + layer_per_block - 1] for i in range(student_layer_num)]
        # teacher_qry_attention = new_teacher_attns[-k_layer:]
        # new_teacher_attns = [teacher_pos_attention[i * layer_per_block + layer_per_block - 1] for i in range(student_layer_num)]
        # teacher_pos_attention = new_teacher_attns[-k_layer:]

        teacher_qry_first = teacher_qry_attention[0]
        teacher_pos_first = teacher_pos_attention[0]
        student_qry_first = student_qry_attention[0]
        student_pos_first = student_pos_attention[0]
        
        student_idx = self.extract_student_indices(input_data, topk_results)
        
        for i in range(batch_size):
            # print(f"Top k tokens for instance {i}: QRY - {topk_results[i]['qry_topk']}, POS - {topk_results[i]['pos_topk']}")
            qry_topk_idx = [idx for idx, _, _ in topk_results[i]['qry_topk']]
            pos_topk_idx = [idx for idx, _, _ in topk_results[i]['pos_topk']]
            
            if len(qry_topk_idx) == 0 or len(pos_topk_idx) == 0:
                print("Warning: No valid top-k tokens found for instance {}, skipping attention loss computation.".format(i))
                continue
            
            s_qry_topk_idx = [idx for idx in student_idx[i]['qry'] if idx < student_qry_first.size(2)]
            s_pos_topk_idx = [idx for idx in student_idx[i]['pos'] if idx < student_pos_first.size(2)]
            # print(f"Mapped student indices for instance {i}: QRY - {s_qry_topk_idx}, POS - {s_pos_topk_idx}")
            
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

            # calculate CKA loss
            att_loss = cka_fn_loss(tq_mean, sq_mean) + cka_fn_loss(tp_mean, sp_mean)
            att_loss_total += att_loss / 2
        # print("Total attention loss before averaging:", att_loss_total.item())
        return att_loss_total / batch_size
    
    