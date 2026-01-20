import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F

class CKD(nn.Module):
    def __init__(self, args):
        super(CKD, self).__init__()
        self.args = args
        self.kd_loss_weight = 0.3
        self.distance_weight = 0.5
        self.angle_weight = 0.5
        self.mse = nn.MSELoss(reduction='mean')
        
    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']
        
        teacher_input_qry = input_data['teacher_inputs']['qry']
        teacher_input_pos = input_data['teacher_inputs']['pos']
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_reps = teacher_model.encode_input(teacher_input_qry)[0]
            teacher_pos_reps = teacher_model.encode_input(teacher_input_pos)[0]

        student_qry_reps = student_model.encode_input(student_input_qry)[0]
        student_pos_reps = student_model.encode_input(student_input_pos)[0]

        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_qry_reps.size(0) // student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)
        
        distance_loss = self.compute_distance_loss(student_qry_reps, teacher_qry_reps)
        total_loss = contrastive_loss + self.kd_loss_weight * distance_loss

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": distance_loss,
        }
    
    def pairwise_distance(self, x):
        norm = (x**2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2.0 * torch.mm(x, x.t())
        return dist
    
    def compute_mse(self, student_qry, teacher_qry):
        num_samples = student_qry.size(0)
        if num_samples <= 1:
            return 0.0

        student_diffs = student_qry.unsqueeze(1) - student_qry.unsqueeze(0)
        teacher_diffs = teacher_qry.unsqueeze(1) - teacher_qry.unsqueeze(0)

        per_pair_mse = ((student_diffs - teacher_diffs) ** 2).mean(dim=2)

        mask = ~torch.eye(num_samples, dtype=torch.bool, device=student_qry.device)
        return per_pair_mse.masked_select(mask).mean()
    
    def compute_distance_loss(self, student_qry, teacher_qry):

        student_repr = student_qry
        teacher_repr = self.distiller.t2s_ckd(teacher_qry)
        loss = self.compute_mse(student_repr, teacher_repr)
        
        return loss