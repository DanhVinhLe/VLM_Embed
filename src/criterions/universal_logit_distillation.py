import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalLogitDistillation(nn.Module):
    def __init__(self, args):
        super(UniversalLogitDistillation, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        
    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']
        
        teacher_input_qry = input_data['teacher_inputs']['qry']
        teacher_input_pos = input_data['teacher_inputs']['pos']
        with torch.no_grad():
            teacher_qry_reps = teacher_model.encode_input(teacher_input_qry)
            teacher_pos_reps = teacher_model.encode_input(teacher_input_pos)
            
        student_qry_reps = student_model.encode_input(student_input_qry)
        student_pos_reps = student_model.encode_input(student_input_pos)
        
        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_qry_reps.size(0) // student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)
        
    def compute_universal_logit_loss(self, student_qry, student_pos, teacher_qry, teacher_pos):
        # Todo
        pass