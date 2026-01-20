import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, model_trainer, input_data):
        self.model_trainer = model_trainer
        model = model_trainer.model
        
        student_input_qry = input_data['qry']
        student_input_pos = input_data['pos']

        student_qry_reps = model.encode_input(student_input_qry)[0]
        student_pos_reps = model.encode_input(student_input_pos)[0]

        scores = model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_qry_reps.size(0) // student_pos_reps.size(0))
        contrastive_loss = self.cross_entropy(scores / self.model_trainer.temperature, target)

        total_loss = contrastive_loss
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            # "kd_loss": kd_loss,
        }
        