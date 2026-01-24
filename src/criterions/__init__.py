from torch import nn
from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .emo_loss import EMOLoss
from .em_kd import EMKDLoss
from .ckd import CKDLoss
from .em_kd_llava_ov import EMKDLLavaLoss
from .ckd import CKDLoss
from .holo import HoloDistillLoss

criterion_list = {
    "contrastive_rkd": ContrastiveLossWithRKD,
    "universal_logit": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "emo_loss": EMOLoss,
    "em_kd": EMKDLoss,
    "em_kd_llava_ov": EMKDLLavaLoss,
    "ckd": CKDLoss,
    "mse": nn.MSELoss,
    "ckd": CKDLoss,
    "holo": HoloDistillLoss
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)