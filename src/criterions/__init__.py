from .contrastive_loss_with_RKD import ContrastiveLossWithRKD

criterion_list = {
    "contrastive_rkd": ContrastiveLossWithRKD,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list:
        raise ValueError(f"Criterion {args.criterion} not found.")
    return criterion_list[args.criterion](args)