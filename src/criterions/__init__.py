from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .emo_loss import EMOLoss
from .em_kd import EMKDLoss
from .em_kd_llava_ov import EMKDLLavaLoss
from .span_propose import SpanProposeCriterion
from .span_propose_attn import SpanProposeCriterionWeighted
from .span_propose_attn_only_phrase import SpanProposeCriterionWeightedOnlyPhrase
from .span_propose_wo_hid_cross import SpanProposeCriterionWeightedWOHidCross
from .span_propose_wo_hid_intra import SpanProposeCriterionWeightedWOHidIntra
from .span_propose_wo_intra_cross import SpanProposeCriterionWeightedWOIntraCross
from .span_propose_attn_fix import SpanProposeCriterionWeightedFix
from .span_propose_wo_hid import SpanProposeCriterionWeightedWOHid
from .span_propose_wo_intra import SpanProposeCriterionWeightedWOIntra
from .span_propose_wo_cross import SpanProposeCriterionWeightedWOCross

criterion_list = {
    "contrastive_rkd": ContrastiveLossWithRKD,
    "proposal_dtw": ProposalLossWithDTW,
    "universal_logit": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "emo_loss": EMOLoss,
    "em_kd": EMKDLoss,
    "em_kd_llava_ov": EMKDLLavaLoss,
    "span_propose": SpanProposeCriterion,
    "span_propose_attn": SpanProposeCriterionWeighted,
    "span_propose_attn_only_phrase": SpanProposeCriterionWeightedOnlyPhrase,
    "span_propose_wo_hid_cross": SpanProposeCriterionWeightedWOHidCross,
    "span_propose_wo_hid_intra": SpanProposeCriterionWeightedWOHidIntra,
    "span_propose_wo_intra_cross": SpanProposeCriterionWeightedWOIntraCross,
    "span_propose_attn_fix": SpanProposeCriterionWeightedFix,
    "span_propose_wo_hid": SpanProposeCriterionWeightedWOHid,
    "span_propose_wo_intra": SpanProposeCriterionWeightedWOIntra,
    "span_propose_wo_cross": SpanProposeCriterionWeightedWOCross
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)