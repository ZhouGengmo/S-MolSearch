import math
import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from ot import smooth
import logging
logger = logging.getLogger(__name__)


@register_loss("smolsearch")
class SMolSearchLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed

    def forward(self, model, sample, reduce=True):
        input_key = 'net_input_sup_pos1'
        # [B, clip_dim], [B, clip_dim], [B, clip_dim], [B, clip_dim]
        supd_supm_pos1_emd, supd_supm_pos2_emd, unsupd_unsupm_emb, unsupd_supm_emb  = model(**sample)
        sup_clip_loss = self.calc_clip_loss(supd_supm_pos1_emd, supd_supm_pos2_emd)
         # [B, B], [B, B]
        unsupd_supm_emb_detach = unsupd_supm_emb.detach().clone()
        unsupd_unsupm_logits = self.calc_sim_logits(unsupd_unsupm_emb, unsupd_unsupm_emb, self.args.tem_logit)
        unsupd_supm_logits = self.calc_sim_logits(unsupd_supm_emb_detach, unsupd_supm_emb_detach, self.args.tem_soft)
        unsupd_supm_logits_labels = self.calc_ot_labels_ot_dual(unsupd_supm_logits, self.args.batch_mul_factor)

        soft_loss = F.cross_entropy(
            unsupd_unsupm_logits.float(),
            unsupd_supm_logits_labels.float(),
        )
        sample_size = sample[input_key]["src_tokens"].shape[0]
        loss = sup_clip_loss + self.args.soft_loss * soft_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample_size,
            "sup_clip_loss": sup_clip_loss.data,
            "soft_loss": soft_loss.data,
        }

        if self.args.reg_loss > 0:
            pdist = nn.PairwiseDistance(2)
            unsupd_unsupm_emb = unsupd_unsupm_emb / unsupd_unsupm_emb.norm(dim=1, keepdim=True)
            I = self.pairwise_NNs_inner(unsupd_unsupm_emb)
            distances = pdist(unsupd_unsupm_emb, unsupd_unsupm_emb[I])
            loss_uniform = - torch.log(sample_size * distances).mean()
            loss += self.args.reg_loss * loss_uniform
            logging_output['reg_loss'] = loss_uniform.data
        logging_output['loss'] = loss.data

        return loss, sample_size, logging_output

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
        _, I = torch.max(dots, 1)  # max inner prod -> min distance
        return I

    def calc_ot_labels_ot_dual(self, unsupd_supm_logits, batch_mul_factor = 1.0, cost_factor = 1.0, cost_mul_factor = 1.0, lambda_factor = 5.0, max_iter = 200, eps = 1e-5, max_nz = 20):
        cost_matrix = cost_factor - cost_mul_factor * unsupd_supm_logits
        row_num = cost_matrix.shape[0]
        col_num = cost_matrix.shape[1]
        row_marginal = torch.ones(row_num).cuda().to(torch.float64) * batch_mul_factor
        col_marginal = torch.ones(col_num).cuda().to(torch.float64) * batch_mul_factor
        cost_matrix = cost_matrix.to(torch.float64)
        P_label = smooth.smooth_ot_dual(row_marginal, col_marginal, cost_matrix, lambda_factor, reg_type = 'l2', stopThr = eps,  numItermax = max_iter)
        P_label = P_label.type_as(unsupd_supm_logits)
        return P_label

    def calc_sim_logits(self, mean_emd: torch.Tensor, mean_emd2: torch.Tensor, logit_scale: float) -> torch.Tensor:
        """ Computes clip loss from model ouput emb
        Inputs:
        * mean_emd: (B, clip_dim)
        * mean_emd2: (B, clip_dim)
        Outputs: (...,)
        """
        mean_emd = mean_emd / mean_emd.norm(dim=1, keepdim=True)
        mean_emd2 = mean_emd2 / mean_emd2.norm(dim=1, keepdim=True)  
        logits_emd = (mean_emd @ mean_emd2.t()) / logit_scale
        return logits_emd

    def calc_clip_loss(self, mean_emd: torch.Tensor, mean_emd2: torch.Tensor ) -> torch.Tensor:
        """ Computes clip loss from model ouput emb
        Inputs:
        * mean_emd: (B, clip_dim)
        * mean_emd2: (B, clip_dim)
        Outputs: (...,)
        """
        mean_emd = mean_emd / mean_emd.norm(dim=1, keepdim=True)
        mean_emd2 = mean_emd2 / mean_emd2.norm(dim=1, keepdim=True)
        logit_scale = self.args.tem_clip
        logits_emd = (mean_emd @ mean_emd2.t()) / logit_scale
        logits_emd2 = (mean_emd2 @ mean_emd.t()) / logit_scale
        logits_shape = logits_emd.shape[0]  
        labels = torch.arange(logits_shape, dtype=torch.long).cuda()
        ### clip loss
        return (F.cross_entropy(logits_emd, labels) + F.cross_entropy(logits_emd2, labels))/2

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        # """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)
        # # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )

        sup_clip_loss = sum(log.get('sup_clip_loss', 0) for log in logging_outputs)
        metrics.log_scalar('avg_sup_clip_loss', sup_clip_loss / sample_size , sample_size, round=3)

        soft_loss = sum(log.get('soft_loss', 0) for log in logging_outputs)
        metrics.log_scalar('avg_soft_loss', soft_loss / sample_size , sample_size, round=3)

        reg_loss = sum(log.get('reg_loss', 0) for log in logging_outputs)
        if reg_loss != 0:
            metrics.log_scalar('avg_reg_loss', reg_loss / sample_size , sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
