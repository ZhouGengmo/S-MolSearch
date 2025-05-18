# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from .unimol import UniMolModel, base_architecture
import argparse

logger = logging.getLogger(__name__)


@register_model("smolsearch")
class SMolSearchModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--sup-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--unsup-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--sup-clip-dim",
            type=int,
            default=1024,
        )
        parser.add_argument(
            "--unsup-clip-dim",
            type=int,
            default=1024,
        )
        parser.add_argument(
            "--soft-loss",
            type=float,
            metavar="D",
            help="unsup soft loss ratio",
        )
        parser.add_argument(
            "--reg-loss",
            type=float,
            metavar="D",
            help="unsup reg loss ratio",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        smolsearch_architecture(args)

        self.args = args
        self.sup_model = UniMolModel(args.sup, dictionary)
        self.unsup_model = UniMolModel(args.unsup, dictionary)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        net_input_sup_pos1,
        net_input_sup_pos2,
        net_input_unsup,
        **kwargs
    ):
        supd_supm_pos1_emd = self.sup_model(**net_input_sup_pos1)
        supd_supm_pos2_emd = self.sup_model(**net_input_sup_pos2)
        unsupd_unsupm_emb = self.unsup_model(**net_input_unsup)
        unsupd_supm_emb = self.sup_model(**net_input_unsup)
        return supd_supm_pos1_emd, supd_supm_pos2_emd, unsupd_unsupm_emb, unsupd_supm_emb

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


@register_model_architecture("smolsearch", "smolsearch")
def smolsearch_architecture(args):
    parser = argparse.ArgumentParser()
    args.sup = parser.parse_args([])
    args.unsup = parser.parse_args([])
    args.soft_loss = getattr(args, "soft_loss", 1.0)
    args.reg_loss = getattr(args, "reg_loss", 0.1)

    args.sup.encoder_layers = getattr(args, "sup_encoder_layers", 15)
    args.sup.encoder_embed_dim = getattr(args, "sup_encoder_embed_dim", 512)
    args.sup.encoder_ffn_embed_dim = getattr(args, "sup_encoder_ffn_embed_dim", 2048)
    args.sup.encoder_attention_heads = getattr(args, "sup_encoder_attention_heads", 64)
    args.sup.dropout = getattr(args, "sup_dropout", 0.1)
    args.sup.emb_dropout = getattr(args, "sup_emb_dropout", 0.1)
    args.sup.attention_dropout = getattr(args, "sup_attention_dropout", 0.1)
    args.sup.activation_dropout = getattr(args, "sup_activation_dropout", 0.0)
    args.sup.pooler_dropout = getattr(args, "sup_pooler_dropout", 0.0)
    args.sup.max_seq_len = getattr(args, "sup_max_seq_len", 512)
    args.sup.activation_fn = getattr(args, "sup_activation_fn", "gelu")
    args.sup.pooler_activation_fn = getattr(args, "sup_pooler_activation_fn", "tanh")
    args.sup.post_ln = getattr(args, "sup_post_ln", False)
    args.sup.clip_dim = getattr(args, "sup_clip_dim", 768)
    args.sup.delta_pair_repr_norm_loss = -1.0

    args.unsup.encoder_layers = getattr(args, "unsup_encoder_layers", 15)
    args.unsup.encoder_embed_dim = getattr(args, "unsup_encoder_embed_dim", 512)
    args.unsup.encoder_ffn_embed_dim = getattr(args, "unsup_encoder_ffn_embed_dim", 2048)
    args.unsup.encoder_attention_heads = getattr(args, "unsup_encoder_attention_heads", 64)
    args.unsup.dropout = getattr(args, "unsup_dropout", 0.1)
    args.unsup.emb_dropout = getattr(args, "unsup_emb_dropout", 0.1)
    args.unsup.attention_dropout = getattr(args, "unsup_attention_dropout", 0.1)
    args.unsup.activation_dropout = getattr(args, "unsup_activation_dropout", 0.0)
    args.unsup.pooler_dropout = getattr(args, "unsup_pooler_dropout", 0.0)
    args.unsup.max_seq_len = getattr(args, "unsup_max_seq_len", 512)
    args.unsup.activation_fn = getattr(args, "unsup_activation_fn", "gelu")
    args.unsup.pooler_activation_fn = getattr(args, "unsup_pooler_activation_fn", "tanh")
    args.unsup.post_ln = getattr(args, "unsup_post_ln", False)
    args.unsup.clip_dim = getattr(args, "unsup_clip_dim", 768)
    args.unsup.delta_pair_repr_norm_loss = -1.0

    base_architecture(args)
