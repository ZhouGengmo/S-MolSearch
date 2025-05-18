# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
)

from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    TTADataset,
    KeyListDataset,
    MixDataset,
)

from unicore import checkpoint_utils
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("smolsearch")
class SMolSearchTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )
        parser.add_argument(
            "--tem-clip",
            type=float,
            default=20,
            metavar="D",
            help="contrastive loss temperature",
        )
        parser.add_argument(
            "--tem-soft",
            type=float,
            default=20,
            metavar="D",
            help="contrastive loss temperature",
        )
        parser.add_argument(
            "--tem-logit",
            type=float,
            default=1,
            metavar="D",
            help="contrastive loss temperature",
        )
        parser.add_argument(
            "--finetune-sup-model",
            default=None,
            type=str,
            help="pretrained sup model path",
        )
        parser.add_argument(
            "--finetune-unsup-model",
            default=None,
            type=str,
            help="pretrained unsup model path",
        )
        parser.add_argument(
            "--unsup-data-path",
            default=None,
            type=str,
            help="unsup data lmdb file path",
        )
        parser.add_argument(
            "--min-lr",
            type=float,
            default=0.0,
            metavar="D",
            help="min lr for cosine",
        ) 
        parser.add_argument(
            "--batch-mul-factor",
            type=float,
            default=1.0,
            metavar="D",
            help="batch-mul-factor",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """

        split_path = os.path.join(self.args.data, split + '.lmdb')
        raw_dataset = LMDBDataset(split_path)
        unsup_raw_dataset = LMDBDataset(self.args.unsup_data_path)
        def one_dataset(raw_dataset, coord_seed, mask_seed):
            if split == "train":
                dataset = ConformerSampleDataset(
                    raw_dataset, coord_seed, "atoms", "coordinates"
                )
                dataset = AtomTypeDataset(raw_dataset, dataset)
            else:
                raw_dataset = TTADataset(
                    raw_dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
                )
                dataset = AtomTypeDataset(raw_dataset, raw_dataset)

            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            dataset = CroppingDataset(
                dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            }


        
        if split in ['train', 'train.small']:
            raw_sup_pos1_dataset = KeyDataset(raw_dataset, 'pos1')
            raw_sup_pos2_dataset = KeyListDataset(raw_dataset, 'pos2_list')
            mix_dataset = MixDataset(raw_sup_pos1_dataset, raw_sup_pos2_dataset, unsup_raw_dataset)
            sup_pos1_dataset = KeyDataset(mix_dataset, 'sup_pos1')
            sup_pos2_dataset = KeyDataset(mix_dataset, 'sup_pos2')
            unsup_dataset = KeyDataset(mix_dataset, 'unsup_data')
        elif split == 'valid':
            sup_dataset = KeyDataset(raw_dataset, 'sup_data')
            sup_pos1_dataset = KeyDataset(sup_dataset, 'pos1')
            sup_pos2_dataset = KeyListDataset(sup_dataset, 'pos2_list')
            unsup_dataset = KeyDataset(raw_dataset, 'unsup_data')
        
        net_input_sup_pos1 = one_dataset(sup_pos1_dataset, self.args.seed, self.args.seed)
        net_input_sup_pos2 = one_dataset(sup_pos2_dataset, self.args.seed, self.args.seed)
        net_input_unsup = one_dataset(unsup_dataset, self.args.seed, self.args.seed)

        dataset = {
            'net_input_sup_pos1': net_input_sup_pos1,
            'net_input_sup_pos2': net_input_sup_pos2,
            'net_input_unsup': net_input_unsup,
        }

        dataset =  NestedDictionaryDataset(
            dataset
        )
        if split == "train":
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.finetune_sup_model is not None:
            logger.info(f"load pretrain model weight from {args.finetune_sup_model} for sup model")
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_sup_model, 
            )
            model.sup_model.load_state_dict(state["model"], strict=False)
        if args.finetune_unsup_model is not None:
            logger.info(f"load pretrain model weight from {args.finetune_unsup_model} for unsup model")
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_unsup_model, 
            )
            model.unsup_model.load_state_dict(state["model"], strict=False)
        return model

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output
