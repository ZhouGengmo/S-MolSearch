# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset


class MixDataset(BaseWrapperDataset):
    def __init__(self, sup_dataset, sup_pos2_dataset, unsup_dataset):
        self.dataset = sup_dataset
        self.sup_pos2_dataset = sup_pos2_dataset
        self.unsup_dataset = unsup_dataset
        self.len_sup = len(self.dataset)
        self.len_unsup = len(self.unsup_dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return max(self.len_sup, self.len_unsup)  # for 1 epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        sup_item = self.dataset[index % self.len_sup]
        sup_pos2_item = self.sup_pos2_dataset[index % self.len_sup]
        unsup_item = self.unsup_dataset[index % self.len_unsup]
        return  {
                'sup_pos1': sup_item,
                'unsup_data': unsup_item,
                'sup_pos2': sup_pos2_item,
            }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
