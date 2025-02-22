# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss


@MODELS.register_module()
class CountLoss(nn.Module):

    def __init__(self,
                 
                 loss_weight=1.0,
                 loss_name='loss_count',
                 ):
        super().__init__()
        
        self.loss_weight = loss_weight

        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                **kwargs):
        # cls_score: Tensor torch.Size([2, 19])
        # label: Tensor     torch.Size([2, 768, 768])
        B = label.shape[0]
        label = torch.unique(label.reshape(B, -1), dim=-1) # [B, 768**2]
        #remove duplicate
        class_num_label = torch.zeros(B, dtype=torch.long).to(cls_score.device)
        for i in range(B):
            unique_elements = torch.unique(label[i])
            class_num_label[i] = len(unique_elements)
        # print(cls_score)
        # print(class_num_label)
        loss_count = F.cross_entropy(cls_score, class_num_label)

        return loss_count

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
