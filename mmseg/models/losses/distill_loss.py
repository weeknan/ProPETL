# Copyright (c) OpenMMLab. All rights reserved.
# the region label is a soft label, which mean the one-hot label of all piexl in a patch
# and use the MSE loss to pull the logit and the label together
# the output of backbone is done with softmax first
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from ..losses import accuracy

@MODELS.register_module()
class DistillLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 loss_name='distill_cls',
                 ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                x,
                teacher_x,
                **kwargs):
        
        distill_loss = []
        for stage in range(len(x)):
            loss = F.mse_loss(x[stage], teacher_x[stage].detach())
            distill_loss.append(self.loss_weight * loss)
        
        return distill_loss

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
