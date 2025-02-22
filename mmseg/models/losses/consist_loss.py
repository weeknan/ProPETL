# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss

@MODELS.register_module()
class ConsistLoss(nn.Module):
    """ConsistLoss.
    force the features has high simility between ajcent patch; 2023.12
    which is really no sense when rethink in 2.27.2024 haha


    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 loss_weight=1.0,
                 loss_name='loss_consist',
                 ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self._loss_name = loss_name


    def forward(self,
                sim_map,
                **kwargs):
        
        # sim_map has shape: [B, L, L]
        diag = torch.diagonal(sim_map, offset=1, dim1=-2, dim2=-1) # [B, L-1]
        return -torch.mean(diag).unsqueeze(0)


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
