# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss

@MODELS.register_module()
class MagConsistLoss(nn.Module):
    """MagConsistLoss.
    force the magnitude of diff features has consist magnitude


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
                 loss_name='loss_diff_consist',
                 ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                x,
                pre_x,
                proj_layer,
                label,
                **kwargs):

    
        B, _, patch_num, _ = x.shape
        B, H, W = label.shape
        patch_size = H // patch_num
        # x has shape: [B, D, N, N]
        # pre_x has shape: [B, D, N, N]
        # print(x.shape)
        # print(pre_x.shape)
        diff_feature = (x - pre_x).permute(0, 2, 3, 1) # [B, N, N, D]
        mag_diff_feature = (diff_feature ** 2).mean(dim=-1) # [B, N, N]

        mag_diff_feature = mag_diff_feature.reshape(mag_diff_feature.shape[0], -1) # [B, N**2]
        mean_mag_diff_feature = mag_diff_feature.mean(dim=-1) # [B]
        print(mean_mag_diff_feature)
        mag_consist_loss = weight_reduce_loss(F.mse_loss(mag_diff_feature, 
                                                         mean_mag_diff_feature.unsqueeze(1).repeat(1, mag_diff_feature.shape[-1])))

        mag_consist_loss = self.loss_weight * mag_consist_loss

        return mag_consist_loss


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
