# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss

@MODELS.register_module()
class DiffConsistLoss(nn.Module):
    """DiffConsistLoss.
    force the diff features has high simility in same semantic class


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
                 num_classes=None,
                 ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.num_classes = num_classes


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
        # [B, N, N, C]
        proj_diff_feature = proj_layer(diff_feature)

        label = label.view(B, H//patch_size, patch_size, W//patch_size, patch_size).permute(0, 1, 3, 2, 4)
        # [B, patch_num, patch_num, N]
        # N means how many pixel are there in a patch 
        label = label.reshape(B, H//patch_size, W//patch_size, -1)
        # [B, patch_num, patch_num, N, C]
        label = F.one_hot(label, num_classes=260)
        # print(label.shape)
        # remove the none classes
        # print(self.num_classes)
        # assert False
        label = label[:, :, :, :, :self.num_classes]
        # [B, patch_num, patch_num, C]
        label = (label*1.0).mean(dim=3)
        # most appearence class [B, patch_num, patch_num]
        label = torch.argmax(label, dim=-1)

        proj_diff_feature = proj_diff_feature.reshape(-1, proj_diff_feature.shape[-1])
        label = label.reshape(-1)

        diff_consist_loss = weight_reduce_loss(F.cross_entropy(proj_diff_feature, label))

        diff_consist_loss = self.loss_weight * diff_consist_loss

        return diff_consist_loss


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
