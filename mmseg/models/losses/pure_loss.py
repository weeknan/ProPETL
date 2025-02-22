# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from ..losses import accuracy

@MODELS.register_module()
class PureLoss(nn.Module):

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
        # cls_score: Tensor torch.Size([2, L, 2])
        # label: Tensor     torch.Size([2, 768, 768])
        _, L, _ = cls_score.shape
        B, H, W = label.shape
        # 16 =  786 // 48
        patch_size = H // int(pow(L, 0.5))
        # [B, patch_num, patch_num, patch_size, patch_size]
        label = label.view(B, H//patch_size, patch_size, W//patch_size, patch_size).permute(0, 1, 3, 2, 4)
        # [B, patch_num, patch_num, N]
        # N means how many pixel are there in a patch 
        # print(label.shape)
        label = label.reshape(B, H//patch_size, W//patch_size, -1)
        # print(label.shape)
        # [B, patch_num, patch_num]
        pure_or_not = torch.all((label == label[:, :, :, 0].unsqueeze(3)), dim=-1) 
        pure_idx = (pure_or_not == True)
        label = torch.zeros(pure_or_not.shape, dtype=torch.int64).to(cls_score.device)
        label[pure_idx] = 1
        label = label.view(B, -1)

        cls_score = cls_score.reshape(B*L, -1)
        label = label.reshape(-1)
        # print(cls_score.shape)
        # print(label.shape)
        loss_count = F.cross_entropy(cls_score, label)
        #print(torch.sum(pure_idx) / len(pure_idx.view(-1)))
        return loss_count # , accuracy(cls_score, label)

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
