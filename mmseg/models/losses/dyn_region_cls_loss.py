# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from ..losses import accuracy

@MODELS.register_module()
class DynRegionClsLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 loss_name='loss_region_cls',
                 window_size=None,
                 alpha=None,
                 num_classes=None,
                 ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.num_classes = num_classes
        self.window_size=window_size
        self.alpha=alpha

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                **kwargs):
        # cls_score: Tensor torch.Size([2, L, C])
        # label: Tensor     torch.Size([2, 768, 768])
        # print(cls_score.shape)
        _, L, _ = cls_score.shape
        B, H, W = label.shape
        # 16 =  786 // 48

        patch_size = H // int(pow(L, 0.5))
        
        # [B, patch_num, patch_num, patch_size, patch_size]
        label = label.view(B, H//self.window_size, self.window_size, W//self.window_size, self.window_size).permute(0, 1, 3, 2, 4)
        # [B, patch_num, patch_num, N]
        # N means how many pixel are there in a patch 
        label = label.reshape(B, H//self.window_size, W//self.window_size, -1)
        # [B, patch_num, patch_num, N, C]
        label = F.one_hot(label, num_classes=260)
        # print(label.shape)
        # remove the none classes
        label = label[:, :, :, :, :self.num_classes]
        # [B, patch_num, patch_num, C]
        label = (label*1.0).mean(dim=3)


        # [B, C, patch_num, patch_num]
        cls_score = cls_score.view(B, int(pow(L, 0.5)), int(pow(L, 0.5)), -1)
        cls_score = cls_score.permute(0, 3, 1, 2)
        cls_score = F.interpolate(cls_score, size=(H//self.window_size, W//self.window_size), mode='bilinear')
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(-1, cls_score.shape[-1])
        label = label.reshape(-1, label.shape[-1])
        if self.alpha != None:
            most_freq_label = torch.argmax(label, dim=-1)
            one_hot_label = F.one_hot(most_freq_label, num_classes=self.num_classes)
            # print(one_hot_label[1000:1002])
            # print(label[1000:1002])
            # print('---------------------')
            # #assert False
            label = self.alpha * label + (1 - self.alpha) * one_hot_label

        loss_count = F.cross_entropy(cls_score, label)

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
