# Copyright (c) OpenMMLab. All rights reserved.
# the region label is a soft label, which mean the one-hot label of all piexl in a patch
# and use the CE loss to pull the logit and the label together
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from ..losses import accuracy

@MODELS.register_module()
class GlobalClsLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 loss_name='loss_global_cls',
                 num_classes=None,
                 alpha=None,
                 hard_region_label=False
                 ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.num_classes = num_classes
        self.hard_region_label = hard_region_label
        self.alpha = alpha

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                **kwargs):
        # cls_score: Tensor torch.Size([2, C])
        # label: Tensor     torch.Size([2, 768, 768])
        # print(cls_score.shape)
        # _, L, _ = cls_score.shape
        B, H, W = label.shape
        label = F.one_hot(label, num_classes=260) # [B, H, W, 260]
        label = label[:, :, :, :self.num_classes] # [B, H, W, C]
        if len(cls_score.shape) == 2:
            # global part
            label = (label * 1.0).mean(dim=1) # [B, W, C]
            label = label.mean(dim=1) # [B, C]
            if self.alpha != None:
                most_freq_label = torch.argmax(label, dim=-1)
                one_hot_label = F.one_hot(most_freq_label, num_classes=self.num_classes)
                # print(one_hot_label[:2])
                # print(label[:2])
                # print('---------------------')
                # assert False
                label = self.alpha * label + (1 - self.alpha) * one_hot_label
            # print(cls_score.shape)
            # print(label.shape)
            # assert False
            loss_cls = F.cross_entropy(cls_score, label)
            return loss_cls
        
        else:
            assert len(cls_score.shape) == 4
            # local part
            n_region = cls_score.shape[1]
            label = label.reshape(B, n_region, H // n_region, n_region, W // n_region, -1) #[B, 2, 10, 2, 10, -1]
            label = (label * 1.0).mean(dim=2) #[B, 2, 2, 10, -1]
            label = label.mean(dim=3) #[B, 2, 2, -1]
            label = label.reshape(-1, label.shape[-1])
            if self.hard_region_label:
                label = torch.argmax(label, dim=-1)

            if self.alpha != None:
                most_freq_label = torch.argmax(label, dim=-1)
                one_hot_label = F.one_hot(most_freq_label, num_classes=self.num_classes)
                # print(one_hot_label[1000:1002])
                # print(label[1000:1002])
                # print('---------------------')
                #assert False
                label = self.alpha * label + (1 - self.alpha) * one_hot_label
                
            cls_score = cls_score.reshape(-1, self.num_classes)
            
            loss_cls = F.cross_entropy(cls_score, label)

            return loss_cls


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
