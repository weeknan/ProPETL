# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.registry import MODELS
from ..utils import Upsample
from mmengine.model import BaseModule
from .decode_head import BaseDecodeHead
from typing import List, Tuple
from torch import Tensor
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
import torch
from ..utils import resize

@MODELS.register_module()
class MLPHead(BaseModule):
    """two layer mlp head.

    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 num_classes,
                 in_index,
                 loss_decode=None,
                 **kwargs):

        super().__init__(**kwargs)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, num_classes),
        )
        self.region_cls_loss = build_loss(loss_decode)
        self.in_index = in_index
        self.align_corners = False
        self.num_classes = num_classes
        self.out_channels = num_classes

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    def loss(self, inputs: Tuple[Tensor], 
             batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:

        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def loss_by_feat(self, seg_logits,
                     batch_data_samples) -> dict:

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        
        seg_label = seg_label.squeeze(1)
        # print(seg_feature.shape)
        # print(seg_feature)
        # print(seg_label)
        loss['region_loss'] = self.region_cls_loss(
                    seg_logits,
                    seg_label,
                    )
        
        return loss

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        inputs = inputs[self.in_index]

        return inputs
    
    def forward(self, x):

        x = self._transform_inputs(x)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous() # [n, L, c]
        x = self.mlp(x)

        return x


@MODELS.register_module()
class BNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

from ..losses import accuracy
@MODELS.register_module()
class VariableGranularityBNHead(BNHead):
    """Just a batchnorm."""

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size # n means an image has H/n * W/n patches
    
    def transform_label(self, label, size):
        # label [B, 512, 512]
        label = label.unsqueeze(1).float()
        label = resize(label, size=size)
        label = label.squeeze(1)
        return label.long()

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        H, W = seg_label.shape[-2:]
        seg_logits = resize(
            input=seg_logits,
            size=[H // self.patch_size, W // self.patch_size],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1) # [B, 512, 512]
        seg_label = self.transform_label(seg_label, [H // self.patch_size, W // self.patch_size]) 
        seg_weight = None
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

import torch.nn.functional as F
@MODELS.register_module()
class ClsBNHead(BNHead):
    """Just a batchnorm."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_seg = nn.Linear(self.in_channels, self.num_classes)
    
    def transform_label(self, label):
        # label [B, 512, 512]
        #print(label[0])
        label = F.one_hot(label, num_classes=260) # [B, 512, 512, 260]
        B, H, W = label.shape[:3]
        label = label.reshape(B, H*W, -1)
        label = (label*1.0).mean(dim=1) # [B, C]
        #print(label[0])
        label = label[:, :172]
        label = torch.argmax(label, dim=-1) # [B]
        #print(label)
        return label.long()

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        # B, C = seg_logits.shape[:2]
        # seg_logits = seg_logits.reshape(B, C, -1)
        # seg_logits = seg_logits.mean(dim=-1) # [B, C]


        seg_label = seg_label.squeeze(1) # [B, 512, 512]
        seg_label = self.transform_label(seg_label) # [B]
        seg_weight = None
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss
    
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        B, C, H, W = output.shape
        output = output.reshape(B, C, -1).mean(dim=-1)
        output = self.cls_seg(output)
        return output
    
    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        
        #print(seg_logits.shape)
        return seg_logits

    