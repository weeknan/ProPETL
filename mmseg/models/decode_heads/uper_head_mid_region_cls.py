# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..losses import accuracy
import torch.nn.functional as F

@MODELS.register_module()
class UPerHeadMidCls(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        region_shape = inputs[-1].shape[-2:]
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        # print(output.shape)
        # print(region_shape)
        # assert False
        output = resize(
            input=output,
            size=region_shape,
            mode='bilinear',
            align_corners=self.align_corners)
        #print(output.shape)
        return output
    
    def get_acc(self, cls_score, label):
        # cls_score: Tensor torch.Size([2, L, C])
        # label: Tensor     torch.Size([2, 768, 768])
        #print(cls_score.shape)
        #if len(cls_score.shape) == 4:
        # with shape [B, C, H, W]
        _, C, score_L, score_H = cls_score.shape
        L = score_L * score_H
        cls_score = cls_score.permute(0, 2, 3, 1) # [B, H, W, C]
        # #else:
        #     _, L, _ = cls_score.shape
        #print(label.shape)
        B, H, W = label.shape
        # 16 =  786 // 48
        patch_size = H // int(pow(L, 0.5))
        # [B, patch_num, patch_num, patch_size, patch_size]
        label = label.view(B, H//patch_size, patch_size, W//patch_size, patch_size).permute(0, 1, 3, 2, 4)
        # [B, patch_num, patch_num, N]
        # N means how many pixel are there in a patch 
        label = label.reshape(B, H//patch_size, W//patch_size, -1)
        # [B, patch_num, patch_num, N, C]
        label = F.one_hot(label, num_classes=260)
        # print(label.shape)
        # remove the none classes
        label = label[:, :, :, :, :C]
        # [B, patch_num, patch_num, C]
        label = (label*1.0).mean(dim=3)
        # most appearence class [B, patch_num, patch_num]
        label = torch.argmax(label, dim=-1)

        cls_score = cls_score.reshape(-1, cls_score.shape[-1])
        label = label.reshape(-1)
        
        return accuracy(cls_score, label, ignore_index=self.ignore_index)


    def loss_by_feat(self, seg_logits,
                     batch_data_samples) -> dict:

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

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

        loss['acc_seg'] = self.get_acc(seg_logits, seg_label)
        
        return loss
