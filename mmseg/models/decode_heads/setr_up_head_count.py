# Copyright (c) OpenMMLab. All rights reserved.
# support the count loss, the output will be pooled to [B, 1, D]
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.registry import MODELS
from ..utils import Upsample
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class SETRUPHead_count(BaseDecodeHead):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

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
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 num_convs=1,
                 up_scale=4,
                 kernel_size=3,
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))
                 ],
                 **kwargs):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super().__init__(init_cfg=init_cfg, **kwargs)

        assert isinstance(self.in_channels, int)

        _, self.norm = build_norm_layer(norm_layer, self.in_channels)

        self.up_convs = nn.ModuleList()
        in_channels = self.in_channels
        out_channels = self.channels
        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    Upsample(
                        scale_factor=up_scale,
                        mode='bilinear',
                        align_corners=self.align_corners)))
            in_channels = out_channels

    def loss_by_feat(self, seg_logits,
                     batch_data_samples) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        # seg_logits = resize(
        #     input=seg_logits,
        #     size=seg_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
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

        # #remove duplicate
        # B = seg_label.shape[0]
        # class_num_label = torch.zeros(B, dtype=torch.long).to(seg_logits.device)
        # for i in range(B):
        #     unique_elements = torch.unique(label[i])
        #     class_num_label[i] = len(unique_elements)
            
        # loss['acc_count'] = accuracy(
        #     seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def forward(self, x):

        x = self._transform_inputs(x)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        out = self.cls_seg(x)
        num_clssses = out.shape[1]
        out = torch.mean(out.reshape(n, num_clssses, -1), dim=-1) # [n, num_classes]

        return out
