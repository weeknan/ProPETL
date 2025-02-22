# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder, EncoderDecoder_cls
from .seg_tta import SegTTAModel, SegTTAModel_kitti_logit

from .encoder_pure_region import EncoderPureRegion


__all__ = [
    'BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'SegTTAModel',
      'EncoderPureRegion',
]
