# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

@DATASETS.register_module()
class CityscapesDataset_road(BaseSegDataset):
    """Cityscapes dataset.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!only has road!!!!!!!!!!!!!!!!!!!!!!!!!

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'others'),
        palette=[[128, 64, 128], [70, 70, 70]]
        )

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

@DATASETS.register_module()
class CityscapesDataset_single_scene(BaseSegDataset):
    """Cityscapes dataset.
    with road, other and only one scene

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = None

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 metainfo=None,
                 **kwargs) -> None:
        assert metainfo != None
        self._load_own_info(metainfo)

        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    
    @classmethod
    def _load_own_info(cls,
                       metainfo):
        cls.METAINFO = metainfo

@DATASETS.register_module()
class CityscapesDataset_multi_scene(BaseSegDataset):
    """Cityscapes dataset.
    with road, other and multiple scene

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = None

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 metainfo=None,
                 **kwargs) -> None:
        assert metainfo != None
        self._load_own_info(metainfo)
        #print(self.METAINFO)
        # for k in self.METAINFO.keys():
        #     self.METAINFO[k].append(metainfo[k])
        # self.METAINFO.update(metainfo)
        # print(self.METAINFO)
        # assert False
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    
    @classmethod
    def _load_own_info(cls,
                       metainfo):
        cls.METAINFO = metainfo