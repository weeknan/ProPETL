# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from mmseg.registry import TRANSFORMS
from mmseg.utils import datafrombytes

try:
    from osgeo import gdal
except ImportError:
    gdal = None

import torch
@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotations_with_ignore(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        
        gt_semantic_seg[gt_semantic_seg == 255] = 19
        #print(torch.unique(torch.tensor(gt_semantic_seg)))
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotations_part_classes(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.
    
    FIX classes for cityscapes

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
        fixed_class_id = None,
        ignore_as_other = False,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

        assert fixed_class_id != None
        self.fixed_class_id = fixed_class_id
        self.ignore_as_other = ignore_as_other

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        
        fixed_class_index = (gt_semantic_seg == self.fixed_class_id)
        if self.ignore_as_other:
            other_index = (gt_semantic_seg != self.fixed_class_id)
        else: # ignore (255) is still 255
            other_index = (gt_semantic_seg != self.fixed_class_id) * (gt_semantic_seg != 255)
        gt_semantic_seg[fixed_class_index] = 0
        gt_semantic_seg[other_index] = 1

        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotations_two_classes(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.
    
    FIX classes for cityscapes
    load 'road' and one scene class

    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
        fixed_class_id = None,
        fixed_scene_id = None,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

        assert fixed_class_id != None
        self.fixed_class_id = fixed_class_id
        assert fixed_scene_id != None
        self.fixed_scene_id = fixed_scene_id

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        
        
        fixed_class_index = (gt_semantic_seg == self.fixed_class_id)
        fixed_scene_index = (gt_semantic_seg == self.fixed_scene_id)

        other_index = ((gt_semantic_seg != self.fixed_class_id) * (gt_semantic_seg != self.fixed_scene_id) * (gt_semantic_seg != 255))

        gt_semantic_seg[fixed_class_index] = 0
        gt_semantic_seg[fixed_scene_index] = 1
        gt_semantic_seg[other_index] = 2
        #print(torch.unique(torch.tensor(gt_semantic_seg)))
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotations_multi_classes(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.
    
    FIX classes for cityscapes
    load 'road' and multiple scene class

    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
        fixed_class_id = None,
        fixed_scene_id = None,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

        assert fixed_class_id != None
        self.fixed_class_id = fixed_class_id
        assert fixed_scene_id != None
        self.fixed_scene_id = fixed_scene_id

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        
        
        fixed_class_index = (gt_semantic_seg == self.fixed_class_id)
        other_index = (gt_semantic_seg != self.fixed_class_id) * (gt_semantic_seg != 255)
        # print(type(other_index))
        # print(other_index)
        # assert False
        #print(torch.unique(torch.tensor(gt_semantic_seg)))
        gt_semantic_seg[fixed_class_index] = 0
        #print(torch.unique(torch.tensor(gt_semantic_seg)))
        fixed_scene_index = []
        for i in range(len(self.fixed_scene_id)):
            original_scene_id = self.fixed_scene_id[i]
            
            fixed_scene_index.append((gt_semantic_seg == original_scene_id))
            other_index = other_index * (gt_semantic_seg != original_scene_id)
        for i in range(len(self.fixed_scene_id)):
            original_scene_id = self.fixed_scene_id[i]
            new_scene_id = i + 1 #
            gt_semantic_seg[fixed_scene_index[i]] = new_scene_id
        # other
        gt_semantic_seg[other_index] = len(self.fixed_scene_id) + 1
        #print(torch.unique(torch.tensor(gt_semantic_seg)))
        #assert False
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotationsColor(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.
    support color annotations

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
        fix_path=None,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend
        self.fix_path = fix_path

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.fix_path: # to cheat and avoid the caluate the metrix
            gt_semantic_seg = np.zeros(results['ori_shape'].__add__((3,))).astype(np.uint8)
        else:
            img_bytes = fileio.get(
                results['seg_map_path'], backend_args=self.backend_args)
            # shape (375, 1242, 3)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
        
        gt_semantic_seg = gt_semantic_seg.sum(axis = -1)
        road_index = (gt_semantic_seg == 510)
        other_index = (gt_semantic_seg != 510)
        gt_semantic_seg[road_index] = 1
        gt_semantic_seg[other_index] = 0

        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotationsCamvid(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by Camvid
    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
        fix_path=None,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend
        self.fix_path = fix_path

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.fix_path: # to cheat and avoid the caluate the metrix
            gt_semantic_seg = np.zeros(results['ori_shape'].__add__((3,))).astype(np.uint8)
        else:
            img_bytes = fileio.get(
                results['seg_map_path'], backend_args=self.backend_args)
            # shape (375, 1242, 3)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)           
        # print(results['seg_map_path'])
        # print(gt_semantic_seg.shape)
        # print(gt_semantic_seg[330, 450:])
        # print('-----------------------')
        road_index = (gt_semantic_seg == 3)
        other_index = (gt_semantic_seg != 3)
        gt_semantic_seg[road_index] = 1
        gt_semantic_seg[other_index] = 0
        #print(gt_semantic_seg.shape)
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

import torch.nn.functional as F
@TRANSFORMS.register_module()
class LoadAnnotations_cls(LoadAnnotations):
    # turn a pixel-label to one label
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
    
    def _load_seg_map(self, results: dict) -> None:
        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        H, W = gt_semantic_seg.shape[-2:]
        gt_semantic_seg = torch.tensor(gt_semantic_seg) 
        label = F.one_hot(gt_semantic_seg.long(), num_classes=260)
        label = label.reshape(H*W, -1)
        label = (label*1.0).mean(dim=-1) # [B, C]
        #print(label[0])
        label = label[:172]
        label = torch.argmax(label, dim=-1) # [B]
        gt_semantic_seg = np.array(label)

        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')


@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadBiomedicalImageFromFile(BaseTransform):
    """Load an biomedical mage from file.

    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities, and data type is float32
        if set to_float32 = True, or float64 if decode_backend is 'nifti' and
        to_float32 is False.
    - img_shape
    - ori_shape

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']

        data_bytes = fileio.get(filename, self.backend_args)
        img = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img[None, ...]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalAnnotation(BaseTransform):
    """Load ``seg_map`` annotation provided by biomedical dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'gt_seg_map': np.ndarray (X, Y, Z) or (Z, Y, X)
        }

    Required Keys:

    - seg_map_path

    Added Keys:

    - gt_seg_map (np.ndarray): Biomedical seg map with shape (Z, Y, X) by
        default, and data type is float32 if set to_float32 = True, or
        float64 if decode_backend is 'nifti' and to_float32 is False.

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded seg map to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See :class:`mmengine.fileio` for details.
            Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['seg_map_path'], self.backend_args)
        gt_seg_map = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            gt_seg_map = gt_seg_map.astype(np.float32)

        if self.decode_backend == 'nifti':
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        if self.to_xyz:
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalData(BaseTransform):
    """Load an biomedical image and annotation from file.

    The loading data format is as the following:

    .. code-block:: python

        {
            'img': np.ndarray data[:-1, X, Y, Z]
            'seg_map': np.ndarray data[-1, X, Y, Z]
        }


    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.
    - img_shape
    - ori_shape

    Args:
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 with_seg=False,
                 decode_backend: str = 'numpy',
                 to_xyz: bool = False,
                 backend_args: Optional[dict] = None) -> None:  # noqa
        self.with_seg = with_seg
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['img_path'], self.backend_args)
        data = datafrombytes(data_bytes, backend=self.decode_backend)
        # img is 4D data (N, X, Y, Z), N is the number of protocol
        img = data[:-1, :]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]

        if self.with_seg:
            gt_seg_map = data[-1, :]
            if self.decode_backend == 'nifti':
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)

            if self.to_xyz:
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)
            results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'with_seg={self.with_seg}, '
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class InferencerLoader(BaseTransform):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: Union[str, np.ndarray, dict]) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if isinstance(single_input, str):
            inputs = dict(img_path=single_input)
        elif isinstance(single_input, np.ndarray):
            inputs = dict(img=single_input)
        elif isinstance(single_input, dict):
            inputs = single_input
        else:
            raise NotImplementedError

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class LoadSingleRSImageFromFile(BaseTransform):
    """Load a Remote Sensing mage from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        img = np.einsum('ijk->jki', ds.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class LoadMultipleRSImageFromFile(BaseTransform):
    """Load two Remote Sensing mage from file.

    Required Keys:

    - img_path
    - img_path2

    Modified Keys:

    - img
    - img2
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.to_float32 = to_float32

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        filename2 = results['img_path2']

        ds = gdal.Open(filename)
        ds2 = gdal.Open(filename2)

        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        if ds2 is None:
            raise Exception(f'Unable to open file: {filename2}')

        img = np.einsum('ijk->jki', ds.ReadAsArray())
        img2 = np.einsum('ijk->jki', ds2.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)
            img2 = img2.astype(np.float32)

        if img.shape != img2.shape:
            raise Exception(f'Image shapes do not match:'
                            f' {img.shape} vs {img2.shape}')

        results['img'] = img
        results['img2'] = img2
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str

@TRANSFORMS.register_module()
class LoadImageFromFile_fixed(BaseTransform):
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None, 
                 file_path = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_path = file_path

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:

        filename = self.file_path
        
        print('------------------------------------')
        print(filename)
        print('------------------------------------')
        
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
