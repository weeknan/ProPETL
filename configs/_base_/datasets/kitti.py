crop_size = (
    768,
    768,
)

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=0.5, keep_ratio=True),
                dict(type='Resize', scale_factor=0.75, keep_ratio=True),
                dict(type='Resize', scale_factor=1.0, keep_ratio=True),
                dict(type='Resize', scale_factor=1.25, keep_ratio=True),
                dict(type='Resize', scale_factor=1.5, keep_ratio=True),
                dict(type='Resize', scale_factor=1.75, keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='KittiRoadDataset',
        data_root='data/kitti_road/data_road',
        data_prefix=dict(
            img_path='training/image_2', 
            seg_map_path='training/gt_image_2_wo_lane'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotationsColor'),
            dict(
                type='RandomResize',
                scale=(
                    2484,
                    750,
                ),
                ratio_range=(
                    0.5,
                    2.0,
                ),
                keep_ratio=True),
            dict(
                type='RandomCrop', crop_size=crop_size, 
                cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiRoadDataset',
        data_root='data/kitti_road/data_road',
        data_prefix=dict(
            img_path='training/image_2', 
            seg_map_path='training/gt_image_2_wo_lane'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                2980,
                900,
            ), keep_ratio=True),
            dict(type='LoadAnnotationsColor'),
            dict(type='PackSegInputs'),
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiRoadDataset',
        data_root='data/kitti_road/data_road',
        data_prefix=dict(
            img_path='training/image_2', 
            seg_map_path='training/gt_image_2_wo_lane'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                2049,
                1025,
            ), keep_ratio=True),
            dict(type='LoadAnnotationsColor'),
            dict(type='PackSegInputs'),
        ]))

val_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
        'mFscore',
    ])
test_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
    ])