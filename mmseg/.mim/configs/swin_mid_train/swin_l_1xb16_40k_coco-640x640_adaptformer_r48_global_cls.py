_base_ = [
    '../_base_/datasets/coco-stuff10k_640x640.py',
]
num_classes = 171
backbone_norm_cfg = dict(type='LN', eps=1e-06, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (640, 640)

model = dict(
    type='EncoderPureRegion',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size),
    pretrained=None,
    backbone=dict(
        type='SwinTransformer_adaptformer_mid_global_cls',
        pretrain_img_size=224,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[
            2,
            2,
            18,
            2,
        ],
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        strides=(
            4,
            2,
            2,
            2,
        ),
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'pretrain/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
        ),
        adapter_bottleneck_reduction=48,
        num_classes=num_classes,
        semantic_loss_decode=dict(
            type='GlobalClsLoss',
            loss_weight=1.0,
            num_classes=num_classes)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(
        768,
        768,
    ), stride=(
        512,
        512,
    )))

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-05, betas=(
            0.9,
            0.999,
        ), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=40000,
        by_epoch=False),
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=99999999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
checkpoint_file = 'pretrain/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
launcher = 'pytorch'

train_dataloader = dict(batch_size=16)