_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/coco-stuff10k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'pretrain/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer_adaptformer_2concat',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        adapter_bottleneck_reduction=48,
        saved_adapter_ckpt='./work_dirs/swin_l_1xb16_40k_coco-640x640_adaptformer_r48_global_cls/iter_40000.pth',
        ),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=171),
    auxiliary_head=dict(in_channels=768, num_classes=171))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=80000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

train_cfg = dict(val_interval=1000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=1000,
        save_best='mIoU'),
    )