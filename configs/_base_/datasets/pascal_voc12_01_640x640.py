_base_ = './pascal_voc12_640x640.py'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        ann_file='ImageSets/Segmentation/train_01_a.txt',
        ))

