_base_ = [
    '../_base_/datasets/SARDet_100k.py',
    '../_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

model = dict(
    type='GFL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[36.50463548417378, 36.50467669785022, 36.50465618752095],
        std=[52.12157845801164, 52.12163789765554, 52.12160835551052],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='groksar.FrequencySpatialFPN',  # FPN을 FrequencySpatialFPN으로 변경
        in_channels=[256, 512, 1024, 2048],  # ResNet 백본 출력 채널
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)  # GroupNorm 추가
    ),
    bbox_head=dict(
        type='GFLHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='DAdaptAdam',
        lr=1.0,
        weight_decay=0.05,
        decouple=True),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True)
)

# Scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.333, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# Dataloader
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='groksar.SAR_Det_Finegrained_Dataset',
        data_root='/root/GrokSAR/datasets/SARDet-100K',
        data_prefix=dict(img='JPEGImages/'),
        ann_file='Annotations/train.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='groksar.SAR_Det_Finegrained_Dataset',
        data_root='/root/GrokSAR/datasets/SARDet-100K',
        data_prefix=dict(img='JPEGImages/'),
        ann_file='Annotations/val.json',
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    classwise=False,
    format_only=False,
    ann_file='/root/GrokSAR/datasets/SARDet-100K/Annotations/val.json'
)

test_evaluator = val_evaluator

# Training and Validation Configs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Visualization
visualizer = dict(
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)
