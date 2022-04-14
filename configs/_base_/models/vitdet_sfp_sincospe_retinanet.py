# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ViTDetVisionTransformer',
        arch='b',
        img_size=1024,
        patch_size=16,
        window_size=16,
        drop_path_rate=0.1,
        out_indices=[11],
        final_norm=True,
        sincos_pos_embed=True,
        init_cfg=dict(type='Pretrained', checkpoint='/home/yunji.cjy/pretrain/warpper_mae_vit-base-p16-1600e.pth')),
    neck=dict(
        type='SFP',
        in_channels=768,
        out_channels=256,
        norm_cfg=dict(type='LN')),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
