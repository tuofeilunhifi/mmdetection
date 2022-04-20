# optimizer
paramwise_cfg={
    'weight_decay': 0.1,
    'weight_decay_norm': 0.0,
    'base_lr': 2e-5,
    'skip_list': ("pos_embed", "decoder_pos_embed"),
    'multiplier': 2.0,
}
optimizer = dict(type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.1,
                    constructor='MIMDetOptimizerConstructor',
                    paramwise_cfg=paramwise_cfg)

cumulative_iters = 2
optimizer_config = dict(grad_clip=None, cumulative_iters=cumulative_iters)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=0.25,
    warmup_ratio=0.0001,
    step=[27, 33])

runner = dict(type='EpochBasedRunner', max_epochs=36)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
