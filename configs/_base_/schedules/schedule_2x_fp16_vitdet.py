# optimizer
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)
cumulative_iters = 4
optimizer_config = dict(grad_clip=None, cumulative_iters=cumulative_iters)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250 * cumulative_iters,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
