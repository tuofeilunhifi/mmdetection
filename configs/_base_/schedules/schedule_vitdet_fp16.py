# optimizer
optimizer = dict(
    #_delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                        num_layers=12, 
                        layer_decay_rate=0.7,
                        custom_keys={
                            'bias': dict(decay_multi=0.),
                            'pos_embed': dict(decay_mult=0.),
                            'relative_position_bias_table': dict(decay_mult=0.),
                            'norm': dict(decay_mult=0.),
                            "rel_pos_h": dict(decay_mult=0.),
                            "rel_pos_w": dict(decay_mult=0.),
                            }
                            )
                 )

cumulative_iters = 4
optimizer_config = dict(grad_clip=None, cumulative_iters=cumulative_iters)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250 * cumulative_iters,
    warmup_ratio=0.067,
    step=[22, 24])
runner = dict(type='EpochBasedRunner', max_epochs=25)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
