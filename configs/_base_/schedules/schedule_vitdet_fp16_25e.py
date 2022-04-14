# optimizer
paramwise_cfg={
    'norm': dict(weight_decay=0.),
    'bias': dict(weight_decay=0.),
    'pos_embed': dict(weight_decay=0.),
    'cls_token': dict(weight_decay=0.)
}
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1, 
                    # constructor='TransformerFinetuneConstructor',
                    # model_type='vit',
                    # layer_decay=0.70,
                    paramwise_cfg=paramwise_cfg)

cumulative_iters = 4
optimizer_config = dict(grad_clip=None, cumulative_iters=cumulative_iters)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250 * cumulative_iters * 4,
    warmup_ratio=0.001,
    step=[22, 24])
runner = dict(type='EpochBasedRunner', max_epochs=25)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
