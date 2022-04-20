_base_ = './mimdet_Op5_3x_fp16_coco.py'

model = dict(backbone=dict(sample_ratio=0.25))
optimizer = dict(lr=2e-5 * 0.5)