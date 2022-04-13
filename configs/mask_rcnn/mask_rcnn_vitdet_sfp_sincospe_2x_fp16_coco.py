_base_ = [
    '../_base_/models/mask_rcnn_vitdet_sfp_sincospe.py',
    '../_base_/datasets/coco_instance_vitdet.py',
    '../_base_/schedules/schedule_2x_fp16_vitdet.py', '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=6)
find_unused_parameters=True