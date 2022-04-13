_base_ = [
    '../_base_/models/mask_rcnn_vitdet_sfp_sincospe.py',
    '../_base_/datasets/coco_instance_vitdet_100e.py',
    '../_base_/schedules/schedule_100e_fp16_vitdet.py', '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)
find_unused_parameters=True