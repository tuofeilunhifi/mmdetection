_base_ = [
    '../_base_/models/mask_rcnn_vitdet_sfp.py',
    '../_base_/datasets/coco_instance_vitdet.py',
    '../_base_/schedules/schedule_25e_cosine_fp16_vitdet.py', '../_base_/default_runtime.py'
]
