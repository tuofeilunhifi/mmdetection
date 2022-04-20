_base_ = [
    './_base_/models/mimdet.py',
    './_base_/datasets/coco_instance_mimdet_3x.py',
    './_base_/schedules/schedule_mimdet_fp16_3x.py', '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)
find_unused_parameters=True