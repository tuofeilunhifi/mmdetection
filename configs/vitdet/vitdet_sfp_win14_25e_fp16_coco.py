_base_ = [
    '../_base_/models/vitdet_sfp_win14.py',
    '../_base_/datasets/coco_instance_vitdet_25e.py',
    '../_base_/schedules/schedule_vitdet_fp16_25e.py', '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)
find_unused_parameters=True