_base_ = [
    '../_base_/models/vitdet_sfp_sincospe.py',
    '../_base_/datasets/coco_instance_vitdet_50e.py',
    '../_base_/schedules/schedule_vitdet_fp16.py', '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)
find_unused_parameters=True