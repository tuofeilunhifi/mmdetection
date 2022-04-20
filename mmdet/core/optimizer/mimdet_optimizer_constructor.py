import json
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info


@OPTIMIZER_BUILDERS.register_module()
class MIMDetOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options

        parameter_groups = {}
        print(self.paramwise_cfg)
        weight_decay = self.paramwise_cfg.get('weight_decay')
        weight_decay_norm = self.paramwise_cfg.get('weight_decay_norm')
        base_lr = self.paramwise_cfg.get('base_lr')
        skip_list = self.paramwise_cfg.get('skip_list')
        multiplier = self.paramwise_cfg.get('multiplier')
        print("Build MIMDetOptimizerConstructor")

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = "no_decay"
                this_weight_decay = 0.0
            elif "norm" in name and weight_decay_norm is not None:
                group_name = "decay"
                this_weight_decay = weight_decay_norm
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            if name.startswith("backbone.bottom_up.encoder.patch_embed"):
                group_name = "backbone.bottom_up.encoder.patch_embed_%s" % (group_name)
                if group_name not in parameter_groups:
                    parameter_groups[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "param_names": [],
                        "lr": base_lr,
                    }
            elif name.startswith("backbone.bottom_up.encoder"):
                group_name = "backbone.bottom_up.encoder_%s" % (group_name)
                if group_name not in parameter_groups:
                    parameter_groups[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "param_names": [],
                        "lr": base_lr / multiplier,
                    }
            else:
                group_name = "others_%s" % (group_name)
                if group_name not in parameter_groups:
                    parameter_groups[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "param_names": [],
                        "lr": base_lr * multiplier,
                    }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"], 
                    "lr": parameter_groups[key]["lr"], 
                    "weight_decay": parameter_groups[key]["weight_decay"], 
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())