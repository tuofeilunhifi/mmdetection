# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class SFP(BaseModule):
    r"""Simple Feature Pyramid.

    This is an implementation of paper `Exploring Plain Vision Transformer Backbonesfor Object 
    Detection <https://arxiv.org/abs/2203.16527>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=[
                     dict(type='Xavier', layer=['Conv2d', 'ConvTranspose2d'], distribution='uniform'), 
                     dict(type='Constant', layer=['GroupNorm'], val=1)
                ]]):
        super(SFP, self).__init__(init_cfg)
        assert isinstance(in_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_level = 4
        self.fp16_enabled = False

        self.top_downs = nn.ModuleList()
        self.sfp_outs = nn.ModuleList()

        for i in range(self.num_level):
            if i == 0:
                top_down = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2, padding=0),
                    nn.GroupNorm(1, in_channels, eps=1e-6),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2, padding=0)
                )
            elif i == 1:
                top_down = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2, padding=0)
            elif i == 2:
                top_down = nn.Identity()
            elif i == 3:
                top_down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            sfp_out = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(1, out_channels, eps=1e-6),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(1, out_channels, eps=1e-6)
            )

            self.top_downs.append(top_down)
            self.sfp_outs.append(sfp_out)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == 1

        # build outputs
        outs = []
        for i in range(self.num_level):

            x = self.top_downs[i](inputs[0])
            x = self.sfp_outs[i](x)

            outs.append(x)

        return tuple(outs)
