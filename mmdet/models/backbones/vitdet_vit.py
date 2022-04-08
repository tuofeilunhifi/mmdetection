# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import VisionTransformer
from mmcv.cnn import build_norm_layer
from mmcv.utils import to_2tuple
from mmdet.models.utils import resize

from ..builder import BACKBONES
from ..utils.transformer import PatchEmbed

@BACKBONES.register_module()
class ViTDetVisionTransformer(VisionTransformer):
    """Vision Transformer for MIM-style model (Mask Image Modeling)
    classification (fine-tuning or linear probe).
    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_
    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        finetune (bool): Whether or not do fine-tuning. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 patch_norm=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 finetune=True,
                 init_cfg=None):
        super().__init__(
            arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.embed_dims = self.arch_settings['embed_dims']
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            in_channels=3,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        if not self.final_norm:
            _, self.fc_norm = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)

        self.finetune = finetune
        if not self.finetune:
            self._freeze_stages()

    def train(self, mode=True):
        super(ViTDetVisionTransformer, self).train(mode)
        if not self.finetune:
            self._freeze_stages()

    def _freeze_stages(self):
        """Freeze params in backbone when linear probing."""
        for _, param in self.named_parameters():
            param.requires_grad = False

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positiong embeding method.
        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return patched_img + pos_embed

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def window_partition(self, x):
        B, L, C = x.shape
        x = x.reshape(64 * B , -1, C)
        return x

    def window_reverse(self, x):
        B, L, C = x.shape
        x = x.reshape(B // 64, -1, C)
        return x 

    def forward(self, x):
        # print("1", x.shape)
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)
        # print("2", x.shape)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)
        x = self.drop_after_pos(x)

        # remove class token
        x = x[:, 1:]

        # window_partition
        x = self.window_partition(x)

        outs = []
        for i, layer in enumerate(self.layers):
            # local self-attention & global self-attention
            if (self.layers==12 and (i+1) % 3 == 0) or (self.layers==24 and (i+1) % 6 == 0):
                x  = self.window_reverse(x)
                x = layer(x)
                x = self.window_partition(x)
            else:
                x = layer(x)

            if i == len(self.layers) - 1:
                # window_reverse
                x = self.window_reverse(x)
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                B, _, C = x.shape
                x = x.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(x)

        return tuple(outs)