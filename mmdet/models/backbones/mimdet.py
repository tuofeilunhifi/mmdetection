# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES

from .vitdet_vit import Block

from ..utils.pos_embed import (
    get_2d_sincos_pos_embed,
    interpolate_pos_embed,
    interpolate_pos_embed_online,
)

class ConvStem(nn.Module):
    """ConvStem, from Early Convolutions Help Transformers See Better, Tete et
    al.
    https://arxiv.org/abs/2106.14881
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=4,
        norm_layer=None,
    ):
        super().__init__()

        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.depth = depth

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // (2 ** (depth - 1))
        for idx in range(depth):
            stage_list = [
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, output_dim, eps=1e-6),
                nn.GELU(),
            ]
            if idx == depth - 1:
                stage_list.append(nn.Conv2d(output_dim, embed_dim, kernel_size=1))
            stage = nn.Sequential(*stage_list)
            input_dim = output_dim
            output_dim *= 2
            stem.append(stage)
        self.proj = nn.ModuleList(stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        outputs = []
        for i, stage in enumerate(self.proj):
            x = stage(x)
            if i >= 1:
                if i == (len(self.proj) - 1):
                    outputs.append(self.norm(x))
                else:
                    outputs.append(x)
        return outputs


class MIMDetEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        dpr=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-5),
        pretrained=None,
    ):
        super().__init__()

        self.patch_embed = ConvStem(
            img_size, patch_size, in_chans, embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )

        dpr = [
            x.item() for x in torch.linspace(0, dpr, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        #self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        if pretrained:
            checkpoint_model = torch.load(pretrained, map_location="cpu")["model"]
            new_checkpoint_model = {}
            for k, v in checkpoint_model.items():
                if "encoder" in k:
                    new_checkpoint_model[k.replace("encoder.", "")] = v
                elif "module" in k:
                    new_checkpoint_model[k.replace("module.", "")] = v
                else:
                    new_checkpoint_model[k] = v
            interpolate_pos_embed(self, new_checkpoint_model, "pos_embed")
            print(self.load_state_dict(new_checkpoint_model, strict=False))
            print(f"Loading ViT Encoder pretrained weights from {pretrained}.")
        else:
            print("Loading ViT Encoder pretrained weights from scratch.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, sample_ratio, masks):
        N, L, D = x.shape
        masks_flatten = masks[0].flatten(1)
        assert masks_flatten.shape[1] == L
        len_keep = int(L * sample_ratio)

        noise = torch.rand(N, L, device=x.device)
        noise = noise.masked_fill(masks_flatten, 100)

        # sort noise for each sample
        ids_keep = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_keep, dim=1)

        # keep the first subset
        ids_keep = ids_keep[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, ids_restore

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones(
                (N, H, W), dtype=torch.bool, device=device
            )
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.patch_embed.patch_size[0])),
                    : int(np.ceil(float(w) / self.patch_embed.patch_size[1])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks

    def forward(self, imgs, sample_ratio):
        outputs = self.patch_embed(imgs)
        x = outputs[-1]
        H, W = x.shape[-2:]
        masks = self.mask_out_padding([x.shape], [imgs.shape[-2:]], imgs.device)
        x = x.flatten(2).transpose(1, 2)
        pos_embed = interpolate_pos_embed_online(
            self.pos_embed, self.patch_embed.grid_size, (H, W), 1
        )[:, 1:, :]
        x = x + pos_embed

        x, ids_restore = self.random_masking(x, sample_ratio, masks)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        outputs[-1] = x
        x = outputs

        return x, ids_restore, (H, W)


class MIMDetDecoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        decoder_embed_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        dpr=0.0,
        norm_layer=nn.LayerNorm,
        pretrained=None,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        #self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        if pretrained:
            checkpoint_model = torch.load(pretrained, map_location="cpu")["model"]
            new_checkpoint_model = {}
            for k, v in checkpoint_model.items():
                if "decoder" in k:
                    new_checkpoint_model[k.replace("decoder.", "")] = v
                elif "module" in k:
                    new_checkpoint_model[k.replace("module.", "")] = v
                else:
                    new_checkpoint_model[k] = v
            interpolate_pos_embed(self, new_checkpoint_model, "decoder_pos_embed")
            print(self.load_state_dict(new_checkpoint_model, strict=False))
            print(f"Loading ViT Decoder pretrained weights from {pretrained}.")
        else:
            print("Loading ViT Decoder pretrained weights from scratch.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore, new_size):
        x = self.decoder_embed(x)
        B, L, C = x.shape
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - L, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )  # unshuffle

        pos_embed = interpolate_pos_embed_online(
            self.decoder_pos_embed, self.grid_size, new_size, 1
        )[:, 1:, :]

        x = x + pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x.transpose(1, 2).reshape(B, C, *new_size)

@BACKBONES.register_module()
class MIMDet(BaseModule):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
    """

    def __init__(self,
                 encoder,
                 decoder,
                 sample_ratio,
                 init_cfg=None):
        super(MIMDet, self).__init__(init_cfg)

        self.encoder = MIMDetEncoder(**encoder)
        self.decoder = MIMDetDecoder(**decoder)
        self.enc_pretrained = encoder['pretrained']
        self.dec_pretrained = decoder['pretrained']
        self.sample_ratio = sample_ratio

    def init_weights(self):
        self.encoder.initialize_weights(self.enc_pretrained)
        self.decoder.initialize_weights(self.dec_pretrained)

    def forward(self, x):
        outs = []
        latent, ids_restore, new_size = self.encoder(x, self.sample_ratio)
        latent[-1] = self.decoder(latent[-1], ids_restore, new_size)
        outs += latent
        outs.append(F.max_pool2d(latent[-1], 2))

        return tuple(outs)
