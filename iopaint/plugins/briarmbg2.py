# copy from https://huggingface.co/briaai/RMBG-2.0/tree/main
import os
import math
import numpy as np

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers import PretrainedConfig

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.utils.checkpoint as checkpoint

from collections import OrderedDict

from torchvision.models import (
    vgg16,
    vgg16_bn,
    VGG16_Weights,
    VGG16_BN_Weights,
    resnet50,
    ResNet50_Weights,
)


class Config:
    def __init__(self) -> None:
        # PATH settings
        self.sys_home_dir = os.path.expanduser(
            "~"
        )  # Make up your file system as: SYS_HOME_DIR/codes/dis/BiRefNet, SYS_HOME_DIR/datasets/dis/xx, SYS_HOME_DIR/weights/xx

        # TASK settings
        self.task = ["DIS5K", "COD", "HRSOD", "DIS5K+HRSOD+HRS10K", "P3M-10k"][0]
        self.training_set = {
            "DIS5K": ["DIS-TR", "DIS-TR+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4"][0],
            "COD": "TR-COD10K+TR-CAMO",
            "HRSOD": [
                "TR-DUTS",
                "TR-HRSOD",
                "TR-UHRSD",
                "TR-DUTS+TR-HRSOD",
                "TR-DUTS+TR-UHRSD",
                "TR-HRSOD+TR-UHRSD",
                "TR-DUTS+TR-HRSOD+TR-UHRSD",
            ][5],
            "DIS5K+HRSOD+HRS10K": "DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4+DIS-TR+TE-HRS10K+TE-HRSOD+TE-UHRSD+TR-HRS10K+TR-HRSOD+TR-UHRSD",  # leave DIS-VD for evaluation.
            "P3M-10k": "TR-P3M-10k",
        }[self.task]
        self.prompt4loc = ["dense", "sparse"][0]

        # Faster-Training settings
        self.load_all = True
        self.compile = True  # 1. Trigger CPU memory leak in some extend, which is an inherent problem of PyTorch.
        #   Machines with > 70GB CPU memory can run the whole training on DIS5K with default setting.
        # 2. Higher PyTorch version may fix it: https://github.com/pytorch/pytorch/issues/119607.
        # 3. But compile in Pytorch > 2.0.1 seems to bring no acceleration for training.
        self.precisionHigh = True

        # MODEL settings
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = [0, 3][1]  # multi-scale skip connections from encoder
        self.mul_scl_ipt = ["", "add", "cat"][2]
        self.dec_att = ["", "ASPP", "ASPPDeformable"][2]
        self.squeeze_block = [
            "",
            "BasicDecBlk_x1",
            "ResBlk_x4",
            "ASPP_x3",
            "ASPPDeformable_x3",
        ][1]
        self.dec_blk = ["BasicDecBlk", "ResBlk", "HierarAttDecBlk"][0]

        # TRAINING settings
        self.batch_size = 4
        self.IoU_finetune_last_epochs = [
            0,
            {
                "DIS5K": -50,
                "COD": -20,
                "HRSOD": -20,
                "DIS5K+HRSOD+HRS10K": -20,
                "P3M-10k": -20,
            }[self.task],
        ][1]  # choose 0 to skip
        self.lr = (1e-4 if "DIS5K" in self.task else 1e-5) * math.sqrt(
            self.batch_size / 4
        )  # DIS needs high lr to converge faster. Adapt the lr linearly
        self.size = 1024
        self.num_workers = max(
            4, self.batch_size
        )  # will be decrease to min(it, batch_size) at the initialization of the data_loader

        # Backbone settings
        self.bb = [
            "vgg16",
            "vgg16bn",
            "resnet50",  # 0, 1, 2
            "swin_v1_t",
            "swin_v1_s",  # 3, 4
            "swin_v1_b",
            "swin_v1_l",  # 5-bs9, 6-bs4
            "pvt_v2_b0",
            "pvt_v2_b1",  # 7, 8
            "pvt_v2_b2",
            "pvt_v2_b5",  # 9-bs10, 10-bs5
        ][6]
        self.lateral_channels_in_collection = {
            "vgg16": [512, 256, 128, 64],
            "vgg16bn": [512, 256, 128, 64],
            "resnet50": [1024, 512, 256, 64],
            "pvt_v2_b2": [512, 320, 128, 64],
            "pvt_v2_b5": [512, 320, 128, 64],
            "swin_v1_b": [1024, 512, 256, 128],
            "swin_v1_l": [1536, 768, 384, 192],
            "swin_v1_t": [768, 384, 192, 96],
            "swin_v1_s": [768, 384, 192, 96],
            "pvt_v2_b0": [256, 160, 64, 32],
            "pvt_v2_b1": [512, 320, 128, 64],
        }[self.bb]
        if self.mul_scl_ipt == "cat":
            self.lateral_channels_in_collection = [
                channel * 2 for channel in self.lateral_channels_in_collection
            ]
        self.cxt = (
            self.lateral_channels_in_collection[1:][::-1][-self.cxt_num :]
            if self.cxt_num
            else []
        )

        # MODEL settings - inactive
        self.lat_blk = ["BasicLatBlk"][0]
        self.dec_channels_inter = ["fixed", "adap"][0]
        self.refine = ["", "itself", "RefUNet", "Refiner", "RefinerPVTInChannels4"][0]
        self.progressive_ref = self.refine and True
        self.ender = self.progressive_ref and False
        self.scale = self.progressive_ref and 2
        self.auxiliary_classification = (
            False  # Only for DIS5K, where class labels are saved in `dataset.py`.
        )
        self.refine_iteration = 1
        self.freeze_bb = False
        self.model = [
            "BiRefNet",
        ][0]
        if self.dec_blk == "HierarAttDecBlk":
            self.batch_size = 2 ** [0, 1, 2, 3, 4][2]

        # TRAINING settings - inactive
        self.preproc_methods = ["flip", "enhance", "rotate", "pepper", "crop"][:4]
        self.optimizer = ["Adam", "AdamW"][1]
        self.lr_decay_epochs = [
            1e5
        ]  # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5
        # Loss
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            "bce": 30 * 1,  # high performance
            "iou": 0.5 * 1,  # 0 / 255
            "iou_patch": 0.5 * 0,  # 0 / 255, win_size = (64, 64)
            "mse": 150 * 0,  # can smooth the saliency map
            "triplet": 3 * 0,
            "reg": 100 * 0,
            "ssim": 10 * 1,  # help contours,
            "cnt": 5 * 0,  # help contours
            "structure": 5
            * 0,  # structure loss from codes of MVANet. A little improvement on DIS-TE[1,2,3], a bit more decrease on DIS-TE4.
        }
        self.lambdas_cls = {"ce": 5.0}
        # Adv
        self.lambda_adv_g = 10.0 * 0  # turn to 0 to avoid adv training
        self.lambda_adv_d = 3.0 * (self.lambda_adv_g > 0)

        # PATH settings - inactive
        self.data_root_dir = os.path.join(self.sys_home_dir, "datasets/dis")
        self.weights_root_dir = os.path.join(self.sys_home_dir, "weights")
        self.weights = {
            "pvt_v2_b2": os.path.join(self.weights_root_dir, "pvt_v2_b2.pth"),
            "pvt_v2_b5": os.path.join(
                self.weights_root_dir, ["pvt_v2_b5.pth", "pvt_v2_b5_22k.pth"][0]
            ),
            "swin_v1_b": os.path.join(
                self.weights_root_dir,
                [
                    "swin_base_patch4_window12_384_22kto1k.pth",
                    "swin_base_patch4_window12_384_22k.pth",
                ][0],
            ),
            "swin_v1_l": os.path.join(
                self.weights_root_dir,
                [
                    "swin_large_patch4_window12_384_22kto1k.pth",
                    "swin_large_patch4_window12_384_22k.pth",
                ][0],
            ),
            "swin_v1_t": os.path.join(
                self.weights_root_dir,
                ["swin_tiny_patch4_window7_224_22kto1k_finetune.pth"][0],
            ),
            "swin_v1_s": os.path.join(
                self.weights_root_dir,
                ["swin_small_patch4_window7_224_22kto1k_finetune.pth"][0],
            ),
            "pvt_v2_b0": os.path.join(self.weights_root_dir, ["pvt_v2_b0.pth"][0]),
            "pvt_v2_b1": os.path.join(self.weights_root_dir, ["pvt_v2_b1.pth"][0]),
        }

        # Callbacks - inactive
        self.verbose_eval = True
        self.only_S_MAE = False
        self.use_fp16 = False  # Bugs. It may cause nan in training.
        self.SDPA_enabled = False  # Bugs. Slower and errors occur in multi-GPUs

        # others
        self.device = [0, "cpu"][0]  # .to(0) == .to('cuda:0')

        self.batch_size_valid = 1
        self.rand_seed = 7
        # run_sh_file = [f for f in os.listdir('.') if 'train.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'train.sh' == f]
        # with open(run_sh_file[0], 'r') as f:
        #     lines = f.readlines()
        #     self.save_last = int([l.strip() for l in lines if '"{}")'.format(self.task) in l and 'val_last=' in l][0].split('val_last=')[-1].split()[0])
        #     self.save_step = int([l.strip() for l in lines if '"{}")'.format(self.task) in l and 'step=' in l][0].split('step=')[-1].split()[0])
        # self.val_step = [0, self.save_step][0]

    def print_task(self) -> None:
        # Return task for choosing settings in shell scripts.
        print(self.task)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        if config.SDPA_enabled:
            x = (
                torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.attn_drop_prob,
                    is_causal=False,
                )
                .transpose(1, 2)
                .reshape(B, N, C)
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=7, stride=4, in_channels=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_channels=in_channels,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

        # return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


## @register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, in_channels=3, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            in_channels=in_channels,
        )


## @register_model
class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


## @register_model
class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


## @register_model
class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


### models/backbones/swin_v1.py

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        if config.SDPA_enabled:
            x = (
                torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.attn_drop_prob,
                    is_causal=False,
                )
                .transpose(1, 2)
                .reshape(B_, N, C)
            )
        else:
            attn = q @ k.transpose(-2, -1)

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                    1
                ).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = x + absolute_pos_embed  # B Wh*Ww C

        outs = []  # x.contiguous()]
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = (
                    x_out.view(-1, H, W, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


def swin_v1_t():
    model = SwinTransformer(
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7
    )
    return model


def swin_v1_s():
    model = SwinTransformer(
        embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=7
    )
    return model


def swin_v1_b():
    model = SwinTransformer(
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12
    )
    return model


def swin_v1_l():
    model = SwinTransformer(
        embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12
    )
    return model


### models/modules/deform_conv.py

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = (
            kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        )
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


### utils.py

import torch.nn as nn


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


def build_norm_layer(
    dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6
):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


### dataset.py

_class_labels_TR_sorted = (
    "Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, "
    "BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, "
    "CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, "
    "Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, "
    "Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, "
    "Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, "
    "KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, "
    "Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, "
    "OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, "
    "RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, "
    "ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, "
    "Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, "
    "TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, "
    "UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht"
)
class_labels_TR_sorted = _class_labels_TR_sorted.split(", ")


### models/backbones/build_backbones.py

config = Config()


def build_backbone(bb_name, pretrained=True, params_settings=""):
    if bb_name == "vgg16":
        bb_net = list(
            vgg16(pretrained=VGG16_Weights.DEFAULT if pretrained else None).children()
        )[0]
        bb = nn.Sequential(
            OrderedDict(
                {
                    "conv1": bb_net[:4],
                    "conv2": bb_net[4:9],
                    "conv3": bb_net[9:16],
                    "conv4": bb_net[16:23],
                }
            )
        )
    elif bb_name == "vgg16bn":
        bb_net = list(
            vgg16_bn(
                pretrained=VGG16_BN_Weights.DEFAULT if pretrained else None
            ).children()
        )[0]
        bb = nn.Sequential(
            OrderedDict(
                {
                    "conv1": bb_net[:6],
                    "conv2": bb_net[6:13],
                    "conv3": bb_net[13:23],
                    "conv4": bb_net[23:33],
                }
            )
        )
    elif bb_name == "resnet50":
        bb_net = list(
            resnet50(
                pretrained=ResNet50_Weights.DEFAULT if pretrained else None
            ).children()
        )
        bb = nn.Sequential(
            OrderedDict(
                {
                    "conv1": nn.Sequential(*bb_net[0:3]),
                    "conv2": bb_net[4],
                    "conv3": bb_net[5],
                    "conv4": bb_net[6],
                }
            )
        )
    else:
        bb = eval("{}({})".format(bb_name, params_settings))
        if pretrained:
            bb = load_weights(bb, bb_name)
    return bb


def load_weights(model, model_name):
    save_model = torch.load(config.weights[model_name], map_location="cpu")
    model_dict = model.state_dict()
    state_dict = {
        k: v if v.size() == model_dict[k].size() else model_dict[k]
        for k, v in save_model.items()
        if k in model_dict.keys()
    }
    # to ignore the weights with mismatched size when I modify the backbone itself.
    if not state_dict:
        save_model_keys = list(save_model.keys())
        sub_item = save_model_keys[0] if len(save_model_keys) == 1 else None
        state_dict = {
            k: v if v.size() == model_dict[k].size() else model_dict[k]
            for k, v in save_model[sub_item].items()
            if k in model_dict.keys()
        }
        if not state_dict or not sub_item:
            print(
                "Weights are not successully loaded. Check the state dict of weights file."
            )
            return None
        else:
            print(
                'Found correct weights in the "{}" item of loaded state_dict.'.format(
                    sub_item
                )
            )
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model


### models/modules/decoder_blocks.py

import torch
import torch.nn as nn
# from models.aspp import ASPP, ASPPDeformable
# from config import Config


# config = Config()


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicDecBlk, self).__init__()
        inter_channels = in_channels // 4 if config.dec_channels_inter == "adap" else 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att == "ASPP":
            self.dec_att = ASPP(in_channels=inter_channels)
        elif config.dec_att == "ASPPDeformable":
            self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = (
            nn.BatchNorm2d(inter_channels) if config.batch_size > 1 else nn.Identity()
        )
        self.bn_out = (
            nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        if hasattr(self, "dec_att"):
            x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class ResBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=None, inter_channels=64):
        super(ResBlk, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_channels = in_channels // 4 if config.dec_channels_inter == "adap" else 64

        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.bn_in = (
            nn.BatchNorm2d(inter_channels) if config.batch_size > 1 else nn.Identity()
        )
        self.relu_in = nn.ReLU(inplace=True)

        if config.dec_att == "ASPP":
            self.dec_att = ASPP(in_channels=inter_channels)
        elif config.dec_att == "ASPPDeformable":
            self.dec_att = ASPPDeformable(in_channels=inter_channels)

        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_out = (
            nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()
        )

        self.conv_resi = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        _x = self.conv_resi(x)
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        if hasattr(self, "dec_att"):
            x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x + _x


### models/modules/lateral_blocks.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# from config import Config


# config = Config()


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicLatBlk, self).__init__()
        inter_channels = in_channels // 4 if config.dec_channels_inter == "adap" else 64
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


### models/modules/aspp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.deform_conv import DeformableConv2d
# from config import Config


# config = Config()


class _ASPPModule(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            in_channels,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes) if config.batch_size > 1 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, in_channels=64, out_channels=None, output_stride=16):
        super(ASPP, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(
            in_channels, self.in_channelster, 1, padding=0, dilation=dilations[0]
        )
        self.aspp2 = _ASPPModule(
            in_channels,
            self.in_channelster,
            3,
            padding=dilations[1],
            dilation=dilations[1],
        )
        self.aspp3 = _ASPPModule(
            in_channels,
            self.in_channelster,
            3,
            padding=dilations[2],
            dilation=dilations[2],
        )
        self.aspp4 = _ASPPModule(
            in_channels,
            self.in_channelster,
            3,
            padding=dilations[3],
            dilation=dilations[3],
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_channelster)
            if config.batch_size > 1
            else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(self.in_channelster * 5, out_channels, 1, bias=False)
        self.bn1 = (
            nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


##################### Deformable
class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = DeformableConv2d(
            in_channels,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes) if config.batch_size > 1 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=[1, 3, 7]):
        super(ASPPDeformable, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(
            in_channels, self.in_channelster, 1, padding=0
        )
        self.aspp_deforms = nn.ModuleList(
            [
                _ASPPModuleDeformable(
                    in_channels,
                    self.in_channelster,
                    conv_size,
                    padding=int(conv_size // 2),
                )
                for conv_size in parallel_block_sizes
            ]
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_channelster)
            if config.batch_size > 1
            else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(
            self.in_channelster * (2 + len(self.aspp_deforms)),
            out_channels,
            1,
            bias=False,
        )
        self.bn1 = (
            nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


### models/refinement/refiner.py


class RefinerPVTInChannels4(nn.Module):
    def __init__(self, in_channels=3 + 1):
        super(RefinerPVTInChannels4, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.bb = build_backbone(self.config.bb, params_settings="in_channels=4")

        lateral_channels_in_collection = {
            "vgg16": [512, 256, 128, 64],
            "vgg16bn": [512, 256, 128, 64],
            "resnet50": [1024, 512, 256, 64],
            "pvt_v2_b2": [512, 320, 128, 64],
            "pvt_v2_b5": [512, 320, 128, 64],
            "swin_v1_b": [1024, 512, 256, 128],
            "swin_v1_l": [1536, 768, 384, 192],
        }
        channels = lateral_channels_in_collection[self.config.bb]
        self.squeeze_module = BasicDecBlk(channels[0], channels[0])

        self.decoder = Decoder(channels)

        if 0:
            for key, value in self.named_parameters():
                if "bb." in key:
                    value.requires_grad = False

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        ########## Encoder ##########
        if self.config.bb in ["vgg16", "vgg16bn", "resnet50"]:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)

        x4 = self.squeeze_module(x4)

        ########## Decoder ##########

        features = [x, x1, x2, x3, x4]
        scaled_preds = self.decoder(features)

        return scaled_preds


class Refiner(nn.Module):
    def __init__(self, in_channels=3 + 1):
        super(Refiner, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.stem_layer = StemLayer(
            in_channels=in_channels,
            inter_channels=48,
            out_channels=3,
            norm_layer="BN" if self.config.batch_size > 1 else "LN",
        )
        self.bb = build_backbone(self.config.bb)

        lateral_channels_in_collection = {
            "vgg16": [512, 256, 128, 64],
            "vgg16bn": [512, 256, 128, 64],
            "resnet50": [1024, 512, 256, 64],
            "pvt_v2_b2": [512, 320, 128, 64],
            "pvt_v2_b5": [512, 320, 128, 64],
            "swin_v1_b": [1024, 512, 256, 128],
            "swin_v1_l": [1536, 768, 384, 192],
        }
        channels = lateral_channels_in_collection[self.config.bb]
        self.squeeze_module = BasicDecBlk(channels[0], channels[0])

        self.decoder = Decoder(channels)

        if 0:
            for key, value in self.named_parameters():
                if "bb." in key:
                    value.requires_grad = False

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        x = self.stem_layer(x)
        ########## Encoder ##########
        if self.config.bb in ["vgg16", "vgg16bn", "resnet50"]:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)

        x4 = self.squeeze_module(x4)

        ########## Decoder ##########

        features = [x, x1, x2, x3, x4]
        scaled_preds = self.decoder(features)

        return scaled_preds


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        DecoderBlock = eval("BasicDecBlk")
        LateralBlock = eval("BasicLatBlk")

        self.decoder_block4 = DecoderBlock(channels[0], channels[1])
        self.decoder_block3 = DecoderBlock(channels[1], channels[2])
        self.decoder_block2 = DecoderBlock(channels[2], channels[3])
        self.decoder_block1 = DecoderBlock(channels[3], channels[3] // 2)

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3] // 2, 1, 1, 1, 0))

    def forward(self, features):
        x, x1, x2, x3, x4 = features
        outs = []
        p4 = self.decoder_block4(x4)
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        p3 = self.decoder_block3(_p3)
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        p2 = self.decoder_block2(_p2)
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode="bilinear", align_corners=True)
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision:
            outs.append(self.conv_ms_spvn_4(p4))
            outs.append(self.conv_ms_spvn_3(p3))
            outs.append(self.conv_ms_spvn_2(p2))
        outs.append(p1_out)
        return outs


class RefUNet(nn.Module):
    # Refinement
    def __init__(self, in_channels=3 + 1):
        super(RefUNet, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        #####
        self.decoder_5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        #####
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.conv_d0 = nn.Conv2d(64, 1, 3, 1, 1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        outs = []
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        hx = x

        hx1 = self.encoder_1(hx)
        hx2 = self.encoder_2(hx1)
        hx3 = self.encoder_3(hx2)
        hx4 = self.encoder_4(hx3)

        hx = self.decoder_5(self.pool4(hx4))
        hx = torch.cat((self.upscore2(hx), hx4), 1)

        d4 = self.decoder_4(hx)
        hx = torch.cat((self.upscore2(d4), hx3), 1)

        d3 = self.decoder_3(hx)
        hx = torch.cat((self.upscore2(d3), hx2), 1)

        d2 = self.decoder_2(hx)
        hx = torch.cat((self.upscore2(d2), hx1), 1)

        d1 = self.decoder_1(hx)

        x = self.conv_d0(d1)
        outs.append(x)
        return outs


### models/stem_layer.py


class StemLayer(nn.Module):
    r"""Stem layer of InternImage
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(
        self,
        in_channels=3 + 1,
        inter_channels=48,
        out_channels=96,
        act_layer="GELU",
        norm_layer="BN",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, inter_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = build_norm_layer(
            inter_channels, norm_layer, "channels_first", "channels_first"
        )
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(
            inter_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = build_norm_layer(
            out_channels, norm_layer, "channels_first", "channels_first"
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


### models/birefnet.py


class BiRefNetConfig(PretrainedConfig):
    model_type = "SegformerForSemanticSegmentation"

    def __init__(self, bb_pretrained=False, **kwargs):
        self.bb_pretrained = bb_pretrained
        super().__init__(**kwargs)


class BiRefNet(PreTrainedModel):
    config_class = BiRefNetConfig

    def __init__(self, bb_pretrained=True, config=BiRefNetConfig()):
        super(BiRefNet, self).__init__(config)
        print(1)
        bb_pretrained = config.bb_pretrained
        self.config = Config()
        self.epoch = 1
        self.bb = build_backbone(self.config.bb, pretrained=bb_pretrained)

        channels = self.config.lateral_channels_in_collection

        if self.config.auxiliary_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_head = nn.Sequential(
                nn.Linear(channels[0], len(class_labels_TR_sorted))
            )

        if self.config.squeeze_block:
            self.squeeze_module = nn.Sequential(
                *[
                    eval(self.config.squeeze_block.split("_x")[0])(
                        channels[0] + sum(self.config.cxt), channels[0]
                    )
                    for _ in range(eval(self.config.squeeze_block.split("_x")[1]))
                ]
            )

        self.decoder = Decoder(channels)

        if self.config.ender:
            self.dec_end = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),
                nn.Conv2d(16, 1, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

        # refine patch-level segmentation
        if self.config.refine:
            if self.config.refine == "itself":
                self.stem_layer = StemLayer(
                    in_channels=3 + 1,
                    inter_channels=48,
                    out_channels=3,
                    norm_layer="BN" if self.config.batch_size > 1 else "LN",
                )
            else:
                self.refiner = eval(
                    "{}({})".format(self.config.refine, "in_channels=3+1")
                )

        if self.config.freeze_bb:
            # Freeze the backbone...
            print(self.named_parameters())
            for key, value in self.named_parameters():
                if "bb." in key and "refiner." not in key:
                    value.requires_grad = False

    def forward_enc(self, x):
        if self.config.bb in ["vgg16", "vgg16bn", "resnet50"]:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)
            if self.config.mul_scl_ipt == "cat":
                B, C, H, W = x.shape
                x1_, x2_, x3_, x4_ = self.bb(
                    F.interpolate(
                        x, size=(H // 2, W // 2), mode="bilinear", align_corners=True
                    )
                )
                x1 = torch.cat(
                    [
                        x1,
                        F.interpolate(
                            x1_, size=x1.shape[2:], mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
                x2 = torch.cat(
                    [
                        x2,
                        F.interpolate(
                            x2_, size=x2.shape[2:], mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
                x3 = torch.cat(
                    [
                        x3,
                        F.interpolate(
                            x3_, size=x3.shape[2:], mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
                x4 = torch.cat(
                    [
                        x4,
                        F.interpolate(
                            x4_, size=x4.shape[2:], mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
            elif self.config.mul_scl_ipt == "add":
                B, C, H, W = x.shape
                x1_, x2_, x3_, x4_ = self.bb(
                    F.interpolate(
                        x, size=(H // 2, W // 2), mode="bilinear", align_corners=True
                    )
                )
                x1 = x1 + F.interpolate(
                    x1_, size=x1.shape[2:], mode="bilinear", align_corners=True
                )
                x2 = x2 + F.interpolate(
                    x2_, size=x2.shape[2:], mode="bilinear", align_corners=True
                )
                x3 = x3 + F.interpolate(
                    x3_, size=x3.shape[2:], mode="bilinear", align_corners=True
                )
                x4 = x4 + F.interpolate(
                    x4_, size=x4.shape[2:], mode="bilinear", align_corners=True
                )
        class_preds = (
            self.cls_head(self.avgpool(x4).view(x4.shape[0], -1))
            if self.training and self.config.auxiliary_classification
            else None
        )
        if self.config.cxt:
            x4 = torch.cat(
                (
                    *[
                        F.interpolate(
                            x1, size=x4.shape[2:], mode="bilinear", align_corners=True
                        ),
                        F.interpolate(
                            x2, size=x4.shape[2:], mode="bilinear", align_corners=True
                        ),
                        F.interpolate(
                            x3, size=x4.shape[2:], mode="bilinear", align_corners=True
                        ),
                    ][-len(self.config.cxt) :],
                    x4,
                ),
                dim=1,
            )
        return (x1, x2, x3, x4), class_preds

    def forward_ori(self, x):
        ########## Encoder ##########
        (x1, x2, x3, x4), class_preds = self.forward_enc(x)
        if self.config.squeeze_block:
            x4 = self.squeeze_module(x4)
        ########## Decoder ##########
        features = [x, x1, x2, x3, x4]
        # if self.training and self.config.out_ref:
        #     features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        scaled_preds = self.decoder(features)
        return scaled_preds, class_preds

    def forward(self, x):
        scaled_preds, class_preds = self.forward_ori(x)
        class_preds_lst = [class_preds]
        return [scaled_preds, class_preds_lst] if self.training else scaled_preds


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        DecoderBlock = eval(self.config.dec_blk)
        LateralBlock = eval(self.config.lat_blk)

        if self.config.dec_ipt:
            self.split = self.config.dec_ipt_split
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            self.ipt_blk5 = DBlock(
                2**10 * 3 if self.split else 3,
                [N_dec_ipt, channels[0] // 8][ipt_cha_opt],
                inter_channels=ic,
            )
            self.ipt_blk4 = DBlock(
                2**8 * 3 if self.split else 3,
                [N_dec_ipt, channels[0] // 8][ipt_cha_opt],
                inter_channels=ic,
            )
            self.ipt_blk3 = DBlock(
                2**6 * 3 if self.split else 3,
                [N_dec_ipt, channels[1] // 8][ipt_cha_opt],
                inter_channels=ic,
            )
            self.ipt_blk2 = DBlock(
                2**4 * 3 if self.split else 3,
                [N_dec_ipt, channels[2] // 8][ipt_cha_opt],
                inter_channels=ic,
            )
            self.ipt_blk1 = DBlock(
                2**0 * 3 if self.split else 3,
                [N_dec_ipt, channels[3] // 8][ipt_cha_opt],
                inter_channels=ic,
            )
        else:
            self.split = None

        self.decoder_block4 = DecoderBlock(
            channels[0]
            + (
                [N_dec_ipt, channels[0] // 8][ipt_cha_opt] if self.config.dec_ipt else 0
            ),
            channels[1],
        )
        self.decoder_block3 = DecoderBlock(
            channels[1]
            + (
                [N_dec_ipt, channels[0] // 8][ipt_cha_opt] if self.config.dec_ipt else 0
            ),
            channels[2],
        )
        self.decoder_block2 = DecoderBlock(
            channels[2]
            + (
                [N_dec_ipt, channels[1] // 8][ipt_cha_opt] if self.config.dec_ipt else 0
            ),
            channels[3],
        )
        self.decoder_block1 = DecoderBlock(
            channels[3]
            + (
                [N_dec_ipt, channels[2] // 8][ipt_cha_opt] if self.config.dec_ipt else 0
            ),
            channels[3] // 2,
        )
        self.conv_out1 = nn.Sequential(
            nn.Conv2d(
                channels[3] // 2
                + (
                    [N_dec_ipt, channels[3] // 8][ipt_cha_opt]
                    if self.config.dec_ipt
                    else 0
                ),
                1,
                1,
                1,
                0,
            )
        )

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

            if self.config.out_ref:
                _N = 16
                self.gdt_convs_4 = nn.Sequential(
                    nn.Conv2d(channels[1], _N, 3, 1, 1),
                    nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(),
                    nn.ReLU(inplace=True),
                )
                self.gdt_convs_3 = nn.Sequential(
                    nn.Conv2d(channels[2], _N, 3, 1, 1),
                    nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(),
                    nn.ReLU(inplace=True),
                )
                self.gdt_convs_2 = nn.Sequential(
                    nn.Conv2d(channels[3], _N, 3, 1, 1),
                    nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(),
                    nn.ReLU(inplace=True),
                )

                self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

                self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def get_patches_batch(self, x, p):
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [
                    p.unsqueeze(0)
                    for p in torch.split(
                        column_x, split_size_or_sections=_size_h, dim=-2
                    )
                ]
            patch_sample = torch.cat(patches_x, dim=1)
            patches_batch.append(patch_sample)
        return torch.cat(patches_batch, dim=0)

    def forward(self, features):
        if self.training and self.config.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features
        outs = []

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, x4) if self.split else x
            x4 = torch.cat(
                (
                    x4,
                    self.ipt_blk5(
                        F.interpolate(
                            patches_batch,
                            size=x4.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    ),
                ),
                1,
            )
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.config.ms_supervision else None
        if self.config.out_ref:
            p4_gdt = self.gdt_convs_4(p4)
            if self.training:
                # >> GT:
                m4_dia = m4
                gdt_label_main_4 = gdt_gt * F.interpolate(
                    m4_dia, size=gdt_gt.shape[2:], mode="bilinear", align_corners=True
                )
                outs_gdt_label.append(gdt_label_main_4)
                # >> Pred:
                gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
                outs_gdt_pred.append(gdt_pred_4)
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            # >> Finally:
            p4 = p4 * gdt_attn_4
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p3) if self.split else x
            _p3 = torch.cat(
                (
                    _p3,
                    self.ipt_blk4(
                        F.interpolate(
                            patches_batch,
                            size=x3.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    ),
                ),
                1,
            )
        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.config.ms_supervision else None
        if self.config.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                m3_dia = m3
                gdt_label_main_3 = gdt_gt * F.interpolate(
                    m3_dia, size=gdt_gt.shape[2:], mode="bilinear", align_corners=True
                )
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p2) if self.split else x
            _p2 = torch.cat(
                (
                    _p2,
                    self.ipt_blk3(
                        F.interpolate(
                            patches_batch,
                            size=x2.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    ),
                ),
                1,
            )
        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.config.ms_supervision else None
        if self.config.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                # >> GT:
                m2_dia = m2
                gdt_label_main_2 = gdt_gt * F.interpolate(
                    m2_dia, size=gdt_gt.shape[2:], mode="bilinear", align_corners=True
                )
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat(
                (
                    _p1,
                    self.ipt_blk2(
                        F.interpolate(
                            patches_batch,
                            size=x1.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    ),
                ),
                1,
            )
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode="bilinear", align_corners=True)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat(
                (
                    _p1,
                    self.ipt_blk1(
                        F.interpolate(
                            patches_batch,
                            size=x.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    ),
                ),
                1,
            )
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        return (
            outs
            if not (self.config.out_ref and self.training)
            else ([outs_gdt_pred, outs_gdt_label], outs)
        )


class SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels=64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


def create_briarmbg2_session():
    birefnet = BiRefNet.from_pretrained("briaai/RMBG-2.0")
    return birefnet


def briarmbg2_process(device, bgr_np_image, session, only_mask=False):
    from torchvision import transforms
    from PIL import Image

    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.fromarray(bgr_np_image)
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0)
    input_images = input_images.to(device)

    # Prediction
    preds = session(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    if only_mask:
        return np.array(mask)

    image.putalpha(mask)
    return np.array(image)
