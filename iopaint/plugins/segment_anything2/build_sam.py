# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from pathlib import Path

from .modeling.backbones.hieradet import Hiera
from .modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from .modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from .modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from .modeling.position_encoding import PositionEmbeddingSine
from .modeling.sam.transformer import RoPEAttention
from .modeling.sam2_base import SAM2Base

CURRENT_DIR = Path(__file__).parent
CONFIG_DIR = CURRENT_DIR / "sam2_configs"

common_kwargs = dict(
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=False,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    compile_image_encoder=False,
)


def build_memory_attention():
    return MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            self_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[32, 32],
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
            ),
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            cross_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[32, 32],
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64,
            ),
        ),
        num_layers=4,
    )


def build_memory_encoder():
    return MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64, normalize=True, scale=None, temperature=10000
        ),
        mask_downsampler=MaskDownSampler(
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        fuser=Fuser(
            layer=CXBlock(
                dim=256,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True,
            ),
            num_layers=2,
        ),
    )


def build_sam2_tiny():
    return SAM2Base(
        **common_kwargs,
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=Hiera(
                embed_dim=96,
                num_heads=1,
                stages=(1, 2, 7, 2),
                global_att_blocks=(5, 7, 9),
                window_pos_embed_bkg_spatial_size=(7, 7),
                window_spec=(8, 4, 14, 7),
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=256,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                d_model=256,
                backbone_channel_list=[768, 384, 192, 96],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
        ),
        memory_attention=build_memory_attention(),
        memory_encoder=build_memory_encoder(),
    )


def build_sam2_small():
    return SAM2Base(
        **common_kwargs,
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=Hiera(
                embed_dim=96,
                num_heads=1,
                stages=(1, 2, 11, 2),
                global_att_blocks=(7, 10, 13),
                window_pos_embed_bkg_spatial_size=(7, 7),
                window_spec=(8, 4, 14, 7),
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=256,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                d_model=256,
                backbone_channel_list=[768, 384, 192, 96],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
        ),
        memory_attention=build_memory_attention(),
        memory_encoder=build_memory_encoder(),
    )


def build_sam2_base():
    return SAM2Base(
        **common_kwargs,
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=Hiera(
                embed_dim=112,
                num_heads=2,
                stages=(2, 3, 16, 3),
                global_att_blocks=(12, 16, 20),
                window_pos_embed_bkg_spatial_size=(14, 14),
                window_spec=(8, 4, 14, 7),
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=256,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                d_model=256,
                backbone_channel_list=[896, 448, 224, 112],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
        ),
        memory_attention=build_memory_attention(),
        memory_encoder=build_memory_encoder(),
    )


def build_sam2_large():
    return SAM2Base(
        **common_kwargs,
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=Hiera(
                embed_dim=144,
                num_heads=2,
                stages=(2, 6, 36, 4),
                global_att_blocks=(23, 33, 43),
                window_pos_embed_bkg_spatial_size=(7, 7),
                window_spec=(8, 4, 16, 8),
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=256,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                d_model=256,
                backbone_channel_list=[1152, 576, 288, 144],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
        ),
        memory_attention=build_memory_attention(),
        memory_encoder=build_memory_encoder(),
    )


sam2_model_registry = {
    "sam2_tiny": build_sam2_tiny,
    "sam2_small": build_sam2_small,
    "sam2_base": build_sam2_base,
    "sam2_large": build_sam2_large,
}


def build_sam2(
    name,
    ckpt_path=None,
    device="cuda",
    mode="eval",
):
    model = sam2_model_registry[name]()
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
