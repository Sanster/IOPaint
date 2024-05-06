# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import apply_freeu

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def CrossAttnDownBlock2D_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    additional_residuals: Optional[torch.FloatTensor] = None,
    down_block_add_samples: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
    output_states = ()

    lora_scale = (
        cross_attention_kwargs.get("scale", 1.0)
        if cross_attention_kwargs is not None
        else 1.0
    )

    blocks = list(zip(self.resnets, self.attentions))

    for i, (resnet, attn) in enumerate(blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        # apply additional residuals to the output of the last pair of resnet and attention blocks
        if i == len(blocks) - 1 and additional_residuals is not None:
            hidden_states = hidden_states + additional_residuals

        if down_block_add_samples is not None:
            hidden_states = hidden_states + down_block_add_samples.pop(0)

        output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states, scale=lora_scale)

        if down_block_add_samples is not None:
            hidden_states = hidden_states + down_block_add_samples.pop(
                0
            )  # todo: add before or after

        output_states = output_states + (hidden_states,)

    return hidden_states, output_states


def DownBlock2D_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: Optional[torch.FloatTensor] = None,
    scale: float = 1.0,
    down_block_add_samples: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
    output_states = ()

    for resnet in self.resnets:
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    use_reentrant=False,
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
        else:
            hidden_states = resnet(hidden_states, temb, scale=scale)

        if down_block_add_samples is not None:
            hidden_states = hidden_states + down_block_add_samples.pop(0)

        output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states, scale=scale)

        if down_block_add_samples is not None:
            hidden_states = hidden_states + down_block_add_samples.pop(
                0
            )  # todo: add before or after

        output_states = output_states + (hidden_states,)

    return hidden_states, output_states


def CrossAttnUpBlock2D_forward(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
    temb: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    return_res_samples: Optional[bool] = False,
    up_block_add_samples: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    lora_scale = (
        cross_attention_kwargs.get("scale", 1.0)
        if cross_attention_kwargs is not None
        else 1.0
    )
    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )
    if return_res_samples:
        output_states = ()

    for resnet, attn in zip(self.resnets, self.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        if return_res_samples:
            output_states = output_states + (hidden_states,)
        if up_block_add_samples is not None:
            hidden_states = hidden_states + up_block_add_samples.pop(0)

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)
        if return_res_samples:
            output_states = output_states + (hidden_states,)
        if up_block_add_samples is not None:
            hidden_states = hidden_states + up_block_add_samples.pop(0)

    if return_res_samples:
        return hidden_states, output_states
    else:
        return hidden_states


def UpBlock2D_forward(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
    temb: Optional[torch.FloatTensor] = None,
    upsample_size: Optional[int] = None,
    scale: float = 1.0,
    return_res_samples: Optional[bool] = False,
    up_block_add_samples: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )
    if return_res_samples:
        output_states = ()

    for resnet in self.resnets:
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    use_reentrant=False,
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
        else:
            hidden_states = resnet(hidden_states, temb, scale=scale)

        if return_res_samples:
            output_states = output_states + (hidden_states,)
        if up_block_add_samples is not None:
            hidden_states = hidden_states + up_block_add_samples.pop(
                0
            )  # todo: add before or after

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

        if return_res_samples:
            output_states = output_states + (hidden_states,)
        if up_block_add_samples is not None:
            hidden_states = hidden_states + up_block_add_samples.pop(
                0
            )  # todo: add before or after

    if return_res_samples:
        return hidden_states, output_states
    else:
        return hidden_states
