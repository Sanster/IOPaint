import os
from pathlib import Path

import einops
import torch
import torch as th
import torch.nn as nn
import copy
from easydict import EasyDict as edict

from iopaint.model.anytext.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from iopaint.model.anytext.ldm.modules.attention import SpatialTransformer
from iopaint.model.anytext.ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from iopaint.model.anytext.ldm.models.diffusion.ddpm import LatentDiffusion
from iopaint.model.anytext.ldm.util import log_txt_as_img, exists, instantiate_from_config
from iopaint.model.anytext.ldm.models.diffusion.ddim import DDIMSampler
from iopaint.model.anytext.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from .recognizer import TextRecognizer, create_predictor

CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            if self.use_fp16:
                t_emb = t_emb.half()
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            glyph_channels,
            position_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.glyph_block = TimestepEmbedSequential(
            conv_nd(dims, glyph_channels, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
        )

        self.position_block = TimestepEmbedSequential(
            conv_nd(dims, position_channels, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
        )

        self.fuse_block = zero_module(conv_nd(dims, 256+64+4, model_channels, 3, padding=1))

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, text_info, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if self.use_fp16:
            t_emb = t_emb.half()
        emb = self.time_embed(t_emb)

        # guided_hint from text_info
        B, C, H, W = x.shape
        glyphs = torch.cat(text_info['glyphs'], dim=1).sum(dim=1, keepdim=True)
        positions = torch.cat(text_info['positions'], dim=1).sum(dim=1, keepdim=True)
        enc_glyph = self.glyph_block(glyphs, emb, context)
        enc_pos = self.position_block(positions, emb, context)
        guided_hint = self.fuse_block(torch.cat([enc_glyph, enc_pos, text_info['masked_x']], dim=1))

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, glyph_key, position_key, only_mid_control, loss_alpha=0, loss_beta=0, with_step_weight=False, use_vae_upsample=False, latin_weight=1.0, embedding_manager_config=None, *args, **kwargs):
        self.use_fp16 = kwargs.pop('use_fp16', False)
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.glyph_key = glyph_key
        self.position_key = position_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.with_step_weight = with_step_weight
        self.use_vae_upsample = use_vae_upsample
        self.latin_weight = latin_weight

        if embedding_manager_config is not None and embedding_manager_config.params.valid:
            self.embedding_manager = self.instantiate_embedding_manager(embedding_manager_config, self.cond_stage_model)
            for param in self.embedding_manager.embedding_parameters():
                param.requires_grad = True
        else:
            self.embedding_manager = None
        if self.loss_alpha > 0 or self.loss_beta > 0 or self.embedding_manager:
            if embedding_manager_config.params.emb_type == 'ocr':
                self.text_predictor = create_predictor().eval()
                args = edict()
                args.rec_image_shape = "3, 48, 320"
                args.rec_batch_num = 6
                args.rec_char_dict_path = str(CURRENT_DIR.parent / "ocr_recog" / "ppocr_keys_v1.txt")
                args.use_fp16 = self.use_fp16
                self.cn_recognizer = TextRecognizer(args, self.text_predictor)
                for param in self.text_predictor.parameters():
                    param.requires_grad = False
                if self.embedding_manager:
                    self.embedding_manager.recog = self.cn_recognizer

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        if self.embedding_manager is None:  # fill in full caption
            self.fill_caption(batch)
        x, c, mx = super().get_input(batch, self.first_stage_key, mask_k='masked_img', *args, **kwargs)
        control = batch[self.control_key]  # for log_images and loss_alpha, not real control
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        inv_mask = batch['inv_mask']
        if bs is not None:
            inv_mask = inv_mask[:bs]
        inv_mask = inv_mask.to(self.device)
        inv_mask = einops.rearrange(inv_mask, 'b h w c -> b c h w')
        inv_mask = inv_mask.to(memory_format=torch.contiguous_format).float()

        glyphs = batch[self.glyph_key]
        gly_line = batch['gly_line']
        positions = batch[self.position_key]
        n_lines = batch['n_lines']
        language = batch['language']
        texts = batch['texts']
        assert len(glyphs) == len(positions)
        for i in range(len(glyphs)):
            if bs is not None:
                glyphs[i] = glyphs[i][:bs]
                gly_line[i] = gly_line[i][:bs]
                positions[i] = positions[i][:bs]
                n_lines = n_lines[:bs]
            glyphs[i] = glyphs[i].to(self.device)
            gly_line[i] = gly_line[i].to(self.device)
            positions[i] = positions[i].to(self.device)
            glyphs[i] = einops.rearrange(glyphs[i], 'b h w c -> b c h w')
            gly_line[i] = einops.rearrange(gly_line[i], 'b h w c -> b c h w')
            positions[i] = einops.rearrange(positions[i], 'b h w c -> b c h w')
            glyphs[i] = glyphs[i].to(memory_format=torch.contiguous_format).float()
            gly_line[i] = gly_line[i].to(memory_format=torch.contiguous_format).float()
            positions[i] = positions[i].to(memory_format=torch.contiguous_format).float()
        info = {}
        info['glyphs'] = glyphs
        info['positions'] = positions
        info['n_lines'] = n_lines
        info['language'] = language
        info['texts'] = texts
        info['img'] = batch['img']  # nhwc, (-1,1)
        info['masked_x'] = mx
        info['gly_line'] = gly_line
        info['inv_mask'] = inv_mask
        return x, dict(c_crossattn=[c], c_concat=[control], text_info=info)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        _cond = torch.cat(cond['c_crossattn'], 1)
        _hint = torch.cat(cond['c_concat'], 1)
        if self.use_fp16:
            x_noisy = x_noisy.half()
        control = self.control_model(x=x_noisy, timesteps=t, context=_cond, hint=_hint, text_info=cond['text_info'])
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = diffusion_model(x=x_noisy, timesteps=t, context=_cond, control=control, only_mid_control=self.only_mid_control)

        return eps

    def instantiate_embedding_manager(self, config, embedder):
        model = instantiate_from_config(config, embedder=embedder)
        return model

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning(dict(c_crossattn=[[""] * N], text_info=None))

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                if self.embedding_manager is not None and c['text_info'] is not None:
                    self.embedding_manager.encode_text(c['text_info'])
                if isinstance(c, dict):
                    cond_txt = c['c_crossattn'][0]
                else:
                    cond_txt = c
                if self.embedding_manager is not None:
                    cond_txt = self.cond_stage_model.encode(cond_txt, embedding_manager=self.embedding_manager)
                else:
                    cond_txt = self.cond_stage_model.encode(cond_txt)
                if isinstance(c, dict):
                    c['c_crossattn'][0] = cond_txt
                else:
                    c = cond_txt
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def fill_caption(self, batch, place_holder='*'):
        bs = len(batch['n_lines'])
        cond_list = copy.deepcopy(batch[self.cond_stage_key])
        for i in range(bs):
            n_lines = batch['n_lines'][i]
            if n_lines == 0:
                continue
            cur_cap = cond_list[i]
            for j in range(n_lines):
                r_txt = batch['texts'][j][i]
                cur_cap = cur_cap.replace(place_holder, f'"{r_txt}"', 1)
            cond_list[i] = cur_cap
        batch[self.cond_stage_key] = cond_list

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        if self.cond_stage_trainable:
            with torch.no_grad():
                c = self.get_learned_conditioning(c)
        c_crossattn = c["c_crossattn"][0][:N]
        c_cat = c["c_concat"][0][:N]
        text_info = c["text_info"]
        text_info['glyphs'] = [i[:N] for i in text_info['glyphs']]
        text_info['gly_line'] = [i[:N] for i in text_info['gly_line']]
        text_info['positions'] = [i[:N] for i in text_info['positions']]
        text_info['n_lines'] = text_info['n_lines'][:N]
        text_info['masked_x'] = text_info['masked_x'][:N]
        text_info['img'] = text_info['img'][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["masked_image"] = self.decode_first_stage(text_info['masked_x'])
        log["control"] = c_cat * 2.0 - 1.0
        log["img"] = text_info['img'].permute(0, 3, 1, 2)  # log source image if needed
        # get glyph
        glyph_bs = torch.stack(text_info['glyphs'])
        glyph_bs = torch.sum(glyph_bs, dim=0) * 2.0 - 1.0
        log["glyph"] = torch.nn.functional.interpolate(glyph_bs, size=(512, 512), mode='bilinear', align_corners=True,)
        # fill caption
        if not self.embedding_manager:
            self.fill_caption(batch)
        captions = batch[self.cond_stage_key]
        log["conditioning"] = log_txt_as_img((512, 512), captions, size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "text_info": text_info},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross['c_crossattn'][0]], "text_info": text_info}
            samples_cfg, tmps = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c_crossattn], "text_info": text_info},
                                                batch_size=N, ddim=use_ddim,
                                                ddim_steps=ddim_steps, eta=ddim_eta,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc_full,
                                                )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            pred_x0 = False  # wether log pred_x0
            if pred_x0:
                for idx in range(len(tmps['pred_x0'])):
                    pred_x0 = self.decode_first_stage(tmps['pred_x0'][idx])
                    log[f"pred_x0_{tmps['index'][idx]}"] = pred_x0

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, log_every_t=5, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if self.embedding_manager:
            params += list(self.embedding_manager.embedding_parameters())
        if not self.sd_locked:
            # params += list(self.model.diffusion_model.input_blocks.parameters())
            # params += list(self.model.diffusion_model.middle_block.parameters())
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.unlockKV:
            nCount = 0
            for name, param in self.model.diffusion_model.named_parameters():
                if 'attn2.to_k' in name or 'attn2.to_v' in name:
                    params += [param]
                    nCount += 1
            print(f'Cross attention is unlocked, and {nCount} Wk or Wv are added to potimizers!!!')

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
