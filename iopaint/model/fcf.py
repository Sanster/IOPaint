import os
import random

import cv2
import torch
import numpy as np
import torch.fft as fft

from iopaint.schema import InpaintRequest

from iopaint.helper import (
    load_model,
    get_cache_path_by_url,
    norm_img,
    boxes_from_mask,
    resize_max_size,
    download_model,
)
from .base import InpaintModel
from torch import conv2d, nn
import torch.nn.functional as F

from .utils import (
    setup_filter,
    _parse_scaling,
    _parse_padding,
    Conv2dLayer,
    FullyConnectedLayer,
    MinibatchStdLayer,
    activation_funcs,
    conv2d_resample,
    bias_act,
    upsample2d,
    normalize_2nd_moment,
    downsample2d,
)


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl="cuda"):
    assert isinstance(x, torch.Tensor)
    return _upfirdn2d_ref(
        x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain
    )


def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops."""
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(
        x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)]
    )
    x = x[
        :,
        :,
        max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
        max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
    ]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


class EncoderEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
        z_dim,  # Output Latent (Z) dimensionality.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == "skip":
            self.fromrgb = Conv2dLayer(
                self.img_channels, in_channels, kernel_size=1, activation=activation
            )
        self.mbstd = (
            MinibatchStdLayer(
                group_size=mbstd_group_size, num_channels=mbstd_num_channels
            )
            if mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=activation,
            conv_clamp=conv_clamp,
        )
        self.fc = FullyConnectedLayer(
            in_channels * (resolution**2), z_dim, activation=activation
        )
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, cmap, force_fp32=False):
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        const_e = self.conv(x)
        x = self.fc(const_e.flatten(1))
        x = self.dropout(x)

        # Conditioning.
        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x, const_e


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        tmp_channels,  # Number of intermediate channels.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        first_layer_idx,  # Index of the first layer.
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        freeze_layers=0,  # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels + 1
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0:
            self.fromrgb = Conv2dLayer(
                self.img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == "resnet":
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, force_fp32=False):
        # dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        dtype = torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )

        # Input.
        if x is not None:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0:
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = (
                downsample2d(img, self.resample_filter)
                if self.architecture == "skip"
                else None
            )

        # Main layers.
        if self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            feat = x.clone()
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            feat = x.clone()
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img, feat


class EncoderNetwork(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        z_dim,  # Input latent (Z) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        architecture="orig",  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=16384,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={},  # Arguments for EncoderEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(
            img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp
        )
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution
            use_fp16 = False
            block = EncoderBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs,
            )
            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0,
                c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None,
                **mapping_kwargs,
            )
        self.b4 = EncoderEpilogue(
            channels_dict[4],
            cmap_dim=cmap_dim,
            z_dim=z_dim * 2,
            resolution=4,
            **epilogue_kwargs,
            **common_kwargs,
        )

    def forward(self, img, c, **block_kwargs):
        x = None
        feats = {}
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            x, img, feat = block(x, img, **block_kwargs)
            feats[res] = feat

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x, const_e = self.b4(x, cmap)
        feats[4] = const_e

        B, _ = x.shape
        z = torch.zeros(
            (B, self.z_dim), requires_grad=False, dtype=x.dtype, device=x.device
        )  ## Noise for Co-Modulation
        return x, z, feats


def fma(a, b, c):  # => a * b + c
    return _FusedMultiplyAdd.apply(a, b, c)


class _FusedMultiplyAdd(torch.autograd.Function):  # a * b + c
    @staticmethod
    def forward(ctx, a, b, c):  # pylint: disable=arguments-differ
        out = torch.addcmul(c, a, b)
        ctx.save_for_backward(a, b)
        ctx.c_shape = c.shape
        return out

    @staticmethod
    def backward(ctx, dout):  # pylint: disable=arguments-differ
        a, b = ctx.saved_tensors
        c_shape = ctx.c_shape
        da = None
        db = None
        dc = None

        if ctx.needs_input_grad[0]:
            da = _unbroadcast(dout * b, a.shape)

        if ctx.needs_input_grad[1]:
            db = _unbroadcast(dout * a, b.shape)

        if ctx.needs_input_grad[2]:
            dc = _unbroadcast(dout, c_shape)

        return da, db, dc


def _unbroadcast(x, shape):
    extra_dims = x.ndim - len(shape)
    assert extra_dims >= 0
    dim = [
        i
        for i in range(x.ndim)
        if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)
    ]
    if len(dim):
        x = x.sum(dim=dim, keepdim=True)
    if extra_dims:
        x = x.reshape(-1, *x.shape[extra_dims + 1 :])
    assert x.shape == shape
    return x


def modulated_conv2d(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,  # Optional noise tensor to add to the output activations.
    up=1,  # Integer upsampling factor.
    down=1,  # Integer downsampling factor.
    padding=0,  # Padding with respect to the upsampled image.
    resample_filter=None,
    # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,  # Apply weight demodulation?
    flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1
            / np.sqrt(in_channels * kh * kw)
            / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True)
        )  # max_Ikk
        styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]
    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight,
        )
        if demodulate and noise is not None:
            x = fma(
                x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype)
            )
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
        flip_weight=flip_weight,
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        use_noise=True,  # Enable noise input?
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last=False,  # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
                memory_format=memory_format
            )
        )
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode="none", fused_modconv=True, gain=1):
        assert noise_mode in ["random", "const", "none"]
        in_resolution = self.resolution // self.up
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == "random":
            noise = (
                torch.randn(
                    [x.shape[0], 1, self.resolution, self.resolution], device=x.device
                )
                * self.noise_strength
            )
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        flip_weight = self.up == 1  # slightly faster
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = F.leaky_relu(x, negative_slope=0.2, inplace=False)
        if act_gain != 1:
            x = x * act_gain
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x


class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=1,
        conv_clamp=None,
        channels_last=False,
    ):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
                memory_format=memory_format
            )
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            demodulate=False,
            fused_modconv=fused_modconv,
        )
        x = bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


class SynthesisForeword(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Output Latent (Z) dimensionality.
        resolution,  # Resolution of this block.
        in_channels,
        img_channels,  # Number of input color channels.
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        self.fc = FullyConnectedLayer(
            self.z_dim, (self.z_dim // 2) * 4 * 4, activation=activation
        )
        self.conv = SynthesisLayer(
            self.in_channels, self.in_channels, w_dim=(z_dim // 2) * 3, resolution=4
        )

        if architecture == "skip":
            self.torgb = ToRGBLayer(
                self.in_channels,
                self.img_channels,
                kernel_size=1,
                w_dim=(z_dim // 2) * 3,
            )

    def forward(self, x, ws, feats, img, force_fp32=False):
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        x_global = x.clone()
        # ToRGB.
        x = self.fc(x)
        x = x.view(-1, self.z_dim // 2, 4, 4)
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        x_skip = feats[4].clone()
        x = x + x_skip

        mod_vector = []
        mod_vector.append(ws[:, 0])
        mod_vector.append(x_global.clone())
        mod_vector = torch.cat(mod_vector, dim=1)

        x = self.conv(x, mod_vector)

        mod_vector = []
        mod_vector.append(ws[:, 2 * 2 - 3])
        mod_vector.append(x_global.clone())
        mod_vector = torch.cat(mod_vector, dim=1)

        if self.architecture == "skip":
            img = self.torgb(x, mod_vector)
            img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)

        assert x.dtype == dtype
        return x, img


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        spatial_scale_factor=None,
        spatial_scale_mode="bilinear",
        spectral_pos_encoding=False,
        use_se=False,
        se_kwargs=None,
        ffc3d=False,
        fft_norm="ortho",
    ):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.relu = torch.nn.ReLU(inplace=False)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(
                x,
                scale_factor=self.spatial_scale_factor,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = (
                torch.linspace(0, 1, height)[None, None, :, None]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            coords_hor = (
                torch.linspace(0, 1, width)[None, None, None, :]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        ffted = (
            ffted.view(
                (
                    batch,
                    -1,
                    2,
                )
                + ffted.size()[2:]
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        if self.spatial_scale_factor is not None:
            output = F.interpolate(
                output,
                size=orig_size,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        return output


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        enable_lfu=True,
        **fu_kwargs,
    ):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False
            ),
            # nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(
                torch.split(x[:, : c // 4], split_s, dim=-2), dim=1
            ).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        enable_lfu=True,
        padding_type="reflect",
        gated=False,
        **spectral_kwargs,
    ):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(
            in_cl,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(
            in_cl,
            out_cg,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(
            in_cg,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg,
            out_cg,
            stride,
            1 if groups == 1 else groups // 2,
            enable_lfu,
            **spectral_kwargs,
        )

        self.gated = gated
        module = (
            nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        )
        self.gate = module(in_channels, 2, 1)

    def forward(self, x, fname=None):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        spec_x = self.convg2g(x_g)

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + spec_x

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.SyncBatchNorm,
        activation_layer=nn.Identity,
        padding_type="reflect",
        enable_lfu=True,
        **kwargs,
    ):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            padding_type=padding_type,
            **kwargs,
        )
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        # self.bn_l = lnorm(out_channels - global_channels)
        # self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x, fname=None):
        x_l, x_g = self.ffc(
            x,
            fname=fname,
        )
        x_l = self.act_l(x_l)
        x_g = self.act_g(x_g)
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        activation_layer=nn.ReLU,
        dilation=1,
        spatial_transform_kwargs=None,
        inline=False,
        ratio_gin=0.75,
        ratio_gout=0.75,
    ):
        super().__init__()
        self.conv1 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
        )
        self.conv2 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
        )
        self.inline = inline

    def forward(self, x, fname=None):
        if self.inline:
            x_l, x_g = (
                x[:, : -self.conv1.ffc.global_in_num],
                x[:, -self.conv1.ffc.global_in_num :],
            )
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g), fname=fname)
        x_l, x_g = self.conv2((x_l, x_g), fname=fname)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCBlock(torch.nn.Module):
    def __init__(
        self,
        dim,  # Number of output/input channels.
        kernel_size,  # Width and height of the convolution kernel.
        padding,
        ratio_gin=0.75,
        ratio_gout=0.75,
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
    ):
        super().__init__()
        if activation == "linear":
            self.activation = nn.Identity
        else:
            self.activation = nn.ReLU
        self.padding = padding
        self.kernel_size = kernel_size
        self.ffc_block = FFCResnetBlock(
            dim=dim,
            padding_type="reflect",
            norm_layer=nn.SyncBatchNorm,
            activation_layer=self.activation,
            dilation=1,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
        )

        self.concat_layer = ConcatTupleLayer()

    def forward(self, gen_ft, mask, fname=None):
        x = gen_ft.float()

        x_l, x_g = (
            x[:, : -self.ffc_block.conv1.ffc.global_in_num],
            x[:, -self.ffc_block.conv1.ffc.global_in_num :],
        )
        id_l, id_g = x_l, x_g

        x_l, x_g = self.ffc_block((x_l, x_g), fname=fname)
        x_l, x_g = id_l + x_l, id_g + x_g
        x = self.concat_layer((x_l, x_g))

        return x + gen_ft.float()


class FFCSkipLayer(torch.nn.Module):
    def __init__(
        self,
        dim,  # Number of input/output channels.
        kernel_size=3,  # Convolution kernel size.
        ratio_gin=0.75,
        ratio_gout=0.75,
    ):
        super().__init__()
        self.padding = kernel_size // 2

        self.ffc_act = FFCBlock(
            dim=dim,
            kernel_size=kernel_size,
            activation=nn.ReLU,
            padding=self.padding,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
        )

    def forward(self, gen_ft, mask, fname=None):
        x = self.ffc_act(gen_ft, mask, fname=fname)
        return x


class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this block.
        img_channels,  # Number of output color channels.
        is_last,  # Is this the last block?
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.res_ffc = {4: 0, 8: 0, 16: 0, 32: 1, 64: 1, 128: 1, 256: 1, 512: 1}

        if in_channels != 0 and resolution >= 8:
            self.ffc_skip = nn.ModuleList()
            for _ in range(self.res_ffc[resolution]):
                self.ffc_skip.append(FFCSkipLayer(dim=out_channels))

        if in_channels == 0:
            self.const = torch.nn.Parameter(
                torch.randn([out_channels, resolution, resolution])
            )

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim * 3,
                resolution=resolution,
                up=2,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim * 3,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == "skip":
            self.torgb = ToRGBLayer(
                out_channels,
                img_channels,
                w_dim=w_dim * 3,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )
            self.num_torgb += 1

        if in_channels != 0 and architecture == "resnet":
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(
        self,
        x,
        mask,
        feats,
        img,
        ws,
        fname=None,
        force_fp32=False,
        fused_modconv=None,
        **layer_kwargs,
    ):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        dtype = torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if fused_modconv is None:
            fused_modconv = (not self.training) and (
                dtype == torch.float32 or int(x.shape[0]) == 1
            )

        x = x.to(dtype=dtype, memory_format=memory_format)
        x_skip = (
            feats[self.resolution].clone().to(dtype=dtype, memory_format=memory_format)
        )

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, ws[1], fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(
                x, ws[0].clone(), fused_modconv=fused_modconv, **layer_kwargs
            )
            if len(self.ffc_skip) > 0:
                mask = F.interpolate(
                    mask,
                    size=x_skip.shape[2:],
                )
                z = x + x_skip
                for fres in self.ffc_skip:
                    z = fres(z, mask)
                x = x + z
            else:
                x = x + x_skip
            x = self.conv1(
                x,
                ws[1].clone(),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs,
            )
            x = y.add_(x)
        else:
            x = self.conv0(
                x, ws[0].clone(), fused_modconv=fused_modconv, **layer_kwargs
            )
            if len(self.ffc_skip) > 0:
                mask = F.interpolate(
                    mask,
                    size=x_skip.shape[2:],
                )
                z = x + x_skip
                for fres in self.ffc_skip:
                    z = fres(z, mask)
                x = x + z
            else:
                x = x + x_skip
            x = self.conv1(
                x, ws[1].clone(), fused_modconv=fused_modconv, **layer_kwargs
            )
        # ToRGB.
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, ws[2].clone(), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        x = x.to(dtype=dtype)
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        z_dim,  # Output Latent (Z) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        channel_base=16384,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(3, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) for res in self.block_resolutions
        }
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.foreword = SynthesisForeword(
            img_channels=img_channels,
            in_channels=min(channel_base // 4, channel_max),
            z_dim=z_dim * 2,
            resolution=4,
        )

        self.num_ws = self.img_resolution_log2 * 2 - 2
        for res in self.block_resolutions:
            if res // 2 in channels_dict.keys():
                in_channels = channels_dict[res // 2] if res > 4 else 0
            else:
                in_channels = min(channel_base // (res // 2), channel_max)
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            use_fp16 = False
            is_last = res == self.img_resolution
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            setattr(self, f"b{res}", block)

    def forward(self, x_global, mask, feats, ws, fname=None, **block_kwargs):
        img = None

        x, img = self.foreword(x_global, ws, feats, img)

        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            mod_vector0 = []
            mod_vector0.append(ws[:, int(np.log2(res)) * 2 - 5])
            mod_vector0.append(x_global.clone())
            mod_vector0 = torch.cat(mod_vector0, dim=1)

            mod_vector1 = []
            mod_vector1.append(ws[:, int(np.log2(res)) * 2 - 4])
            mod_vector1.append(x_global.clone())
            mod_vector1 = torch.cat(mod_vector1, dim=1)

            mod_vector_rgb = []
            mod_vector_rgb.append(ws[:, int(np.log2(res)) * 2 - 3])
            mod_vector_rgb.append(x_global.clone())
            mod_vector_rgb = torch.cat(mod_vector_rgb, dim=1)
            x, img = block(
                x,
                mask,
                feats,
                img,
                (mod_vector0, mod_vector1, mod_vector_rgb),
                fname=fname,
                **block_kwargs,
            )
        return img


class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False
    ):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )
        return x


class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        encoder_kwargs={},  # Arguments for EncoderNetwork.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.encoder = EncoderNetwork(
            c_dim=c_dim,
            z_dim=z_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **encoder_kwargs,
        )
        self.synthesis = SynthesisNetwork(
            z_dim=z_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs
        )

    def forward(
        self,
        img,
        c,
        fname=None,
        truncation_psi=1,
        truncation_cutoff=None,
        **synthesis_kwargs,
    ):
        mask = img[:, -1].unsqueeze(1)
        x_global, z, feats = self.encoder(img, c)
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )
        img = self.synthesis(x_global, mask, feats, ws, fname=fname, **synthesis_kwargs)
        return img


FCF_MODEL_URL = os.environ.get(
    "FCF_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_fcf/places_512_G.pth",
)
FCF_MODEL_MD5 = os.environ.get("FCF_MODEL_MD5", "3323152bc01bf1c56fd8aba74435a211")


class FcF(InpaintModel):
    name = "fcf"
    min_size = 512
    pad_mod = 512
    pad_to_square = True
    is_erase_model = True

    def init_model(self, device, **kwargs):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        kwargs = {
            "channel_base": 1 * 32768,
            "channel_max": 512,
            "num_fp16_res": 4,
            "conv_clamp": 256,
        }
        G = Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=512,
            img_channels=3,
            synthesis_kwargs=kwargs,
            encoder_kwargs=kwargs,
            mapping_kwargs={"num_layers": 2},
        )
        self.model = load_model(G, FCF_MODEL_URL, device, FCF_MODEL_MD5)
        self.label = torch.zeros([1, self.model.c_dim], device=device)

    @staticmethod
    def download():
        download_model(FCF_MODEL_URL, FCF_MODEL_MD5)

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(FCF_MODEL_URL))

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        if image.shape[0] == 512 and image.shape[1] == 512:
            return self._pad_forward(image, mask, config)

        boxes = boxes_from_mask(mask)
        crop_result = []
        config.hd_strategy_crop_margin = 128
        for box in boxes:
            crop_image, crop_mask, crop_box = self._crop_box(image, mask, box, config)
            origin_size = crop_image.shape[:2]
            resize_image = resize_max_size(crop_image, size_limit=512)
            resize_mask = resize_max_size(crop_mask, size_limit=512)
            inpaint_result = self._pad_forward(resize_image, resize_mask, config)

            # only paste masked area result
            inpaint_result = cv2.resize(
                inpaint_result,
                (origin_size[1], origin_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            original_pixel_indices = crop_mask < 127
            inpaint_result[original_pixel_indices] = crop_image[:, :, ::-1][
                original_pixel_indices
            ]

            crop_result.append((inpaint_result, crop_box))

        inpaint_result = image[:, :, ::-1].copy()
        for crop_image, crop_box in crop_result:
            x1, y1, x2, y2 = crop_box
            inpaint_result[y1:y2, x1:x2, :] = crop_image

        return inpaint_result

    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W] mask area == 255
        return: BGR IMAGE
        """

        image = norm_img(image)  # [0, 1]
        image = image * 2 - 1  # [0, 1] -> [-1, 1]
        mask = (mask > 120) * 255
        mask = norm_img(mask)

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        erased_img = image * (1 - mask)
        input_image = torch.cat([0.5 - mask, erased_img], dim=1)

        output = self.model(
            input_image, self.label, truncation_psi=0.1, noise_mode="none"
        )
        output = (
            (output.permute(0, 2, 3, 1) * 127.5 + 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        output = output[0].cpu().numpy()
        cur_res = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return cur_res
