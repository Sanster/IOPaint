import os

import numpy as np
import torch
from loguru import logger

from lama_cleaner.model.base import InpaintModel
from lama_cleaner.schema import Config

torch.manual_seed(42)
import torch.nn as nn
from tqdm import tqdm
from lama_cleaner.helper import download_model, norm_img, get_cache_path_by_url
from lama_cleaner.model.utils import make_beta_schedule, make_ddim_timesteps, make_ddim_sampling_parameters, noise_like, \
    timestep_embedding

LDM_ENCODE_MODEL_URL = os.environ.get(
    "LDM_ENCODE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_encode.pt",
)

LDM_DECODE_MODEL_URL = os.environ.get(
    "LDM_DECODE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_decode.pt",
)

LDM_DIFFUSION_MODEL_URL = os.environ.get(
    "LDM_DIFFUSION_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_ldm/diffusion.pt",
)


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 device,
                 timesteps=1000,
                 beta_schedule="linear",
                 linear_start=0.0015,
                 linear_end=0.0205,
                 cosine_s=0.008,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 use_positional_encodings=False):
        super().__init__()
        self.device = device
        self.parameterization = parameterization
        self.use_positional_encodings = use_positional_encodings

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(self.device, beta_schedule, timesteps, linear_start=linear_start,
                                   linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()


class LatentDiffusion(DDPM):
    def __init__(self,
                 diffusion_model,
                 device,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = 1
        self.scale_by_std = scale_by_std
        super().__init__(device, *args, **kwargs)
        self.diffusion_model = diffusion_model
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 2
        self.scale_factor = scale_factor

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def apply_model(self, x_noisy, t, cond):
        # x_recon = self.model(x_noisy, t, cond['c_concat'][0])  # cond['c_concat'][0].shape 1,4,128,128
        t_emb = timestep_embedding(x_noisy.device, t, 256, repeat_only=False)
        x_recon = self.diffusion_model(x_noisy, t_emb, cond)
        return x_recon


class DDIMSampler(object):
    def __init__(self, model, schedule="linear"):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  # array([1])
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod  # torch.Size([1000])
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, steps, conditioning, batch_size, shape):
        self.make_schedule(ddim_num_steps=steps, ddim_eta=0, verbose=False)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # samples: 1,3,128,128
        return self.ddim_sampling(conditioning,
                                  size,
                                  quantize_denoised=False,
                                  ddim_use_original_steps=False,
                                  noise_dropout=0,
                                  temperature=1.,
                                  )

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      ddim_use_original_steps=False,
                      quantize_denoised=False,
                      temperature=1., noise_dropout=0.):
        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)  # 用了
        timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps  # 用了

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        logger.info(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout)
            img, _ = outs

        return img

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0.):
        b, *_, device = *x.shape, x.device
        e_t = self.model.apply_model(x, t, c)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:  # 没用
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:  # 没用
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


def load_jit_model(url, device):
    model_path = download_model(url)
    logger.info(f"Load LDM model from: {model_path}")
    model = torch.jit.load(model_path).to(device)
    model.eval()
    return model


class LDM(InpaintModel):
    pad_mod = 32

    def __init__(self, device):
        super().__init__(device)
        self.device = device

    def init_model(self, device):
        self.diffusion_model = load_jit_model(LDM_DIFFUSION_MODEL_URL, device)
        self.cond_stage_model_decode = load_jit_model(LDM_DECODE_MODEL_URL, device)
        self.cond_stage_model_encode = load_jit_model(LDM_ENCODE_MODEL_URL, device)

        model = LatentDiffusion(self.diffusion_model, device)
        self.sampler = DDIMSampler(model)

    @staticmethod
    def is_downloaded() -> bool:
        model_paths = [
            get_cache_path_by_url(LDM_DIFFUSION_MODEL_URL),
            get_cache_path_by_url(LDM_DECODE_MODEL_URL),
            get_cache_path_by_url(LDM_ENCODE_MODEL_URL),
        ]
        return all([os.path.exists(it) for it in model_paths])

    def forward(self, image, mask, config: Config):
        """
        image: [H, W, C] RGB
        mask: [H, W, 1]
        return: BGR IMAGE
        """
        # image [1,3,512,512] float32
        # mask: [1,1,512,512] float32
        # masked_image: [1,3,512,512] float32
        steps = config.ldm_steps
        image = norm_img(image)
        mask = norm_img(mask)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        masked_image = (1 - mask) * image

        image = self._norm(image)
        mask = self._norm(mask)
        masked_image = self._norm(masked_image)

        c = self.cond_stage_model_encode(masked_image)

        cc = torch.nn.functional.interpolate(mask, size=c.shape[-2:])  # 1,1,128,128
        c = torch.cat((c, cc), dim=1)  # 1,4,128,128

        shape = (c.shape[1] - 1,) + c.shape[2:]
        samples_ddim = self.sampler.sample(steps=steps,
                                           conditioning=c,
                                           batch_size=c.shape[0],
                                           shape=shape)
        x_samples_ddim = self.cond_stage_model_decode(samples_ddim)  # samples_ddim: 1, 3, 128, 128 float32

        # image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        # mask = torch.clamp((mask + 1.0) / 2.0, min=0.0, max=1.0)
        inpainted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        # inpainted = (1 - mask) * image + mask * predicted_image
        inpainted_image = inpainted_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        inpainted_image = inpainted_image.astype(np.uint8)[:, :, ::-1]
        return inpainted_image

    def _norm(self, tensor):
        return tensor * 2.0 - 1.0
