"""
AnyText: Multilingual Visual Text Generation And Editing
Paper: https://arxiv.org/abs/2311.03054
Code: https://github.com/tyxsspa/AnyText
Copyright (c) Alibaba, Inc. and its affiliates.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import random
import re
import numpy as np
import cv2
import einops
import time
from PIL import ImageFont
from iopaint.model.anytext.cldm.model import create_model, load_state_dict
from iopaint.model.anytext.cldm.ddim_hacked import DDIMSampler
from iopaint.model.anytext.utils import (
    resize_image,
    check_channels,
    draw_glyph,
    draw_glyph2,
)


BBOX_MAX_NUM = 8
PLACE_HOLDER = "*"
max_chars = 20


class AnyTextPipeline:
    def __init__(self, cfg_path, model_dir, font_path, device, use_fp16=True):
        self.cfg_path = cfg_path
        self.model_dir = model_dir
        self.font_path = font_path
        self.use_fp16 = use_fp16
        self.device = device
        self.init_model()

    """
    return:
        result: list of images in numpy.ndarray format
        rst_code: 0: normal -1: error 1:warning
        rst_info: string of error or warning
        debug_info: string for debug, only valid if show_debug=True
    """

    def __call__(self, input_tensor, **forward_params):
        tic = time.time()
        str_warning = ""
        # get inputs
        seed = input_tensor.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 99999999)
        # seed_everything(seed)
        prompt = input_tensor.get("prompt")
        draw_pos = input_tensor.get("draw_pos")
        ori_image = input_tensor.get("ori_image")

        mode = forward_params.get("mode")
        sort_priority = forward_params.get("sort_priority", "↕")
        show_debug = forward_params.get("show_debug", False)
        revise_pos = forward_params.get("revise_pos", False)
        img_count = forward_params.get("image_count", 4)
        ddim_steps = forward_params.get("ddim_steps", 20)
        w = forward_params.get("image_width", 512)
        h = forward_params.get("image_height", 512)
        strength = forward_params.get("strength", 1.0)
        cfg_scale = forward_params.get("cfg_scale", 9.0)
        eta = forward_params.get("eta", 0.0)
        a_prompt = forward_params.get(
            "a_prompt",
            "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks",
        )
        n_prompt = forward_params.get(
            "n_prompt",
            "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
        )

        prompt, texts = self.modify_prompt(prompt)
        if prompt is None and texts is None:
            return (
                None,
                -1,
                "You have input Chinese prompt but the translator is not loaded!",
                "",
            )
        n_lines = len(texts)
        if mode in ["text-generation", "gen"]:
            edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
        elif mode in ["text-editing", "edit"]:
            if draw_pos is None or ori_image is None:
                return (
                    None,
                    -1,
                    "Reference image and position image are needed for text editing!",
                    "",
                )
            if isinstance(ori_image, str):
                ori_image = cv2.imread(ori_image)[..., ::-1]
                assert (
                    ori_image is not None
                ), f"Can't read ori_image image from{ori_image}!"
            elif isinstance(ori_image, torch.Tensor):
                ori_image = ori_image.cpu().numpy()
            else:
                assert isinstance(
                    ori_image, np.ndarray
                ), f"Unknown format of ori_image: {type(ori_image)}"
            edit_image = ori_image.clip(1, 255)  # for mask reason
            edit_image = check_channels(edit_image)
            edit_image = resize_image(
                edit_image, max_length=768
            )  # make w h multiple of 64, resize if w or h > max_length
            h, w = edit_image.shape[:2]  # change h, w by input ref_img
        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if draw_pos is None:
            pos_imgs = np.zeros((w, h, 1))
        if isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
            pos_imgs = 255 - draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        else:
            assert isinstance(
                draw_pos, np.ndarray
            ), f"Unknown format of draw_pos: {type(draw_pos)}"
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # seprate pos_imgs
        pos_imgs = self.separate_pos_imgs(pos_imgs, sort_priority)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == " ":
                pass  # text-to-image without text
            else:
                return (
                    None,
                    -1,
                    f"Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!",
                    "",
                )
        elif len(pos_imgs) > n_lines:
            str_warning = f"Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt."
        # get pre_pos, poly_list, hint that needed for anytext
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = (
                    input_pos[..., np.newaxis]
                    if len(input_pos.shape) == 2
                    else input_pos
                )
                poly, pos_img = self.find_polygon(input_pos)
                pre_pos += [pos_img / 255.0]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
        # prepare info dict
        info = {}
        info["glyphs"] = []
        info["gly_line"] = []
        info["positions"] = []
        info["n_lines"] = [len(texts)] * img_count
        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = (
                    f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                )
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(
                    self.font,
                    text,
                    poly_list[i],
                    scale=gly_scale,
                    width=w,
                    height=h,
                    add_space=False,
                )
                gly_pos_img = cv2.drawContours(
                    glyphs * 255, [poly_list[i] * gly_scale], 0, (255, 255, 255), 1
                )
                if revise_pos:
                    resize_gly = cv2.resize(
                        glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0])
                    )
                    new_pos = cv2.morphologyEx(
                        (resize_gly * 255).astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        kernel=np.ones(
                            (resize_gly.shape[0] // 10, resize_gly.shape[1] // 10),
                            dtype=np.uint8,
                        ),
                        iterations=1,
                    )
                    new_pos = (
                        new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                    )
                    contours, _ = cv2.findContours(
                        new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    if len(contours) != 1:
                        str_warning = f"Fail to revise position {i} to bounding rect, remain position unchanged..."
                    else:
                        rect = cv2.minAreaRect(contours[0])
                        poly = np.int0(cv2.boxPoints(rect))
                        pre_pos[i] = (
                            cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.0
                        )
                        gly_pos_img = cv2.drawContours(
                            glyphs * 255, [poly * gly_scale], 0, (255, 255, 255), 1
                        )
                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                glyphs = np.zeros((h * gly_scale, w * gly_scale, 1))
                gly_line = np.zeros((80, 512, 1))
                gly_pos_imgs += [
                    np.zeros((h * gly_scale, w * gly_scale, 1))
                ]  # for show
            pos = pre_pos[i]
            info["glyphs"] += [self.arr2tensor(glyphs, img_count)]
            info["gly_line"] += [self.arr2tensor(gly_line, img_count)]
            info["positions"] += [self.arr2tensor(pos, img_count)]
        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0) * (1 - np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().to(self.device)
        if self.use_fp16:
            masked_img = masked_img.half()
        encoder_posterior = self.model.encode_first_stage(masked_img[None, ...])
        masked_x = self.model.get_first_stage_encoding(encoder_posterior).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        info["masked_x"] = torch.cat([masked_x for _ in range(img_count)], dim=0)

        hint = self.arr2tensor(np_hint, img_count)
        cond = self.model.get_learned_conditioning(
            dict(
                c_concat=[hint],
                c_crossattn=[[prompt + " , " + a_prompt] * img_count],
                text_info=info,
            )
        )
        un_cond = self.model.get_learned_conditioning(
            dict(c_concat=[hint], c_crossattn=[[n_prompt] * img_count], text_info=info)
        )
        shape = (4, h // 8, w // 8)
        self.model.control_scales = [strength] * 13
        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            img_count,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=un_cond,
        )
        if self.use_fp16:
            samples = samples.half()
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        results = [x_samples[i] for i in range(img_count)]
        if (
            mode == "edit" and False
        ):  # replace backgound in text editing but not ideal yet
            results = [r * np_hint + edit_image * (1 - np_hint) for r in results]
            results = [r.clip(0, 255).astype(np.uint8) for r in results]
        if len(gly_pos_imgs) > 0 and show_debug:
            glyph_bs = np.stack(gly_pos_imgs, axis=2)
            glyph_img = np.sum(glyph_bs, axis=2) * 255
            glyph_img = glyph_img.clip(0, 255).astype(np.uint8)
            results += [np.repeat(glyph_img, 3, axis=2)]
        # debug_info
        if not show_debug:
            debug_info = ""
        else:
            input_prompt = prompt
            for t in texts:
                input_prompt = input_prompt.replace("*", f'"{t}"', 1)
            debug_info = f'<span style="color:black;font-size:18px">Prompt: </span>{input_prompt}<br> \
                           <span style="color:black;font-size:18px">Size: </span>{w}x{h}<br> \
                           <span style="color:black;font-size:18px">Image Count: </span>{img_count}<br> \
                           <span style="color:black;font-size:18px">Seed: </span>{seed}<br> \
                           <span style="color:black;font-size:18px">Use FP16: </span>{self.use_fp16}<br> \
                           <span style="color:black;font-size:18px">Cost Time: </span>{(time.time()-tic):.2f}s'
        rst_code = 1 if str_warning else 0
        return results, rst_code, str_warning, debug_info

    def init_model(self):
        font_path = self.font_path
        self.font = ImageFont.truetype(font_path, size=60)
        cfg_path = self.cfg_path
        ckpt_path = os.path.join(self.model_dir, "anytext_v1.1.ckpt")
        clip_path = os.path.join(self.model_dir, "clip-vit-large-patch14")
        self.model = create_model(
            cfg_path,
            device=self.device,
            cond_stage_path=clip_path,
            use_fp16=self.use_fp16,
        )
        if self.use_fp16:
            self.model = self.model.half()
        self.model.load_state_dict(
            load_state_dict(ckpt_path, location=self.device), strict=False
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.ddim_sampler = DDIMSampler(self.model, device=self.device)

    def modify_prompt(self, prompt):
        prompt = prompt.replace("“", '"')
        prompt = prompt.replace("”", '"')
        p = '"(.*?)"'
        strs = re.findall(p, prompt)
        if len(strs) == 0:
            strs = [" "]
        else:
            for s in strs:
                prompt = prompt.replace(f'"{s}"', f" {PLACE_HOLDER} ", 1)
        # if self.is_chinese(prompt):
        #     if self.trans_pipe is None:
        #         return None, None
        #     old_prompt = prompt
        #     prompt = self.trans_pipe(input=prompt + " .")["translation"][:-1]
        #     print(f"Translate: {old_prompt} --> {prompt}")
        return prompt, strs

    # def is_chinese(self, text):
    #     text = checker._clean_text(text)
    #     for char in text:
    #         cp = ord(char)
    #         if checker._is_chinese_char(cp):
    #             return True
    #     return False

    def separate_pos_imgs(self, img, sort_priority, gap=102):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        components = []
        for label in range(1, num_labels):
            component = np.zeros_like(img)
            component[labels == label] = 255
            components.append((component, centroids[label]))
        if sort_priority == "↕":
            fir, sec = 1, 0  # top-down first
        elif sort_priority == "↔":
            fir, sec = 0, 1  # left-right first
        components.sort(key=lambda c: (c[1][fir] // gap, c[1][sec] // gap))
        sorted_components = [c[0] for c in components]
        return sorted_components

    def find_polygon(self, image, min_rect=False):
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
        if min_rect:
            # get minimum enclosing rectangle
            rect = cv2.minAreaRect(max_contour)
            poly = np.int0(cv2.boxPoints(rect))
        else:
            # get approximate polygon
            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            poly = cv2.approxPolyDP(max_contour, epsilon, True)
            n, _, xy = poly.shape
            poly = poly.reshape(n, xy)
        cv2.drawContours(image, [poly], -1, 255, -1)
        return poly, image

    def arr2tensor(self, arr, bs):
        arr = np.transpose(arr, (2, 0, 1))
        _arr = torch.from_numpy(arr.copy()).float().to(self.device)
        if self.use_fp16:
            _arr = _arr.half()
        _arr = torch.stack([_arr for _ in range(bs)], dim=0)
        return _arr
