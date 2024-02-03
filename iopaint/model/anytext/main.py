import cv2
import os

from anytext_pipeline import AnyTextPipeline
from utils import save_images

seed = 66273235
# seed_everything(seed)

pipe = AnyTextPipeline(
    ckpt_path="/Users/cwq/code/github/IOPaint/iopaint/model/anytext/anytext_v1.1_fp16.ckpt",
    font_path="/Users/cwq/code/github/AnyText/anytext/font/SourceHanSansSC-Medium.otf",
    use_fp16=False,
    device="mps",
)

img_save_folder = "SaveImages"
rgb_image = cv2.imread(
    "/Users/cwq/code/github/AnyText/anytext/example_images/ref7.jpg"
)[..., ::-1]

masked_image = cv2.imread(
    "/Users/cwq/code/github/AnyText/anytext/example_images/edit7.png"
)[..., ::-1]

rgb_image = cv2.resize(rgb_image, (512, 512))
masked_image = cv2.resize(masked_image, (512, 512))

# results: list of rgb ndarray
results, rtn_code, rtn_warning = pipe(
    prompt='A cake with colorful characters that reads "EVERYDAY", best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks',
    negative_prompt="low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
    image=rgb_image,
    masked_image=masked_image,
    num_inference_steps=20,
    strength=1.0,
    guidance_scale=9.0,
    height=rgb_image.shape[0],
    width=rgb_image.shape[1],
    seed=seed,
    sort_priority="y",
)
if rtn_code >= 0:
    save_images(results, img_save_folder)
    print(f"Done, result images are saved in: {img_save_folder}")
