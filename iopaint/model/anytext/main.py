from anytext_pipeline import AnyTextPipeline
from utils import save_images

seed = 66273235
# seed_everything(seed)

pipe = AnyTextPipeline(
    cfg_path="/Users/cwq/code/github/AnyText/anytext/models_yaMl/anytext_sd15.yaml",
    model_dir="/Users/cwq/.cache/modelscope/hub/damo/cv_anytext_text_generation_editing",
    # font_path="/Users/cwq/code/github/AnyText/anytext/font/Arial_Unicode.ttf",
    # font_path="/Users/cwq/code/github/AnyText/anytext/font/SourceHanSansSC-VF.ttf",
    font_path="/Users/cwq/code/github/AnyText/anytext/font/SourceHanSansSC-Medium.otf",
    use_fp16=False,
    device="mps",
)

img_save_folder = "SaveImages"
params = {
    "show_debug": True,
    "image_count": 2,
    "ddim_steps": 20,
}

# # 1. text generation
# mode = "text-generation"
# input_data = {
#     "prompt": 'photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream',
#     "seed": seed,
#     "draw_pos": "/Users/cwq/code/github/AnyText/anytext/example_images/gen9.png",
# }
# results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
# if rtn_code >= 0:
#     save_images(results, img_save_folder)
#     print(f"Done, result images are saved in: {img_save_folder}")
# if rtn_warning:
#     print(rtn_warning)
#
# exit()
# 2. text editing
mode = "text-editing"
input_data = {
    "prompt": 'A cake with colorful characters that reads "EVERYDAY"',
    "seed": seed,
    "draw_pos": "/Users/cwq/code/github/AnyText/anytext/example_images/edit7.png",
    "ori_image": "/Users/cwq/code/github/AnyText/anytext/example_images/ref7.jpg",
}
results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
if rtn_code >= 0:
    save_images(results, img_save_folder)
    print(f"Done, result images are saved in: {img_save_folder}")
if rtn_warning:
    print(rtn_warning)
