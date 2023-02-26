import os
import tempfile
from pathlib import Path


def test_load_model():
    from lama_cleaner.interactive_seg import InteractiveSeg
    from lama_cleaner.model_manager import ModelManager

    interactive_seg_model = InteractiveSeg()

    models = [
        "lama",
        "ldm",
        "zits",
        "mat",
        "fcf",
        "manga",
    ]
    for m in models:
        ModelManager(
            name=m,
            device="cpu",
            no_half=False,
            hf_access_token="",
            disable_nsfw=False,
            sd_cpu_textencoder=True,
            sd_run_local=True,
            local_files_only=True,
            cpu_offload=True,
            enable_xformers=False,
        )


# def create_empty_file(tmp_dir, name):
#     tmp_model_dir = os.path.join(tmp_dir, "torch", "hub", "checkpoints")
#     Path(tmp_model_dir).mkdir(exist_ok=True, parents=True)
#     path = os.path.join(tmp_model_dir, name)
#     with open(path, "w") as f:
#         f.write("1")
#
#
# def test_load_model_error():
#     MODELS = [
#         ("big-lama.pt", "e3aa4aaa15225a33ec84f9f4bc47e500"),
#         ("cond_stage_model_encode.pt", "23239fc9081956a3e70de56472b3f296"),
#         ("cond_stage_model_decode.pt", "fe419cd15a750d37a4733589d0d3585c"),
#         ("diffusion.pt", "b0afda12bf790c03aba2a7431f11d22d"),
#     ]
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         os.environ["XDG_CACHE_HOME"] = tmp_dir
#         for name, md5 in MODELS:
#             create_empty_file(tmp_dir, name)
#             test_load_model()
