import os
import importlib.util
import shutil
import ctypes
import logging

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# https://github.com/pytorch/pytorch/issues/27971#issuecomment-1768868068
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "1"
os.environ["LRU_CACHE_CAPACITY"] = "1"
# prevent CPU memory leak when run model on GPU
# https://github.com/pytorch/pytorch/issues/98688#issuecomment-1869288431
# https://github.com/pytorch/pytorch/issues/108334#issuecomment-1752763633
os.environ["TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"] = "1"

import warnings

warnings.simplefilter("ignore", UserWarning)


def fix_window_pytorch():
    # copy from: https://github.com/comfyanonymous/ComfyUI/blob/5cbaa9e07c97296b536f240688f5a19300ecf30d/fix_torch.py#L4
    import platform

    try:
        if platform.system() != "Windows":
            return
        torch_spec = importlib.util.find_spec("torch")
        for folder in torch_spec.submodule_search_locations:
            lib_folder = os.path.join(folder, "lib")
            test_file = os.path.join(lib_folder, "fbgemm.dll")
            dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
            if os.path.exists(dest):
                break

            with open(test_file, "rb") as f:
                contents = f.read()
                if b"libomp140.x86_64.dll" not in contents:
                    break
            try:
                mydll = ctypes.cdll.LoadLibrary(test_file)
            except FileNotFoundError:
                logging.warning("Detected pytorch version with libomp issue, patching.")
                shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
    except:
        pass


def entry_point():
    # To make os.environ["XDG_CACHE_HOME"] = args.model_cache_dir works for diffusers
    # https://github.com/huggingface/diffusers/blob/be99201a567c1ccd841dc16fb24e88f7f239c187/src/diffusers/utils/constants.py#L18
    from iopaint.cli import typer_app

    fix_window_pytorch()

    typer_app()
