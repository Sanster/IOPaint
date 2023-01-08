import os
import imghdr
import argparse
from pathlib import Path

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument(
        "--model",
        default="lama",
        choices=["lama", "ldm", "zits", "mat", "fcf", "sd1.5", "cv2", "manga", "sd2", "paint_by_example"],
    )
    parser.add_argument("--no-half", action="store_true", help="sd/paint_by_example model no half precision")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="sd/paint_by_example model, offloads all models to CPU, significantly reducing vRAM usage.")
    parser.add_argument(
        "--hf_access_token",
        default="",
        help="SD model no more need token: https://github.com/huggingface/diffusers/issues/1447",
    )
    parser.add_argument(
        "--sd-disable-nsfw",
        action="store_true",
        help="Disable Stable Diffusion NSFW checker",
    )
    parser.add_argument(
        "--sd-cpu-textencoder",
        action="store_true",
        help="Always run Stable Diffusion TextEncoder model on CPU",
    )
    parser.add_argument(
        "--sd-run-local",
        action="store_true",
        help="SD model no more need token, use --local-files-only to set not connect to huggingface server",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="sd/paint_by_example model. Use local files only, not connect to huggingface server",
    )
    parser.add_argument(
        "--sd-enable-xformers",
        action="store_true",
        help="Enable xFormers optimizations. Requires that xformers package has been installed. See: https://github.com/facebookresearch/xformers"
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu", "mps"])
    parser.add_argument("--gui", action="store_true", help="Launch as desktop app")
    parser.add_argument(
        "--gui-size",
        default=[1600, 1000],
        nargs=2,
        type=int,
        help="Set window size for GUI",
    )
    parser.add_argument(
        "--input", type=str,
        help="If input is image, it will be load by default. If input is directory, all images will be loaded to file manager"
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Only required when --input is directory. Output directory for all processed images"
    )
    parser.add_argument("--disable-model-switch", action="store_true", help="Disable model switch in frontend")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.input is not None:
        if not os.path.exists(args.input):
            parser.error(f"invalid --input: {args.input} not exists")
        if os.path.isfile(args.input):
            if imghdr.what(args.input) is None:
                parser.error(f"invalid --input: {args.input} is not a valid image file")
        else:
            if args.output_dir is None:
                parser.error(f"invalid --input: {args.input} is a directory, --output-dir is required")
            else:
                output_dir = Path(args.output_dir)
                if not output_dir.exists():
                    logger.info(f"Creating output directory: {output_dir}")
                    output_dir.mkdir(parents=True)
                else:
                    if not output_dir.is_dir():
                        parser.error(f"invalid --output-dir: {output_dir} is not a directory")

    return args
