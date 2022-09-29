import os
import imghdr
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument(
        "--model",
        default="lama",
        choices=["lama", "ldm", "zits", "mat", "fcf", "sd1.4", "cv2"],
    )
    parser.add_argument(
        "--hf_access_token",
        default="",
        help="Huggingface access token. Check how to get token from: https://huggingface.co/docs/hub/security-tokens",
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
        help="After first time Stable Diffusion model downloaded, you can add this arg and remove --hf_access_token",
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument("--gui", action="store_true", help="Launch as desktop app")
    parser.add_argument(
        "--gui-size",
        default=[1600, 1000],
        nargs=2,
        type=int,
        help="Set window size for GUI",
    )
    parser.add_argument(
        "--input", type=str, help="Path to image you want to load by default"
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.input is not None:
        if not os.path.exists(args.input):
            parser.error(f"invalid --input: {args.input} not exists")
        if imghdr.what(args.input) is None:
            parser.error(f"invalid --input: {args.input} is not a valid image file")

    if args.model.startswith("sd") and not args.sd_run_local:
        if not args.hf_access_token.startswith("hf_"):
            parser.error(
                f"sd(stable-diffusion) model requires huggingface access token. Check how to get token from: https://huggingface.co/docs/hub/security-tokens"
            )

    return args
