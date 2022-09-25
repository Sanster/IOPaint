import os
import imghdr
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--model", default="lama", choices=["lama", "ldm", "zits", "mat", 'fcf'])
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

    return args
