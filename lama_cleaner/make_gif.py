import io
import math
from pathlib import Path

from PIL import Image, ImageDraw


def keep_ratio_resize(img, size, resample=Image.BILINEAR):
    if img.width > img.height:
        w = size
        h = int(img.height * size / img.width)
    else:
        h = size
        w = int(img.width * size / img.height)
    return img.resize((w, h), resample)


def cubic_bezier(p1, p2, duration: int, frames: int):
    """

    Args:
        p1:
        p2:
        duration: Total duration of the curve
        frames:

    Returns:

    """
    x0, y0 = (0, 0)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = (1, 1)

    def cal_y(t):
        return math.pow(1 - t, 3) * y0 + \
               3 * math.pow(1 - t, 2) * t * y1 + \
               3 * (1 - t) * math.pow(t, 2) * y2 + \
               math.pow(t, 3) * y3

    def cal_x(t):
        return math.pow(1 - t, 3) * x0 + \
               3 * math.pow(1 - t, 2) * t * x1 + \
               3 * (1 - t) * math.pow(t, 2) * x2 + \
               math.pow(t, 3) * x3

    res = []
    for t in range(0, 1 * frames, duration):
        t = t / frames
        res.append((cal_x(t), cal_y(t)))

    res.append((1, 0))
    return res


def make_compare_gif(
    clean_img: Image.Image,
    src_img: Image.Image,
    max_side_length: int = 600,
    splitter_width: int = 5,
    splitter_color=(255, 203, 0, int(255 * 0.73))
):
    if clean_img.size != src_img.size:
        clean_img = clean_img.resize(src_img.size, Image.BILINEAR)

    duration_per_frame = 20
    num_frames = 50
    # erase-in-out
    cubic_bezier_points = cubic_bezier((0.33, 0), (0.66, 1), 1, num_frames)
    cubic_bezier_points.reverse()

    max_side_length = min(max_side_length, max(clean_img.size))

    src_img = keep_ratio_resize(src_img, max_side_length)
    clean_img = keep_ratio_resize(clean_img, max_side_length)
    width, height = src_img.size

    # Generate images to make Gif from right to left
    images = []

    for i in range(num_frames):
        new_frame = Image.new('RGB', (width, height))
        new_frame.paste(clean_img, (0, 0))

        left = int(cubic_bezier_points[i][0] * width)
        cropped_src_img = src_img.crop((left, 0, width, height))
        new_frame.paste(cropped_src_img, (left, 0, width, height))
        if i != num_frames - 1:
            # draw a yellow splitter on the edge of the cropped image
            draw = ImageDraw.Draw(new_frame)
            draw.line([(left, 0), (left, height)], width=splitter_width, fill=splitter_color)
        images.append(new_frame)

    for i in range(10):
        images.append(src_img)

    cubic_bezier_points.reverse()
    # Generate images to make Gif from left to right
    for i in range(num_frames):
        new_frame = Image.new('RGB', (width, height))
        new_frame.paste(src_img, (0, 0))

        right = int(cubic_bezier_points[i][0] * width)
        cropped_src_img = clean_img.crop((0, 0, right, height))
        new_frame.paste(cropped_src_img, (0, 0, right, height))
        if i != num_frames - 1:
            # draw a yellow splitter on the edge of the cropped image
            draw = ImageDraw.Draw(new_frame)
            draw.line([(right, 0), (right, height)], width=splitter_width, fill=splitter_color)
        images.append(new_frame)

    images.append(clean_img)

    img_byte_arr = io.BytesIO()
    clean_img.save(
        img_byte_arr,
        format='GIF',
        save_all=True,
        include_color_table=True,
        append_images=images,
        optimize=False,
        duration=duration_per_frame,
        loop=0
    )
    return img_byte_arr.getvalue()
