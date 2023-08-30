# code copy from: https://github.com/parlance-zz/g-diffuser-bot
import cv2
import numpy as np


def np_img_grey_to_rgb(data):
    if data.ndim == 3:
        return data
    return np.expand_dims(data, 2) * np.ones((1, 1, 3))


def convolve(data1, data2):  # fast convolution with fft
    if data1.ndim != data2.ndim:  # promote to rgb if mismatch
        if data1.ndim < 3:
            data1 = np_img_grey_to_rgb(data1)
        if data2.ndim < 3:
            data2 = np_img_grey_to_rgb(data2)
    return ifft2(fft2(data1) * fft2(data2))


def fft2(data):
    if data.ndim > 2:  # multiple channels
        out_fft = np.zeros(
            (data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128
        )
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
            out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
    else:  # single channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
        out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

    return out_fft


def ifft2(data):
    if data.ndim > 2:  # multiple channels
        out_ifft = np.zeros(
            (data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128
        )
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
            out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
    else:  # single channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
        out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

    return out_ifft


def get_gradient_kernel(width, height, std=3.14, mode="linear"):
    window_scale_x = float(
        width / min(width, height)
    )  # for non-square aspect ratios we still want a circular kernel
    window_scale_y = float(height / min(width, height))
    if mode == "gaussian":
        x = (np.arange(width) / width * 2.0 - 1.0) * window_scale_x
        kx = np.exp(-x * x * std)
        if window_scale_x != window_scale_y:
            y = (np.arange(height) / height * 2.0 - 1.0) * window_scale_y
            ky = np.exp(-y * y * std)
        else:
            y = x
            ky = kx
        return np.outer(kx, ky)
    elif mode == "linear":
        x = (np.arange(width) / width * 2.0 - 1.0) * window_scale_x
        if window_scale_x != window_scale_y:
            y = (np.arange(height) / height * 2.0 - 1.0) * window_scale_y
        else:
            y = x
        return np.clip(1.0 - np.sqrt(np.add.outer(x * x, y * y)) * std / 3.14, 0.0, 1.0)
    else:
        raise Exception("Error: Unknown mode in get_gradient_kernel: {0}".format(mode))


def image_blur(data, std=3.14, mode="linear"):
    width = data.shape[0]
    height = data.shape[1]
    kernel = get_gradient_kernel(width, height, std, mode=mode)
    return np.real(convolve(data, kernel / np.sqrt(np.sum(kernel * kernel))))


def soften_mask(np_rgba_image, softness, space):
    if softness == 0:
        return np_rgba_image
    softness = min(softness, 1.0)
    space = np.clip(space, 0.0, 1.0)
    original_max_opacity = np.max(np_rgba_image[:, :, 3])
    out_mask = np_rgba_image[:, :, 3] <= 0.0
    blurred_mask = image_blur(np_rgba_image[:, :, 3], 3.5 / softness, mode="linear")
    blurred_mask = np.maximum(blurred_mask - np.max(blurred_mask[out_mask]), 0.0)
    np_rgba_image[
        :, :, 3
    ] *= blurred_mask  # preserve partial opacity in original input mask
    np_rgba_image[:, :, 3] /= np.max(np_rgba_image[:, :, 3])  # renormalize
    np_rgba_image[:, :, 3] = np.clip(
        np_rgba_image[:, :, 3] - space, 0.0, 1.0
    )  # make space
    np_rgba_image[:, :, 3] /= np.max(np_rgba_image[:, :, 3])  # and renormalize again
    np_rgba_image[:, :, 3] *= original_max_opacity  # restore original max opacity
    return np_rgba_image


def expand_image(
    cv2_img, top: int, right: int, bottom: int, left: int, softness: float, space: float
):
    origin_h, origin_w = cv2_img.shape[:2]
    new_width = cv2_img.shape[1] + left + right
    new_height = cv2_img.shape[0] + top + bottom
    new_img = np.zeros((new_height, new_width, 4), np.uint8)  # expanded image is rgba

    print(
        "Expanding input image from {0}x{1} to {2}x{3}".format(
            cv2_img.shape[1], cv2_img.shape[0], new_width, new_height
        )
    )
    if cv2_img.shape[2] == 3:  # rgb input image
        new_img[
            top : top + cv2_img.shape[0], left : left + cv2_img.shape[1], 0:3
        ] = cv2_img
        new_img[
            top : top + cv2_img.shape[0], left : left + cv2_img.shape[1], 3
        ] = 255  # fully opaque
    elif cv2_img.shape[2] == 4:  # rgba input image
        new_img[top : top + cv2_img.shape[0], left : left + cv2_img.shape[1]] = cv2_img
    else:
        raise Exception(
            "Unsupported image format: {0} channels".format(cv2_img.shape[2])
        )

    if softness > 0.0:
        new_img = soften_mask(new_img / 255.0, softness / 100.0, space / 100.0)
        new_img = (np.clip(new_img, 0.0, 1.0) * 255.0).astype(np.uint8)

    mask_image = 255.0 - new_img[:, :, 3]  # extract mask from alpha channel and invert
    rgb_init_image = (
        0.0 + new_img[:, :, 0:3]
    )  # strip mask from init_img leaving only rgb channels

    hard_mask = np.zeros_like(cv2_img[:, :, 0])
    if top != 0:
        hard_mask[0 : origin_h // 2, :] = 255
    if bottom != 0:
        hard_mask[origin_h // 2 :, :] = 255
    if left != 0:
        hard_mask[:, 0 : origin_w // 2] = 255
    if right != 0:
        hard_mask[:, origin_w // 2 :] = 255
    hard_mask = cv2.copyMakeBorder(
        hard_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255
    )
    mask_image = np.where(hard_mask > 0, mask_image, 0)
    return rgb_init_image.astype(np.uint8), mask_image.astype(np.uint8)


if __name__ == "__main__":
    from pathlib import Path

    current_dir = Path(__file__).parent.absolute().resolve()
    image_path = current_dir.parent / "tests" / "bunny.jpeg"
    init_image = cv2.imread(str(image_path))
    init_image, mask_image = expand_image(
        init_image,
        200,
        200,
        0,
        0,
        60,
        50,
    )
    print(mask_image.dtype, mask_image.min(), mask_image.max())
    print(init_image.dtype, init_image.min(), init_image.max())
    mask_image = mask_image.astype(np.uint8)
    init_image = init_image.astype(np.uint8)
    cv2.imwrite("expanded_image.png", init_image)
    cv2.imwrite("expanded_mask.png", mask_image)
