import cv2
import numpy as np

from lama_cleaner.helper import boxes_from_mask


def test_boxes_from_mask():
    mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
    mask = mask[:, :, np.newaxis]
    mask = (mask / 255).transpose(2, 0, 1)
    boxes = boxes_from_mask(mask)
    print(boxes)


test_boxes_from_mask()
