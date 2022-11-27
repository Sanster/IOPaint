from pathlib import Path

import pytest
import cv2
import numpy as np

from lama_cleaner.interactive_seg import InteractiveSeg, Click

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / 'result'
save_dir.mkdir(exist_ok=True, parents=True)
img_p = current_dir / "overture-creations-5sI6fQgYIuo.png"


def test_interactive_seg():
    interactive_seg_model = InteractiveSeg()
    img = cv2.imread(str(img_p))
    pred = interactive_seg_model(img, clicks=[Click(coords=(256, 256), indx=0, is_positive=True)])
    cv2.imwrite(str(save_dir / "test_interactive_seg.png"), pred)


def test_interactive_seg_with_negative_click():
    interactive_seg_model = InteractiveSeg()
    img = cv2.imread(str(img_p))
    pred = interactive_seg_model(img, clicks=[
        Click(coords=(256, 256), indx=0, is_positive=True),
        Click(coords=(384, 256), indx=1, is_positive=False)
    ])
    cv2.imwrite(str(save_dir / "test_interactive_seg_negative.png"), pred)


def test_interactive_seg_with_prev_mask():
    interactive_seg_model = InteractiveSeg()
    img = cv2.imread(str(img_p))
    mask = np.zeros_like(img)[:, :, 0]
    pred = interactive_seg_model(img, clicks=[Click(coords=(256, 256), indx=0, is_positive=True)], prev_mask=mask)
    cv2.imwrite(str(save_dir / "test_interactive_seg_with_mask.png"), pred)
