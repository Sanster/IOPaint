import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler
from iopaint.tests.utils import assert_equal, get_config, current_dir, check_device


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
def test_lama(device, strategy):
    check_device(device)
    model = ModelManager(name="lama", device=device)
    assert_equal(
        model,
        get_config(strategy=strategy),
        f"lama_{strategy[0].upper() + strategy[1:]}_result.png",
    )

    fx = 1.3
    assert_equal(
        model,
        get_config(strategy=strategy),
        f"lama_{strategy[0].upper() + strategy[1:]}_fx_{fx}_result.png",
        fx=1.3,
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
@pytest.mark.parametrize("ldm_sampler", [LDMSampler.ddim, LDMSampler.plms])
def test_ldm(device, strategy, ldm_sampler):
    check_device(device)
    model = ModelManager(name="ldm", device=device)
    cfg = get_config(strategy=strategy, ldm_sampler=ldm_sampler)
    assert_equal(
        model, cfg, f"ldm_{strategy[0].upper() + strategy[1:]}_{ldm_sampler}_result.png"
    )

    fx = 1.3
    assert_equal(
        model,
        cfg,
        f"ldm_{strategy[0].upper() + strategy[1:]}_{ldm_sampler}_fx_{fx}_result.png",
        fx=fx,
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
@pytest.mark.parametrize("zits_wireframe", [False, True])
def test_zits(device, strategy, zits_wireframe):
    check_device(device)
    model = ModelManager(name="zits", device=device)
    cfg = get_config(strategy=strategy, zits_wireframe=zits_wireframe)
    assert_equal(
        model,
        cfg,
        f"zits_{strategy[0].upper() + strategy[1:]}_wireframe_{zits_wireframe}_result.png",
    )

    fx = 1.3
    assert_equal(
        model,
        cfg,
        f"zits_{strategy.capitalize()}_wireframe_{zits_wireframe}_fx_{fx}_result.png",
        fx=fx,
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("no_half", [True, False])
def test_mat(device, strategy, no_half):
    check_device(device)
    model = ModelManager(name="mat", device=device, no_half=no_half)
    cfg = get_config(strategy=strategy)

    assert_equal(
        model,
        cfg,
        f"mat_{strategy.capitalize()}_result.png",
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_fcf(device, strategy):
    check_device(device)
    model = ModelManager(name="fcf", device=device)
    cfg = get_config(strategy=strategy)

    assert_equal(model, cfg, f"fcf_{strategy.capitalize()}_result.png", fx=2, fy=2)
    assert_equal(model, cfg, f"fcf_{strategy.capitalize()}_result.png", fx=3.8, fy=2)


@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
@pytest.mark.parametrize("cv2_flag", ["INPAINT_NS", "INPAINT_TELEA"])
@pytest.mark.parametrize("cv2_radius", [3, 15])
def test_cv2(strategy, cv2_flag, cv2_radius):
    model = ModelManager(
        name="cv2",
        device=torch.device("cpu"),
    )
    cfg = get_config(strategy=strategy, cv2_flag=cv2_flag, cv2_radius=cv2_radius)
    assert_equal(
        model,
        cfg,
        f"cv2_{strategy.capitalize()}_{cv2_flag}_{cv2_radius}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
def test_manga(device, strategy):
    check_device(device)
    model = ModelManager(
        name="manga",
        device=torch.device(device),
    )
    cfg = get_config(strategy=strategy)
    assert_equal(
        model,
        cfg,
        f"manga_{strategy.capitalize()}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_mi_gan(device, strategy):
    check_device(device)
    model = ModelManager(
        name="migan",
        device=torch.device(device),
    )
    cfg = get_config(strategy=strategy)
    assert_equal(
        model,
        cfg,
        f"migan_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.5,
        fy=1.7
    )
