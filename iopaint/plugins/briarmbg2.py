import numpy as np


def create_briarmbg2_session():
    from transformers import AutoModelForImageSegmentation

    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    )
    return birefnet


def briarmbg2_process(device, bgr_np_image, session, only_mask=False):
    from torchvision import transforms
    from PIL import Image

    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.fromarray(bgr_np_image)
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0)
    input_images = input_images.to(device)

    # Prediction
    preds = session(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    if only_mask:
        return np.array(mask)

    image.putalpha(mask)
    return np.array(image)
