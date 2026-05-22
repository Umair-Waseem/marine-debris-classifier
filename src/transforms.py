import torch
from torchvision import transforms as T

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


# =========================================================
# Runtime Transforms
# =========================================================

def get_train_transforms():
    """
    Training transforms.

    These are applied at runtime only.
    They do not save resized or augmented images to disk.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(
            degrees=10,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=(128, 128, 128),
        ),
        T.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02,
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    """
    Validation/test transforms.

    No random augmentation is applied to validation or test data.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms(split: str):
    """
    Return transforms according to split.

    Valid split values:
    - train
    - validation
    - test
    """
    split = str(split).strip().lower()

    if split == "train":
        return get_train_transforms()

    if split in {"validation", "valid", "val", "test"}:
        return get_eval_transforms()

    raise ValueError(
        f"Invalid split for transforms: {split}. "
        "Expected one of: train, validation, test."
    )


def denormalize_tensor_image(tensor_image: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor image for plotting.

    Input shape:
        [3, H, W]

    Output:
        Tensor clipped to [0, 1].
    """
    if tensor_image.ndim != 3:
        raise ValueError(
            f"Expected tensor image with shape [3, H, W], "
            f"but got {tuple(tensor_image.shape)}"
        )

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor_image.dtype,
        device=tensor_image.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor_image.dtype,
        device=tensor_image.device,
    ).view(3, 1, 1)

    image = tensor_image * std + mean
    image = torch.clamp(image, 0.0, 1.0)

    return image
