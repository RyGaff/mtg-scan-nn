import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from src.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_train_transform() -> A.Compose:
    """Return a stochastic augmentation pipeline suitable for training.

    Augmentation categories (albumentations 2.0.x API):
      - Perspective warp
      - Rotation
      - Random resized crop
      - Color jitter
      - Gamma adjustment
      - Gaussian blur
      - Gaussian noise
      - JPEG compression artefacts
      - Sun-flare / glare
      - RGB channel shift
      - Normalize + ToTensor
    """
    pad_size = int(INPUT_SIZE * 1.15)
    return A.Compose([
        A.LongestMaxSize(max_size=pad_size),
        A.PadIfNeeded(
            min_height=pad_size,
            min_width=pad_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Perspective(scale=(0.02, 0.08), p=0.7),
        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.8,
        ),
        A.RandomResizedCrop(
            size=(INPUT_SIZE, INPUT_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.95, 1.05),
            p=1.0,
        ),
        A.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15,
            hue=0.03,
            p=0.8,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # std_range is relative to max dtype value (255 for uint8).
        # Equivalent to old var_limit=(5.0, 30.0) pixel-variance:
        #   pixel std = sqrt(5..30) ≈ 2.2..5.5  →  normalized std ≈ 0.009..0.022
        A.GaussNoise(std_range=(0.009, 0.022), p=0.3),
        A.ImageCompression(quality_range=(40, 95), p=0.5),
        A.RandomSunFlare(
            flare_roi=(0.1, 0.1, 0.9, 0.9),
            src_radius=40,
            angle_range=(0.0, 1.0),
            num_flare_circles_range=(1, 3),
            p=0.25,
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=10, b_shift_limit=15, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform() -> A.Compose:
    """Return a deterministic transform for evaluation / inference.

    Only resize, pad, normalize, and convert to tensor — no stochastic ops.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=INPUT_SIZE),
        A.PadIfNeeded(
            min_height=INPUT_SIZE,
            min_width=INPUT_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
