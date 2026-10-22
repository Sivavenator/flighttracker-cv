"""
Albumentations augmentation pipelines for detector and classifier training.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector augmentation (YOLO, 640-px input)
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 640):
    """
    Full augmentation pipeline for YOLOv8 detector training.
    Returns an albumentations Compose with YOLO bounding-box support.
    """
    try:
        import albumentations as A
    except ImportError as exc:
        raise RuntimeError("albumentations not installed") from exc

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.2),
            A.RandomScale(scale_limit=(-0.2, 0.2), p=0.3),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=0.15),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=64,
            min_visibility=0.3,
        ),
    )


def get_val_transforms(image_size: int = 640):
    """Minimal transforms for validation: just resize + pad."""
    try:
        import albumentations as A
    except ImportError as exc:
        raise RuntimeError("albumentations not installed") from exc

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


# ---------------------------------------------------------------------------
# Classifier augmentation (224-px input)
# ---------------------------------------------------------------------------

def get_classifier_train_transforms(image_size: int = 224):
    """
    Augmentation pipeline for EfficientNet classifier training.
    Returns an albumentations Compose that works on raw BGR crops.
    """
    try:
        import albumentations as A
    except ImportError as exc:
        raise RuntimeError("albumentations not installed") from exc

    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.6),
        A.ToGray(p=0.05),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.CoarseDropout(max_holes=8, max_height=image_size // 8, max_width=image_size // 8, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        _ToTensorTransform(),
    ])


def get_classifier_val_transforms(image_size: int = 224):
    """Centre-crop + normalise for classifier validation."""
    try:
        import albumentations as A
    except ImportError as exc:
        raise RuntimeError("albumentations not installed") from exc

    return A.Compose([
        A.SmallestMaxSize(max_size=int(image_size * 1.143)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        _ToTensorTransform(),
    ])


# ---------------------------------------------------------------------------
# Tensor transform shim (albumentations doesn't include ToTensor natively)
# ---------------------------------------------------------------------------

class _ToTensorTransform(object):
    """Converts HWC numpy array to CHW torch tensor after Normalize."""

    def __call__(self, **data):
        import torch
        img = data["image"]
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)  # HWC -> CHW
        data["image"] = torch.from_numpy(np.ascontiguousarray(img)).float()
        return data

    @property
    def targets(self):
        return {"image": self.__call__}


# ---------------------------------------------------------------------------
# Mosaic augmentation
# ---------------------------------------------------------------------------

class AircraftMosaic:
    """
    Combines 4 aircraft images into a 2x2 mosaic for YOLOv8 training.

    This replicates the mosaic augmentation concept but as a standalone
    utility that can be applied before feeding images into the YOLO trainer.
    (YOLOv8 already has built-in mosaic; this class is for custom pipelines.)
    """

    def __init__(self, mosaic_size: int = 640) -> None:
        self.mosaic_size = mosaic_size

    def create(
        self,
        images: List[np.ndarray],
        all_bboxes: List[List[List[float]]],
        all_labels: List[List[int]],
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Combine exactly 4 images into one mosaic.

        images: list of 4 BGR images
        all_bboxes: list of 4 lists, each containing YOLO [xc,yc,w,h] boxes
        all_labels: list of 4 lists of class IDs

        Returns: (mosaic_img, merged_bboxes, merged_labels)
        """
        assert len(images) == 4, "AircraftMosaic requires exactly 4 images."
        s = self.mosaic_size
        hs = s // 2  # half-size

        mosaic = np.full((s, s, 3), 114, dtype=np.uint8)
        merged_bboxes: List[List[float]] = []
        merged_labels: List[int] = []

        positions = [(0, 0), (hs, 0), (0, hs), (hs, hs)]
        for i, (img, bboxes, labels) in enumerate(zip(images, all_bboxes, all_labels)):
            x_off, y_off = positions[i]
            tile = cv2.resize(img, (hs, hs))
            mosaic[y_off:y_off + hs, x_off:x_off + hs] = tile

            for bbox, lbl in zip(bboxes, labels):
                xc, yc, bw, bh = bbox
                # Convert from [0,1] to mosaic pixel coords
                xc_px = xc * hs + x_off
                yc_px = yc * hs + y_off
                bw_px = bw * hs
                bh_px = bh * hs
                # Normalise to full mosaic
                merged_bboxes.append([
                    xc_px / s, yc_px / s, bw_px / s, bh_px / s
                ])
                merged_labels.append(lbl)

        return mosaic, merged_bboxes, merged_labels


# ---------------------------------------------------------------------------
# Night simulation
# ---------------------------------------------------------------------------

class NightSimulator:
    """
    Converts a daytime BGR frame to a simulated night-time frame for
    training a detector that works in low-light conditions.
    """

    def __init__(
        self,
        base_gain: float = 1.5,
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        noise_sigma: float = 15.0,
    ) -> None:
        self.base_gain = base_gain
        self.noise_sigma = noise_sigma
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Darken the frame, add gaussian noise, and apply CLAHE to simulate
        a camera operating in low-light conditions.
        """
        dark = np.clip(frame.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)

        # Add realistic sensor noise
        noise = np.random.normal(0, self.noise_sigma, dark.shape).astype(np.int16)
        noisy = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Apply CLAHE per channel
        channels = cv2.split(noisy)
        clahe_channels = [self.clahe.apply(c) for c in channels]
        enhanced = cv2.merge(clahe_channels)

        # Apply gain
        result = np.clip(enhanced.astype(np.float32) * self.base_gain, 0, 255).astype(np.uint8)
        return result

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.apply(frame)
