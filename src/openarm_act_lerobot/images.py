"""Camera frames as the dataset stores them and as the model reads them.

One encoding for training and serving alike:

    chest        RGB, uint8 [H, W, 3]
    chest_depth  raw depth, uint16 mm [H, W] (0 = no reading) -> a gray
                 uint8 [H, W, 3] image, scaled as ACT scales depth

The converter writes these into the LeRobot dataset; the server applies the
same functions to what openarm_act_lerobot_client sends, then hands the model
a float [3, H, W] in 0..1, which is what LeRobot's video decoder gives it in
training; the model resizes and normalises it (modeling_openarm_act). Numpy only: the tests and the converter's planning run without torch.
"""

from __future__ import annotations

import numpy as np

from openarm_act_lerobot import constants as C


def depth_to_image(depth_mm) -> np.ndarray:
    """Recorder depth (uint16 mm, 0 = no reading) -> uint8 [H, W, 3] gray.

    The scaling ACT uses (openarm_act/policy.py): metres clipped to
    [DEPTH_MIN_M, DEPTH_MAX_M] and mapped to 0..1, here 0..255 on all three
    channels: an image a backbone can take on its own (depth_as=image), whose
    first channel is act_pipeline's 0..1 depth (depth_as=channel).
    """
    depth_m = np.nan_to_num(np.asarray(depth_mm, dtype=np.float32) / 1000.0, nan=0.0)
    scaled = (np.clip(depth_m, C.DEPTH_MIN_M, C.DEPTH_MAX_M) - C.DEPTH_MIN_M) / (C.DEPTH_MAX_M - C.DEPTH_MIN_M)
    gray = np.round(scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def depth_to_metres(depth_mm) -> np.ndarray:
    """Recorder depth (uint16 mm, 0 = no reading) -> float32 [1, H, W] in
    metres: what LeRobot's depth feature yields in training (data.depth_format=raw)."""
    depth_m = np.asarray(depth_mm, dtype=np.float32) / 1000.0
    return np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)[None]


def parse_image(image) -> np.ndarray:
    """uint8 [H, W, 3] from any layout the recorder, LeRobot or the client
    hands over: uint8 HWC, float CHW in 0..1, or a raw uint16 depth frame [H, W]."""
    image = np.asarray(image)
    if image.ndim == 2:
        return depth_to_image(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.round(255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"cannot read an image of shape {image.shape} and dtype {image.dtype}")
    return image


def to_model_input(image) -> np.ndarray:
    """-> float32 [3, H, W] in 0..1, as LeRobot's decoder yields frames."""
    return np.ascontiguousarray(np.transpose(parse_image(image), (2, 0, 1)), dtype=np.float32) / 255.0
