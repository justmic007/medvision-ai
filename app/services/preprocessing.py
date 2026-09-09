"""Image preprocessing: raw scan -> model-ready tensor.

Turns a chest X-ray (DICOM or PNG/JPG, per D-06) into the exact tensor the
TorchXRayVision DenseNet expects: shape (1, 1, 224, 224), pixel values in
roughly [-1024, 1024] (xrv's training convention).

Design (D-11): a model is fed by a *pluggable* preprocessor. `Preprocessor` is
the interface; `ChestXrayPreprocessor` is the one concrete implementation we
ship. A future modality (e.g. brain MRI) implements the same interface without
touching callers. This is a deliberately thin seam — one ABC, not a framework.

All heavy lifting uses torchxrayvision's own verified utilities so our tensor
matches what the pretrained model was trained on:
  - xrv.utils.load_image / xrv.utils.read_xray_dcm  (loading)
  - xrv.utils.normalize                              (scale to ~[-1024, 1024])
  - xrv.datasets.XRayCenterCrop                      (center-crop to square)
  - xrv.datasets.XRayResizer                         (resize to 224)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv

# The pretrained `*-res224-*` models expect 224x224 input.
TARGET_SIZE = 224


class Preprocessor(ABC):
    """Turns a raw image file into a model-ready (1, 1, H, W) float tensor.

    Implementations own their modality's loading + normalization. The contract:
    given a path, return a torch.FloatTensor of shape (1, 1, TARGET_SIZE,
    TARGET_SIZE) in the range the target model expects.
    """

    @abstractmethod
    def process(self, path: str | Path) -> torch.Tensor:
        ...


class ChestXrayPreprocessor(Preprocessor):
    """Chest X-ray preprocessor for the TorchXRayVision DenseNet.

    Accepts DICOM (.dcm) or PNG/JPG. Both are normalized to xrv's ~[-1024, 1024]
    convention, center-cropped to a square, resized to 224, and shaped to
    (1, 1, 224, 224).
    """

    def __init__(self, size: int = TARGET_SIZE) -> None:
        self._crop = xrv.datasets.XRayCenterCrop()
        self._resize = xrv.datasets.XRayResizer(size)

    def _load(self, path: Path) -> np.ndarray:
        """Load to a single-channel float array, normalized to ~[-1024, 1024].

        Returns shape (1, H, W) — channel-first, single channel — which is what
        xrv's crop/resize transforms operate on.
        """
        suffix = path.suffix.lower()
        if suffix == ".dcm":
            # read_xray_dcm handles DICOM photometric interpretation and returns
            # a normalized (1, H, W) float array in xrv's ~[-1024, 1024] range.
            img = xrv.utils.read_xray_dcm(str(path))
        else:
            # load_image reads PNG/JPG and ALREADY returns a normalized (1, H, W)
            # float array in ~[-1024, 1024] (verified: it does its own scaling,
            # so no separate normalize() call is needed).
            img = xrv.utils.load_image(str(path))

        # Defensive: guarantee channel-first single-channel shape (1, H, W).
        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 3 and img.shape[0] != 1:
            img = img.mean(axis=0, keepdims=True)
        return img

    def process(self, path: str | Path) -> torch.Tensor:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = self._load(path)          # (1, H, W), ~[-1024, 1024]
        img = self._crop(img)           # (1, S, S) center square
        img = self._resize(img)         # (1, 224, 224)

        tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1, 1, 224, 224)
        return tensor
