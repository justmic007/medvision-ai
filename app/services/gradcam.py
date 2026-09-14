"""Explainability: GradCAM heatmaps over the X-ray.

Answers "where did the model look?" for a chosen finding. Given a preprocessed
image and a target pathology, it produces a heatmap (values in [0, 1], sized to
the input image) highlighting the regions that drove that finding.

How it works:
  1. Forward pass with gradients enabled; capture the `features` activations
     (1, 1024, 7, 7) and retain their gradient.
  2. Backpropagate from the target finding's score.
  3. Weight each of the 1024 channels by its mean gradient (channel importance),
     sum into a 7x7 map, ReLU (keep positive contributions only), normalize.
  4. Upsample the 7x7 map to the input size -> the heatmap.

Gradient capture uses `Tensor.retain_grad()` on the forward-captured activations
rather than a backward hook: torchxrayvision's model applies an in-place ReLU in
its `features2` step, which collides with backward hooks on `features`. Reading
`.grad` off a retained tensor sidesteps that entirely (verified on torch 2.2.2).

Design (D-11): explainability is defined by an interface; the chest X-ray
implementation is one concrete case. GradCAM manages its own gradient-enabled
forward pass, independent of the classifier's no_grad inference path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

TARGET_SIZE = 224


class Explainer(ABC):
    """Produces a heatmap for a target finding on a preprocessed image."""

    @abstractmethod
    def heatmap(self, tensor: torch.Tensor, finding: str) -> np.ndarray:
        ...


class GradCAM(Explainer):
    """GradCAM for the TorchXRayVision DenseNet.

    Hooks the `features` block's output, backpropagates a chosen finding, and
    builds a normalized heatmap sized to the input image.
    """

    def __init__(self, weights: str = "densenet121-res224-all") -> None:
        self._model = xrv.models.DenseNet(weights=weights)
        self._model.eval()
        self.pathologies: list[str] = list(self._model.pathologies)
        self._weights = weights

    def _finding_index(self, finding: str) -> int:
        if finding not in self.pathologies:
            raise ValueError(
                f"Unknown finding {finding!r}. "
                f"Must be one of the model's {len(self.pathologies)} pathologies."
            )
        return self.pathologies.index(finding)

    def heatmap(self, tensor: torch.Tensor, finding: str) -> np.ndarray:
        """Return a (H, W) float heatmap in [0, 1] for `finding` on `tensor`.

        `tensor` is a (1, 1, H, W) preprocessed image from a Preprocessor.
        """
        idx = self._finding_index(finding)

        captured: dict[str, torch.Tensor] = {}

        def fwd_hook(_m, _i, output):
            output.retain_grad()
            captured["act"] = output

        handle = self._model.features.register_forward_hook(fwd_hook)
        try:
            inp = tensor.clone().requires_grad_(True)
            score = self._model(inp)[0][idx]
            self._model.zero_grad()
            score.backward()

            act = captured["act"]          # (1, 1024, 7, 7)
            grad = act.grad                # (1, 1024, 7, 7)

            # Channel importance = mean gradient per channel -> (1, 1024, 1, 1)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            # Weighted sum over channels -> (1, 1, 7, 7)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)              # keep positive contributions

            # Upsample to input size, drop batch/channel dims -> (H, W)
            size = tensor.shape[-2:]
            cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()

            # Normalize to [0, 1] (guard against a flat map)
            cam -= cam.min()
            peak = cam.max()
            if peak > 0:
                cam /= peak
            return cam
        finally:
            handle.remove()


def render_overlay(
    tensor: torch.Tensor,
    heatmap: np.ndarray,
    out_path: str,
    alpha: float = 0.4,
) -> str:
    """Render `heatmap` as a colored overlay on the X-ray and save a PNG.

    `tensor` is the (1, 1, H, W) preprocessed image; `heatmap` is the (H, W)
    map from GradCAM.heatmap(). Saves to out_path and returns it.

    Kept as a standalone function (not on the class) so rendering is decoupled
    from the CAM computation — a caller can use the raw heatmap array without
    ever touching matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; no display needed
    import matplotlib.pyplot as plt

    # The X-ray as a 2D grayscale array in display range.
    img = tensor[0, 0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    return out_path


def render_overlay_base64(
    tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> str:
    """Render the overlay to an in-memory PNG and return base64 (no file).

    Same rendering as render_overlay(), but returns a base64-encoded PNG string
    suitable for embedding in a JSON API response or an <img src="data:..."> —
    used by the orchestrator / /analyze endpoint (D-14 era: demo returns base64;
    storage-backed URLs come with the clinical-workflow phase, D-10).
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = tensor[0, 0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
