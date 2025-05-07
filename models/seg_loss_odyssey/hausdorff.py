import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf

copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super().__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert pred.dim() == target.dim(), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt**self.alpha + target_dt**self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.erosions = erosions

        # Register kernels as buffers
        cross = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)
        bound = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        kernel2D = cross.unsqueeze(0).unsqueeze(0) / 5
        kernel3D = torch.stack([bound, cross, bound]).unsqueeze(0).unsqueeze(0) / 7

        self.register_buffer("kernel2D", kernel2D)
        self.register_buffer("kernel3D", kernel3D)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred and target can have shapes:
        - 2D: (B, 1, H, W) or (B, C, H, W)
        - 3D: (B, 1, D, H, W) or (B, C, D, H, W)

        In case of multi-channel predictions, `argmax` is applied to get binary predictions.
        """
        # If multi-channel prediction, apply argmax and convert to binary (1 - foreground)
        if pred.shape[1] > 1:
            pred = torch.argmax(pred, dim=1, keepdim=True).float()
        if target.shape[1] > 1:
            target = torch.argmax(target, dim=1, keepdim=True).float()

        assert pred.shape == target.shape, f"Shapes must match, got {pred.shape=} and {target.shape=}"
        assert pred.dim() in (4, 5), f"Only 2D and 3D inputs are supported, got {pred.dim()=}"

        bound = (pred - target) ** 2
        eroded = torch.zeros_like(bound)
        kernel = self.kernel3D if bound.dim() == 5 else self.kernel2D

        for k in range(self.erosions):
            # Perform convolution to simulate erosion/dilation
            bound = F.conv3d(bound, kernel, padding=1) if bound.dim() == 5 else F.conv2d(bound, kernel, padding=1)

            # Soft thresholding and normalization
            erosion = bound - 0.5
            erosion = F.relu(erosion)

            ptp = erosion.amax(dim=[2, 3, 4] if bound.dim() == 5 else [2, 3], keepdim=True) - erosion.amin(
                dim=[2, 3, 4] if bound.dim() == 5 else [2, 3], keepdim=True
            )
            ptp = ptp + 1e-6  # Avoid division by zero
            erosion = erosion / ptp

            eroded += erosion * (k + 1) ** self.alpha

        return eroded.mean()
