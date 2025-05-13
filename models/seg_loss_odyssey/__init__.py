import torch

from .boundary_loss import BDLoss, DC_and_BD_loss, SoftDiceLoss
from .dice_loss import (
    DC_and_CE_loss,
    DC_and_topk_loss,
    ExpLog_loss,
    FocalTversky_loss,
    GDiceLoss,
    GDiceLossV2,
    IoULoss,
    PenaltyGDiceLoss,
    SSLoss,
    TverskyLoss,
)
from .hausdorff import HausdorffDTLoss, HausdorffERLoss
from .lovasz_loss import LovaszSoftmax

__all__ = [
    "available_losses",
    "setup_odyssey_criterion",
    "BDLoss",
    "DC_and_BD_loss",
    "SoftDiceLoss",
    "DC_and_CE_loss",
    "DC_and_topk_loss",
    "ExpLog_loss",
    "FocalTversky_loss",
    "GDiceLoss",
    "GDiceLossV2",
    "IoULoss",
    "PenaltyGDiceLoss",
    "SSLoss",
    "TverskyLoss",
    "HausdorffDTLoss",
    "HausdorffERLoss",
    "LovaszSoftmax",
]


available_losses = {
    name: obj
    for name, obj in globals().items()
    if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and name not in ["SegLossOdyssey", "SegLossOdysseyTransforms"]
}


class SegLossOdysseyTransforms(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inp, gt):
        # (B, H, W, D, C) -> (B, C, H, W, D)
        inp = inp.permute(0, 4, 1, 2, 3)
        # (B, H, W, D, C) -> (B, H, W, D, 1) -> (B, 1, H, W, D)
        gt = gt.argmax(-1, keepdim=True).permute(0, 4, 1, 2, 3)
        return inp, gt


class SegLossOdyssey(torch.nn.Module):

    def __init__(self, loss, **loss_kw):
        super().__init__()
        self.transform = SegLossOdysseyTransforms()
        self.criterion = eval(loss)(**loss_kw)

    def forward(self, inp, gt, *extra_tensors):
        inp, gt = self.transform(inp, gt)
        return self.criterion(inp, gt, *extra_tensors)


def setup_odyssey_criterion(loss, **loss_kw):
    return SegLossOdyssey(loss, **loss_kw)
