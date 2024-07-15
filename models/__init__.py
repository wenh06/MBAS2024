"""
MBAS2024 models

It is a multi-head model for MBAS2024 challenge.
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG, DEFAULTS  # noqa: F401
from torch_ecg.models.loss import AsymmetricLoss, BCEWithLogitsWithClassWeightLoss, FocalLoss, MaskedBCEWithLogitsLoss
from torch_ecg.utils.download import url_is_reachable
from torch_ecg.utils.misc import CitationMixin, add_docstring  # noqa: F401
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg
from const import INPUT_IMAGE_TYPES, MODEL_CACHE_DIR
from outputs import MBAS2024Outputs

from .nested_vnet import NestedVNet
from .vnet import VNet

__all__ = [
    "MultiHead_MBAS2024",
]


if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable("https://huggingface.co")):
    # workaround for using huggingface hub in China
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)


class MultiHead_MBAS2024(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Multi-head model for MBAS2024.

    Parameters
    ----------
    stage : {0, 1}
        Stage 0 or 1 of the pipeline.
        0 for raw localization, 1 for fine segmentation.
    config : dict
        Hyper-parameters, including backbone_name, etc.
        ref. the corresponding config file.

    """

    __DEBUG__ = True
    __name__ = "MultiHead_MBAS2024"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update((config or {}).copy())
        assert self.config.stage in [0, 1], "stage must be 0 or 1"
        nums_classes = 2 if self.config.stage == 0 else 4
        # self.preprocessor = None
        # self.augmentor = None
        if self.config.seg_model_name.lower() in ["nestedvnet", "nested_vnet"]:
            self.segmentation_head = NestedVNet(nums_classes, self.config.nested_vnet)
        elif self.config.seg_model_name.lower() in ["vnet"]:
            self.segmentation_head = VNet(nums_classes, self.config.vnet)
        self.segmentation_loss = self._setup_criterion(self.config.seg_loss, self.config.seg_loss_kw)
        # TODO: add bounding box regression head
        self.bbox_head = None
        self.bb_loss = None

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        img : torch.Tensor
            Input LGE-MRI image tensor.
        labels : dict, optional
            Labels for training, including
            - "mask": optional for training the segmentation head.

        Returns
        -------
        dict
            Predictions, including "seg_logits" and "seg_mask" for segmentation,
            and the loss if any of the labels is provided.

        """
        seg_logits = self.segmentation_head(img)  # (B, H, W, D, C), C is the number of classes
        seg_mask = torch.softmax(seg_logits, dim=-1).argmax(dim=-1)  # (B, H, W, D)
        output = {"seg_logits": seg_logits, "seg_mask": seg_mask}
        if self.bbox_head is not None:
            raise NotImplementedError
        if labels is not None:
            output["total_loss"] = 0
            output["seg_loss"] = self.segmentation_loss(seg_logits, labels["mask"])
            output["total_loss"] += output["seg_loss"]
            if self.bbox_head is not None and "bbox" in labels:
                raise NotImplementedError
        return output

    # def freeze_backbone(self, freeze: bool = True) -> None:
    #     raise NotImplementedError

    @torch.no_grad()
    def inference(self, img: INPUT_IMAGE_TYPES) -> MBAS2024Outputs:
        """Inference on a single image or a batch of images.

        Parameters
        ----------
        img : numpy.ndarray or torch.Tensor or list
            Input LGE-MRI image.

        Returns
        -------
        MBAS2024Outputs
            Predictions, including "pred_mask" for segmentation.

        """
        original_mode = self.training
        self.eval()
        input_tensors = self.get_input_tensors(img)
        output = self.forward(input_tensors)
        self.train(original_mode)
        return MBAS2024Outputs(
            pred_mask=output["seg_mask"].cpu().numpy(),
        )

    @property
    def config(self) -> CFG:
        return self.__config

    def get_input_tensors(self, x: INPUT_IMAGE_TYPES) -> torch.Tensor:
        """Make the input tensor into a batched tensor,
        and perform necessary preprocessing.
        """
        if isinstance(x, (list, tuple)):
            x = torch.stack([item if isinstance(item, torch.Tensor) else torch.from_numpy(item) for item in x])
        elif not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        # x = torch.tensor(x, dtype=self.dtype, device=self.device)
        x = x.to(device=self.device, dtype=self.dtype)
        for _ in range(5 - x.ndim):
            x = x.unsqueeze(0)
        # x of shape (B, C, H, W, D)
        # sample-wise normalization
        x = (x - x.mean(dim=(1, 2, 3, 4), keepdim=True)) / (x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6)
        # TODO: other preprocessing steps including CLAHE, etc.
        return x

    def _setup_criterion(self, loss: str, loss_kw: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Setup the loss function.

        Parameters
        ----------
        loss : str
            Name of the loss function.
        loss_kw : dict
            Keyword arguments for the loss function.

        Returns
        -------
        nn.Module
            The loss function.

        """
        if loss_kw is None:
            loss_kw = {}
        for k, v in loss_kw.items():
            if isinstance(v, torch.Tensor):
                loss_kw[k] = v.to(device=self.device, dtype=self.dtype)
        if loss == "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(**loss_kw)
        elif loss == "BCEWithLogitsWithClassWeightLoss":
            criterion = BCEWithLogitsWithClassWeightLoss(**loss_kw)
        elif loss == "BCELoss":
            criterion = nn.BCELoss(**loss_kw)
        elif loss == "MaskedBCEWithLogitsLoss":
            criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif loss == "MaskedBCEWithLogitsLoss":
            criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif loss == "FocalLoss":
            criterion = FocalLoss(**loss_kw)
        elif loss == "AsymmetricLoss":
            criterion = AsymmetricLoss(**loss_kw)
        elif loss == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss(**loss_kw)
        else:
            raise NotImplementedError(
                f"loss `{loss}` not implemented! "
                "Please use one of the following: `BCEWithLogitsLoss`, `BCEWithLogitsWithClassWeightLoss`, "
                "`BCELoss`, `MaskedBCEWithLogitsLoss`, `MaskedBCEWithLogitsLoss`, `FocalLoss`, "
                "`AsymmetricLoss`, `CrossEntropyLoss`, or override this method to setup your own criterion."
            )
        criterion = criterion.to(device=self.device, dtype=self.dtype)
        return criterion
