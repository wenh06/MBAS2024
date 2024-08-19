"""
MBAS2024 model.

It is a multi-head model for MBAS2024 challenge.
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
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
from utils.mclahe_tf import mclahe  # noqa: F401

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
            - "weight_mask": optional for training the segmentation head.

        Returns
        -------
        dict
            Predictions, including "seg_logits" and "seg_mask" for segmentation,
            and the loss if any of the labels is provided.

        """
        seg_logits = self.segmentation_head(img.to(device=self.device))  # (B, C, H, W, D)
        if isinstance(seg_logits, torch.Tensor):
            seg_mask = torch.softmax(seg_logits, dim=-1).argmax(dim=-1)  # (B, H, W, D)
            seg_logits = [seg_logits]
        else:
            # list of tensors, via deep supervision
            # TODO: final mask by voting or the last one?
            seg_mask = torch.softmax(seg_logits[-1], dim=-1).argmax(dim=-1)  # (B, H, W, D)
        output = {"seg_logits": seg_logits, "seg_mask": seg_mask}
        if self.bbox_head is not None:
            raise NotImplementedError
        if labels is not None:
            output["total_loss"] = 0
            output["seg_loss"] = 0
            for logits_tensor in seg_logits:
                if self.config.seg_loss == "MaskedBCEWithLogitsLoss":
                    output["seg_loss"] += self.segmentation_loss(
                        logits_tensor, labels["mask"].to(self.device), labels["weight_mask"].to(self.device)
                    )
                else:
                    output["seg_loss"] += self.segmentation_loss(logits_tensor, labels["mask"].to(self.device))
            output["total_loss"] += output["seg_loss"]
            if self.bbox_head is not None and "bbox" in labels:
                raise NotImplementedError
        return output

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
        # if self.config.apply_mclahe:
        #     if isinstance(img, (list, tuple)):
        #         img = [mclahe(item.cpu().numpy()) if isinstance(item, torch.Tensor) else mclahe(item) for item in img]
        #     elif isinstance(img, np.ndarray):
        #         # apply mclahe to the last 3 dimensions
        #         shape = img.shape
        #         img = np.array([mclahe(item) for item in img.reshape(-1, *shape[-3:])]).reshape(shape)
        #     elif isinstance(img, torch.Tensor):
        #         shape = tuple(img.shape)
        #         img = np.array([mclahe(item) for item in img.cpu().numpy().reshape(-1, *shape[-3:])]).reshape(shape)
        #     else:
        #         raise ValueError(f"Unsupported input type: {type(img)}")
        input_tensors = self.get_input_tensors(img)
        output = self.forward(input_tensors)
        self.train(original_mode)
        return MBAS2024Outputs(
            pred_mask=output["seg_mask"].cpu().numpy(),
        )

    @property
    def config(self) -> CFG:
        return self.__config

    def get_input_tensors(self, img: INPUT_IMAGE_TYPES) -> torch.Tensor:
        """Make the input tensor into a batched tensor,
        and perform necessary preprocessing.

        Parameters
        ----------
        img : numpy.ndarray or torch.Tensor or list
            Input LGE-MRI image(s).

        Returns
        -------
        torch.Tensor
            The input tensor.

        """
        if isinstance(img, (list, tuple)):
            img = torch.stack([item if isinstance(item, torch.Tensor) else torch.from_numpy(item) for item in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        elif isinstance(img, torch.Tensor):
            pass
        else:
            raise ValueError(f"Unsupported input type: {type(img)}")
        img = img.to(device=self.device, dtype=self.dtype)
        for _ in range(5 - img.ndim):
            img = img.unsqueeze(0)
        # img of shape (B, C, H, W, D)
        # sample-wise normalization
        img = (img - img.mean(dim=(1, 2, 3, 4), keepdim=True)) / (img.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-8)
        return img

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
