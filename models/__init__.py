"""
MBAS2024 models

It is a multi-head model for MBAS2024 challenge.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG, DEFAULTS  # noqa: F401
from torch_ecg.utils.download import url_is_reachable
from torch_ecg.utils.misc import CitationMixin, add_docstring  # noqa: F401
from torch_ecg.utils.utils_nn import SizeMixin

from cfg import ModelCfg
from const import INPUT_IMAGE_TYPES, MODEL_CACHE_DIR
from outputs import MBAS2024Outputs

__all__ = [
    "MultiHead_MBAS2024",
]


if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable("https://huggingface.co")):
    # workaround for using huggingface hub in China
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)


class MultiHead_MBAS2024(nn.Module, SizeMixin, CitationMixin):
    """Multi-head model for MBAS2024.

    Parameters
    ----------
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
            self.__config.update(deepcopy(config))
        raise NotImplementedError

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
            Predictions, including "pred_mask" for segmentation,
            and the loss if any of the labels is provided.

        """
        raise NotImplementedError

    def get_input_tensors(self, x: INPUT_IMAGE_TYPES) -> torch.Tensor:
        raise NotImplementedError

    def freeze_backbone(self, freeze: bool = True) -> None:
        raise NotImplementedError

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
        raise NotImplementedError
        # self.train(original_mode)
        # return MBAS2024Outputs(
        #     pred_mask=None,
        # )

    @property
    def config(self) -> CFG:
        return self.__config

    def save(self, path: Union[str, bytes, os.PathLike], train_config: CFG) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : `path-like`
            Path to save the model.
        train_config : CFG
            Config for training the model,
            used when one restores the model.

        Returns
        -------
        None

        """
        if not self.config.backbone_freeze:
            super().save(path, train_config)
            return

        # if the backbone is frozen, we need to save the heads only
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        to_save = {
            "model_config": self.config,
            "train_config": train_config,
        }
        raise NotImplementedError
        # torch.save(to_save, path)
