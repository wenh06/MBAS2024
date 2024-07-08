"""
Modified from
https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py

More variants of VNet can be found at
https://github.com/junqiangchen/VNetFamily
"""

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils import SizeMixin

from .layers import DownTransition, InputTransition, OutputTransition, UpTransition

__all__ = [
    "VNet",
]


class VNet(nn.Module, SizeMixin):
    """VNet model.

    Parameters
    ----------
    num_classes: int
        Number of classes.
    activation: str, default "mish"
        Activation function name, e.g.
        "relu", "prelu", "elu", "leaky_relu", "mish", "swish", etc.
    conv_ordering: str, default "cna"
        Convolutional block ordering, e.g. "cna" or "can".
        "c" stands for Conv3d, "n" stands for BatchNorm3d, "a" stands for activation.

    The output tensor is the logits map,
    of shape ``(batch_size, height, width, depth, num_classes)``.

    """

    __DEFAULT_CONFIG__ = CFG(
        activation="mish",
        conv_ordering="cna",
        input_conv={"channels": 16, "kernel_size": 5},
        down_conv={  # down transitions (convolutional blocks)
            "channels": [32, 64, 128, 256],  # out channels
            "blocks": [1, 2, 2, 2],
            "kernel_size": [5, 5, 5, 5],
            "dropout": [False, False, True, True],
        },
        up_conv={  # up transitions (transposed convolutional blocks)
            "channels": [256, 128, 64, 32],  # out channels
            "blocks": [2, 2, 1, 1],
            "kernel_size": [5, 5, 5, 5],
            "dropout": [True, True, False, False],
        },
        output_conv={"kernel_size": 5},
    )

    def __init__(self, num_classes: int, config: Optional[CFG] = None) -> None:
        super().__init__()
        self.config = deepcopy(self.__DEFAULT_CONFIG__)
        self.config.update((config or {}).copy())
        self.__check_config_validity()
        self.in_tr = InputTransition(
            self.config.input_conv.channels, self.config.input_conv.kernel_size, self.config.activation
        )
        self.down_tr = nn.ModuleList()
        input_channels = self.config.input_conv.channels
        for i, (nconv, dropout) in enumerate(zip(self.config.down_conv.blocks, self.config.down_conv.dropout)):
            self.down_tr.append(
                DownTransition(
                    input_channels,
                    self.config.down_conv.channels[i],
                    nconv,
                    self.config.down_conv.kernel_size[i],
                    self.config.activation,
                    dropout,
                )
            )
            input_channels = self.config.down_conv.channels[i]
        self.up_tr = nn.ModuleList()
        input_channels = self.config.down_conv.channels[-1]
        for i, (nconv, dropout) in enumerate(zip(self.config.up_conv.blocks, self.config.up_conv.dropout)):
            self.up_tr.append(
                UpTransition(
                    input_channels,
                    self.config.up_conv.channels[i],
                    nconv,
                    self.config.up_conv.kernel_size[i],
                    self.config.activation,
                    dropout,
                )
            )
            input_channels = self.config.up_conv.channels[i]
        self.out_tr = OutputTransition(input_channels, num_classes, self.config.output_conv.kernel_size, self.config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_outputs = [self.in_tr(x)]
        for down_tr in self.down_tr:
            down_outputs.append(down_tr(down_outputs[-1]))
        out = down_outputs.pop()
        for i, up_tr in enumerate(self.up_tr):
            out = up_tr(out, down_outputs[-1 - i])
        out = self.out_tr(out)
        return out

    def __check_config_validity(self) -> None:
        assert (
            len(self.config.down_conv.channels)
            == len(self.config.down_conv.blocks)
            == len(self.config.down_conv.kernel_size)
            == len(self.config.down_conv.dropout)
        )
        assert (
            len(self.config.up_conv.channels)
            == len(self.config.up_conv.blocks)
            == len(self.config.up_conv.kernel_size)
            == len(self.config.up_conv.dropout)
        )
        assert len(self.config.down_conv.channels) == len(self.config.up_conv.channels)
