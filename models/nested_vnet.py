"""
Modified from
https://github.com/4uiiurz1/pytorch-nested-unet

More variants of VNet can be found at
https://github.com/junqiangchen/VNetFamily
"""

from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.models._nets import get_activation
from torch_ecg.utils import SizeMixin

from .layers import DownTransition, InputTransition, OutputTransition, make_convs

__all__ = ["NestedVNet"]


class UpTransition(nn.Module, SizeMixin):
    """Up transition block.

    Parameters
    ----------
    in_chans: int
        Number of input channels.
    out_chans: int
        Number of output channels.
    num_convs: int
        Number of convolutional layers.
    num_skips: int
        Number of skip connections.
    kernel_size: int
        Kernel size of the convolutional layers.
    activation: str
        Activation function name, e.g.
        "relu", "prelu", "elu", "leaky_relu", "mish", "swish", etc.
    dropout: Union[bool, float], default False
        Whether to use dropout or not if is boolean, or the dropout rate if is float.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_convs: int,
        num_skips: int,
        kernel_size: int,
        activation: str,
        dropout: Union[bool, float] = False,
    ) -> None:
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(in_chans, out_chans // 2, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm3d(out_chans // 2)
        if isinstance(dropout, bool) and dropout:
            self.dropout1 = nn.Dropout3d()
        elif isinstance(dropout, float) and dropout > 0:
            self.dropout1 = nn.Dropout3d(dropout)
        else:
            self.dropout1 = nn.Identity()
        kw_act = dict()
        if activation.lower() == "elu":
            kw_act["inplace"] = True
        elif activation.lower() == "prelu":
            kw_act["num_parameters"] = out_chans // 2
        self.act1 = get_activation(activation, kw_act.copy())
        if activation.lower() == "prelu":
            kw_act["num_parameters"] = out_chans
        self.act2 = get_activation(activation, kw_act.copy())
        if num_skips == 1:
            self.dropout2 = nn.Dropout3d()
        else:  # > 1
            self.dropout2 = nn.Sequential(
                *[
                    nn.Conv3d(out_chans // 2 * (2 * num_skips - 1), out_chans // 2, kernel_size=1),
                    nn.BatchNorm3d(out_chans // 2),
                    nn.Dropout3d(),
                ]
            )
        self.ops = make_convs(out_chans, num_convs, kernel_size, activation)

    def forward(self, x: torch.Tensor, skipx: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Features from the previous layer.
        skipx: torch.Tensor
            Skip connection tensor. Features from the encoder layer.

        Returns
        -------
        torch.Tensor
            Output tensor.

        """
        out = self.dropout1(x)
        skipxdo = self.dropout2(skipx)
        out = self.act1(self.norm(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)  # along the channel dimension
        out = self.ops(xcat)
        out = self.act2(torch.add(out, xcat))
        return out


class NestedVNet(nn.Module, SizeMixin):
    """Nested V-Net (V-Net ++) model.

    References
    ----------
    https://arxiv.org/abs/1807.10165

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
        deep_supervision=False,  # whether to use deep supervision
    )

    def __init__(self, num_classes: int, config: Optional[CFG] = None) -> None:
        super().__init__()
        self.config = deepcopy(self.__DEFAULT_CONFIG__)
        self.config.update((config or {}).copy())
        self.__check_config_validity()
        self.n_levels = len(self.config.down_conv.channels)
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
        self.out_tr = nn.ModuleList()
        for lv in range(self.n_levels):
            self.up_tr.append(nn.ModuleList())
            channels = self.config.up_conv.channels[-1 - lv :]
            blocks = self.config.up_conv.blocks[-1 - lv :]
            kernel_size = self.config.up_conv.kernel_size[-1 - lv :]
            dropout = self.config.up_conv.dropout[-1 - lv :]
            input_channels = self.config.down_conv.channels[lv]
            for i, (nconv, dropout) in enumerate(zip(blocks, dropout)):
                num_skips = i + 1
                self.up_tr[-1].append(
                    UpTransition(
                        input_channels,
                        channels[i],
                        nconv,
                        num_skips,
                        kernel_size[i],
                        self.config.activation,
                        dropout,
                    )
                )
                input_channels = channels[i]
            self.out_tr.append(
                OutputTransition(input_channels, num_classes, self.config.output_conv.kernel_size, self.config.activation)
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        tensor_container = [[None for _ in range(lv + 1)] for lv in range(self.n_levels + 1)]
        outputs = []
        # down transitions
        for idx in range(self.n_levels + 1):
            if idx == 0:
                tensor_container[idx][0] = self.in_tr(x)
            else:
                tensor_container[idx][0] = self.down_tr[idx - 1](tensor_container[idx - 1][0])
        # up transitions and out transitions
        for idx in range(1, self.n_levels + 1):
            for i in range(len(self.up_tr[idx - 1])):
                tensor_container[idx][i + 1] = self.up_tr[idx - 1][i](
                    x=tensor_container[idx][i],
                    skipx=torch.cat([tensor_container[lv][lv - (idx - i - 1)] for lv in range(idx - i - 1, idx)], dim=1),
                )
            if self.config.deep_supervision:
                outputs.append(self.out_tr[idx - 1](tensor_container[idx][-1]))
            else:
                if idx == self.n_levels:
                    outputs.append(self.out_tr[idx - 1](tensor_container[idx][-1]))
        return outputs

    @torch.no_grad()
    def inference(self, x: torch.Tensor, level: int = -1) -> torch.Tensor:
        """Inference on a single image or a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input LGE-MRI image.
        level : int, default -1
            Level of the output tensor,
            or equivalently, the depth of the network.

        Returns
        -------
        torch.Tensor
            Segmentation mask.

        """
        original_mode = self.training
        self.eval()
        if level < 0:
            level = self.n_levels + level
        assert 0 <= level < self.n_levels
        tensor_container = [[None for _ in range(lv + 1)] for lv in range(level + 1)]
        outputs = []
        # down transitions
        for idx in range(level + 1):
            if idx == 0:
                tensor_container[idx][0] = self.in_tr(x)
            else:
                tensor_container[idx][0] = self.down_tr[idx - 1](tensor_container[idx - 1][0])
        # up transitions and out transitions
        for idx in range(1, level + 1):
            for i in range(len(self.up_tr[idx - 1])):
                tensor_container[idx][i + 1] = self.up_tr[idx - 1][i](
                    x=tensor_container[idx][i],
                    skipx=torch.cat([tensor_container[lv][lv - (idx - i - 1)] for lv in range(idx - i - 1, idx)], dim=1),
                )
            outputs.append(self.out_tr[idx - 1](tensor_container[idx][-1]))
        self.train(original_mode)
        return outputs[-1]

    def __check_config_validity(self) -> None:
        """Check the validity of the configuration."""
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
