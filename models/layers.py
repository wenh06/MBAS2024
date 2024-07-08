"""
Common layers for 3D volumetric V-Net models.
"""

from typing import Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch_ecg.models._nets import get_activation
from torch_ecg.utils import SizeMixin


class LUConv(nn.Sequential, SizeMixin):
    """Convolutional block with activation and normalization.

    Parameters
    ----------
    nchan: int
        Number of input and output channels.
    kernel_size: int
        Kernel size of the convolutional layer.
    activation: str
        Activation function name, e.g.
        "relu", "prelu", "elu", "leaky_relu", "mish", "swish", etc.

    """

    def __init__(self, nchan: int, kernel_size: int, activation: str, ordering: str = "cna") -> None:
        super().__init__()
        kw_act = dict()
        if activation.lower() == "elu":
            kw_act["inplace"] = True
        elif activation.lower() == "prelu":
            kw_act["num_parameters"] = nchan
        if ordering.lower() == "cna":
            self.add_module("conv", nn.Conv3d(nchan, nchan, kernel_size=kernel_size, padding="same"))
            self.add_module("bn", nn.BatchNorm3d(nchan))
            self.add_module("act", get_activation(activation, kw_act))
        elif ordering.lower() == "can":
            self.add_module("conv", nn.Conv3d(nchan, nchan, kernel_size=kernel_size, padding="same"))
            self.add_module("act", get_activation(activation, kw_act))
            self.add_module("bn", nn.BatchNorm3d(nchan))
        else:
            raise ValueError(f"Ordering \042{ordering}\042 is not supported yet.")


def make_convs(nchan: int, depth: int, kernel_size: int, activation: str, ordering: str = "cna") -> nn.Sequential:
    """Create a sequence of n convolutional layers.

    Parameters
    ----------
    nchan: int
        Number of input and output channels.
    depth: int
        Number of convolutional layers.
    kernel_size: int
        Kernel size of the convolutional layer.
    activation: str
        Activation function name, e.g.
        "relu", "prelu", "elu", "leaky_relu", "mish", "swish", etc.
    ordering: str, default "cna"
        Convolutional block ordering, e.g. "cna" or "can".

    Returns
    -------
    nn.Sequential
        A sequence of n convolutional layers.

    """
    return nn.Sequential(*[LUConv(nchan, kernel_size, activation, ordering) for _ in range(depth)])


class InputConcat(nn.Module):
    """Concatenate input tensor with itself along the channel dimension.

    Parameters
    ----------
    nchan: int
        Number of copies to concatenate.

    """

    def __init__(self, nchan: int) -> None:
        super().__init__()
        self.nchan = nchan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x for _ in range(self.nchan)], 1)


class InputTransition(nn.Module, SizeMixin):
    """Input transition block.

    The operation of this block is `conv(x) + x (out_chans copies along the channel dimension)`.

    Parameters
    ----------
    out_chans: int
        Number of output channels.
    kernel_size: int
        Kernel size of the convolutional layer.
    activation: str
        Activation function name, e.g.
        "relu", "prelu", "elu", "leaky_relu", "mish", "swish", etc.

    """

    def __init__(self, out_chans: int, kernel_size: int, activation: str) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, out_chans, kernel_size=kernel_size, padding="same")
        self.norm = nn.BatchNorm3d(out_chans)
        kw_act = dict()
        if activation.lower() == "elu":
            kw_act["inplace"] = True
        elif activation.lower() == "prelu":
            kw_act["num_parameters"] = out_chans
        self.act = get_activation(activation, kw_act)
        self.concat = InputConcat(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(torch.add(self.norm(self.conv(x)), self.concat(x)))
        return out


class DownTransition(nn.Module, SizeMixin):
    """Down transition block.

    Parameters
    ----------
    in_chans: int
        Number of input channels.
    num_convs: int
        Number of convolutional layers.
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
        kernel_size: int,
        activation: str,
        dropout: Union[bool, float] = False,
    ) -> None:
        super().__init__()
        # out_chans = 2*in_chans
        self.down_conv = nn.Conv3d(in_chans, out_chans, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm3d(out_chans)
        if isinstance(dropout, bool) and dropout:
            self.dropout = nn.Dropout3d()
        elif isinstance(dropout, float) and dropout > 0:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = nn.Identity()
        kw_act = dict()
        if activation.lower() == "elu":
            kw_act["inplace"] = True
        elif activation.lower() == "prelu":
            kw_act["num_parameters"] = out_chans
        self.act1 = get_activation(activation, kw_act)
        self.act2 = get_activation(activation, kw_act)
        self.ops = make_convs(out_chans, num_convs, kernel_size, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.act1(self.norm(self.down_conv(x)))
        out = self.ops(self.dropout(down))
        out = self.act2(torch.add(out, down))
        return out


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
        self.dropout2 = nn.Dropout3d()
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
        out = self.dropout2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Sequential, SizeMixin):
    """Output transition block.

    Parameters
    ----------
    in_chans: int
        Number of input channels.
    out_chans: int
        Number of output channels.
    kernel_size: int
        Kernel size of the convolutional layer.
    activation: str
        Activation function name, e.g.
        "relu", "prelu", "elu", "leaky_relu", "mish", "swish", etc.

    """

    def __init__(self, in_chans: int, out_chans: int, kernel_size: int, activation: str) -> None:
        super().__init__()
        self.add_module("conv1", nn.Conv3d(in_chans, out_chans, kernel_size=kernel_size, padding="same"))
        self.add_module("norm", nn.BatchNorm3d(out_chans))
        kw_act = dict()
        if activation.lower() == "elu":
            kw_act["inplace"] = True
        elif activation.lower() == "prelu":
            kw_act["num_parameters"] = out_chans
        self.add_module("act", get_activation(activation, kw_act))
        self.add_module("conv2", nn.Conv3d(out_chans, out_chans, kernel_size=1))
        self.add_module("permute", Rearrange("b c h w d -> b h w d c"))
        # self.add_module("flatten", Rearrange("b h w d c -> (b h w d) c"))
        # self.add_module("softmax", nn.Softmax(dim=-1))
