""" """

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "MBAS2024Outputs",
]


@dataclass
class MBAS2024Outputs:
    """Output class for MBAS2024."""

    pred_mask: Optional[Sequence[np.ndarray]] = None

    def __post_init__(self) -> None:
        pass
