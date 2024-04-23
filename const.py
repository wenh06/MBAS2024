"""Constants for the project."""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

__all__ = [
    "INPUT_IMAGE_TYPES",
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
]


INPUT_IMAGE_TYPES = Union[
    torch.Tensor,
    List[torch.Tensor],
    np.ndarray,
    List[np.ndarray],
]


MODEL_CACHE_DIR = str(
    Path(
        # "~/.cache/revenger_model_dir_cinc2024"
        os.environ.get("MODEL_CACHE_DIR", "~/.cache/mbas2024/revenger_model_dir")
    )
    .expanduser()
    .resolve()
)
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


DATA_CACHE_DIR = str(
    Path(
        # "~/.cache/revenger_data_dir_cinc2024"
        os.environ.get("DATA_CACHE_DIR", "~/.cache/mbas2024/revenger_data_dir")
    )
    .expanduser()
    .resolve()
)
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
