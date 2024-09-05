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
    "REMOTE_MODELS",
]


INPUT_IMAGE_TYPES = Union[
    torch.Tensor,
    List[torch.Tensor],
    np.ndarray,
    List[np.ndarray],
]


MODEL_CACHE_DIR = str(Path(os.environ.get("MODEL_CACHE_DIR", "~/.cache/mbas2024/revenger_model_dir")).expanduser().resolve())
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


DATA_CACHE_DIR = str(Path(os.environ.get("DATA_CACHE_DIR", "~/.cache/mbas2024/revenger_data_dir")).expanduser().resolve())
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


REMOTE_MODELS = {
    "vnet-v2": {
        "url": {
            "google-drive": "https://drive.google.com/u/0/uc?id=1pAFU6OTY-j4TEq--cQkQmeDjwAKQjN4p",
            "deep-psp": None,
        },
        "filename": "vnet-v2.zip",
    },
    "vnet-v1": {
        "url": {
            "google-drive": None,
            "deep-psp": None,
        },
        "filename": "vnet-v1.zip",
    },
    "nested-vnet-v3": {
        "url": {
            "google-drive": None,
            "deep-psp": None,
        },
        "filename": "nested-vnet-v3.zip",
    },
}
