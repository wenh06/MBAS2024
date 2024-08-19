"""
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin
from torch_ecg.utils.utils_nn import default_collate_fn  # noqa: F401
from tqdm.auto import tqdm

from cfg import TrainCfg
from data_reader import MBAS2024
from utils.mclahe_tf import mclahe

__all__ = [
    "MBAS2024Dataset",
]


class MBAS2024Dataset(Dataset, ReprMixin):
    """Dataset for the MBAS2024 Challenge.

    Parameters
    ----------
    stage : {0, 1}
        Stage 0 or 1 of the pipeline.
        0 for raw localization, 1 for fine segmentation.
    config : CFG, optional
        Configuration for the dataset.
    reader_kwargs : Dict
        Keyword arguments for the data reader.

    """

    def __init__(self, stage: int, config: Optional[CFG] = None, training: bool = True, **reader_kwargs) -> None:
        super().__init__()
        self.config = CFG(deepcopy(TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.stage = stage
        self.training = training

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
            assert self.config.db_dir is not None, "db_dir must be specified"
        else:
            reader_kwargs.pop("db_dir", None)
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()
        self.reader = MBAS2024(db_dir=self.config.db_dir, **reader_kwargs)

        if self.config.use_tio_transforms and self.training:
            aug_prob = self.config.get("aug_prob", 0.5)
            augmentation = tio.Compose(
                [
                    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, image_interpolation="linear", p=aug_prob),
                    tio.RandomElasticDeformation(max_displacement=(10, 10, 10), p=aug_prob),
                    tio.RandomNoise(std=(0, 0.1), p=aug_prob),
                    tio.RandomBiasField(coefficients=(0, 0.1), p=aug_prob),
                    tio.RandomBlur(std=(0, 0.5), p=aug_prob),
                    tio.RandomGamma(log_gamma=(-0.3, 0.3), p=aug_prob),
                    tio.RandomMotion(degrees=10, p=aug_prob),
                    tio.RandomSpike(num_spikes=(0, 10), p=aug_prob),
                    tio.RandomGhosting(num_ghosts=(0, 10), p=aug_prob),
                ],
                p=0.8,
            )
            self.transform = tio.Compose(
                [
                    augmentation,
                    tio.ZNormalization(),
                ],
                p=1,
            )
            raise NotImplementedError("torchio transforms are not supported yet")
        else:
            self.transform = None

        self.__cache = {}  # "coarse_data", "coarse_mask", "fine_data", "fine_mask"
        self.__load_all_data()

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: Union[int, Sequence[int], slice]) -> Dict[str, np.ndarray]:
        if len(self.__cache) == 0:
            self.__load_all_data()
        if self.stage == 0:
            return {
                "image": self.__cache["coarse_data"][idx],
                "mask": self.__cache["coarse_mask"][idx],
            }
        elif self.stage == 1:
            return {
                "image": self.__cache["fine_data"][idx],
                "mask": self.__cache["fine_mask"][idx],
            }
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def __load_all_data(self):
        """Load all data and cache them."""
        # self.config.data_shape = (576, 576, 48)
        # self.config.coarse_shape = (144, 144, 48)
        # self.config.fine_shape = (256, 256, 48)
        # the extra dimension (dimension 1) is for the channel
        self.__cache["coarse_data"] = np.zeros((len(self.reader), 1, *self.config.coarse_shape), dtype=np.float32)
        self.__cache["coarse_mask"] = np.zeros((len(self.reader), *self.config.coarse_shape), dtype=np.float32)
        self.__cache["fine_data"] = np.zeros((len(self.reader), 1, *self.config.fine_shape), dtype=np.float32)
        self.__cache["fine_mask"] = np.zeros((len(self.reader), *self.config.fine_shape), dtype=np.float32)
        with tqdm(range(len(self.reader)), desc="Loading data", unit="image", miniters=1, dynamic_ncols=True) as pbar:
            for idx in pbar:
                # load data and annotation
                data = self.reader.load_data(idx)
                mask = self.reader.load_ann(idx)
                (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.reader.load_ann_box(idx)

                if self.config.apply_mclahe:
                    data = mclahe(data)

                if self.transform is not None:
                    # data, mask = self.transform(data, mask)
                    raise NotImplementedError("torchio transforms are not supported yet")
                else:
                    # normalize the data
                    data = (data - np.mean(data)) / (np.std(data) + 1e-8)  # avoid division by zero

                # adjust the shape of the data
                x_size, y_size, z_size = data.shape
                # the training data has shape (576, 576, 44) or (640, 640, 44)
                if x_size > self.config.data_shape[0]:
                    x_crop = (x_size - self.config.data_shape[0]) // 2
                else:
                    x_crop = 0
                if y_size > self.config.data_shape[1]:
                    y_crop = (y_size - self.config.data_shape[1]) // 2
                else:
                    y_crop = 0
                if z_size < self.config.data_shape[2]:
                    z_pad = (self.config.data_shape[2] - z_size) // 2
                    data = np.pad(
                        data,
                        ((0, 0), (0, 0), (z_pad, self.config.data_shape[2] - z_size - z_pad)),
                        mode="constant",
                        constant_values=0,
                    )
                    mask = np.pad(
                        mask,
                        ((0, 0), (0, 0), (z_pad, self.config.data_shape[2] - z_size - z_pad)),
                        mode="constant",
                        constant_values=0,
                    )
                    z_min += z_pad
                    z_max += z_pad
                data = data[x_crop : x_crop + self.config.data_shape[0], y_crop : y_crop + self.config.data_shape[1], :]
                mask = mask[x_crop : x_crop + self.config.data_shape[0], y_crop : y_crop + self.config.data_shape[1], :]
                x_min = max(x_min - x_crop, 0)
                x_max = min(x_max - x_crop, self.config.data_shape[0])
                y_min = max(y_min - y_crop, 0)
                y_max = min(y_max - y_crop, self.config.data_shape[1])

                # stage 0 data: raw localization
                # down-sample the data to from (576, 576, 48) to (144, 144, 48)
                step = 4
                self.__cache["coarse_data"][idx, 0] = data[::step, ::step, :]
                # the coarse mask is to make the cube defined by the bounding box have value 1
                # self.config.coarse_pad = 2
                x_min_coarse = max(0, x_min // step - self.config.coarse_pad)
                x_max_coarse = min(self.config.coarse_shape[0], x_max // step + self.config.coarse_pad)
                y_min_coarse = max(0, y_min // step - self.config.coarse_pad)
                y_max_coarse = min(self.config.coarse_shape[1], y_max // step + self.config.coarse_pad)
                self.__cache["coarse_mask"][idx, x_min_coarse:x_max_coarse, y_min_coarse:y_max_coarse, :] = 1

                # stage 1 data: fine segmentation
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                x_min_fine = max(0, x_center - self.config.fine_shape[0] // 2)
                x_max_fine = min(self.config.data_shape[0], x_center + self.config.fine_shape[0] // 2)
                y_min_fine = max(0, y_center - self.config.fine_shape[1] // 2)
                y_max_fine = min(self.config.data_shape[1], y_center + self.config.fine_shape[1] // 2)
                self.__cache["fine_data"][idx, 0] = data[x_min_fine:x_max_fine, y_min_fine:y_max_fine, :]
                self.__cache["fine_mask"][idx] = mask[x_min_fine:x_max_fine, y_min_fine:y_max_fine, :]

        # if using the binary cross-entropy (BCE) loss or loss function based on BCE loss
        # the mask should be binary
        self.__cache["coarse_mask"] = F.one_hot(
            torch.tensor(self.__cache["coarse_mask"], dtype=torch.int64), num_classes=2
        ).numpy()
        self.__cache["fine_mask"] = F.one_hot(torch.tensor(self.__cache["fine_mask"], dtype=torch.int64), num_classes=4).numpy()

    @property
    def cache(self) -> Dict[str, np.ndarray]:
        return self.__cache
