import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import nibabel as nib
import numpy as np
import pandas as pd
from torch_ecg.databases.base import DataBaseInfo, _DataBase
from torch_ecg.utils.misc import add_docstring, get_record_list_recursive3

__all__ = ["MBAS2024"]


_MBAS2024_INFO = DataBaseInfo(
    title="""
    MICCAI MBAS 2024 - Multi-class Bi-Atrial Segmentation Challenge
    """,
    about="""
    1. The dataset consists of 3D LGE-MRI scans of the left atrium and right atrium, and their walls.
    2. The dataset is divided into 3 subsets: training (70 samples), validation (30 samples), and testing (100 samples, hidden).
    3. The data as well as the segmentation annotations are in NIfTI format.
    4. The segmentation annotations are multi-class, with 4 classes: right atrium (1), left atrium (2), left & right atrial walls (3), and background (0).
    """,
    usage=[
        "Atrium Segmentation",
    ],
    references=[
        "https://codalab.lisn.upsaclay.fr/competitions/18516",
        "https://www.sciencedirect.com/science/article/abs/pii/S1361841520301961",
    ],
    doi=[
        "10.1016/j.media.2020.101832",
    ],
)


@add_docstring(_MBAS2024_INFO.format_database_docstring(), mode="prepend")
class MBAS2024(_DataBase):
    """
    Parameters
    ----------
    db_dir : `path-like`, optional
        Storage path of the database.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments

    """

    __class_map__ = {
        0: "background",
        1: "right atrium",
        2: "left atrium",
        3: "left & right atrial walls",
    }
    __label2id__ = {v: k for k, v in __class_map__.items()}
    __id2label__ = {k: v for k, v in __class_map__.items()}
    __palette__ = {
        0: (0, 0, 0, 0),  # "background" (transparent)
        1: "red",  # "right atrium"
        2: "blue",  # "left atrium"
        3: "yellow",  # "left & right atrial walls"
    }

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="MBAS2024",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "nii.gz"
        self.ann_ext = "nii.gz"

        self._data_file_pattern = "MBAS_\\d{3}_gt\\.nii\\.gz"
        self._ann_file_pattern = "MBAS_\\d{3}_label\\.nii\\.gz"
        self._ls_rec()

    def _ls_rec(self):
        """Find all records in the database."""
        self._df_records = pd.DataFrame(
            get_record_list_recursive3(self.db_dir, self._data_file_pattern, relative=False, with_suffix=True),
            columns=["path"],
        )
        self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))
        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.parents[0].name)
        self._df_records["subset"] = self._df_records["path"].apply(lambda x: x.parents[1].name)
        self._df_records["ann_path"] = self._df_records["path"].apply(lambda x: Path(str(x).replace("gt", "label")))
        self._df_records["shape"] = self._df_records["path"].apply(lambda x: nib.load(str(x)).header.get_data_shape())
        self._df_records["channels"] = self._df_records["shape"].apply(lambda x: x[-1])
        self._df_records["best_affine"] = self._df_records["path"].apply(lambda x: nib.load(str(x)).header.get_best_affine())
        self._df_records["base_affine"] = self._df_records["path"].apply(lambda x: nib.load(str(x)).header.get_base_affine())
        self._df_records["qform"] = self._df_records["path"].apply(lambda x: nib.load(str(x)).header.get_qform())
        self._df_records["qform_quaternion"] = self._df_records["path"].apply(
            lambda x: nib.load(str(x)).header.get_qform_quaternion()
        )
        self._df_records["sform"] = self._df_records["path"].apply(lambda x: nib.load(str(x)).header.get_sform())
        self._df_records["zooms"] = self._df_records["path"].apply(lambda x: nib.load(str(x)).header.get_zooms())
        self._df_records.set_index("record", inplace=True)

        if self._df_records.empty:
            self.logger.warning("No records found in the database directory")
            # fmt: off
            self._df_records = pd.DataFrame(
                columns=[
                    "path", "record", "subset", "ann_path", "shape",
                    "best_affine", "base_affine", "qform", "qform_quaternion", "sform", "zooms"
                ]
            )
            # fmt: on

        self._all_records = self._df_records.index.unique().tolist()

    def get_data_path(self, rec: Union[str, int]) -> Path:
        """Get the path of the 3D LGE-MRI of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.

        Returns
        -------
        Path
            The path of the 3D LGE-MRI data.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        return self._df_records.loc[rec, "path"]

    def get_ann_path(self, rec: Union[str, int]) -> Path:
        """Get the path of the segmentation annotation of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.

        Returns
        -------
        Path
            The path of the segmentation annotation.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        return self._df_records.loc[rec, "ann_path"]

    def load_data(self, rec: Union[str, int]) -> np.ndarray:
        """Load the 3D LGE-MRI of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.

        Returns
        -------
        numpy.ndarray
            The 3D LGE-MRI data.

        """
        return nib.load(str(self.get_data_path(rec))).get_fdata()

    def load_ann(self, rec: Union[str, int]) -> np.ndarray:
        """Load the segmentation annotation of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.

        Returns
        -------
        numpy.ndarray
            The segmentation annotation,
            typically a 3D mask of the same shape as the LGE-MRI.

        """
        return nib.load(str(self.get_ann_path(rec))).get_fdata()

    def view_data(
        self,
        rec: Union[str, int],
        channels: Optional[Union[Sequence[int], int]] = None,
        with_ann: bool = True,
        orthoview: bool = False,
    ) -> None:
        """View the 3D LGE-MRI of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        channels : int or Sequence[int], optional
            The channel(s) to view. If None, view all channels.
            Valid only when orthoview is False.
        with_ann : bool, default True
            Whether to overlay the segmentation annotation on the MRI.
        orthoview : bool, default False
            Whether to view the MRI in orthogonal view.

        """
        if "plt" not in globals():
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        if isinstance(rec, int):
            rec = self._all_records[rec]
        data = self.load_data(rec)
        if orthoview:
            data.orthoview()
            return
        if with_ann:
            seg_ann = self.load_ann(rec)
        else:
            seg_ann = np.full_like(data, self.__label2id__["background"])
        if channels is None:
            channels = list(range(self._df_records.loc[rec, "channels"]))
        elif isinstance(channels, int):
            channels = [channels]
        num_channels = len(channels)

        fig_height = int(np.ceil(num_channels / 4).item()) * 5
        fig, ax = plt.subplots(fig_height // 5, min(4, num_channels), figsize=(20, fig_height))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if num_channels == 1:
            ax = [ax]
        else:
            ax = ax.ravel()
        seg_cmap = ListedColormap([self.__palette__[cls_] for cls_ in self.__class_map__ if cls_ in self.__palette__])
        for ax_idx, chan_idx in enumerate(channels):
            ax[ax_idx].set_axis_off()
            ax[ax_idx].imshow(data[..., chan_idx], cmap="gray")
            ax[ax_idx].set_title(f"Channel {chan_idx}")
            # add annotation
            chan_ann = seg_ann[..., chan_idx]
            # for cls_ in np.unique(chan_ann):
            #     if cls_ == self.__label2id__["background"]:
            #         continue
            #     ax[ax_idx].contour(chan_ann == cls_, colors=self.__palette__[cls_], linewidths=1, hatches=["//"])
            ax[ax_idx].imshow(chan_ann, cmap=seg_cmap, alpha=0.1)

    @property
    def database_info(self) -> DataBaseInfo:
        return _MBAS2024_INFO

    @property
    def url(self) -> str:
        return "https://codalab.lisn.upsaclay.fr/competitions/18516"
