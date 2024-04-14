import os
from pathlib import Path
from typing import Any, Optional, Union

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
    1.
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

    def view_data(self, rec: Union[str, int], orthoview: bool = False) -> None:
        """View the 3D LGE-MRI of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        orthoview : bool, default False
            Whether to view the MRI in orthogonal view.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        if orthoview:
            raise NotImplementedError
        raise NotImplementedError

    @property
    def database_info(self) -> DataBaseInfo:
        return _MBAS2024_INFO

    @property
    def url(self) -> str:
        return "https://codalab.lisn.upsaclay.fr/competitions/18516"
