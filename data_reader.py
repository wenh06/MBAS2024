from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_ecg.databases.base import DataBaseInfo, _DataBase
from torch_ecg.utils.download import http_get
from torch_ecg.utils.misc import add_docstring, get_record_list_recursive3

if TYPE_CHECKING:
    from matplotlib.colors import ListedColormap

__all__ = ["MBAS2024"]


# ---------------------------------------------------------------------------
# Shared visualisation helpers (notebook-friendly)
# ---------------------------------------------------------------------------


def _is_notebook() -> bool:
    """Return True when running inside a Jupyter notebook / IPython kernel."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return True
    except Exception:
        pass
    return False


def _build_seg_cmap(palette: Dict[int, str], n_classes: int) -> ListedColormap:
    """Build a discrete colormap from a class-id→colour palette."""
    from matplotlib.colors import ListedColormap

    colors = [palette.get(i, (0, 0, 0, 0)) for i in range(n_classes)]
    return ListedColormap(colors)


def _slice_view_interactive(
    image: np.ndarray,
    masks: Optional[Dict[int, np.ndarray]] = None,
    palette: Optional[Dict[int, str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """Interactive single-panel slice viewer with checkboxes and legend.

    In Jupyter notebooks, displays:
    - An integer slider to scrub through z-slices.
    - One checkbox per label class to toggle contour overlay.
    - A colour legend.

    Parameters
    ----------
    image : (H, W, D) float32 array
    masks : dict of ``class_id → (H, W, D) uint8 array``, optional
    palette : dict of ``class_id → colour``, optional
    class_names : dict of ``class_id → str``, optional
        Human-readable names for the legend and checkbox labels.
    title : str
    figsize : (int, int)
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from IPython.display import display
    from ipywidgets import Checkbox, Dropdown, HBox, IntSlider, Output, VBox, interactive_output

    if palette is None:
        palette = {}
    if class_names is None:
        class_names = {}
    if masks is None:
        masks = {}

    n_slices = image.shape[-1]
    mid = n_slices // 2
    mask_ids = sorted(masks.keys())

    # -- widgets ---------------------------------------------------------------
    slider = IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice")
    overlay_dd = Dropdown(
        options=["contour", "filled", "filled+hatch"],
        value="filled+hatch",
        description="Overlay:",
    )
    show_cbs: Dict[int, Checkbox] = {}
    for cls_id in mask_ids:
        label = class_names.get(cls_id, f"Class {cls_id}")
        show_cbs[cls_id] = Checkbox(value=True, description=label, indent=False)

    out = Output()

    # -- plot function ---------------------------------------------------------
    def _plot(slice_idx: int, overlay_mode: str, **show: bool) -> None:
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(image[..., slice_idx], cmap="gray", origin="lower")

            is_filled = overlay_mode.startswith("filled")
            use_hatch = overlay_mode == "filled+hatch"

            legend_handles = []
            for cls_id in mask_ids:
                if not show.get(str(cls_id), True):
                    continue
                mask = masks[cls_id]
                if mask.max() == 0:
                    continue
                mask_slice = mask[..., slice_idx]
                color = palette.get(cls_id, "white")

                if is_filled:
                    ax.contourf(
                        mask_slice,
                        levels=[0.5, 1],
                        colors=[color],
                        alpha=0.25,
                        antialiased=True,
                        hatches=["//"] if use_hatch else [],
                    )
                    legend_handle = mpatches.Patch(
                        facecolor=color,
                        alpha=0.5,
                        edgecolor=color,
                        label=class_names.get(cls_id, f"Class {cls_id}"),
                    )
                else:
                    ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1.5)
                    legend_handle = mpatches.Patch(
                        color=color,
                        label=class_names.get(cls_id, f"Class {cls_id}"),
                    )

                legend_handles.append(legend_handle)

            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", framealpha=0.7, fontsize="small")

            ax.set_title(f"{title}  (slice {slice_idx + 1}/{n_slices})")
            ax.axis("off")
            fig.tight_layout()
            plt.show()

    # -- wire widgets ----------------------------------------------------------
    controls: Dict = {"slice_idx": slider, "overlay_mode": overlay_dd}
    controls.update({str(cls_id): cb for cls_id, cb in show_cbs.items()})

    checkbox_row = HBox(list(show_cbs.values()))
    ui = VBox([slider, overlay_dd, checkbox_row, out])
    display(ui)

    # Hold a reference so the widget isn't garbage-collected
    _plot._widget = interactive_output(_plot, controls)


def _slice_view_static(
    image: np.ndarray,
    masks: Optional[Dict[int, np.ndarray]] = None,
    palette: Optional[Dict[int, str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    channels: Optional[List[int]] = None,
    title: str = "",
    overlay_mode: str = "contour",
    max_cols: int = 4,
) -> None:
    """Static multi-slice grid view (fallback when not in a notebook).

    Parameters
    ----------
    overlay_mode : str, default "contour"
        ``"contour"``, ``"filled"``, or ``"filled+hatch"``.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n_slices = image.shape[-1]
    if channels is None:
        channels = list(range(n_slices))
    if palette is None:
        palette = {}
    if class_names is None:
        class_names = {}
    if masks is None:
        masks = {}

    mask_ids = sorted(masks.keys())
    is_filled = overlay_mode.startswith("filled")
    use_hatch = overlay_mode == "filled+hatch"

    n = len(channels)
    n_rows = int(np.ceil(n / max_cols))
    n_cols = min(max_cols, n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for ax_idx, sl in enumerate(channels):
        axes_flat[ax_idx].set_axis_off()
        axes_flat[ax_idx].imshow(image[..., sl], cmap="gray", origin="lower")
        axes_flat[ax_idx].set_title(f"Slice {sl}")
        for cls_id in mask_ids:
            mask = masks[cls_id]
            if mask.max() == 0:
                continue
            mask_slice = mask[..., sl]
            color = palette.get(cls_id, "white")
            if is_filled:
                axes_flat[ax_idx].contourf(
                    mask_slice,
                    levels=[0.5, 1],
                    colors=[color],
                    alpha=0.25,
                    antialiased=True,
                    hatches=["//"] if use_hatch else [],
                )
            else:
                axes_flat[ax_idx].contour(mask_slice, levels=[0.5], colors=[color], linewidths=1)

    # Shared legend
    legend_handles = []
    for cls_id in mask_ids:
        if masks[cls_id].max() > 0:
            color = palette.get(cls_id, "white")
            legend_handles.append(
                mpatches.Patch(
                    facecolor=color if is_filled else "none",
                    alpha=0.5 if is_filled else 1.0,
                    edgecolor=color,
                    label=class_names.get(cls_id, f"Class {cls_id}"),
                )
            )
    if legend_handles:
        axes_flat[min(n - 1, len(axes_flat) - 1)].legend(
            handles=legend_handles, loc="upper right", framealpha=0.7, fontsize="small"
        )

    for ax_idx in range(n, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)
    fig.suptitle(title)
    plt.show()


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
        1: "left & right atrial walls",
        2: "right atrium",
        3: "left atrium",
    }
    __label2id__ = {v: k for k, v in __class_map__.items()}
    __id2label__ = {k: v for k, v in __class_map__.items()}
    __palette__ = {
        0: (0, 0, 0, 0),  # "background" (transparent)
        1: "yellow",  # "left & right atrial walls"
        2: "red",  # "right atrium"
        3: "blue",  # "left atrium"
    }
    __default_crop_pad__ = [7, 7, 3]

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

        self._data_file_pattern = "(?P<record>MBAS_\\d{3})_gt\\.nii\\.gz"
        self._ann_file_pattern = "(?P<record>MBAS_\\d{3})_label\\.nii\\.gz"
        self._df_records_all = None
        self._ls_rec()

    def _ls_rec(self):
        """Find all records in the database."""
        self._df_records = pd.DataFrame(
            get_record_list_recursive3(self.db_dir, self._data_file_pattern, relative=False, with_suffix=True),
            columns=["path"],
        )
        self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))
        # self._df_records["record"] = self._df_records["path"].apply(lambda x: x.parents[0].name)
        self._df_records["record"] = self._df_records["path"].apply(
            lambda x: re.search(self._data_file_pattern, x.name).group("record")
        )
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

        self._df_records_all = self._df_records.copy()

        # exclude the validation set in which the records are not annotated
        self._df_records = self._df_records[self._df_records["subset"] == "Training"]

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
        return self._df_records_all.loc[rec, "path"]

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

    def load_data(self, rec: Union[str, int], output_shape: Optional[Sequence[int]] = None) -> np.ndarray:
        """Load the 3D LGE-MRI of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        output_shape : Sequence[int], optional
            The output shape of the 3D LGE-MRI data.
            If None, the original shape is returned.

        Returns
        -------
        numpy.ndarray
            The 3D LGE-MRI data.

        """
        if output_shape is None:
            return nib.load(str(self.get_data_path(rec))).get_fdata()
        return self.resample_data(nib.load(str(self.get_data_path(rec))).get_fdata(), output_shape)

    def load_ann(self, rec: Union[str, int], output_shape: Optional[Sequence[int]] = None) -> np.ndarray:
        """Load the segmentation annotation of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        output_shape : Sequence[int], optional
            The output shape of the segmentation annotation.
            If None, the original shape is returned.

        Returns
        -------
        numpy.ndarray
            The segmentation annotation,
            typically a 3D mask of the same shape as the LGE-MRI.

        """
        if output_shape is None:
            return nib.load(str(self.get_ann_path(rec))).get_fdata()
        return self.resample_data(nib.load(str(self.get_ann_path(rec))).get_fdata(), output_shape)

    def load_ann_box(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        ann_shape: Optional[Sequence[int]] = None,
        ann_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Load the bounding box of the segmentation annotation of a record.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        pad : int or Sequence[int], optional
            Padding around the bounding box, in each dimension.
            Defaults to self.__default_crop_pad__.
        ann_shape : Sequence[int], optional
            The shape of the segmentation annotation.
            If None, the original shape is used.
        ann_mask : numpy.ndarray, optional
            The pre-loaded segmentation annotation.
            If None, it will be loaded.

        Returns
        -------
        numpy.ndarray
            The bounding box of the segmentation annotation,
            of shape [[x_min, x_max], [y_min, y_max], [z_min, z_max]].

        """
        if ann_mask is None:
            ann_mask = self.load_ann(rec, output_shape=ann_shape)
        elif ann_shape is not None and ann_mask.shape != ann_shape:
            ann_mask = self.resample_data(ann_mask, ann_shape)
        x, y, z = np.where(ann_mask != self.__label2id__["background"])
        if pad is None:
            pad = self.__default_crop_pad__
        if isinstance(pad, int):
            pad = [pad] * 3
        x_min, x_max = max(0, x.min() - pad[0]), min(ann_mask.shape[0], x.max() + pad[0])
        y_min, y_max = max(0, y.min() - pad[1]), min(ann_mask.shape[1], y.max() + pad[1])
        z_min, z_max = max(0, z.min() - pad[2]), min(ann_mask.shape[2], z.max() + pad[2])
        return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    def load_data_cropped(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the 3D LGE-MRI of a record cropped by the bounding box of the segmentation annotation.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        pad : int or Sequence[int], optional
            Padding around the bounding box, in each dimension.
            Defaults to self.__default_crop_pad__.
        output_shape : Sequence[int], optional
            The output shape of the 3D LGE-MRI data.
            If None, the original shape is returned.

        Returns
        -------
        numpy.ndarray
            The cropped 3D LGE-MRI data.

        """
        data = self.load_data(rec)
        box = self.load_ann_box(rec, pad=pad if pad is not None else self.__default_crop_pad__)
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = box
        if output_shape is None:
            return data[x_min:x_max, y_min:y_max, z_min:z_max]
        return self.resample_data(data[x_min:x_max, y_min:y_max, z_min:z_max], output_shape)

    def load_ann_cropped(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the segmentation annotation of a record cropped by the bounding box of the segmentation annotation.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        pad : int or Sequence[int], optional
            Padding around the bounding box, in each dimension.
            Defaults to self.__default_crop_pad__.
        output_shape : Sequence[int], optional
            The output shape of the segmentation annotation.
            If None, the original shape is returned.

        Returns
        -------
        numpy.ndarray
            The cropped segmentation annotation,
            typically a 3D mask of the same shape as the LGE-MRI.

        """
        ann_mask = self.load_ann(rec)
        box = self.load_ann_box(rec, pad=pad if pad is not None else self.__default_crop_pad__)
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = box
        if output_shape is None:
            return ann_mask[x_min:x_max, y_min:y_max, z_min:z_max]
        return self.resample_data(ann_mask[x_min:x_max, y_min:y_max, z_min:z_max], output_shape)

    def view_data(
        self,
        rec: Union[str, int],
        channels: Optional[Union[Sequence[int], int]] = None,
        with_ann: bool = True,
        orthoview: bool = False,
        output_shape: Optional[Sequence[int]] = None,
        crop: bool = False,
        crop_pad: Optional[Union[int, Sequence[int]]] = None,
        data: Optional[np.ndarray] = None,
        interactive: Optional[bool] = None,
        overlay_mode: str = "filled+hatch",
    ) -> None:
        """View the 3D LGE-MRI of a record.

        In Jupyter notebooks the default is an interactive slider-based view
        with per-class checkboxes and a colour legend.  Outside notebooks the
        default is a static grid of all (or selected) slices.

        Parameters
        ----------
        rec : str or int
            The record name or index in self._all_records.
        channels : int or Sequence[int], optional
            The channel(s) to view (static mode only). If None, view all channels.
        with_ann : bool, default True
            Whether to overlay the segmentation annotation on the MRI.
        orthoview : bool, default False
            Whether to view the MRI in orthogonal view.
        output_shape : Sequence[int], optional
            The output shape of the 3D LGE-MRI data.
            If None, the original shape is returned.
        crop : bool, default False
            Whether to crop the MRI by the bounding box of the segmentation
            annotation.
        crop_pad : int or Sequence[int], optional
            Padding around the bounding box, in each dimension.
            Defaults to self.__default_crop_pad__.
        data : numpy.ndarray, optional
            The pre-loaded 3D LGE-MRI data.
            If None, it will be loaded.
        interactive : bool, optional
            Force interactive (``True``) or static (``False``) mode.
            Defaults to auto-detection based on the runtime environment.
        overlay_mode : str, default ``"filled+hatch"``
            Overlay style.  In interactive mode this is controlled by a
            dropdown; this value sets the initial selection.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        if data is not None:
            pass
        elif crop:
            data = self.load_data_cropped(rec, pad=crop_pad, output_shape=output_shape)
        else:
            data = self.load_data(rec, output_shape=output_shape)
        if orthoview:
            data.orthoview()
            return

        # Build per-class binary masks for contour overlay
        masks: Dict[int, np.ndarray] = {}
        title = f"LGE-MRI — {rec}"
        if with_ann:
            seg_ann = (
                self.load_ann_cropped(rec, pad=crop_pad, output_shape=output_shape)
                if crop
                else self.load_ann(rec, output_shape=output_shape)
            )
            for cls_id in sorted(self.__class_map__):
                if cls_id == 0:
                    continue
                binary = (seg_ann == cls_id).astype(np.uint8)
                if binary.max() > 0:
                    masks[cls_id] = binary
            title += " + annotation"
        else:
            seg_ann = np.zeros(data.shape[:3], dtype=np.uint8)

        # Choose interactive vs static
        if interactive is None:
            interactive = _is_notebook()

        if interactive:
            _slice_view_interactive(data, masks, self.__palette__, self.__id2label__, title=title)
        else:
            _slice_view_static(
                data, masks, self.__palette__, self.__id2label__, channels=channels, title=title, overlay_mode=overlay_mode
            )

    @property
    def database_info(self) -> DataBaseInfo:
        return _MBAS2024_INFO

    @property
    def webpage(self) -> str:
        return "https://codalab.lisn.upsaclay.fr/competitions/18516"

    @property
    def url(self) -> str:
        return "https://drive.google.com/u/0/uc?id=1QbeGGrrTmKi4220BbJTA7FoC-gmHgwhc"

    def download(self) -> None:
        """Download the database."""
        http_get(self.url, self.db_dir, filename="MBAS2024.zip", extract=True)

    @property
    def validation_set(self) -> List[str]:
        return self._df_records_all[self._df_records_all["subset"] == "Validation"].index.tolist()

    @staticmethod
    def resample_data(data: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Resample 3D LGE-MRI data.

        Parameters
        ----------
        data : numpy.ndarray
            The 3D LGE-MRI data or the segmentation annotation (mask).
        zoom_factor : float or Sequence[float]
            The zoom factor for each dimension.

        Returns
        -------
        numpy.ndarray
            The resampled 3D LGE-MRI data.

        """
        dtype = data.dtype
        rsmp_data = (
            F.interpolate(torch.from_numpy(data).unsqueeze(0).unsqueeze(0), size=shape, mode="trilinear", align_corners=True)
            .squeeze()
            .numpy()
        )
        # if is of integer type, round to the nearest integer
        if np.issubdtype(dtype, np.integer):
            rsmp_data = np.round(rsmp_data).astype(dtype)
        return rsmp_data
