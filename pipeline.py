"""
Inference pipeline for the prediction of the segmentation mask:

- reshape (crop or zero-pad) the input image to a specific shape
- get a downsampled version of the input image
- use the stage-0 model to predict the coarse region
- crop the coarse region from the input image: find the center of the coarse region and crop a region of fixed size around it
- use the stage-1 model to predict the fine region
- transform the predicted fine region back to the original shape

"""

import argparse
import time
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

from cfg import TrainCfg
from data_reader import MBAS2024
from models import MultiHead_MBAS2024
from utils.mclahe_tf import mclahe

__all__ = ["run_pipeline"]


RSMP_STEP = 4


def run_pipeline(
    img: Union[np.ndarray, Sequence[np.ndarray]],
    stage0_model: MultiHead_MBAS2024,
    stage1_model: MultiHead_MBAS2024,
    parallel: bool = False,
    batch_size: int = 1,
) -> Union[List[np.ndarray], np.ndarray]:
    """Run the inference pipeline for the prediction of the segmentation mask
    on a batch of images.

    Parameters
    ----------
    img : numpy.ndarray
        Input LGE-MRI images. The batch dimension should be the first dimension.
    stage0_model : nn.Module
        The stage-0 model for coarse segmentation.
    stage1_model : nn.Module
        The stage-1 model for fine segmentation.
    parallel : bool, default False
        Whether to run the inference in parallel.
        Valid when the input is a sequence of images.
    batch_size : int, default 1
        Batch size for parallel inference.
        Valid when parallel is True and the input is a sequence of images.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Predicted segmentation mask(s).

    """
    assert stage0_model.config.get("apply_mclahe", False) == stage1_model.config.get(
        "apply_mclahe", False
    ), "apply_mclahe should be consistent"
    assert stage0_model.config.stage == 0 and stage1_model.config.stage == 1, "stage should be consistent"

    if isinstance(img, np.ndarray) and img.ndim == 3:
        return _run_pipeline(img, stage0_model, stage1_model)

    pred_masks = []
    if parallel:
        raise NotImplementedError("Parallel inference is not supported yet.")
        with tqdm(total=len(img), desc="Inference", unit="record", dynamic_ncols=True, mininterval=1.0) as pbar:
            for i in range(0, len(img), batch_size):
                pred_masks.extend(_run_pipeline_parallel(img[i : i + batch_size], stage0_model, stage1_model))
                pbar.update(len(img[i : i + batch_size]))
        # for i in tqdm(range(0, len(img), batch_size), desc="Inference", unit="batch", dynamic_ncols=True, mininterval=1.0):
        #     pred_masks.extend(_run_pipeline_parallel(img[i : i + batch_size], stage0_model, stage1_model))
    else:
        for i in tqdm(range(len(img)), desc="Inference", unit="record", dynamic_ncols=True, mininterval=1.0):
            pred_masks.append(_run_pipeline(img[i], stage0_model, stage1_model))

    return pred_masks


def _run_pipeline_parallel(
    img: Sequence[np.ndarray],
    stage0_model: MultiHead_MBAS2024,
    stage1_model: MultiHead_MBAS2024,
) -> List[np.ndarray]:
    """Run the inference pipeline for the prediction of the segmentation mask
    on a batch of images in parallel.

    Parameters
    ----------
    img : Sequence[numpy.ndarray]
        Input LGE-MRI images.
    stage0_model : nn.Module
        The stage-0 model for coarse segmentation.
    stage1_model : nn.Module
        The stage-1 model for fine segmentation.

    Returns
    -------
    List[numpy.ndarray]
        Predicted segmentation masks.

    """
    original_shape = [x.shape for x in img]
    if stage0_model.config.get("apply_mclahe", False):
        img_raw = [mclahe(x, use_gpu=False) for x in img]

    img_raw, shifts = zip(*[reshape_input_image(x) for x in img_raw])

    pred_masks = [np.zeros_like(x, dtype=np.uint8) for x in img_raw]

    # get the downsampled version of the input images
    img_coarse = [x[::RSMP_STEP, ::RSMP_STEP, :] for x in img_raw]

    # stage 0: coarse segmentation
    pred_coarse = stage0_model.inference(img_coarse).pred_mask
    img_fine, x_min, x_max, y_min, y_max = [], [], [], [], []
    for i, pred in enumerate(pred_coarse):
        x_coords, y_coords, z_coords = np.where(pred == 1)
        x_center = x_coords.mean().astype(int) * RSMP_STEP
        y_center = y_coords.mean().astype(int) * RSMP_STEP

        # crop the fine region
        x_min.append(max(0, x_center - TrainCfg.fine_shape[0] // 2))
        x_max.append(x_min[-1] + TrainCfg.fine_shape[0])
        y_min.append(max(0, y_center - TrainCfg.fine_shape[1] // 2))
        y_max.append(y_min[-1] + TrainCfg.fine_shape[1])
        img_fine.append(img_raw[i][x_min[-1] : x_max[-1], y_min[-1] : y_max[-1], :])

    # stage 1: fine segmentation
    stage1_masks = stage1_model.inference(img_fine).pred_mask
    for i, pred in enumerate(stage1_masks):
        pred_masks[i][x_min[i] : x_max[i], y_min[i] : y_max[i], :] = pred

    # transform the predicted masks back to the original shapes
    pred_masks = [
        transform_mask_to_original_shape(pred, shape, *shift) for pred, shape, shift in zip(pred_masks, original_shape, shifts)
    ]

    return pred_masks


def _run_pipeline(
    img: np.ndarray,
    stage0_model: MultiHead_MBAS2024,
    stage1_model: MultiHead_MBAS2024,
) -> np.ndarray:
    """Run the inference pipeline for the prediction of the segmentation mask
    on a single image.

    Parameters
    ----------
    img : numpy.ndarray
        Input LGE-MRI image.
    stage0_model : nn.Module
        The stage-0 model for coarse segmentation.
    stage1_model : nn.Module
        The stage-1 model for fine segmentation.

    Returns
    -------
    numpy.ndarray
        Predicted segmentation mask.

    """
    original_shape = img.shape
    # img_raw = img.copy()
    img_raw = img
    if stage0_model.config.get("apply_mclahe", False):
        img_raw = mclahe(img_raw, use_gpu=False)

    img_raw, (shift_x, shift_y, shift_z) = reshape_input_image(img_raw)

    pred_mask = np.zeros_like(img_raw, dtype=np.uint8)

    # get the downsampled version of the input image
    img_coarse = img_raw[::RSMP_STEP, ::RSMP_STEP, :]

    # stage 0: coarse segmentation
    pred_coarse = stage0_model.inference(img_coarse).pred_mask[0]
    x_coords, y_coords, z_coords = np.where(pred_coarse == 1)
    x_center = x_coords.mean().astype(int) * RSMP_STEP
    y_center = y_coords.mean().astype(int) * RSMP_STEP

    # crop the fine region
    x_min = max(0, x_center - TrainCfg.fine_shape[0] // 2)
    x_max = x_min + TrainCfg.fine_shape[0]
    y_min = max(0, y_center - TrainCfg.fine_shape[1] // 2)
    y_max = y_min + TrainCfg.fine_shape[1]
    img_fine = img_raw[x_min:x_max, y_min:y_max, :]
    # print(f"coarse region center: ({x_center}, {y_center})")
    # print(f"box (x_min, x_max, y_min, y_max): ({x_min}, {x_max}, {y_min}, {y_max})")

    # stage 1: fine segmentation
    pred_mask[x_min:x_max, y_min:y_max, :] = stage1_model.inference(img_fine).pred_mask[0]

    # transform the predicted mask back to the original shape
    pred_mask = transform_mask_to_original_shape(pred_mask, original_shape, shift_x, shift_y, shift_z)

    return pred_mask


def transform_mask_to_original_shape(
    pred_mask: np.ndarray, original_shape: Sequence[int], shift_x: int, shift_y: int, shift_z: int
) -> np.ndarray:
    """Transform the predicted mask back to the original shape.

    Parameters
    ----------
    pred_mask : numpy.ndarray
        Predicted mask.
    original_shape : Sequence[int]
        Original shape of the input image.
    shift_x : int
        Shift in the x direction.
    shift_y : int
        Shift in the y direction.
    shift_z : int
        Shift in the z direction.

    Returns
    -------
    numpy.ndarray
        The mask in the original shape.

    """
    if shift_x < 0:
        # original -> raw by padding
        # then raw -> original by cropping
        pred_mask = pred_mask[-shift_x : original_shape[0] - shift_x, :, :]
    elif shift_x > 0:
        # original -> raw by cropping
        # then raw -> original by padding
        pred_mask = np.pad(
            pred_mask,
            ((shift_x, original_shape[0] - pred_mask.shape[0] - shift_x), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    if shift_y < 0:
        pred_mask = pred_mask[:, -shift_y : original_shape[1] - shift_y, :]
    elif shift_y > 0:
        pred_mask = np.pad(
            pred_mask,
            ((0, 0), (shift_y, original_shape[1] - pred_mask.shape[1] - shift_y), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    if shift_z < 0:
        pred_mask = pred_mask[:, :, -shift_z : original_shape[2] - shift_z]
    elif shift_z > 0:
        pred_mask = np.pad(
            pred_mask,
            ((0, 0), (0, 0), (shift_z, original_shape[2] - pred_mask.shape[2] - shift_z), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # print(f"predicted mask shape: {pred_mask.shape}")

    return pred_mask


def reshape_input_image(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Reshape the input image for the stage-0 model.

    Parameters
    ----------
    img : numpy.ndarray
        Input LGE-MRI image.

    Returns
    -------
    numpy.ndarray
        Downsampled version of the input image.
    Tuple[int, int, int]
        Shifts in the x, y, and z directions.

    """
    original_shape = img.shape
    if original_shape[0] < TrainCfg.data_shape[0]:
        # pad by zeros
        pad_x = (TrainCfg.data_shape[0] - original_shape[0]) // 2
        shift_x = -pad_x
        img = np.pad(
            img,
            ((pad_x, TrainCfg.data_shape[0] - original_shape[0] - pad_x), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif original_shape[0] > TrainCfg.data_shape[0]:
        # crop
        crop_x = (original_shape[0] - TrainCfg.data_shape[0]) // 2
        shift_x = crop_x
        img = img[crop_x : crop_x + TrainCfg.data_shape[0]]
    else:
        shift_x = 0
    if original_shape[1] < TrainCfg.data_shape[1]:
        # pad by zeros
        pad_y = (TrainCfg.data_shape[1] - original_shape[1]) // 2
        shift_y = -pad_y
        img = np.pad(
            img,
            ((0, 0), (pad_y, TrainCfg.data_shape[1] - original_shape[1] - pad_y), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif original_shape[1] > TrainCfg.data_shape[1]:
        # crop
        crop_y = (original_shape[1] - TrainCfg.data_shape[1]) // 2
        shift_y = crop_y
        img = img[:, crop_y : crop_y + TrainCfg.data_shape[1]]
    else:
        shift_y = 0
    if original_shape[2] < TrainCfg.data_shape[2]:
        # pad by zeros
        pad_z = (TrainCfg.data_shape[2] - original_shape[2]) // 2
        shift_z = -pad_z
        img = np.pad(
            img,
            ((0, 0), (0, 0), (pad_z, TrainCfg.data_shape[2] - original_shape[2] - pad_z)),
            mode="constant",
            constant_values=0,
        )
    elif original_shape[2] > TrainCfg.data_shape[2]:
        # crop
        crop_z = (original_shape[2] - TrainCfg.data_shape[2]) // 2
        shift_z = crop_z
        img = img[:, :, crop_z : crop_z + TrainCfg.data_shape[2]]
    else:
        shift_z = 0

    # print(f"image shape: {original_shape} -> {img.shape}")
    # print(f"shift_x: {shift_x}, shift_y: {shift_y}, shift_z: {shift_z}")

    return img, (shift_x, shift_y, shift_z)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Run the inference pipeline for the prediction of the segmentation mask.")
    parser.add_argument(
        "--db-dir",
        type=str,
        help="Directory of the database.",
        dest="db_dir",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).parent / "saved_models"),
        help="Directory of the trained models.",
        dest="model_dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mask_predictions",
        help="Directory of the predicted masks.",
        dest="output_dir",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference.",
        dest="device",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the inference in parallel.",
        dest="parallel",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel inference.",
        dest="batch_size",
    )
    args = parser.parse_args()

    dr = MBAS2024(args.db_dir)

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA is not available. Using CPU for inference.")
    device = torch.device(args.device)

    model_dir = Path(args.model_dir).expanduser().resolve()
    stage0_model = MultiHead_MBAS2024.from_checkpoint(model_dir / "stage0-model.pth.tar", device=device)[0]
    stage1_model = MultiHead_MBAS2024.from_checkpoint(model_dir / "stage1-model.pth.tar", device=device)[0]
    stage0_model = stage0_model.to(device).eval()
    stage1_model = stage1_model.to(device).eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    if len(dr.validation_set) == 0:
        # perhaps a custom subset, e.g. the hidden test set
        records = dr._df_records_all.index.tolist()
        warnings.warn("No validation set found. Inference on the entire dataset.")
    else:
        records = dr.validation_set

    if args.parallel:
        pred_masks = run_pipeline(
            [dr.load_data(rec) for rec in records],
            stage0_model,
            stage1_model,
            parallel=True,
            batch_size=args.batch_size,
        )
        for rec, pred_mask in zip(records, pred_masks):
            nib.save(nib.Nifti1Image(pred_mask, affine=np.eye(4)), output_dir / f"{rec}_label.nii.gz")
    else:
        for rec in tqdm(records, desc="Inference", unit="record", dynamic_ncols=True, mininterval=1.0):
            img = dr.load_data(rec)
            pred_mask = run_pipeline(img, stage0_model, stage1_model, parallel=False)
            nib.save(nib.Nifti1Image(pred_mask, affine=np.eye(4)), output_dir / f"{rec}_label.nii.gz")

    end = time.time()
    min_, sec_ = divmod(end - start, 60)
    min_ = int(min_)

    print(f"Time elapsed: {int(min_):02d} min and {sec_:.2f} sec")

    # example usage:
    # python pipeline.py --db-dir /path/to/db --model-dir /path/to/models --output-dir /path/to/output --device cuda:0
