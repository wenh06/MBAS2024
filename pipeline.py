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
from pathlib import Path
from typing import List, Sequence, Union

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

from cfg import TrainCfg
from data_reader import MBAS2024
from models import MultiHead_MBAS2024
from utils.mclahe_tf import mclahe  # noqa: F401


def run_pipeline(
    img: Union[np.ndarray, Sequence[np.ndarray]],
    stage0_model: MultiHead_MBAS2024,
    stage1_model: MultiHead_MBAS2024,
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

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Predicted segmentation mask(s).

    """
    assert stage0_model.config.get("apply_mclahe", False) == stage1_model.config.get(
        "apply_mclahe", False
    ), "apply_mclahe should be consistent"
    if isinstance(img, np.ndarray) and img.ndim == 3:
        return _run_pipeline(img, stage0_model, stage1_model)

    pred_masks = []
    for i in tqdm(len(img)):
        pred_masks.append(_run_pipeline(img[i], stage0_model, stage1_model))
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
    img_raw = img.copy()
    if stage0_model.config.get("apply_mclahe", False):
        img_raw = mclahe(img_raw)

    if original_shape[0] < TrainCfg.data_shape[0]:
        # pad by zeros
        pad_x = (TrainCfg.data_shape[0] - original_shape[0]) // 2
        shift_x = -pad_x
        img_raw = np.pad(
            img_raw,
            ((pad_x, TrainCfg.data_shape[0] - original_shape[0] - pad_x), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif original_shape[0] > TrainCfg.data_shape[0]:
        # crop
        crop_x = (original_shape[0] - TrainCfg.data_shape[0]) // 2
        shift_x = crop_x
        img_raw = img_raw[crop_x : crop_x + TrainCfg.data_shape[0]]
    else:
        shift_x = 0
    if original_shape[1] < TrainCfg.data_shape[1]:
        # pad by zeros
        pad_y = (TrainCfg.data_shape[1] - original_shape[1]) // 2
        shift_y = -pad_y
        img_raw = np.pad(
            img_raw,
            ((0, 0), (pad_y, TrainCfg.data_shape[1] - original_shape[1] - pad_y), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif original_shape[1] > TrainCfg.data_shape[1]:
        # crop
        crop_y = (original_shape[1] - TrainCfg.data_shape[1]) // 2
        shift_y = crop_y
        img_raw = img_raw[:, crop_y : crop_y + TrainCfg.data_shape[1]]
    else:
        shift_y = 0
    if original_shape[2] < TrainCfg.data_shape[2]:
        # pad by zeros
        pad_z = (TrainCfg.data_shape[2] - original_shape[2]) // 2
        shift_z = -pad_z
        img_raw = np.pad(
            img_raw,
            ((0, 0), (0, 0), (pad_z, TrainCfg.data_shape[2] - original_shape[2] - pad_z)),
            mode="constant",
            constant_values=0,
        )
    elif original_shape[2] > TrainCfg.data_shape[2]:
        # crop
        crop_z = (original_shape[2] - TrainCfg.data_shape[2]) // 2
        shift_z = crop_z
        img_raw = img_raw[:, :, crop_z : crop_z + TrainCfg.data_shape[2]]
    else:
        shift_z = 0

    print(f"image shape: {original_shape} -> {img_raw.shape}")
    print(f"shift_x: {shift_x}, shift_y: {shift_y}, shift_z: {shift_z}")

    pred_mask = np.zeros_like(img_raw, dtype=np.uint8)

    # get the downsampled version of the input image
    step = 4
    img_coarse = img_raw[::step, ::step, :]

    # stage 0: coarse segmentation
    pred_coarse = stage0_model.inference(img_coarse).pred_mask[0]
    x_coords, y_coords, z_coords = np.where(pred_coarse == 1)
    x_center = x_coords.mean().astype(int) * step
    y_center = y_coords.mean().astype(int) * step

    # crop the fine region
    x_min = max(0, x_center - TrainCfg.fine_shape[0] // 2)
    x_max = x_min + TrainCfg.fine_shape[0]
    y_min = max(0, y_center - TrainCfg.fine_shape[1] // 2)
    y_max = y_min + TrainCfg.fine_shape[1]
    img_fine = img_raw[x_min:x_max, y_min:y_max, :]
    print(f"coarse region center: ({x_center}, {y_center})")
    print(f"box (x_min, x_max, y_min, y_max): ({x_min}, {x_max}, {y_min}, {y_max})")

    # stage 1: fine segmentation
    pred_mask[x_min:x_max, y_min:y_max, :] = stage1_model.inference(img_fine).pred_mask[0]

    # transform the predicted mask back to the original shape
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

    print(f"predicted mask shape: {pred_mask.shape}")

    return pred_mask


if __name__ == "__main__":
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
    args = parser.parse_args()

    dr = MBAS2024(args.db_dir)

    model_dir = Path(args.model_dir).expanduser().resolve()
    stage0_model = MultiHead_MBAS2024.from_checkpoint(model_dir / "stage0-model.pth.tar")[0]
    stage1_model = MultiHead_MBAS2024.from_checkpoint(model_dir / "stage1-model.pth.tar")[0]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    for rec in tqdm(dr.validation_set):
        img = dr.load_data(rec)
        pred_mask = run_pipeline(img, stage0_model, stage1_model)
        nib.save(nib.Nifti1Image(pred_mask, affine=np.eye(4)), output_dir / f"{rec}_label.nii.gz")

    # example usage:
    # python pipeline.py --db-dir /path/to/db --model-dir /path/to/models --output-dir /path/to/output
