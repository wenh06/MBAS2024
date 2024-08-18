import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from medpy import metric as medpy_metric
from scipy.spatial import cKDTree
from sklearn.metrics import jaccard_score
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cfg import BaseCfg
from outputs import MBAS2024Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[np.ndarray],
    outputs: Sequence[Union[MBAS2024Outputs, np.ndarray]],
    ignore_index: Union[int, Sequence[int]] = 0,
    class_mapping: Optional[Dict[int, str]] = None,
    average: Literal["samples", "classes", "macro", None] = "samples",
    use_official_metric: bool = True,
    progress: bool = True,
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """Compute the challenge metrics.

    The challenge (segmentation) metrics include:
    - Dice Similarity Coefficient (Dice)
    - 95% Hausdorff distance (HD95)

    And some auxiliary metrics:
    - Intersection over Union (IoU)

    Parameters
    ----------
    labels : Sequence[np.ndarray]
        The mask labels for the records.
        For segmentation, the mask label is one-hot encoded,
        of shape (H, W, D, C), where C is the number of classes;
        or of shape (H, W, D) for categorical labels.
    outputs : Sequence[MBAS2024Outputs] or Sequence[np.ndarray]
        The outputs for the records.
    ignore_index : Union[int, Sequence[int]], default 0
        The index of the class to ignore, by default the background class.
    average : str, default "samples"
        The averaging strategy for the metrics.
        - None: do not average the metrics.
        - "samples": average the metrics for all samples under each class.
        - "classes": average the metrics for all classes for each sample.
        - "macro": compute each metric as the unweighted mean of the per-class metrics.
    use_official_metric : bool, default True
        Whether to use the official metric implementation.
    progress : bool, default False
        Whether to show the progress bar.

    Returns
    -------
    Dict[Dict[str, float]]
        The computed challenge metrics for each class.

    Examples
    --------
    >>> labels = [np.random.randint(0, 4, size=(10, 10, 10)) for _ in range(2)]
    >>> outputs = [
            MBAS2024Outputs(pred_mask=np.random.randint(0, 4, size=(10, 10, 10))),
            MBAS2024Outputs(pred_mask=np.random.randint(0, 4, size=(10, 10, 10)))
        ]
    >>> compute_challenge_metrics(labels, outputs)

    """
    assert len(labels) == len(outputs), "The number of labels and outputs must be the same."
    # convert one-hot encoded labels to categorical labels
    if labels[0].ndim == 4:
        labels = [np.argmax(label, axis=-1) for label in labels]
    # convert the ignore_index to a list
    if isinstance(ignore_index, int):
        ignore_index = [ignore_index]
    elif ignore_index is None:
        ignore_index = []
    if class_mapping is None:
        class_mapping = {idx: cls_ for idx, cls_ in enumerate(BaseCfg.stage1_classes)}

    # initialize the metrics
    metrics_list = ["Dice", "HD95", "IoU"]
    metrics = {
        cls_: {m: [] for m in metrics_list} for idx, cls_ in enumerate(BaseCfg.stage1_classes) if idx not in ignore_index
    }

    for label, output in tqdm(
        zip(labels, outputs),
        total=len(labels),
        desc="Computing challenge metrics",
        unit="sample",
        dynamic_ncols=True,
        mininterval=1,
        disable=not progress,
    ):
        if isinstance(output, MBAS2024Outputs):
            pred_mask = output.pred_mask
        else:
            pred_mask = output
        if pred_mask.ndim == 4:
            pred_mask = np.argmax(pred_mask, axis=-1)
        gt_mask = label.astype(pred_mask.dtype)

        if use_official_metric:
            for idx, cls_ in class_mapping.items():
                if idx in ignore_index:
                    continue
                dice, hd95 = official_calculate_metric_percase(pred_mask == idx, gt_mask == idx)
                metrics[cls_]["Dice"].append(dice)
                metrics[cls_]["HD95"].append(hd95)
                metrics[cls_]["IoU"].append(dice / (2 - dice))
        else:
            # compute the intersection over union (IoU) and Dice similarity coefficient (DSC)
            iou = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average=None)
            for idx, cls_ in class_mapping.items():
                if idx in ignore_index:
                    continue
                metrics[cls_]["IoU"].append(iou[idx])
                metrics[cls_]["Dice"].append(2 * iou[idx] / (1 + iou[idx]))

            # compute the 95% Hausdorff distance (HD95)
            for idx, cls_ in enumerate(BaseCfg.stage1_classes):
                if idx in ignore_index:
                    continue
                distances = _hausdorff_distance(gt_mask == idx, pred_mask == idx, percentile=95)
                metrics[cls_]["HD95"].append(distances)

    if average == "samples":
        for cls_ in metrics:
            for m in metrics_list:
                metrics[cls_][m] = np.mean(metrics[cls_][m])
    elif average == "classes":
        raise NotImplementedError
    elif average == "macro":
        raise NotImplementedError
    else:
        pass

    return metrics


def hausdorff_distance(
    mask1: np.ndarray, mask2: np.ndarray, percentile: float = 95, labels: Optional[Sequence[int]] = None
) -> Dict[int, float]:
    """Compute the Hausdorff distance between two masks.

    Modified from
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py

    Parameters
    ----------
    mask1 : np.ndarray
        The first mask.
    mask2 : np.ndarray
        The second mask.
    percentile : float, default 95
        The percentile for the Hausdorff distance.
    labels : Optional[Sequence[int]], default None
        The labels to consider.
        If not provided, labels are inferred from the masks.

    Returns
    -------
    Dict[int, float]
        The Hausdorff distance for each label.

    """

    if labels is None:
        labels = sorted(set(np.unique(mask1)).union(set(np.unique(mask2))))

    distances = {}
    for label in labels:
        distances[label] = _hausdorff_distance(mask1 == label, mask2 == label, percentile)

    return distances


def _hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray, percentile: float = 95) -> float:
    """Compute the Hausdorff distance between two masks.

    Modified from
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py

    Parameters
    ----------
    mask1 : np.ndarray
        The first mask (boolean).
    mask2 : np.ndarray
        The second mask (boolean).
    percentile : float, default 95
        The percentile for the Hausdorff distance.

    Returns
    -------
    float
        The Hausdorff distance.

    """
    a_points = np.transpose(np.nonzero(mask1))
    b_points = np.transpose(np.nonzero(mask2))

    # Handle empty sets properly:
    # - if both sets are empty, set distance zero
    # - if only one set is empty, set distance to infinity
    if len(a_points) == 0 and len(b_points) == 0:
        return 0
    elif len(a_points) == 0 or len(b_points) == 0:
        return np.inf
    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )
    return np.percentile(np.concatenate((fwd, bwd)), percentile)


def official_calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Calculate the metric per case on binary masks.

    Parameters
    ----------
    pred : np.ndarray
        The predicted mask.
    gt : np.ndarray
        The ground truth mask.

    Returns
    -------
    Tuple[float, float]
        The Dice similarity coefficient and 95% Hausdorff distance.

    """
    if np.argmin(pred.shape) == 2:
        pred = np.moveaxis(pred, -1, 0)
        gt = np.moveaxis(gt, -1, 0)
    dice = medpy_metric.binary.dc(pred, gt)
    hd95 = medpy_metric.binary.hd95(pred, gt, voxelspacing=(2.5, 0.625, 0.625))
    # print('dice, hd95', dice, hd95)
    return dice, hd95


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    import nibabel as nib

    parser = argparse.ArgumentParser(description="Compute the challenge metrics.")
    parser.add_argument("--labels", type=str, help="The path to the labels.")
    parser.add_argument("--outputs", type=str, help="The path to the outputs.")
    parser.add_argument(
        "--ignore-index", nargs="+", type=int, default=[0], help="The index of the class to ignore.", dest="ignore_index"
    )
    parser.add_argument(
        "--average",
        type=str,
        default="samples",
        choices=["samples", "classes", "macro", None],
        help="The averaging strategy for the metrics.",
    )
    parser.add_argument("--use-custom-metric", action="store_true", help="Whether to use the custom metric implementation.")
    parser.add_argument("--parallel", action="store_true", help="Whether to use parallel computation.")

    args = parser.parse_args()

    args.use_official_metric = not args.use_custom_metric

    label_files = sorted(Path(args.labels).expanduser().resolve().rglob("*_label.nii.gz"))
    output_files = sorted(Path(args.outputs).expanduser().resolve().rglob("*_label.nii.gz"))
    assert set([f.name for f in label_files]) == set(
        [f.name for f in output_files]
    ), "Label and output files must correspond to each other."

    if args.parallel:
        raise NotImplementedError(
            "Migth have some issues from scipy.ndimage: RuntimeError: sequence argument must have length equal to input rank"
        )
        args_list = [
            (
                nib.load(str(lf)).get_fdata(),
                nib.load(str(of)).get_fdata(),
                args.ignore_index,
                None,
                None,
                args.use_official_metric,
                False,
            )
            for lf, of in zip(label_files, output_files)
        ]
        with mp.Pool(processes=max(1, mp.cpu_count() - 3)) as pool:
            metrics = pool.starmap(
                compute_challenge_metrics,
                tqdm(
                    args_list,
                    total=len(args_list),
                    desc="Computing challenge metrics",
                    unit="sample",
                    dynamic_ncols=True,
                    mininterval=1,
                ),
            )
    else:
        metrics = []
        for lf, of in tqdm(
            zip(label_files, output_files),
            total=len(label_files),
            desc="Computing challenge metrics",
            unit="sample",
            dynamic_ncols=True,
            mininterval=1,
        ):
            labels = [nib.load(str(lf)).get_fdata()]
            outputs = [nib.load(str(of)).get_fdata()]

            metrics.append(
                compute_challenge_metrics(
                    labels,
                    outputs,
                    ignore_index=args.ignore_index,
                    average=None,
                    use_official_metric=args.use_official_metric,
                    progress=False,
                )
            )

    if args.average == "samples":
        metrics = {cls_: {m: np.mean([metric[cls_][m] for metric in metrics]) for m in metrics[0][cls_]} for cls_ in metrics[0]}
    elif args.average == "classes":
        raise NotImplementedError
    elif args.average == "macro":
        raise NotImplementedError
    else:
        pass

    print(metrics)
