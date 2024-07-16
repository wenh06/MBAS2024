from typing import Dict, Sequence, Union, List, Optional

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import jaccard_score

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
    average: str = "samples",
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
    metrics = {cls_: {
        m: [] for m in metrics_list
    } for idx, cls_ in enumerate(BaseCfg.stage1_classes) if idx not in ignore_index}

    for label, output in zip(labels, outputs):
        if isinstance(output, MBAS2024Outputs):
            pred_mask = output.pred_mask
        else:
            pred_mask = output
        if pred_mask.ndim == 4:
            pred_mask = np.argmax(pred_mask, axis=-1)
        gt_mask = label.astype(pred_mask.dtype)

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
            distances = _hausdorff_distance(gt_mask==idx, pred_mask==idx, percentile=95)
            metrics[cls_]["HD95"].append(distances)

    if average == "samples":
        for cls_ in metrics:
            for m in metrics_list:
                metrics[cls_][m] = np.mean(metrics[cls_][m])
    elif average == "classes":
        raise NotImplementedError

    return metrics


def hausdorff_distance(mask1:np.ndarray, mask2:np.ndarray, percentile:float=95, labels:Optional[Sequence[int]]=None) -> Dict[int, float]:
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


def _hausdorff_distance(mask1:np.ndarray, mask2:np.ndarray, percentile:float=95) -> float:
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
