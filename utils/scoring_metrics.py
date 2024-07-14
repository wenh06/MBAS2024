from typing import Dict, Sequence

import numpy as np
from scipy.spatial import distance  # noqa: F401

from outputs import MBAS2024Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[MBAS2024Outputs],
) -> Dict[str, float]:
    """Compute the challenge metrics.

    The challenge (segmentation) metrics include:
    - Dice Similarity Coefficient (DSC)
    - 95% Hausdorff distance (HD95)

    Parameters
    ----------
    labels : Sequence[Dict[str, np.ndarray]]
        The labels for the records.
        For segmentation, the "mask" label is typically one-hot encoded,
        of shape (H, W, D, C), where C is the number of classes.
    outputs : Sequence[MBAS2024Outputs]
        The outputs for the records.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for "seg" (at least one of them).
        nan values are returned for the metrics that are not computed due to missing outputs.

    Examples
    --------
    >>> labels = [{"mask": np.random.uniform(size=(10, 10, 10, 2))}, {"mask": np.random.uniform(size=(10, 10, 10, 2))}]
    >>> outputs = [
            MBAS2024Outputs(seg_mask=np.random.randint(0, 2, size=(10, 10, 10))),
            MBAS2024Outputs(seg_mask=np.random.randint(0, 2, size=(10, 10, 10)))
        ]
    >>> compute_challenge_metrics(labels, outputs)

    """
    metrics = {}
    # compute the dice similarity coefficient (DSC)
    raise NotImplementedError
