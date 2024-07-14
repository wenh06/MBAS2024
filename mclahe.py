"""
Multidimensional Contrast Limited Adaptive Histogram Equalization

Modified from
https://github.com/VincentStimper/mclahe/blob/master/mclahe/core.py
"""

from typing import Optional, Sequence

import numpy as np


def mclahe(
    image: np.ndarray,
    kernel_size: Optional[Sequence[int]] = None,
    nbins: int = 128,
    clip_limit: float = 0.01,
    adaptive_hist_range=False,
) -> np.ndarray:
    """Apply Multidimensional Contrast Limited Adaptive Histogram Equalization (MCLAHE)
    to a multidimensional image.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    kernel_size : Sequence[int], optional
        The size of the kernel used for the local histogram equalization.
    nbins : int, default 128
        The number of bins used for the histogram.
    clip_limit : float, default 0.01
        The clip limit for the contrast limiting.
    adaptive_hist_range : bool, default False
        Whether to use an adaptive histogram range.

    Returns
    -------
    numpy.ndarray
        The equalized image.

    """
    if kernel_size is None:
        kernel_size = [s // 8 for s in image.shape]
    kernel_size = np.array(kernel_size)

    output = image.copy()
    # Normalize data
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # Pad data
    image_shape = np.array(image.shape)
    pad_size = kernel_size - 1 - np.divmod(image_shape - 1, kernel_size)[1]
    pad_data = np.column_stack(((pad_size + 1) // 2, pad_size // 2))
    pad_hist = np.column_stack((kernel_size // 2, (kernel_size + 1) // 2)) + pad_data
    hist_paddata = np.pad(output, pad_data, mode="symmetric")

    raise NotImplementedError
