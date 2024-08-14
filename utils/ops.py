from typing import Optional, Union

import numpy as np
from scipy.ndimage import convolve


def dilate(image: np.ndarray, kernel: Optional[Union[int, np.ndarray]] = None, iterations: int = 1) -> np.ndarray:
    """Dilate an volumetric (3D) image with a given kernel.

    Parameters
    ----------
    image : np.ndarray
        The input volumetric image.
    kernel : int or np.ndarray, optional
        The kernel to use for dilation. If None, a 3x3x3 cube kernel is used.
        If is an integer, a cube kernel with the given size is used.
    iterations : int, optional
        The number of iterations to perform.

    Returns
    -------
    np.ndarray
        The dilated image.

    """
    if kernel is None:
        kernel = np.ones((3, 3, 3), dtype=float)
    elif isinstance(kernel, int):
        kernel = np.ones((kernel, kernel, kernel), dtype=float)
    else:
        kernel = kernel.astype(float)

    dilated_image = image.copy()

    for _ in range(iterations):
        dilated_image = np.maximum(
            dilated_image, (convolve(dilated_image.astype(float), kernel) / kernel.sum()).astype(image.dtype)
        )

    return dilated_image
