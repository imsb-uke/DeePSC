import numpy as np
from typing import Dict
from skimage import exposure
from monai.transforms import Transform
from monai.config import DtypeLike, KeysCollection


class Clahed(Transform):
    """
    Applies CLAHE to every slice of first dimension of input image.

    Args:
        keys: keys of the corresponding item to be transformed.
        kernel_size: kernel size for CLAHE.
        contrast_clip_limit: upper limit of the contrast enhancement.
        nbins: number of of bins for histogram.
        dtype: output data type
    """

    def __init__(
        self,
        keys: KeysCollection,
        kernel_size=None,
        contrast_clip_limit=0.015,
        nbins=256,
        dtype: DtypeLike = np.float32,
    ) -> None:

        self.keys = keys

        self.kernel_size = kernel_size
        self.contrast_clip_limit = contrast_clip_limit
        self.nbins = nbins

        self.dtype = dtype

    def _clahe(self, img_stack: np.ndarray) -> np.ndarray:
        for i in range(img_stack.shape[0]):
            img = img_stack[i, :, :]
            # CLAHE only works with float values between -1 and 1
            mult = np.abs(img).max()
            img = exposure.equalize_adapthist(
                img / mult,
                kernel_size=self.kernel_size,
                clip_limit=self.contrast_clip_limit,
                nbins=self.nbins,
            )
            # scale back up
            img = img * mult
            img_stack[i, :, :] = img
        return img_stack.astype(self.dtype)

    def __call__(self, input: Dict) -> Dict:

        for key in self.keys:
            self._clahe(input[key])

        return input
