import numpy as np
from warnings import warn
from typing import Union, Dict
from monai.transforms import Transform
from monai.config import DtypeLike, KeysCollection


class Normalized(Transform):
    """
    Apply histogram based intensity scaling to every slice of first dimension of input image.

    Args:
        keys: keys of the corresponding item to be transformed.
        a_upper_quant: upper quantile of the input image to be mapped to b_upper_value
        a_lower_quant: lower quantile of the input image to be mapped to b_lower_value
        b_upper_value: upper value of the output image
        b_lower_value: lower value of the output image
        dtype: output data type
    """

    def __init__(
        self,
        keys: KeysCollection,
        a_upper_quant: Union[float, None] = 0.95,
        a_lower_quant: Union[float, None] = 0.05,
        b_upper_value: float = 1.0,
        b_lower_value: float = 0.0,
        dtype: DtypeLike = np.float32,
    ) -> None:

        self.keys = keys

        self.a_upper_quant = a_upper_quant
        self.a_lower_quant = a_lower_quant

        self.b_upper_value = b_upper_value
        self.b_lower_value = b_lower_value

        self.dtype = dtype

    def _norm_image(self, img_stack: np.ndarray):

        for i in range(img_stack.shape[0]):
            img_stack[i, :, :] = self._normalize(
                img_stack[i, :, :],
                a_upper=np.quantile(img_stack[i, :, :], self.a_upper_quant)
                if self.a_upper_quant not in (None, 1)
                else None,
                a_lower=np.quantile(img_stack[i, :, :], self.a_lower_quant)
                if self.a_lower_quant not in (None, 0)
                else None,
            )

        return img_stack.astype(self.dtype)

    def _normalize(self, img: np.ndarray, a_upper, a_lower):

        if a_upper is None:
            a_upper = img.max()
        if a_lower is None:
            a_lower = img.min()

        if a_upper - a_lower == 0.0:
            warn("Divide by zero (a_lower == a_upper)", Warning)
            return img - a_lower + self.b_lower_value

        img = (img - a_lower) / (a_upper - a_lower)
        img = img * (self.b_upper_value - self.b_lower_value) + self.b_lower_value
        return img

    def __call__(self, input: Dict) -> Dict:

        for key in self.keys:
            input[key] = self._norm_image(input[key])

        return input
