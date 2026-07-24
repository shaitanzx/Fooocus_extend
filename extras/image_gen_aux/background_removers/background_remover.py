from abc import abstractmethod
from typing import Union

import numpy as np
import PIL.Image
import torch

from ..base_model_processor import BaseModelProcessor


class BackgroundRemover(BaseModelProcessor):
    @abstractmethod
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        return_type: str = "pil",
    ):
        """
        Removes background from an image using the underlying model.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`): Input image as PIL Image,
                NumPy array, or PyTorch tensor format.
            return_type (`str`, *optional*, defaults to "pil"): The desired return type, either "pt" for PyTorch tensor,
                "np" for NumPy array, or "pil" for PIL image.

        Returns:
            `Union[PIL.Image.Image, torch.Tensor]`: The processed image with background removed
                in the specified format.
        """
        pass
