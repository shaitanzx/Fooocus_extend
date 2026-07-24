from abc import abstractmethod
from typing import Union

import numpy as np
import PIL.Image
import torch

from ..base_model_processor import BaseModelProcessor


class Preprocessor(BaseModelProcessor):
    """
    This abstract base class defines the interface for image preprocessors.

    Subclasses should implement the abstract methods `from_pretrained` and
    `__call__` to provide specific loading and preprocessing logic for their
    respective models.

    Args:
        model (`nn.Module`): The torch model to use.
    """

    @abstractmethod
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        resolution_scale: float = 1.0,
        invert: bool = True,
        return_type: str = "pil",
    ):
        """
        Preprocesses an image for use with the underlying model.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`): Input image as PIL Image,
                NumPy array, or PyTorch tensor format.
            resolution_scale (`float`, *optional*, defaults to 1.0): Scale factor for image resolution during
                preprocessing and post-processing. Defaults to 1.0 for no scaling.
            invert (`bool`, *optional*, defaults to True): Inverts the generated image if True.
            return_type (`str`, *optional*, defaults to "pil"): The desired return type, either "pt" for PyTorch tensor,
                "np" for NumPy array, or "pil" for PIL image.

        Returns:
            `Union[PIL.Image.Image, torch.Tensor]`: The preprocessed image in the
                specified format.
        """
        pass
