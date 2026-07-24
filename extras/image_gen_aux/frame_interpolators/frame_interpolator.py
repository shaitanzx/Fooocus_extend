from abc import abstractmethod
from typing import Union, List
import numpy as np
import PIL.Image
import torch
from ..base_model_processor import BaseModelProcessor


class FrameInterpolator(BaseModelProcessor):
    """
    Abstract base class for all interpolation models.
    Defines a common API contract.
    """

    def __init__(self):
        super().__init__()
