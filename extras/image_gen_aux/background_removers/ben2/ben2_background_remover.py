# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Union

import numpy as np
import PIL.Image
import torch
from huggingface_hub.utils import validate_hf_hub_args
from PIL import Image

from ...image_processor import ImageMixin
from ..background_remover import BackgroundRemover
from .ben2 import BEN_Base


class BEN2BackgroundRemover(BackgroundRemover, ImageMixin):
    """Background remover using the BEN2 model.

    This class inherits from both `Preprocessor` and `ImageMixin`. Please refer to each
    one to get more information.
    """

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path: Union[str, os.PathLike], **kwargs):
        model = BEN_Base.from_pretrained(pretrained_model_or_path, **kwargs)

        return cls(model)

    @torch.inference_mode
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]],
        return_type: str = "pil",
        return_mask: bool = False,
    ):
        """
        Removes background from an image using the BEN2 model.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]]`): Input image(s) as PIL Image,
                NumPy array, PyTorch tensor, or list of PIL Images.
            return_type (`str`, *optional*, defaults to "pil"): The desired return type, either "pt" for PyTorch tensor,
                "np" for NumPy array, or "pil" for PIL image.
            return_mask (`bool`, *optional*, defaults to False): Whether to return the masks along with the processed images.

        Returns:
            If return_mask is False: List of processed images with background removed.
            If return_mask is True: Tuple of (processed_images, masks).
        """
        # TODO: Add support for tentos and numpy arrays
        if return_type != "pil":
            raise ValueError("For the moment only 'pil' return type is supported.")

        if isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image, list):
            original_size = image[0].size

            for i, img in enumerate(image):
                if img.size != original_size:
                    raise ValueError(
                        f"All images must have the same dimensions for batch processing. "
                        f"Image 0 has size {original_size}, but image {i} has size {img.size}."
                    )

            processed_images = []
            for img in image:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                resized_img = img.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
                processed_images.append(resized_img)

            image_tensor = self.convert_image_to_tensor(processed_images).to(self.model.device)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=self.model.dtype).view(1, 3, 1, 1).to(self.model.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=self.model.dtype).view(1, 3, 1, 1).to(self.model.device)

        image_tensor = (image_tensor - mean) / std

        mask = self.model(image_tensor)

        if isinstance(image, list):
            target_size = (original_size[1], original_size[0])

            if mask.shape[2:4] != target_size:
                mask = torch.nn.functional.interpolate(mask, size=target_size, mode="bicubic", align_corners=False)

        processed_mask = self.post_process_image(mask, return_type)
        foregrounds = []

        if return_type == "pt":
            ...
        elif return_type == "np":
            ...
        else:  # pillow
            for i, img in enumerate(image):
                img_rgba = img.convert("RGBA")
                img_rgba.putalpha(processed_mask[i])
                foregrounds.append(img_rgba)

        if return_mask:
            return foregrounds, processed_mask
        else:
            return foregrounds
