# Copyright (c) Microsoft Corporation

import torch

from .base_dataset import BaseDataset, get_params, get_transform


class FaceTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--no_pairing_check",
            action="store_true",
            help="If specified, skip sanity check of correct label-image file pairing",
        )
        return parser

    def initialize(self, opt, faces: list):
        self.opt = opt
        self.images = faces  # All the images

        self.parts = [
            "skin",
            "hair",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hat",
        ]

        size = len(self.images)
        self.dataset_size = size

    def __getitem__(self, index):
        params = get_params(self.opt, (-1, -1))

        image = self.images[index].convert("RGB")

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        full_label = []

        for _ in self.parts:
            current_part = torch.zeros((self.opt.load_size, self.opt.load_size))
            full_label.append(current_part)

        full_label_tensor = torch.stack(full_label, 0)

        input_dict = {
            "label": full_label_tensor,
            "image": image_tensor,
            "path": "",
        }

        return input_dict

    def __len__(self):
        return self.dataset_size
