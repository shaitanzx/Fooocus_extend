# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

import unittest

import torch


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, x, y):
        adiff = float((x - y).abs().max())
        if (y == 0).all():
            rdiff = "NaN"
        else:
            rdiff = float((adiff / y).abs().max())

        message = ("Tensor close check failed\n" "adiff={}\n" "rdiff={}\n").format(
            adiff, rdiff
        )
        self.assertTrue(torch.allclose(x, y, atol=1e-5, rtol=1e-3), message)
