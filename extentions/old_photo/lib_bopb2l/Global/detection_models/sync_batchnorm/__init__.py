# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

from .batchnorm import (
    SynchronizedBatchNorm1d,
    SynchronizedBatchNorm2d,
    SynchronizedBatchNorm3d,
    convert_model,
    patch_sync_batchnorm,
    set_sbn_eps_mode,
)
from .replicate import DataParallelWithCallback, patch_replication_callback
