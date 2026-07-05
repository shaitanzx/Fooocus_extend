# flake8: noqa
import os.path as osp
from extras.basicsr.train import train_pipeline

import extras.gfpgan.archs
import extras.gfpgan.data
import extras.gfpgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
