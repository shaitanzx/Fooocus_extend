"""
From the CodeFormer project by author sczhou.
"""
import os
import torch
import inspect
from torch import nn
from copy import deepcopy

from extras.facexlib_custom.utils import load_file_from_url
from extras.facexlib_custom.detection.yolov5face.models.common import Conv

from .retinaface import RetinaFace
from .yolov5face.face_detector import YoloDetector


def init_detection_model(model_name, half=False, device='cuda', model_rootpath=None):
    if 'retinaface' in model_name:
        model = init_retinaface_model(model_name, half, device, model_rootpath)
    elif 'YOLOv5' in model_name:
        model = init_yolov5face_model(model_name, device, model_rootpath)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    return model


def init_retinaface_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half)
        model_url = 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/detection_Resnet50_Final.pth'
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half)
        model_url = 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    return model


def init_yolov5face_model(model_name, device='cuda', model_rootpath=None):
    current_file_path = inspect.getfile(inspect.currentframe())
    library_directory = os.path.dirname(os.path.abspath(current_file_path))

    if model_name == 'YOLOv5l':
        model = YoloDetector(config_name=f'{library_directory}/yolov5face/models/yolov5l.yaml', device=device)
        model_url = 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/YOLO5/yolov5l-face.pth'
    elif model_name == 'YOLOv5n':
        model = YoloDetector(config_name=f'{library_directory}/yolov5face/models/yolov5n.yaml', device=device)
        model_url = 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/YOLO5/yolov5n-face.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
    model.detector.load_state_dict(load_net, strict=True)
    model.detector.eval()
    model.detector = model.detector.to(device).float()

    for m in model.detector.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif isinstance(m, Conv):
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    return model