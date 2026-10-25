import cv2
import numpy as np
from extentions.op_edit.body import Body
from extentions.op_edit import util as openpose_util
from controlnet_aux import HEDdetector, LineartAnimeDetector
from modules.config import downloading_pose_preprocessor, downloading_hed_preprocessor
import gc
import torch
import os

def lineart_anime(x: np.ndarray) -> np.ndarray:
    """
    Извлекает anime/manga lineart для SD 1.5 Lineart Anime ControlNet.
    Вход и выход: RGB uint8.
    """
    import gc
    import torch

    model_path = downloading_linart_anime_preprocessor()
    detector = LineartAnimeDetector.from_pretrained(
        os.path.dirname(model_path),
        filename=os.path.basename(model_path))


    resolution = max(x.shape[:2])
    result = detector(
        x,
        detect_resolution=resolution,
        image_resolution=resolution,
        output_type="np",
    )

    if result.shape[:2] != x.shape[:2]:
        result = cv2.resize(
            result,
            (x.shape[1], x.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return result.astype(np.uint8)


def scribble_hed(x: np.ndarray) -> np.ndarray:
    model_path = downloading_hed_preprocessor()
    detector = HEDdetector.from_pretrained(
        os.path.dirname(model_path),
        filename=os.path.basename(model_path))

    resolution = max(x.shape[:2])
    result = detector(
        x,
        detect_resolution=resolution,
        image_resolution=resolution,
        scribble=True,
        output_type="np")


    if result.shape[:2] != x.shape[:2]:
        result = cv2.resize(
            result,
            (x.shape[1], x.shape[0]),
            interpolation=cv2.INTER_NEAREST)

    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return result.astype(np.uint8)

def pose(x: np.ndarray) -> np.ndarray:
    model_path = downloading_pose_preprocessor()
    image_bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    body_estimation = Body(model_path)
    candidate, subset = body_estimation(image_bgr)
    pose_bgr = np.zeros_like(image_bgr)
    pose_bgr = openpose_util.draw_bodypose(pose_bgr,candidate,subset)
    del body_estimation
    gc.collect()
    torch.cuda.empty_cache()
    return cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2RGB)


def centered_canny(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.uint8

    y = cv2.Canny(x, int(canny_low_threshold), int(canny_high_threshold))
    y = y.astype(np.float32) / 255.0
    return y


def centered_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    result = [centered_canny(x[..., i], canny_low_threshold, canny_high_threshold) for i in range(3)]
    result = np.stack(result, axis=2)
    return result


def pyramid_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    H, W, C = x.shape
    acc_edge = None

    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        Hs, Ws = int(H * k), int(W * k)
        small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
        edge = centered_canny_color(small, canny_low_threshold, canny_high_threshold)
        if acc_edge is None:
            acc_edge = edge
        else:
            acc_edge = cv2.resize(acc_edge, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR)
            acc_edge = acc_edge * 0.75 + edge * 0.25

    return acc_edge


def norm255(x, low=4, high=96):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.float32

    v_min = np.percentile(x, low)
    v_max = np.percentile(x, high)

    x -= v_min
    x /= v_max - v_min

    return x * 255.0


def canny_pyramid(x, canny_low_threshold, canny_high_threshold):
    # For some reasons, SAI's Control-lora Canny seems to be trained on canny maps with non-standard resolutions.
    # Then we use pyramid to use all resolutions to avoid missing any structure in specific resolutions.

    color_canny = pyramid_canny_color(x, canny_low_threshold, canny_high_threshold)
    result = np.sum(color_canny, axis=2)

    return norm255(result, low=1, high=99).clip(0, 255).astype(np.uint8)


def cpds(x):
    # cv2.decolor is not "decolor", it is Cewu Lu's method
    # See http://www.cse.cuhk.edu.hk/leojia/projects/color2gray/index.html
    # See https://docs.opencv.org/3.0-beta/modules/photo/doc/decolor.html

    raw = cv2.GaussianBlur(x, (0, 0), 0.8)
    density, boost = cv2.decolor(raw)

    raw = raw.astype(np.float32)
    density = density.astype(np.float32)
    boost = boost.astype(np.float32)

    offset = np.sum((raw - boost) ** 2.0, axis=2) ** 0.5
    result = density + offset

    return norm255(result, low=4, high=96).clip(0, 255).astype(np.uint8)

def recolor_intensity(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Формирует 3-канальную карту интенсивности для Recolor ControlNet.
    Вход и выход — RGB uint8.
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    intensity = hsv[:, :, 2].astype(np.float32) / 255.0
    intensity = np.power(intensity, gamma)
    intensity = (intensity * 255.0).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(intensity, cv2.COLOR_GRAY2RGB)


def recolor_luminance(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Формирует 3-канальную карту светлоты для Recolor ControlNet.
    Вход и выход — RGB uint8.
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    luminance = np.power(luminance, gamma)
    luminance = (luminance * 255.0).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(luminance, cv2.COLOR_GRAY2RGB)
