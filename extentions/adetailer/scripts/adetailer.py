
import re
import sys
import os
import traceback
#!from collections.abc import Sequence
from copy import copy
#!from functools import partial

from typing import TYPE_CHECKING, Any, NamedTuple, cast

import gradio as gr
from PIL import Image, ImageChops
#!from rich import print  # noqa: A004  Shadowing built-in 'print'

import modules


from extentions.adetailer.aaaaaa.ui import WebuiInfo, adui, ordinal, suffix
from extentions.adetailer.adetailer import (
    get_models,
    mediapipe_predict,
    ultralytics_predict,
)
from extentions.adetailer.adetailer.args import (
    BBOX_SORTBY,
    BUILTIN_SCRIPT,
    INPAINT_BBOX_MATCH_MODES,
    SCRIPT_DEFAULT,
    ADetailerArgs,
    InpaintBBoxMatchMode,
    SkipImg2ImgOrig,
)
from extentions.adetailer.adetailer.common import PredictOutput, ensure_pil_image, safe_mkdir
from extentions.adetailer.adetailer.mask import (
    filter_by_ratio,
    filter_k_by,
    has_intersection,
    is_all_black,
    mask_preprocess,
    sort_bboxes,
)

from modules.config import default_adetail_tab
from modules.model_loader import load_file_from_url
if TYPE_CHECKING:
    from fastapi import FastAPI

PARAMS_TXT = "params.txt"

no_huggingface=False

def download_yola(name):
    if 'yolov8' in name:
        load_file_from_url(
            url='https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/YOLO8/'+name,
            model_dir=modules.config.path_yolo,
            file_name=name
        )
    return os.path.join(modules.config.path_yolo, name)

txt2img_submit_button = img2img_submit_button = None
txt2img_submit_button = cast(gr.Button, txt2img_submit_button)
img2img_submit_button = cast(gr.Button, img2img_submit_button)

def ui(is_img2img):
        num_models = default_adetail_tab
        ad_model_list = modules.config.yolo_filenames
        sampler_names = modules.flags.sampler_list
        scheduler_names = modules.flags.scheduler_list

        checkpoint_list = modules.config.model_filenames
        vae_list = [modules.flags.default_vae] + modules.config.vae_filenames

        webui_info = WebuiInfo(
            ad_model_list=ad_model_list,
            sampler_names=sampler_names,
            scheduler_names=scheduler_names,
            #!t2i_button=txt2img_submit_button,
            #!i2i_button=img2img_submit_button,
            checkpoints_list=checkpoint_list,
            vae_list=vae_list,
        )

        only_detect_tur, components,ad_model_dropdowns,ad_ckpt_dropdowns = adui(num_models, is_img2img, webui_info)
        only_detect = only_detect_tur[0]

        return only_detect, components,ad_model_dropdowns,ad_ckpt_dropdowns
def sort_bboxes_from_class(pred: PredictOutput) -> PredictOutput:
        sortby = BBOX_SORTBY[1]
        sortby_idx = BBOX_SORTBY.index(sortby)
        return sort_bboxes(pred, sortby_idx)
def pred_preprocessing(pred: PredictOutput, args):
        pred = filter_by_ratio(
            pred, low=args.ad_mask_min_ratio, high=args.ad_mask_max_ratio
        )
        pred = filter_k_by(pred, k=args.ad_mask_k, by=args.ad_mask_filter_method)
        pred = sort_bboxes_from_class(pred)
        masks = mask_preprocess(
            pred.masks,
            kernel=args.ad_dilate_erode,
            x_offset=args.ad_x_offset,
            y_offset=args.ad_y_offset,
            merge_invert=args.ad_mask_merge_invert,
        )

        return masks
def enabler(ad_component):
    tabs = [item for item in ad_component if isinstance(item, dict)]
    valid_tabs = [
        tab for tab in tabs
        if tab.get("ad_tab_enable") is True and tab.get("ad_model") != "None"
    ]
    return valid_tabs
def prompt_cut(pos_text: str, neg_text: str, target_len: int):    
    def process(s: str):
        parts = re.split(r"\s*\[SEP\]\s*", s.strip()) if s.strip() else []
        lst = [""] if not parts else parts
        return (lst + [lst[-1]] * target_len)[:target_len]
    
    return process(pos_text), process(neg_text)
