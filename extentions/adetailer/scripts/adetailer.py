import re
import sys
import os
from pathlib import Path
import gradio as gr
from PIL import Image, ImageChops
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
from extentions.adetailer.adetailer.common import PredictOutput, ensure_pil_image
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

PARAMS_TXT = "params.txt"

no_huggingface=False
adetailer_dir = Path(modules.config.paths_checkpoints[0]).parent / "detection"
def download_yola(name):
    load_file_from_url(
        url='https://huggingface.co/Bingsu/adetailer/resolve/main/'+name+'.pt',
        model_dir=adetailer_dir,
        file_name=name+'.pt'
    )
    return os.path.join(adetailer_dir, name+'.pt')

#!model_mapping = get_models(
#!    adetailer_dir,
#!    huggingface=not no_huggingface,
#!    download_dir=adetailer_dir,  # ← КЛЮЧЕВОЙ параметр
#!)
yolo_model_list=[
            "face_yolov8n",
            "face_yolov8s",
            "hand_yolov8n",
            "person_yolov8n-seg",
            "person_yolov8s-seg",
            "yolov8x-worldv2",
            "mediapipe_face_full",
            "mediapipe_face_short",
            "mediapipe_face_mesh",
            "mediapipe_face_mesh_eyes_only"
            ]
def ui(is_img2img):
        num_models = default_adetail_tab
        ad_model_list = yolo_model_list
        sampler_names = modules.flags.sampler_list
        scheduler_names = modules.flags.scheduler_list

        checkpoint_list = modules.config.model_filenames
        vae_list =[modules.flags.default_vae] + modules.config.vae_filenames

        webui_info = WebuiInfo(
            ad_model_list=ad_model_list,
            sampler_names=sampler_names,
            scheduler_names=modules.flags.scheduler_list,
            checkpoints_list=checkpoint_list,
            vae_list=vae_list,
            engine=modules.flags.inpaint_engine_versions
        )
        components = adui(num_models, is_img2img, webui_info)
        return components

def enabler(ad_component):
    tabs = [item for item in ad_component if isinstance(item, dict)]
    valid_tabs = [
        tab for tab in tabs
        if tab.get("ad_tab_enable") is True and tab.get("ad_model") != "None"
    ]
    return valid_tabs

def sort_bboxes_main(pred: PredictOutput) -> PredictOutput:
        sortby = BBOX_SORTBY[1]
        sortby_idx = BBOX_SORTBY.index(sortby)
        return sort_bboxes(pred, sortby_idx)
def pred_preprocessing(pred: PredictOutput, args):
        pred = filter_by_ratio(
            pred, low=args.ad_mask_min_ratio, high=args.ad_mask_max_ratio
        )
        pred = filter_k_by(pred, k=args.ad_mask_k, by=args.ad_mask_filter_method)
        pred = sort_bboxes_main(pred)
        masks = mask_preprocess(
            pred.masks,
            kernel=args.ad_dilate_erode,
            x_offset=args.ad_x_offset,
            y_offset=args.ad_y_offset,
            merge_invert=args.ad_mask_merge_invert,
        )

        return masks

def prompt_cut(pos_text: str, neg_text: str, target_len: int):    
    def process(s: str):
        parts = re.split(r"\s*\[SEP\]\s*", s.strip()) if s.strip() else []
        lst = [""] if not parts else parts
        return (lst + [lst[-1]] * target_len)[:target_len]
    
    return process(pos_text), process(neg_text)
