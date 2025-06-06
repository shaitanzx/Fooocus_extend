

import sys
import os
import numpy as np

import cv2
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision.transforms.functional import normalize
from typing import List, Union, Dict, Set, Tuple
sys.path.append(os.path.abspath('extentions/CodeFormer'))
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY



def check_ckpts():
    pretrain_model_url = {
        'codeformer': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/codeformer.pth',
        'detection': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/detection_Resnet50_Final.pth',
        'parsing': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/parsing_parsenet.pth',
        'realesrgan': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/RealESRGAN_x2plus.pth'
    }
    # download weights
    if not os.path.exists('extentions/CodeFormer/weights/CodeFormer/codeformer.pth'):
        load_file_from_url(url=pretrain_model_url['codeformer'], model_dir='extentions/CodeFormer/weights/CodeFormer', progress=True, file_name=None)
    if not os.path.exists('extentions/CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
        load_file_from_url(url=pretrain_model_url['detection'], model_dir='extentions/CodeFormer/weights/facelib', progress=True, file_name=None)
    if not os.path.exists('extentions/CodeFormer/weights/facelib/parsing_parsenet.pth'):
        load_file_from_url(url=pretrain_model_url['parsing'], model_dir='extentions/CodeFormer/weights/facelib', progress=True, file_name=None)
    if not os.path.exists('extentions/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
        load_file_from_url(url=pretrain_model_url['realesrgan'], model_dir='extentions/CodeFormer/weights/realesrgan', progress=True, file_name=None)

def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="extentions/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler
def get_image(input_data: Union[list, np.ndarray]) -> np.ndarray:
    if isinstance(input_data, (list, tuple)) and len(input_data) > 0:        
        return input_data[0],True
    elif isinstance(input_data, np.ndarray):
        
        return input_data,False

def codeformer_process(image,face_align,background_enhance,face_upsample,upscale,codeformer_fidelity):
  codeform_array=[]
  check_ckpts()
  upsampler = set_realesrgan()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
    ).to(device)
  ckpt_path = "extentions/CodeFormer/weights/CodeFormer/codeformer.pth"
  checkpoint = torch.load(ckpt_path)["params_ema"]
  codeformer_net.load_state_dict(checkpoint)
  codeformer_net.eval()
  try: # global try
        # take the default setting for the demo
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

#        print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)
        face_align = face_align if face_align is not None else True
        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        has_aligned = not face_align
        upscale = 1 if has_aligned else upscale
        img_array,generator=get_image(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        print('\timage size:', img.shape)

        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned

            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            if face_helper.is_gray:
                print('\tgrayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )
        else:
            restored_img = restored_face

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        codeform_array.append(restored_img)
        if generator:
            return codeform_array
        else:
            return np.array(restored_img)

  except Exception as error:
        print('Global exception', error)
        return None, None

def codeformer_gen_gui():
    with gr.Row():
          codeformer_gen_enabled = gr.Checkbox(label="Enabled", value=False)
    with gr.Row():
        with gr.Column():      
          codeformer_gen_preface=gr.Checkbox(value=True, label="Pre_Face_Align")
          codeformer_gen_background_enhance=gr.Checkbox(label="Background Enchanced", value=True)
          codeformer_gen_face_upsample=gr.Checkbox(label="Face Upsample", value=True)
        with gr.Column(): 
          codeformer_gen_upscale = gr.Slider(label='Upscale', minimum=1.0, maximum=4.0, step=1.0, value=1,interactive=True)
          codeformer_gen_fidelity =gr.Slider(label='Codeformer_Fidelity', minimum=0, maximum=1, value=0.5, step=0.01, info='0 for better quality, 1 for better identity (default=0.5)')
    with gr.Row():
        gr.HTML('* \"CodeFormer\" is powered by sczhou. <a href="https://github.com/sczhou/CodeFormer" target="_blank">\U0001F4D4 Document</a>')
    return codeformer_gen_enabled,codeformer_gen_preface,codeformer_gen_background_enhance,codeformer_gen_face_upsample,codeformer_gen_upscale,codeformer_gen_fidelity