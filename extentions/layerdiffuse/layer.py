import gradio as gr
import modules.gradio_hijack as grh
import os
import functools
import torch
import numpy as np
import copy
import modules.config
#from modules import scripts
#from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from .lib_layerdiffusion.enums import ResizeMode
from .lib_layerdiffusion.utils import rgba2rgbfp32, to255unit8, crop_and_resize_image, forge_clip_encode
from enum import Enum
#from modules.paths import models_path
#from backend import utils, memory_management

from .lib_layerdiffusion.models import TransparentVAEDecoder, TransparentVAEEncoder
#from backend.sampling.sampling_function import sampling_prepare
from modules.model_loader import load_file_from_url
from .lib_layerdiffusion.attention_sharing import AttentionSharingPatcher
#from modules.canvas.canvas import ForgeCanvas
#from modules import images
from PIL import Image, ImageOps

import ldm_patched.modules.utils

def is_model_loaded(model):
    return any(model == m.model for m in memory_management.current_loaded_models)


layer_model_root = os.path.join(os.path.dirname(modules.config.path_vae), 'layer_model')
os.makedirs(layer_model_root, exist_ok=True)

vae_transparent_encoder = None
vae_transparent_decoder = None


class LayerMethod(Enum):
#    FG_ONLY_ATTN_SD15 = "(SD1.5) Only Generate Transparent Image (Attention Injection)"
#    FG_TO_BG_SD15 = "(SD1.5) From Foreground to Background (need batch size 2)"
#    BG_TO_FG_SD15 = "(SD1.5) From Background to Foreground (need batch size 2)"
#    JOINT_SD15 = "(SD1.5) Generate Everything Together (need batch size 3)"
    FG_ONLY_ATTN = "(SDXL) Only Generate Transparent Image (Attention Injection)"
    FG_ONLY_CONV = "(SDXL) Only Generate Transparent Image (Conv Injection)"
    FG_TO_BLEND = "(SDXL) From Foreground to Blending"
    FG_BLEND_TO_BG = "(SDXL) From Foreground and Blending to Background"
    BG_TO_BLEND = "(SDXL) From Background to Blending"
    BG_BLEND_TO_FG = "(SDXL) From Background and Blending to Foreground"


@functools.lru_cache(maxsize=2)
def load_layer_model_state_dict(filename):
    return ldm_patched.modules.utils.load_torch_file(filename, safe_load=True)


#class LayerDiffusionForForge(scripts.Script):
#    def title(self):
#        return "LayerDiffuse"
#
#    def show(self, is_img2img):
#        return scripts.AlwaysVisible
def prepare_layer(m):
        #ctrls += [method_ld, weight_ld, ending_step_ld, fg_image_ld, bg_image_ld]
        #ctrls += [blend_image_ld, resize_mode_ld, output_origin_ld, fg_additional_prompt_ld, bg_additional_prompt_ld, blend_additional_prompt_ld]
    print('---------------',m) 
    method = LayerMethod(m)
#before_process_init_images
    if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
        vae_encoder = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors',
                model_dir=layer_model_root,
                file_name='vae_transparent_encoder.safetensors'
        )
    else:
        vae_encoder = None
            
#postprocess_image_after_composite
    if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
        vae_decoder = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors',
                model_dir=layer_model_root,
                file_name='vae_transparent_decoder.safetensors'
         )
    else:
        vae_decoder = None

 #process_before_every_sampling

    if method == LayerMethod.FG_ONLY_ATTN:
        model_path = load_file_from_url(
            url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors',
            model_dir=layer_model_root,
            file_name='layer_xl_transparent_attn.safetensors'
        )

    if method == LayerMethod.FG_ONLY_CONV:
        model_path = load_file_from_url(
            url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors',
            model_dir=layer_model_root,
            file_name='layer_xl_transparent_conv.safetensors'
        )

    if method == LayerMethod.BG_TO_BLEND:
        model_path = load_file_from_url(
            url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors',
            model_dir=layer_model_root,
            file_name='layer_xl_bg2ble.safetensors'
        )

    if method == LayerMethod.FG_TO_BLEND:
        model_path = load_file_from_url(
            url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors',
            model_dir=layer_model_root,
            file_name='layer_xl_fg2ble.safetensors'
        )

    if method == LayerMethod.BG_BLEND_TO_FG:
        model_path = load_file_from_url(
            url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bgble2fg.safetensors',
            model_dir=layer_model_root,
            file_name='layer_xl_bgble2fg.safetensors'
        )

    if method == LayerMethod.FG_BLEND_TO_BG:
        model_path = load_file_from_url(
            url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fgble2bg.safetensors',
            model_dir=layer_model_root,
            file_name='layer_xl_fgble2bg.safetensors'
        )
    
    return vae_encoder, vae_decoder, model_path

def vae_layer_decode(method,vae_decoder,latent,pixel):
    mod_number = 1
    method = LayerMethod(method)
    vae_transparent_decoder = TransparentVAEDecoder(ldm_patched.modules.utils.load_torch_file(vae_decoder))


    #i = pp.index

    #if i % mod_number == 0:
    #    latent = p.latents_after_sampling[i]
    #    pixel = p.pixels_after_sampling[i]

    lB, lC, lH, lW = latent.shape
    if lH != pixel.height // 8 or lW != pixel.width // 8:
        print('[LayerDiffuse] VAE zero latent mode.')
        latent = torch.zeros((lC, pixel.height // 8, pixel.width // 8)).to(latent)

    png, vis = vae_transparent_decoder.decode(latent, pixel)
    #pp.image = png
    #p.extra_result_images.append(vis)
    return png, vis



def ui():
    
    enabled = gr.Checkbox(label='Enabled', value=False)
    method = gr.Dropdown(choices=[e.value for e in LayerMethod], value=LayerMethod.FG_ONLY_ATTN.value, label="Method", type='value')
    gr.HTML('</br>')  # some strange gradio problems

    with gr.Row():
        with gr.Column(visible=False) as fg_col:
            gr.Markdown('Foreground fg_co')
            #fg_image = ForgeCanvas(numpy=True, no_scribbles=True, height=300).background
            fg_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=300, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
        with gr.Column(visible=False) as bg_col:
            gr.Markdown('Background bg_col')
            #bg_image = ForgeCanvas(numpy=True, no_scribbles=True, height=300).background
            bg_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=300, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
        with gr.Column(visible=False) as blend_col:
            gr.Markdown('Blending blend_col')
            #blend_image = ForgeCanvas(numpy=True, no_scribbles=True, height=300).background
            blend_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=300, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)

    gr.HTML('</br>')  # some strange gradio problems

    with gr.Row():
        weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=0.001,interactive=True)
        ending_step = gr.Slider(label="Stop At", value=1.0, minimum=0.0, maximum=1.0,interactive=True)

    fg_additional_prompt = gr.Textbox(placeholder="Additional prompt for foreground. fg_additional_prompt", visible=False, label='Foreground Additional Prompt')
    bg_additional_prompt = gr.Textbox(placeholder="Additional prompt for background. bg_additional_prompt", visible=False, label='Background Additional Prompt')
    blend_additional_prompt = gr.Textbox(placeholder="Additional prompt for blended image. blend_additional_prompt", visible=False, label='Blended Additional Prompt')

    resize_mode = gr.Radio(choices=[e.value for e in ResizeMode], value=ResizeMode.CROP_AND_RESIZE.value, label="Resize Mode", type='value', visible=False, interactive=True)
    output_origin = gr.Checkbox(label='Output original mat for img2img', value=False, visible=False)
#    FG_ONLY_ATTN_SD15 = "(SD1.5) Only Generate Transparent Image (Attention Injection)"
#    FG_TO_BG_SD15 = "(SD1.5) From Foreground to Background (need batch size 2)"
#    BG_TO_FG_SD15 = "(SD1.5) From Background to Foreground (need batch size 2)"
#    JOINT_SD15 = "(SD1.5) Generate Everything Together (need batch size 3)"
    FG_ONLY_ATTN = "(SDXL) Only Generate Transparent Image (Attention Injection)"
    FG_ONLY_CONV = "(SDXL) Only Generate Transparent Image (Conv Injection)"
    FG_TO_BLEND = "(SDXL) From Foreground to Blending"
    FG_BLEND_TO_BG = "(SDXL) From Foreground and Blending to Background"
    BG_TO_BLEND = "(SDXL) From Background to Blending"
    BG_BLEND_TO_FG = "(SDXL) From Background and Blending to Foreground"
    def method_changed(m):
        m = LayerMethod(m)
        if m == LayerMethod.FG_TO_BLEND:
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

        if m == LayerMethod.BG_TO_BLEND:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

        if m == LayerMethod.BG_BLEND_TO_FG:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

        if m == LayerMethod.FG_BLEND_TO_BG:
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

    method.change(method_changed, inputs=method, outputs=[fg_col, bg_col, blend_col, resize_mode, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt], show_progress=False, queue=False)

    return enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt



#def postprocess_image_after_composite(self, p: StableDiffusionProcessing, pp, *script_args, **kwargs):
def postprocess_image_after_composite(self, p, pp, *script_args, **kwargs):
    global vae_transparent_decoder, vae_transparent_encoder

    enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt = script_args
    if not enabled:
        return

    mod_number = 1
    method = LayerMethod(method)
    need_process = False

    if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
        need_process = True
        if vae_transparent_decoder is None:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors',
                model_dir=layer_model_root,
                file_name='vae_transparent_decoder.safetensors'
            )
            vae_transparent_decoder = TransparentVAEDecoder(ldm_patched.modules.utils.load_torch_file(model_path))

    if method in [LayerMethod.FG_ONLY_ATTN_SD15, LayerMethod.JOINT_SD15, LayerMethod.BG_TO_FG_SD15]:
        need_process = True
        if vae_transparent_decoder is None:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_vae_transparent_decoder.safetensors',
                model_dir=layer_model_root,
                file_name='layer_sd15_vae_transparent_decoder.safetensors'
            )
            vae_transparent_decoder = TransparentVAEDecoder(ldm_patched.modules.utils.load_torch_file(model_path))
        if method == LayerMethod.JOINT_SD15:
            mod_number = 3
        if method == LayerMethod.BG_TO_FG_SD15:
            mod_number = 2

    if not need_process:
        return

    i = pp.index

    if i % mod_number == 0:
        latent = p.latents_after_sampling[i]
        pixel = p.pixels_after_sampling[i]

        lC, lH, lW = latent.shape
        if lH != pixel.height // 8 or lW != pixel.width // 8:
            print('[LayerDiffuse] VAE zero latent mode.')
            latent = torch.zeros((lC, pixel.height // 8, pixel.width // 8)).to(latent)

        png, vis = vae_transparent_decoder.decode(latent['samples'], pixel)
        pp.image = png
        p.extra_result_images.append(vis)
    return

#def before_process_init_images(self, p: StableDiffusionProcessingImg2Img, pp, *script_args, **kwargs):
def before_process_init_images(method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt):
    global vae_transparent_decoder, vae_transparent_encoder

    #enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt = script_args
    #if not enabled:
    #    return

    method = LayerMethod(method)
    need_process = False

    if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
        need_process = True
        if vae_transparent_encoder is None:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors',
                model_dir=layer_model_root,
                file_name='vae_transparent_encoder.safetensors'
            )
            vae_transparent_encoder = TransparentVAEEncoder(ldm_patched.modules.utils.load_torch_file(model_path))

    #if method in [LayerMethod.FG_ONLY_ATTN_SD15, LayerMethod.JOINT_SD15, LayerMethod.BG_TO_FG_SD15]:
    #    need_process = True
    #    if vae_transparent_encoder is None:
    #        model_path = load_file_from_url(
    #            url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_vae_transparent_encoder.safetensors',
    #            model_dir=layer_model_root,
    #            file_name='layer_sd15_vae_transparent_encoder.safetensors'
    #        )
    #        vae_transparent_encoder = TransparentVAEEncoder(ldm_patched.modules.utils.load_torch_file(model_path))

    if not need_process:
        print('------------------------------------------------------')
        return

    input_png_raw = p.init_images[0]
    input_png_bg_grey = images.flatten(input_png_raw, (127, 127, 127)).convert('RGBA')
    p.init_images = [input_png_bg_grey]

    crop_region = pp['crop_region']
    image = input_png_raw

    if crop_region is None and p.resize_mode != 3:
        image = images.resize_image(p.resize_mode, image, p.width, p.height, force_RGBA=True)

    if crop_region is not None:
        image = image.crop(crop_region)
        image = images.resize_image(2, image, p.width, p.height, force_RGBA=True)

    latent_offset = vae_transparent_encoder.encode(image)

    vae = p.sd_model.forge_objects.vae.clone()

    def vae_regulation(posterior):
        z = posterior.mean + posterior.std * latent_offset.to(posterior.mean)
        return z

    vae.patcher.set_model_vae_regulation(vae_regulation)

    p.sd_model.forge_objects.vae = vae
    return
