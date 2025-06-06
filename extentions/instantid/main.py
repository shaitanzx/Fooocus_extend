import cv2
import torch
import random
import numpy as np
import os

#import spaces

import PIL
from PIL import Image
from typing import Tuple

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from ..pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline, draw_kps
import ldm_patched.modules.model_management as model_management

# from controlnet_aux import OpenposeDetector

import gradio as gr

from .depth_anything.dpt import DepthAnything
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import torch.nn.functional as F
from torchvision.transforms import Compose
from huggingface_hub import hf_hub_download
import gc

def get_canny_image(image, t1=100, t2=200):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image, t1, t2)
    return Image.fromarray(edges, "L")


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image




def start(face_image_path,pose_image_path,num_steps,identitynet_strength_ratio,adapter_strength_ratio,canny_strength,
          depth_strength,controlnet_selection,guidance_scale,task,scheduler,enhance_face_region,base_model,loras,loras_path,async_task,pre_gen,imgs):
    
    model_management.interrupt_processing = False

    def get_depth_map(image):
    
        image = np.array(image) / 255.0

        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to("cuda")

        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        depth_image = Image.fromarray(depth)

        return depth_image


    # global variable
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/1k3d68.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/2d106det.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/genderage.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/glintr100.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/scrfd_10g_bnkps.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="extentions/instantid/checkpoints")
    
    app = FaceAnalysis(name='antelopev2', root='extentions/instantid', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    depth_anything = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14",cache_dir='extentions/instantid/checkpoints/LiheYoung',force_download=False).to(device).eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Path to InstantID models
    face_adapter = f"extentions/instantid/checkpoints/ip-adapter.bin"
    controlnet_path = f"extentions/instantid/checkpoints/ControlNetModel"

    # Load pipeline face ControlNetModel
    controlnet_identitynet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=dtype
    )

    # controlnet-pose/canny/depth
    # controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
    controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0-small"
    controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

    controlnet_canny = ControlNetModel.from_pretrained(
        controlnet_canny_model,cache_dir='extentions/instantid/checkpoints/canny_small',force_download=False,torch_dtype=dtype
    ).to(device)

    controlnet_depth = ControlNetModel.from_pretrained(
        controlnet_depth_model,cache_dir='extentions/instantid/checkpoints/depth_small',force_download=False,torch_dtype=dtype
    ).to(device)


    controlnet_map = {
        #"pose": controlnet_pose,
        "canny": controlnet_canny,
        "depth": controlnet_depth,
    }
    controlnet_map_fn = {
        #"pose": openpose,
        "canny": get_canny_image,
        "depth": get_depth_map,
    }


    print(f"InstantID: Loading {base_model}")
    pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
        base_model,
        controlnet=[controlnet_identitynet],
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

##    pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
##        pipe.scheduler.config
##    )
    loras = [lora for lora in loras if 'None' not in lora]
    adapters = []
    for index, lora in enumerate(loras):
        path_separator = os.path.sep
        lora_filename, lora_weight = lora
        lora_fullpath = loras_path + path_separator + lora_filename
        print(f"InstantID: Loading {lora_fullpath} with weight {lora_weight}")
        try:
            pipe.load_lora_weights(loras_path, weight_name=lora_filename, adapter_name=str(index))
            adapters.append({str(index): lora_weight})
        except ValueError:
            print(f"InstantID: {lora_filename} already loaded, continuing on...")
  


    adapter_names = [list(adapter.keys())[0] for adapter in adapters]
    adapter_weights = [list(adapter.values())[0] for adapter in adapters]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    pipe.fuse_lora() 


    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)
    pipe.image_proj_model.to("cuda")
    pipe.unet.to("cuda")


    scheduler_class_name = scheduler.split("-")[0]

    add_kwargs = {}
    if len(scheduler.split("-")) > 1:
        add_kwargs["use_karras_sigmas"] = True
    if len(scheduler.split("-")) > 2:
        add_kwargs["algorithm_type"] = "sde-dpmsolver++"
    scheduler = getattr(diffusers, scheduler_class_name)
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config, **add_kwargs)


    if face_image_path is None:
        raise gr.Error(
            f"Cannot find any input face image! Please upload the face image"
        )
    prompt=task['positive'][0]
    negative_prompt=task['negative'][0]

    if prompt is None:
        prompt = "a person"

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image, max_side=1024)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        raise gr.Error(
            f"Unable to detect a face in the image. Please upload a different photo with a clear face."
        )

    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
    )[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
    img_controlnet = face_image
    output_path=None
    if pre_gen==True:
        image_gen = Image.fromarray(imgs[0])
        # Путь к папке (если её нет - создаём)
        output_dir = "extentions/instantid"
        filename = "temp.png"
        output_path = os.path.join(output_dir, filename)  # -> "output_images/image.jpg"
        image_gen.save(output_path)
        pose_image_path=output_path
    if pose_image_path is not None:
        pose_image = load_image(pose_image_path)
        if output_path !=  None:
            os.remove(output_path)
        pose_image = resize_img(pose_image, max_side=1024)
        img_controlnet = pose_image
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)

        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(
                f"Cannot find any face in the reference image! Please upload another person image"
            )

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info["kps"])

        width, height = face_kps.size

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    if len(controlnet_selection) > 0:
        controlnet_scales = {
            #"pose": pose_strength,
            "canny": canny_strength,
            "depth": depth_strength,
        }
        pipe.controlnet = MultiControlNetModel(
            [controlnet_identitynet]
            + [controlnet_map[s] for s in controlnet_selection]
        )
        control_scales = [float(identitynet_strength_ratio)] + [
            controlnet_scales[s] for s in controlnet_selection
        ]
        control_images = [face_kps] + [
            controlnet_map_fn[s](img_controlnet).resize((width, height))
            for s in controlnet_selection
        ]
    else:
        pipe.controlnet = controlnet_identitynet
        control_scales = float(identitynet_strength_ratio)
        control_images = face_kps

    generator = torch.Generator(device=device).manual_seed(task['task_seed'])

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    preview_image=None
    def progress_id(pipe, step_index, timestep, callback_kwargs):

      interrupt_processing = model_management.interrupt_processing
      
      if interrupt_processing:
        
        pipe._interrupt =  True
              
      
      
      if step_index % 5 == 0 or step_index == 0:
        latents = callback_kwargs.get("latents")
        global preview_image
        with torch.no_grad():
            needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
            if needs_upcasting:
                    pipe.upcast_vae()
                    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
            
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = image.cpu()
            image_np = image.numpy()
            image_np = image_np.transpose(0, 2, 3, 1)
            image_np = (image_np * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            preview_image = image_np[0]
      async_task.yields.append(['preview', (
            int(15.0 + 85.0 * float(0) / float(num_steps)),
            f'InstatntID step {step_index}/{num_steps}',
            np.array(preview_image))])
      return callback_kwargs
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=control_images,
        control_mask=control_mask,
        controlnet_conditioning_scale=control_scales,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        callback_on_step_end=progress_id,
        generator=generator,
    ).images
    

    del pipe
    del app
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if model_management.interrupt_processing:
         return []

    return [np.array(images[0])]


    
def gui():
    with gr.Row():
        enable_instant = gr.Checkbox(label="Enabled", value=False)
    with gr.Row():
        with gr.Column():
                # upload face image
                face_file = gr.Image(label="Upload a photo of your face", type="filepath")
        with gr.Column():
                # optional: upload a reference pose image
                pose_file = gr.Image(label="Upload a reference pose image (Optional)",type="filepath")
                pre_gen = gr.Checkbox(label="Pregeneration image", value=False)
    with gr.Row():
            # strength
            identitynet_strength_ratio = gr.Slider(label="IdentityNet strength (for fidelity)",minimum=0,maximum=1.5,step=0.05,value=0.80,interactive=True)
            adapter_strength_ratio = gr.Slider(label="Image adapter strength (for detail)",minimum=0,maximum=1.5,step=0.05,value=0.80,interactive=True)
    with gr.Row():
            with gr.Accordion("Controlnet"):
                controlnet_selection = gr.CheckboxGroup(
                    ["canny", "depth"], label="Controlnet", value=["depth"],
                    info="Use pose for skeleton inference, canny for edge detection, and depth for depth map estimation. You can try all three to control the generation process",
                    interactive=True
                )

                canny_strength = gr.Slider(label="Canny strength",minimum=0,maximum=1.5,step=0.05,value=0.40,interactive=True)
                depth_strength = gr.Slider(label="Depth strength",minimum=0,maximum=1.5,step=0.05,value=0.40,interactive=True)
    with gr.Row():
            schedulers = [
                 "DEISMultistepScheduler",
                 "HeunDiscreteScheduler",
                 "EulerDiscreteScheduler",
                 "DPMSolverMultistepScheduler",
                 "DPMSolverMultistepScheduler-Karras",
                 "DPMSolverMultistepScheduler-Karras-SDE"]
            scheduler = gr.Dropdown(
                    label="Schedulers",
                    choices=schedulers,
                    value="EulerDiscreteScheduler",
                    interactive=True
                )
    with gr.Row():
            enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True,interactive=True)
      with gr.Row():
    gr.HTML('* \"InstantID\" is powered by InstantX Research. <a href="https://github.com/instantX-research/InstantID" target="_blank">\U0001F4D4 Document</a>')


    return enable_instant,face_file,pose_file,identitynet_strength_ratio,adapter_strength_ratio,controlnet_selection,canny_strength,depth_strength,scheduler,enhance_face_region,pre_gen


