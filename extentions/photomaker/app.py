import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import sys
from PIL import Image

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter

from huggingface_hub import hf_hub_download
import gradio as gr
import modules.gradio_hijack as grh
import modules.config

from .pipeline_t2i_adapter import PhotoMakerStableDiffusionXLAdapterPipeline
from .face_utils import FaceAnalysis2, analyze_faces

from .style_template import styles
from .aspect_ratio_template import aspect_ratios
import ldm_patched.modules.model_management as model_management
import gc

# global variable


def generate_image(
    upload_images, task,
    output_w,output_h, 
    num_steps,
    style_strength_ratio, 
    guidance_scale, 
    use_doodle,
    sketch_image,
    adapter_conditioning_scale,
    adapter_conditioning_factor,
    base_model_path,
    loras,loras_path,async_task
):
    model_management.interrupt_processing = False
    prompt=task['positive'][0]
    negative_prompt=task['negative'][0]
    seed=task['task_seed']

    face_detector = FaceAnalysis2(providers=['CPUExecutionProvider'],root="",allowed_modules=['detection', 'recognition'])
    face_detector.prepare(ctx_id=0, det_size=(640, 640))

#try:
#    if torch.cuda.is_available():
#        device = "cuda"
#    elif sys.platform == "darwin" and torch.backends.mps.is_available():
#        device = "mps"
#    else:
#        device = "cpu"
#except:
#    device = "cpu"
    device = "cuda"


    enable_doodle_arg = False
    photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", local_dir="extentions/photomaker/model", repo_type="model")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    if device == "mps":
        torch_dtype = torch.float16
    
# load adapter
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch_dtype, variant="fp16",cache_dir='extentions/photomaker/model'
    ).to(device)

    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_single_file(
        base_model_path, 
        adapter=adapter, 
        torch_dtype=torch_dtype,
        use_safetensors=True, 
        variant="fp16",
    ).to(device)

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img",
        pm_version="v2",
    )

    pipe.id_encoder.to(device)

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])

    loras = [lora for lora in loras if 'None' not in lora]
    adapters = []
    for index, lora in enumerate(loras):
        path_separator = os.path.sep
        lora_filename, lora_weight = lora
        lora_fullpath = loras_path + path_separator + lora_filename
        print(f"PhotoMaker: Loading {lora_fullpath} with weight {lora_weight}")
        try:
            pipe.load_lora_weights(loras_path, weight_name=lora_filename, adapter_name=str(index))
            adapters.append({str(index): lora_weight})
        except ValueError:
            print(f"PhotoMaker: {lora_filename} already loaded, continuing on...")
  


    adapter_names = [list(adapter.keys())[0] for adapter in adapters]
    adapter_weights = [list(adapter.values())[0] for adapter in adapters]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    pipe.fuse_lora()
    pipe.to(device)





    if use_doodle:
        sketch_image = sketch_image["composite"]
        r, g, b, a = sketch_image.split()
        sketch_image = a.convert("RGB")
        sketch_image = TF.to_tensor(sketch_image) > 0.5 # Inversion 
        sketch_image = TF.to_pil_image(sketch_image.to(torch.float32))
        adapter_conditioning_scale = adapter_conditioning_scale
        adapter_conditioning_factor = adapter_conditioning_factor
    else:
        adapter_conditioning_scale = 0.
        adapter_conditioning_factor = 0.
        sketch_image = None

    # check the trigger word
    image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    input_ids = pipe.tokenizer.encode(prompt)
    if image_token_id not in input_ids:
        raise gr.Error(f"Cannot find the trigger word '{pipe.trigger_word}' in text prompt! Please refer to step 2️⃣")

    if input_ids.count(image_token_id) > 1:
        raise gr.Error(f"Cannot use multiple trigger words '{pipe.trigger_word}' in text prompt!")



    if upload_images is None:
        raise gr.Error(f"Cannot find any input face image! Please refer to step 1️⃣")

    input_id_images = []
    file_paths = [file.name for file in upload_images]
    for img in file_paths:
        input_id_images.append(load_image(img))
    
    id_embed_list = []

    for img in input_id_images:
        img = np.array(img)
        img = img[:, :, ::-1]
        faces = analyze_faces(face_detector, img)
        if len(faces) > 0:
            id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

    if len(id_embed_list) == 0:
        raise gr.Error(f"No face detected, please update the input face image(s)")
    
    id_embeds = torch.stack(id_embed_list)

    generator = torch.Generator(device=device).manual_seed(seed)

    #print("Start inference...")
    #print(f"[Debug] Seed: {seed}")
    #print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"Style strength on {start_merge_step} step")
    preview_image=None
    def progress_pm(step, timestep, latents):
        global preview_image
        interrupt_processing = model_management.interrupt_processing
        if interrupt_processing:
            pipe._interrupt =  True
        if step % 5 == 0 or step == 0:
            with torch.no_grad():

                latents = 1 / 0.18215 * latents

                image = pipe.vae.decode(latents).sample

                image = (image / 2 + 0.5).clamp(0, 1)

            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # convert to PIL Images
                image = pipe.numpy_to_pil(image)[0]
                preview_image=image

        async_task.yields.append(['preview', (
                            int(15.0 + 85.0 * float(0) / float(num_steps)),
                            f'PhotoMaker Step {step}/{num_steps}',
                            preview_image)])
    images = pipe(
        prompt=prompt,
        width=output_w,
        height=output_h,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        callback=progress_pm,
        generator=generator,
        guidance_scale=guidance_scale,
        id_embeds=id_embeds,
        image=sketch_image,
        adapter_conditioning_scale=adapter_conditioning_scale,
        adapter_conditioning_factor=adapter_conditioning_factor,
    ).images


   
      

    del pipe
    del face_detector
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if model_management.interrupt_processing:
         return []

    return [np.array(images[0])]

def swap_to_gallery(files):
    file_paths = [file.name for file in files]  # Получаем пути из временных файлов
    return gr.update(value=file_paths, visible=True), gr.update(visible=True), gr.update(visible=False)

def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    




def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted([os.path.join(folder_name, basename) for basename in image_basename_list])
    return image_path_list



def gui():

    with gr.Blocks() as demo:
        with gr.Row():
            enable_pm = gr.Checkbox(label="Enabled", value=False)
        with gr.Row():
            gr.HTML('* You MUST USE the trigger word \"img\", eg: \"a photo of a man/woman img\"')
        with gr.Row():
            with gr.Column():
                files = gr.Files(
                        label="Drag (Select) 1 or more photos of your face",
                        file_types=["image"]
                    )
                uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=300)
                with gr.Column(visible=False) as clear_button:
                    remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")

                enable_doodle = gr.Checkbox(
                    label="Enable Drawing Doodle for Control", value=False,
                    info="After enabling this option, PhotoMaker will generate content based on your doodle on the canvas, driven by the T2I-Adapter (Quality may be decreased)",
                    visible=False)
                with gr.Accordion("T2I-Adapter-Doodle (Optional)", visible=False) as doodle_space:
                    with gr.Row():
                        sketch_image = gr.Sketchpad(
                            label="Canvas",
                            type="pil",
                            crop_size=[1024,1024],
                            layers=False,
                            canvas_size=(350, 350),
                            #brush=gr.Brush(default_size=5, colors=["#000000"], color_mode="fixed")
                        )

                    with gr.Row():
                        with gr.Group():
                            adapter_conditioning_scale = gr.Slider(
                                label="Adapter conditioning scale",
                                minimum=0.5,
                                maximum=1,
                                step=0.1,
                                value=0.7,
                            )
                            adapter_conditioning_factor = gr.Slider(
                                label="Adapter conditioning factor",
                                info="Fraction of timesteps for which adapter should be applied",
                                minimum=0.5,
                                maximum=1,
                                step=0.1,
                                value=0.8,
                            )
                with gr.Row():
                    style_strength_ratio = gr.Slider(
                        label="Style strength (%)",
                        minimum=15,
                        maximum=50,
                        step=1,
                        value=20,
                    )

            files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
            remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])



        with gr.Row():
          gr.HTML('* \"PhotoMaker\" is powered by TencentARC. <a href="https://github.com/TencentARC/PhotoMaker" target="_blank">\U0001F4D4 Document</a>')

    return enable_pm,files,style_strength_ratio,enable_doodle,sketch_image,adapter_conditioning_scale,adapter_conditioning_factor
