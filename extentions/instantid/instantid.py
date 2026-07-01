import torch
import os
import ldm_patched.modules.controlnet
import gradio as gr
import modules.gradio_hijack as grh
import numpy as np
import math
import cv2
import PIL.Image
import ldm_patched.modules.conds as conds
import modules.core as core
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention

from insightface.app import FaceAnalysis

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F

def gui():
    with gr.Row():
        enable_instant = gr.Checkbox(label="Enabled", value=False)
    with gr.Row():
        with gr.Column():
            # upload face image
            face_file = grh.Image(label="Upload a photo of your face", type="filepath")
            with gr.Row():
                identitynet_strength_ratio = gr.Slider(label="IdentityNet strength (for fidelity)",minimum=0,maximum=1.5,step=0.001,value=0.80,interactive=True)
                adapter_strength_ratio = gr.Slider(label="Image adapter strength (for detail)",minimum=0,maximum=1.5,step=0.001,value=0.80,interactive=True)
            with gr.Row():
                start_instant = gr.Slider(label="Start at",minimum=0,maximum=1,step=0.001,value=0.0,interactive=True)
                end_instant = gr.Slider(label="End at",minimum=0,maximum=1,step=0.001,value=1.00,interactive=True)


        with gr.Column():
            # optional: upload a reference pose image
            pose_file = grh.Image(label="Upload a reference pose image (Optional)",type="numpy")
            with gr.Column():
                with gr.Row():
                    canny_instant = gr.Checkbox(label='PyraCanny', value=True, container=False, elem_classes='min_check')
                with gr.Row():
                    canny_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=0.5,interactive=True)
                    canny_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=1,interactive=True)
            with gr.Column():
                with gr.Row():
                    cpds_instant = gr.Checkbox(label='CPDS', value=False, container=False, elem_classes='min_check')
                with gr.Row():
                    cdps_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=0.5,interactive=True)
                    cpds_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=1,interactive=True)                    

    with gr.Row():
            tips = r"""
### Usage tips of InstantID
1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."    
2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
3. If you find that text control is not as expected, decrease Adapter strength.
4. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
"""
            gr.Markdown(value=tips)
    with gr.Row():
        gr.HTML('* \"InstantID\" is powered by InstantX Research. <a href="https://github.com/instantX-research/InstantID" target="_blank">\U0001F4D4 Document</a>')


    return enable_instant,face_file,pose_file,identitynet_strength_ratio,adapter_strength_ratio,start_instant, end_instant,canny_instant,canny_stop,canny_weight,cpds_instant,cdps_stop,cpds_weight


def get_or_load_instantid_controlnet():

    ctrl_path = "extentions/instantid/checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors"
    
    print(f"[InstantID CN] Loading ControlNet into memory")
    
    try:
        control_net = core.load_controlnet(ctrl_path)
        print("[InstantID CN] ✅ ControlNet loaded!")
        return control_net
    except Exception as e:
        print(f"[InstantID CN] ❌ Loading error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class InstantID(torch.nn.Module):
    def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()        
        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.image_proj_model = self.init_proj()
        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        self.ip_layers = To_KV(instantid_model["ip_adapter"])


    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim, depth=4, dim_head=64, heads=20,
            num_queries=self.clip_extra_context_tokens, embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim, ff_mult=4
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.to_kvs = torch.nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value

def _set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()
    
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(instantid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(instantid_attention, **patch_kwargs)
def load_instantid_model(ckpt_path):
    print(f"[InstantID] Loading IP-Adaper into memory")


    print("[InstantID] Please wait...")
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("[InstantID] IP-Adaper is loaded")

    def extract_all_tensors(d, prefix=""):
        tensors = {}
        if isinstance(d, dict):
            for k, v in d.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, torch.Tensor):
                    tensors[new_prefix] = v
                else:
                    tensors.update(extract_all_tensors(v, new_prefix))
        return tensors

    sd = extract_all_tensors(pl_sd)


    st_model = {"image_proj": {}, "ip_adapter": {}}

    for key, tensor in sd.items():
        if key.startswith("image_proj."):
            st_model["image_proj"][key.replace("image_proj.", "")] = tensor
        elif key.startswith("ip_adapter."):
            st_model["ip_adapter"][key.replace("ip_adapter.", "")] = tensor
        elif "latents" in key or "proj_in" in key or "layers." in key:
            st_model["image_proj"][key] = tensor
        elif "to_k_ip" in key or "to_v_ip" in key:
            st_model["ip_adapter"][key] = tensor

    output_dim = None
    for key in st_model["ip_adapter"].keys():
        if "to_k_ip" in key:
            output_dim = st_model["ip_adapter"][key].shape[1]
            break
    
    instantid = InstantID(
        st_model,
        cross_attention_dim=1280,
        output_cross_attention_dim=output_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=16,
    )

    return instantid

def apply(image_path, pose_path, cn_strength, ip_weight, unet_model, positive, negative, sigma_min, sigma_max,
          width, height,start,end):
    global instantid_model

    insightface = FaceAnalysis(name='antelopev2', root='extentions/instantid', providers=['CPUExecutionProvider'])
    insightface.prepare(ctx_id=0, det_size=(640, 640))

    instantid_file = "extentions/instantid/checkpoints/ip-adapter.bin"

    combine_embeds='average'
    noise=0.35

    device = torch.device(torch.cuda.current_device())
    dtype = torch.float16

    print(f"[InstantID] ip_weight={ip_weight}, cn_strength={cn_strength}")
    print(f"[InstantID CN]  start={start}, end={end}")
    
    img_bgr = cv2.imread(image_path)

    insightface.det_model.input_size = (640, 640)
    faces = insightface.get(img_bgr)
    if not faces:
        raise Exception('Reference Image: No face detected.')
    print(f"[InstantID] Faces: {len(faces)}")
    
    face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    
    raw_embed = torch.from_numpy(face['embedding'])
    clip_embed = raw_embed.view(1, 1, -1).to(device, dtype=dtype)
    print(f"  -> Форма clip_embed: {clip_embed.shape}")
    image_ref = img_bgr

    if pose_path is not None:
        pose_image = cv2.cvtColor(pose_path, cv2.COLOR_RGB2BGR)
        faces = insightface.get(pose_image)
        face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        image_ref = pose_image

    kps_img_pil = draw_kps(image_ref, face['kps'])
    face_kps = T.ToTensor()(kps_img_pil).permute(1, 2, 0).unsqueeze(0)

    if clip_embed.shape[0] > 1:
        if combine_embeds == 'average':
            clip_embed = torch.mean(clip_embed, dim=0, keepdim=True)
        elif combine_embeds == 'norm average':
            clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0, keepdim=True)

    if noise > 0:
        seed = int(torch.sum(clip_embed).item()) % 1000000007
        torch.manual_seed(seed)
        clip_embed_zeroed = noise * torch.rand_like(clip_embed)
    else:
        clip_embed_zeroed = torch.zeros_like(clip_embed)

    instantid_model.to(device, dtype=dtype)
    image_prompt_embeds, uncond_image_prompt_embeds = instantid_model.get_image_embeds(
        clip_embed, clip_embed_zeroed
    )
    
    image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)

    patch_kwargs = {
        "ipadapter": instantid_model,
        "weight": ip_weight,
        "cond": image_prompt_embeds,
        "uncond": uncond_image_prompt_embeds,
        "mask": None, 
        "sigma_start": sigma_max,
        "sigma_end": sigma_min,
    }

    number = 0
    for id in [4, 5, 7, 8]: 
        block_indices = range(2) if id in [4, 5] else range(10) 
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(unet_model, patch_kwargs, ("input", id, index))
            number += 1
            
    for id in range(6): 
        block_indices = range(2) if id in [3, 4, 5] else range(10) 
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(unet_model, patch_kwargs, ("output", id, index))
            number += 1
            
    for index in range(10):
        patch_kwargs["module_key"] = str(number*2+1)
        _set_model_patch_replace(unet_model, patch_kwargs, ("middle", 1, index))
        number += 1
        
    control_net = get_or_load_instantid_controlnet()   

    face_kps_resized = torch.nn.functional.interpolate(
        face_kps.movedim(-1, 1),  # [1, 3, H, W]
        size=(height, width),
        mode='bilinear',
        align_corners=False
    ).movedim(1, -1)  # [1, gen_height, gen_width, 3]

    control_hint = face_kps_resized.movedim(-1, 1).to(device=device, dtype=dtype)
        
    cnets = {}
    cond_uncond = []
    is_cond = True

    for cond_idx, conditioning in enumerate([positive, negative]):
        cond_name = "positive" if cond_idx == 0 else "negative"
            
        c = []
        for t_idx, t in enumerate(conditioning):
            d = t[1].copy()
              
            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(
                    control_hint, 
                    cn_strength, 
                    (start, end)
                )
                if prev_cnet is not None:
                    c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
                
            embed_to_use = image_prompt_embeds if is_cond else uncond_image_prompt_embeds


                
            d['cross_attn_controlnet'] = conds.CONDCrossAttn(embed_to_use.to(device=device, dtype=dtype))

            n = [t[0], d]
            c.append(n)
            
        cond_uncond.append(c)
        is_cond = False
        
    final_positive, final_negative = cond_uncond[0], cond_uncond[1]

    del insightface, control_net,cnets
    del face_kps, face_kps_resized, control_hint

    del clip_embed, clip_embed_zeroed
   
    return unet_model, final_positive, final_negative
