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
from extentions.tutorial import tutorial
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention

from insightface.app import FaceAnalysis

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F

# SD1.5 InstantID adapter layers. Order must match module_key sequence consumed by
# instantid_attention() (1, 3, 5, ...). Diffusers names identify checkpoint tensors;
# block tuples identify ldm_patched SD1.5 transformer locations.
SD15_INSTANTID_LAYERS = [
    ("down_blocks.0.attentions.0.transformer_blocks.0.attn2", ("input", 1, 0)),
    ("down_blocks.0.attentions.1.transformer_blocks.0.attn2", ("input", 2, 0)),
    ("down_blocks.1.attentions.0.transformer_blocks.0.attn2", ("input", 4, 0)),
    ("down_blocks.1.attentions.1.transformer_blocks.0.attn2", ("input", 5, 0)),
    ("down_blocks.2.attentions.0.transformer_blocks.0.attn2", ("input", 7, 0)),
    ("down_blocks.2.attentions.1.transformer_blocks.0.attn2", ("input", 8, 0)),
    ("mid_block.attentions.0.transformer_blocks.0.attn2", ("middle", 0, 0)),
    ("up_blocks.0.attentions.0.transformer_blocks.0.attn2", ("output", 0, 0)),
    ("up_blocks.0.attentions.1.transformer_blocks.0.attn2", ("output", 1, 0)),
    ("up_blocks.0.attentions.2.transformer_blocks.0.attn2", ("output", 2, 0)),
    ("up_blocks.1.attentions.0.transformer_blocks.0.attn2", ("output", 3, 0)),
    ("up_blocks.1.attentions.1.transformer_blocks.0.attn2", ("output", 4, 0)),
    ("up_blocks.1.attentions.2.transformer_blocks.0.attn2", ("output", 5, 0)),
    ("up_blocks.2.attentions.0.transformer_blocks.0.attn2", ("output", 6, 0)),
    ("up_blocks.2.attentions.1.transformer_blocks.0.attn2", ("output", 7, 0)),
    ("up_blocks.2.attentions.2.transformer_blocks.0.attn2", ("output", 8, 0)),
]

SD15_INSTANTID_DIR = "extentions/instantid/checkpoints/sd15"

def gui():
    with gr.Row():
        enable_instant = gr.Checkbox(label="Enabled", value=False)
        tutorial("instantid/readme.md")      
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


def get_or_load_instantid_controlnet(is_sd15=False):
    if is_sd15:
        ctrl_path = os.path.join(SD15_INSTANTID_DIR, "controlnet.ckpt")
    else:
        ctrl_path = (
            "extentions/instantid/checkpoints/ControlNetModel/"
            "diffusion_pytorch_model.safetensors"
        )

    print(f"[InstantID CN] Loading ControlNet: {ctrl_path}")
    try:
        control_net = core.load_controlnet(ctrl_path)
        if control_net is None:
            raise RuntimeError(f"core.load_controlnet returned None: {ctrl_path}")
        print("[InstantID CN] ControlNet loaded")
        return control_net
    except Exception as e:
        print(f"[InstantID CN] Loading error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

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


def draw_2d_kps(image, kps, point_radius=3, write_numbers=False, overlap_image=False):
    """Draw TheDenk SD1.5's 106 facial landmarks using its trained color groups."""
    kps = np.asarray(kps)
    h, w = image.shape[:2]
    out_img = image.copy() if overlap_image else np.zeros((h, w, 3), dtype=np.uint8)

    face_areas = (
        (range(0, 33), (0, 255, 255)),
        (range(33, 52), (255, 0, 0)),
        (range(52, 72), (255, 255, 0)),
        (range(72, 87), (0, 0, 255)),
        (range(87, 106), (0, 255, 0)),
    )

    if kps.shape != (106, 2):
        raise ValueError(
            f"Expected InsightFace landmark_2d_106 with shape (106, 2), got {kps.shape}"
        )

    for index, (x, y) in enumerate(kps):
        color = next(color for indices, color in face_areas if index in indices)
        x, y = int(x), int(y)
        out_img = cv2.circle(out_img, (x, y), point_radius, color, -1)
        if write_numbers:
            out_img = cv2.putText(
                out_img, str(index), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (255, 255, 255), 1, cv2.LINE_AA
            )
    return out_img


def draw_sd15_control_hint(image, face):
    """Build the SD1.5 InstantID hint: five keypoints plus 106 face landmarks."""
    landmarks_106 = face.get("landmark_2d_106")
    if landmarks_106 is None:
        raise RuntimeError(
            "InsightFace did not return landmark_2d_106; check antelopev2 model files."
        )

    five_points = np.asarray(draw_kps(image, face["kps"]), dtype=np.uint8)
    return draw_2d_kps(
        five_points,
        landmarks_106,
        point_radius=3,
        write_numbers=False,
        overlap_image=True,
    )


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


def load_instantid_model_sd15():
    """Load TheDenk SD1.5 adapter and projection checkpoints."""
    adapter_path = os.path.join(SD15_INSTANTID_DIR, "ip-state.ckpt")
    image_proj_path = os.path.join(SD15_INSTANTID_DIR, "image_proj.ckpt")

    print("[InstantID SD1.5] Loading adapter and image projection weights")
    adapter_data = torch.load(adapter_path, map_location="cpu", weights_only=False)
    image_proj_data = torch.load(image_proj_path, map_location="cpu", weights_only=False)

    adapter_state = adapter_data.get("state_dict", adapter_data)
    image_proj_state = image_proj_data.get("state_dict", image_proj_data)

    # TheDenk checkpoints may carry a leading 'module.' prefix.
    adapter_state = {
        key.removeprefix("module."): value
        for key, value in adapter_state.items()
        if isinstance(value, torch.Tensor)
    }
    image_proj_state = {
        key.removeprefix("module."): value
        for key, value in image_proj_state.items()
        if isinstance(value, torch.Tensor)
    }

    ip_adapter = {}
    for index, (layer_path, _) in enumerate(SD15_INSTANTID_LAYERS):
        module_key = str(index * 2 + 1)
        for projection in ("k", "v"):
            source_key = f"{layer_path}.processor.to_{projection}_ip.weight"
            if source_key not in adapter_state:
                raise KeyError(
                    f"Missing tensor {source_key} in {adapter_path}. "
                    "Check that this is the TheDenk SD1.5 ip-state.ckpt."
                )
            ip_adapter[f"{module_key}.to_{projection}_ip.weight"] = adapter_state[source_key]

    output_dim = int(ip_adapter["1.to_k_ip.weight"].shape[1])
    instantid = InstantID(
        {"image_proj": image_proj_state, "ip_adapter": ip_adapter},
        cross_attention_dim=1280,
        output_cross_attention_dim=output_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=16,
    )
    print(f"[InstantID SD1.5] Loaded adapter; cross-attention output dim={output_dim}")
    return instantid


def apply(image_path, pose_path, cn_strength, ip_weight, unet_model, positive, negative,
          sigma_min, sigma_max, width, height, start, end, is_sd15=False):
    """Apply InstantID while isolating SDXL and SD1.5 model assets and patches."""
    global instantid_model

    if is_sd15:
        instantid_model = load_instantid_model_sd15()
    else:
        instantid_file = "extentions/instantid/checkpoints/ip-adapter.bin"
        instantid_model = load_instantid_model(instantid_file)

    insightface = FaceAnalysis(
        name="antelopev2",
        root="extentions/instantid",
        providers=["CPUExecutionProvider"],
    )
    insightface.prepare(ctx_id=0, det_size=(640, 640))
    insightface.det_model.input_size = (640, 640)

    combine_embeds = "average"
    noise = 0.35
    device = torch.device(torch.cuda.current_device())
    dtype = torch.float16

    print(f"[InstantID] architecture={'SD1.5' if is_sd15 else 'SDXL'}")
    print(f"[InstantID] ip_weight={ip_weight}, cn_strength={cn_strength}")
    print(f"[InstantID CN] start={start}, end={end}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read InstantID face image: {image_path}")

    source_faces = insightface.get(img_bgr)
    if not source_faces:
        raise RuntimeError("Reference Image: No face detected.")
    print(f"[InstantID] Source faces: {len(source_faces)}")

    source_face = max(
        source_faces,
        key=lambda face: (face["bbox"][2] - face["bbox"][0])
        * (face["bbox"][3] - face["bbox"][1]),
    )
    raw_embed = torch.from_numpy(source_face["embedding"])
    clip_embed = raw_embed.view(1, 1, -1).to(device=device, dtype=dtype)

    # Identity comes from source photo; optional pose/reference image supplies geometry.
    image_ref = img_bgr
    face_for_hint = source_face
    if pose_path is not None:
        if not isinstance(pose_path, np.ndarray):
            raise TypeError("pose_path must be a numpy RGB image when provided")
        pose_image = cv2.cvtColor(pose_path, cv2.COLOR_RGB2BGR)
        pose_faces = insightface.get(pose_image)
        if not pose_faces:
            raise RuntimeError("No face detected in the InstantID pose/reference image.")
        face_for_hint = max(
            pose_faces,
            key=lambda face: (face["bbox"][2] - face["bbox"][0])
            * (face["bbox"][3] - face["bbox"][1]),
        )
        image_ref = pose_image

    if is_sd15:
        control_hint_np = draw_sd15_control_hint(image_ref, face_for_hint)
        face_kps = (
            torch.from_numpy(control_hint_np)
            .to(dtype=torch.float32)
            .div_(255.0)
            .unsqueeze(0)
        )
    else:
        # Keep the existing SDXL preprocessing unchanged.
        kps_img_pil = draw_kps(image_ref, face_for_hint["kps"])
        face_kps = T.ToTensor()(kps_img_pil).permute(1, 2, 0).unsqueeze(0)

    if clip_embed.shape[0] > 1:
        if combine_embeds == "average":
            clip_embed = torch.mean(clip_embed, dim=0, keepdim=True)
        elif combine_embeds == "norm average":
            clip_embed = torch.mean(
                clip_embed / torch.norm(clip_embed, dim=0, keepdim=True),
                dim=0,
                keepdim=True,
            )

    if noise > 0:
        seed = int(torch.sum(clip_embed).item()) % 1000000007
        torch.manual_seed(seed)
        clip_embed_zeroed = noise * torch.rand_like(clip_embed)
    else:
        clip_embed_zeroed = torch.zeros_like(clip_embed)

    instantid_model.to(device=device, dtype=dtype)
    image_prompt_embeds, uncond_image_prompt_embeds = instantid_model.get_image_embeds(
        clip_embed, clip_embed_zeroed
    )
    image_prompt_embeds = image_prompt_embeds.to(device=device, dtype=dtype)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device=device, dtype=dtype)

    patch_kwargs = {
        "ipadapter": instantid_model,
        "weight": ip_weight,
        "cond": image_prompt_embeds,
        "uncond": uncond_image_prompt_embeds,
        "mask": None,
        "sigma_start": sigma_max,
        "sigma_end": sigma_min,
    }

    if is_sd15:
        for index, (_, block_key) in enumerate(SD15_INSTANTID_LAYERS):
            patch_kwargs["module_key"] = str(index * 2 + 1)
            _set_model_patch_replace(unet_model, patch_kwargs, block_key)
    else:
        # Original SDXL patch locations. Do not use these keys for SD1.5.
        number = 0
        for block_id in [4, 5, 7, 8]:
            block_indices = range(2) if block_id in [4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                _set_model_patch_replace(
                    unet_model, patch_kwargs, ("input", block_id, index)
                )
                number += 1

        for block_id in range(6):
            block_indices = range(2) if block_id in [3, 4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                _set_model_patch_replace(
                    unet_model, patch_kwargs, ("output", block_id, index)
                )
                number += 1

        for index in range(10):
            patch_kwargs["module_key"] = str(number * 2 + 1)
            _set_model_patch_replace(
                unet_model, patch_kwargs, ("middle", 1, index)
            )
            number += 1

    control_net = get_or_load_instantid_controlnet(is_sd15=is_sd15)

    face_kps_resized = F.interpolate(
        face_kps.movedim(-1, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    control_hint = face_kps_resized.movedim(-1, 1).to(
        device=device, dtype=dtype
    )

    cnets = {}
    cond_uncond = []
    is_cond = True

    for conditioning in (positive, negative):
        current_conditioning = []
        for item in conditioning:
            data = item[1].copy()
            previous_controlnet = data.get("control", None)

            if previous_controlnet in cnets:
                c_net = cnets[previous_controlnet]
            else:
                c_net = control_net.copy().set_cond_hint(
                    control_hint,
                    cn_strength,
                    (start, end),
                )
                if previous_controlnet is not None:
                    c_net.set_previous_controlnet(previous_controlnet)
                cnets[previous_controlnet] = c_net

            data["control"] = c_net
            data["control_apply_to_uncond"] = False
            embed_to_use = (
                image_prompt_embeds if is_cond else uncond_image_prompt_embeds
            )
            data["cross_attn_controlnet"] = conds.CONDCrossAttn(
                embed_to_use.to(device=device, dtype=dtype)
            )
            current_conditioning.append([item[0], data])

        cond_uncond.append(current_conditioning)
        is_cond = False

    final_positive, final_negative = cond_uncond

    del insightface, control_net, cnets
    del face_kps, face_kps_resized, control_hint
    del clip_embed, clip_embed_zeroed

    return unet_model, final_positive, final_negative
