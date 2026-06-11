from huggingface_hub import hf_hub_download
import torch
import os
#import folder_paths
import numpy as np
import math
import cv2
import PIL.Image
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention
# from .utils import tensor_to_image  # Больше не нужно, так как читаем файл напрямую

from insightface.app import FaceAnalysis

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F

def download():
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/1k3d68.onnx", local_dir="extentions/instant2/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/2d106det.onnx", local_dir="extentions/instant2/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/genderage.onnx", local_dir="extentions/instant2/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/glintr100.onnx", local_dir="extentions/instant2/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/scrfd_10g_bnkps.onnx", local_dir="extentions/instant2/models")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="extentions/instant2/checkpoints")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="extentions/instant2/checkpoints")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="extentions/instant2/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="canny_small/config.json", local_dir="extentions/instant2/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="canny_small/diffusion_pytorch_model.safetensors", local_dir="extentions/instant2/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="depth_small/config.json", local_dir="extentions/instant2/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="depth_small/diffusion_pytorch_model.safetensors", local_dir="extentions/instant2/checkpoints")

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

# ==============================================================================
# 2. ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ (АДАПТИРОВАНА ПОД .bin И .safetensors)
# ==============================================================================

def load_instantid_model(ckpt_path):
    """Загружает файл InstantID и возвращает готовый объект класса InstantID."""
    print(f"[InstantID] Загрузка модели из {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Файл модели не найден: {ckpt_path}")

    # ИСПРАВЛЕНО: используем ckpt_path вместо несуществующего ckpt, device и torch_args
    # Загружаем на CPU, чтобы избежать проблем с нехваткой памяти GPU при загрузке
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        if len(pl_sd) == 1:
            key = list(pl_sd.keys())[0]
            sd = pl_sd[key]
            if not isinstance(sd, dict):
                sd = pl_sd
        else:
            sd = pl_sd

    # Разделяем веса на image_proj и ip_adapter
    st_model = {"image_proj": {}, "ip_adapter": {}}
    for key in sd.keys():
        if key.startswith("image_proj."):
            st_model["image_proj"][key.replace("image_proj.", "")] = sd[key]
        elif key.startswith("ip_adapter."):
            st_model["ip_adapter"][key.replace("ip_adapter.", "")] = sd[key]
    
    # Проверка, что веса найдены
    if not st_model["ip_adapter"]:
        raise ValueError("Не найдены веса 'ip_adapter' в файле модели. Убедитесь, что файл корректен.")

    # Определяем output_cross_attention_dim динамически
    output_dim = st_model["ip_adapter"]["1.to_k_ip.weight"].shape[1]

    instantid = InstantID(
        st_model,
        cross_attention_dim=1280,
        output_cross_attention_dim=output_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=16,
    )
    print("[InstantID] Модель успешно загружена.")
    return instantid

# ==============================================================================
# 3. ГЛАВНАЯ ПРОЦЕДУРНАЯ ФУНКЦИЯ (ЦЕПОЧКА ВЫПОЛНЕНИЯ)
# ==============================================================================

def apply_instantid_pipeline(
    image_path,          # str: Путь к файлу изображения
    unet_model,          # object: Ваша модель Fooocus (ModelPatcher)
    insightface,         # object: Инициализированный FaceAnalysis
    instantid_model,     # object: Загруженный объект InstantID (из load_instantid_model)
    positive,            # list: Conditioning Fooocus
    negative,            # list: Conditioning Fooocus
    control_net=None,    # object: ControlNet Fooocus ИЛИ None (если не нужен)
    weight=0.8,
    start_at=0.0,
    end_at=1.0,
    noise=0.35,
    combine_embeds='average',
    device=None,
    dtype=None,
    sigma_min=None,      # float: Из вашего calculate_sigmas (опционально)
    sigma_max=None       # float: Из вашего calculate_sigmas (опционально)
):
    """
    Применяет InstantID к модели и conditioning.
    Возвращает: (patched_unet_model, modified_positive, modified_negative)
    """
    # 1. Инициализация устройства и типа данных
    if device is None:
        device = torch.device(torch.cuda.current_device())
    if dtype is None:
        dtype = torch.float16
        # if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        #     dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

    ip_weight = weight
    cn_strength = weight

    # 2. Загрузка изображения и извлечение признаков (НАПРЯМУЮ ИЗ ФАЙЛА)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from {image_path}")

    insightface.det_model.input_size = (640, 640)
    faces = insightface.get(img_bgr)
    if not faces:
        raise Exception('Reference Image: No face detected.')
    
    # Берем самое крупное лицо (оригинальная логика)
    face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_embed = torch.from_numpy(face['embedding']).unsqueeze(0)

    # Рисуем KPS и конвертируем в тензор ComfyUI [1, H, W, 3]
    kps_img_pil = draw_kps(img_bgr, face['kps'])
    face_kps = T.ToTensor()(kps_img_pil).permute(1, 2, 0).unsqueeze(0)

    # 3. Обработка эмбеддингов (шум и усреднение)
    clip_embed = face_embed.to(device, dtype=dtype)
    if clip_embed.shape[0] > 1:
        if combine_embeds == 'average':
            clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
        elif combine_embeds == 'norm average':
            clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

    if noise > 0:
        seed = int(torch.sum(clip_embed).item()) % 1000000007
        torch.manual_seed(seed)
        clip_embed_zeroed = noise * torch.rand_like(clip_embed)
    else:
        clip_embed_zeroed = torch.zeros_like(clip_embed)

    # 4. Проекция эмбеддингов через Resampler
    instantid_model.to(device, dtype=dtype)
    image_prompt_embeds, uncond_image_prompt_embeds = instantid_model.get_image_embeds(
        clip_embed, clip_embed_zeroed
    )
    image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)

    # 5. Патчинг UNet (ОРИГИНАЛЬНАЯ ЛОГИКА COMFYUI/FOOOCUS)
    work_model = unet_model.clone()

    # Вычисляем сигмы (либо из ваших значений, либо через штатный метод Fooocus)
    if sigma_min is not None and sigma_max is not None:
        sigma_start = sigma_max + start_at * (sigma_min - sigma_max)
        sigma_end = sigma_max + end_at * (sigma_min - sigma_max)
    else:
        sigma_start = unet_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = unet_model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "ipadapter": instantid_model,
        "weight": ip_weight,
        "cond": image_prompt_embeds,
        "uncond": uncond_image_prompt_embeds,
        "mask": None, 
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }

    # Циклы патчинга input, output и middle блоков
    number = 0
    for id in [4, 5, 7, 8]: 
        block_indices = range(2) if id in [4, 5] else range(10) 
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
            number += 1
            
    for id in range(6): 
        block_indices = range(2) if id in [3, 4, 5] else range(10) 
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
            number += 1
            
    for index in range(10):
        patch_kwargs["module_key"] = str(number*2+1)
        _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
        number += 1

    # 6. Обработка ControlNet (ТОЛЬКО если он передан)
    if control_net is not None:
        cnets = {}
        cond_uncond = []
        is_cond = True
        
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                prev_cnet = d.get('control', None)
                
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1, 1), cn_strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False
            
        final_positive, final_negative = cond_uncond[0], cond_uncond[1]
    else:
        # Если ControlNet нет, просто возвращаем оригинальные conditioning
        final_positive = positive
        final_negative = negative

    return work_model, final_positive, final_negative

def apply(image_path,target_unet,positive_cond, negative_cond,sigma_min, sigma_max):
    download()
    insightface_app = FaceAnalysis(name='antelopev2', root='extentions/instant2', providers=['CPUExecutionProvider'])           
    insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    instantid_model = load_instantid_model("extentions/instant2/checkpoints/ControlNetModel")

    patched_unet, new_positive, new_negative = apply_instantid_pipeline(
        image_path=image_path,  # Путь к файлу!
        unet_model=target_unet,
        insightface=insightface_app,
        instantid_model=instantid_model,
        positive=positive_cond,
        negative=negative_cond,
        control_net=None,  # <--- Передайте None, если не хотите использовать ControlNet
        weight=0.8,
        start_at=0.0,
        end_at=1.0,
        sigma_min=sigma_min,  # Из вашего calculate_sigmas
        sigma_max=sigma_max   # Из вашего calculate_sigmas
    )