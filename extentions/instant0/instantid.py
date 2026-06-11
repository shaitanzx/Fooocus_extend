import torch
import numpy as np
import math
import cv2
import PIL.Image
import os

#Убедитесь, что эти импорты ведут к вашим файлам
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T



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


def load_instantid_proj_model(file_path: str, device: torch.device):
    """
    Загружает веса image_proj из файла InstantID и возвращает готовый Resampler.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл модели InstantID не найден: {file_path}")

    # 1. Загружаем веса
    state_dict = torch.load(file_path, map_location=device)

    # 2. Фильтруем только веса для Resampler (убираем префикс "image_proj.")
    proj_weights = {}
    for key, value in state_dict.items():
        if key.startswith("image_proj."):
            clean_key = key.replace("image_proj.", "")
            proj_weights[clean_key] = value

    if not proj_weights:
        raise ValueError("В файле не найдены веса 'image_proj'. Убедитесь, что это правильный файл InstantID.")

    # 3. Создаем экземпляр Resampler с параметрами из оригинального кода InstantID
    # Эти параметры жестко заданы в оригинальном InstantIDModelLoader
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        embedding_dim=512,
        output_dim=1024,
        ff_mult=4
    )

    # 4. Загружаем веса, переводим в режим оценки и на нужное устройство
    image_proj_model.load_state_dict(proj_weights)
    image_proj_model.to(device)
    image_proj_model.eval() # Важно: отключаем dropout и т.д. для инференса

    print(f"[InstantID] Модель Resampler успешно загружена из {file_path}")
    return image_proj_model

def apply(image_path,target_unet,positive_cond, negative_cond):
    download()
    # 1. Загружаем модель проекции (Resampler) один раз при старте
    instantid_path = f"extentions/instantid/checkpoints/ip-adapter.bin"
    image_proj_model = load_instantid_proj_model(instantid_path, device="cuda")

    # 2. Инициализируем InsightFace (тоже один раз)
    insightface = FaceAnalysis(name='antelopev2', root='extentions/instantid', providers=['CPUExecutionProvider'])
    insightface.prepare(ctx_id=0, det_size=(640, 640))



    patched_unet, new_positive, new_negative = apply_instantid_pipeline(
        image_proj_model=my_image_proj_model,
        insightface=my_insightface,
        control_net=None,          # <--- ГЛАВНОЕ ИЗМЕНЕНИЕ
        image_path=image_path,
        unet_model=target_unet,
        positive=positive_cond,
        negative=negative_cond,
    
        weight=0.8,                # Сила влияния на идентичность
        start_at=0.0,
        end_at=1.0,
        noise=0.35,
    
        device=torch.device('cuda'),
        dtype=torch.float16,
        sigma_min=sigma_min,
        sigma_max=sigma_max
    )
def apply_instantid_pipeline(
    # --- Основные объекты ---
    image_proj_model,  # Экземпляр Resampler
    insightface,       # Инициализированный объект FaceAnalysis
    control_net=None,  # <<< ИЗМЕНЕНО: По умолчанию None. Передавайте None, если не нужен.
    image_path=None,   # Путь к файлу изображения (str)
    unet_model=None,   # Ваша модель UNet
    
    positive=None,     # Список conditioning: [[tensor, dict_params], ...]
    negative=None,     # Список conditioning: [[tensor, dict_params], ...]
    
    # --- Параметры настройки ---
    weight=0.8,
    start_at=0.0,
    end_at=1.0,
    ip_weight=None,
    cn_strength=None,
    noise=0.35,
    combine_embeds='average',
    
    # --- Явные параметры окружения ---
    device=None,
    dtype=None,
    sigma_min=None,
    sigma_max=None
):
    # 1. Инициализация
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float16

    ip_weight = weight if ip_weight is None else ip_weight
    # cn_strength игнорируется, если control_net is None

    # 2. Вычисление sigma_start и sigma_end
    if sigma_min is not None and sigma_max is not None:
        sigma_start = sigma_max + start_at * (sigma_min - sigma_max)
        sigma_end = sigma_max + end_at * (sigma_min - sigma_max)
    else:
        sigma_start = 14.6146 * (1.0 - start_at)
        sigma_end = 14.6146 * (1.0 - end_at)
    
    print(f'[InstantID] sigma_start = {sigma_start:.4f}, sigma_end = {sigma_end:.4f}')

    # 3. Загрузка и обработка изображения
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # 4. Извлечение эмбеддинга лица
    faces = insightface.get(img_bgr)
    if not faces:
        raise Exception('Reference Image: No face detected.')
    
    face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_embed = torch.from_numpy(face['embedding']).unsqueeze(0)

    # 5. Извлечение KPS (ТОЛЬКО если есть ControlNet, чтобы сэкономить ресурсы)
    if control_net is not None:
        kps_img_bgr = draw_kps(img_bgr, face['kps'])
        face_kps = T.ToTensor()(kps_img_bgr).permute([1, 2, 0]).unsqueeze(0).to(device, dtype=dtype)
    else:
        face_kps = None

    # 6. Обработка эмбеддингов
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

    # 7. Проекция эмбеддингов
    image_prompt_embeds, uncond_image_prompt_embeds = get_image_embeds(
        image_proj_model, clip_embed, clip_embed_zeroed, device, dtype
    )

    # 8. ПАТЧИНГ МОДЕЛИ (Здесь ваша логика)
    work_model = unet_model # Замените на вашу логику клонирования/патчинга
    
    patch_kwargs = {
        "ipadapter": image_proj_model, 
        "weight": ip_weight,
        "cond": image_prompt_embeds,
        "uncond": uncond_image_prompt_embeds,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }
    # <<< ВАШ ЦИКЛ ПАТЧИНГА ЗДЕСЬ >>>

    # 9. Применение ControlNet к Conditioning (ТОЛЬКО если control_net передан)
    cond_uncond = []
    is_cond = True
    
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy() # Копируем словарь параметров conditioning

            # <<< БЕЗОПАСНАЯ ПРОВЕРКА НА НАЛИЧИЕ CONTROLNET >>>
            if control_net is not None and face_kps is not None:
                prev_cnet = d.get('control', None)
                
                # Здесь должна быть ваша логика применения ControlNet. 
                # Оригинальный код ComfyUI выглядит так, но вам нужно адаптировать его под ваш формат:
                # c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1, 1), cn_strength, (start_at, end_at))
                # if prev_cnet is not None:
                #     c_net.set_previous_controlnet(prev_cnet)
                # d['control'] = c_net
                # d['control_apply_to_uncond'] = False
                
                # target_dtype = getattr(c_net, 'cond_hint_original', None)
                # target_dtype = target_dtype.dtype if target_dtype is not None else dtype
                # d['cross_attn_controlnet'] = (
                #     image_prompt_embeds.to(device, dtype=target_dtype) if is_cond 
                #     else uncond_image_prompt_embeds.to(device, dtype=target_dtype)
                # )
                print("[InstantID] ControlNet logic is stubbed. Implement your specific CN application here.")

            n = [t[0], d]
            c.append(n)
        cond_uncond.append(c)
        is_cond = False

    # 10. Возврат результатов
    return work_model, cond_uncond[0], cond_uncond[1]
