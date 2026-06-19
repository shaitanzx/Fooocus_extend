from huggingface_hub import hf_hub_download
import torch
import os
import ldm_patched.modules.controlnet
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
def get_or_load_instantid_controlnet():
    """
    Загружает ControlNet для InstantID через ldm_patched (Fooocus backend).
    Возвращает объект ControlNet или None, если загрузка не удалась.
    """
    ctrl_path = "extentions/instant2/checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors"
    
    if not os.path.exists(ctrl_path):
        print(f"[InstantID CN] ⚠️ Файл ControlNet не найден: {ctrl_path}")
        return None
    
    print(f"[InstantID CN] Загрузка ControlNet из {ctrl_path}...")
    
    try:
        # Используем функцию load_controlnet из ldm_patched (Fooocus backend)
        control_net = ldm_patched.modules.controlnet.load_controlnet(ctrl_path)
        print("[InstantID CN] ✅ ControlNet успешно загружен через ldm_patched!")
        return control_net
    except Exception as e:
        print(f"[InstantID CN] ❌ Ошибка загрузки ControlNet: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None
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
        print("  [InstantID.__init__] Начало инициализации...")
        
        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        print("  [InstantID.__init__] Создание модели Resampler (image_proj_model)...")
        self.image_proj_model = self.init_proj()
        print("  [InstantID.__init__] Resampler создан успешно.")

        print(f"  [InstantID.__init__] Загрузка весов в Resampler (найдено {len(instantid_model['image_proj'])} ключей)...")
        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        print("  [InstantID.__init__] Веса Resampler загружены успешно.")

        print("  [InstantID.__init__] Создание слоев To_KV (ip_layers)...")
        self.ip_layers = To_KV(instantid_model["ip_adapter"])
        print("  [InstantID.__init__] Инициализация завершена успешно!")

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
    print(f"\n[InstantID] === НАЧАЛО ЗАГРУЗКИ МОДЕЛИ ===")
    print(f"[InstantID] Путь к файлу: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Файл модели не найден: {ckpt_path}")

    print("[InstantID] Чтение файла .bin (это может занять несколько секунд)...")
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("[InstantID] Файл прочитан в память.")

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

    print("[InstantID] Распаковка вложенных словарей...")
    sd = extract_all_tensors(pl_sd)
    print(f"[InstantID] ✅ Успешно извлечено {len(sd)} тензоров.")

    st_model = {"image_proj": {}, "ip_adapter": {}}
    
    print("[InstantID] Распределение тензоров по категориям (image_proj / ip_adapter)...")
    for key, tensor in sd.items():
        if key.startswith("image_proj."):
            st_model["image_proj"][key.replace("image_proj.", "")] = tensor
        elif key.startswith("ip_adapter."):
            st_model["ip_adapter"][key.replace("ip_adapter.", "")] = tensor
        elif "latents" in key or "proj_in" in key or "layers." in key:
            st_model["image_proj"][key] = tensor
        elif "to_k_ip" in key or "to_v_ip" in key:
            st_model["ip_adapter"][key] = tensor

    print(f"[InstantID] Найдено {len(st_model['image_proj'])} ключей для image_proj.")
    print(f"[InstantID] Найдено {len(st_model['ip_adapter'])} ключей для ip_adapter.")

    if not st_model["ip_adapter"]:
        raise ValueError(f"Не найдены веса 'ip_adapter'. Доступные ключи: {list(sd.keys())[:5]}")

    print("[InstantID] Определение output_dim...")
    output_dim = None
    for key in st_model["ip_adapter"].keys():
        if "to_k_ip" in key:
            output_dim = st_model["ip_adapter"][key].shape[1]
            break
    
    if output_dim is None:
        raise ValueError("Не найден ключ 'to_k_ip' в весах ip_adapter.")
    
    print(f"[InstantID] output_dim определен как: {output_dim}")

    print("[InstantID] Передача данных в класс InstantID...")
    instantid = InstantID(
        st_model,
        cross_attention_dim=1280,
        output_cross_attention_dim=output_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=16,
    )
    print("[InstantID] === МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА ===\n")
    return instantid

# ==============================================================================
# 3. ГЛАВНАЯ ПРОЦЕДУРНАЯ ФУНКЦИЯ (ЦЕПОЧКА ВЫПОЛНЕНИЯ)
# ==============================================================================

def apply_instantid_pipeline(
    image_path, unet_model, insightface, instantid_model, positive, negative,
    control_net=None, weight=0.8, start_at=0.0, end_at=1.0, noise=0.35,
    combine_embeds='average', device=None, dtype=None, sigma_min=None, sigma_max=None
):
    print("\n" + "="*60)
    print("[Pipeline] === НАЧАЛО apply_instantid_pipeline ===")
    print("="*60)

    # 1. Инициализация
    print("[Шаг 1/6] Инициализация параметров...")
    if device is None:
        device = torch.device(torch.cuda.current_device())
    if dtype is None:
        dtype = torch.float16
    
    ip_weight = weight
    cn_strength = weight
    print(f"  -> Устройство: {device}, Тип данных: {dtype}")
    print(f"  -> Веса: ip_weight={ip_weight}, cn_strength={cn_strength}")

    # 2. Загрузка изображения и детекция лица
    print("[Шаг 2/6] Загрузка изображения и детекция лица InsightFace...")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from {image_path}")
    print(f"  -> Изображение загружено, размер: {img_bgr.shape}")

    insightface.det_model.input_size = (640, 640)
    faces = insightface.get(img_bgr)
    if not faces:
        raise Exception('Reference Image: No face detected.')
    print(f"  -> Найдено лиц: {len(faces)}")
    
    # Берем самое крупное лицо
    face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    
    # ИСПРАВЛЕНИЕ: Принудительно делаем тензор 3D: [batch=1, seq_len=1, dim=512]
    raw_embed = torch.from_numpy(face['embedding'])
    clip_embed = raw_embed.view(1, 1, -1).to(device, dtype=dtype)
    print(f"  -> Форма clip_embed для Resampler: {clip_embed.shape} (ожидается [1, 1, 512])")

    # Рисуем KPS и конвертируем в тензор [1, H, W, 3]
    print("  -> Генерация карты ключевых точек (KPS)...")
    kps_img_pil = draw_kps(img_bgr, face['kps'])
    face_kps = T.ToTensor()(kps_img_pil).permute(1, 2, 0).unsqueeze(0)
    print(f"  -> Форма face_kps: {face_kps.shape}")

    # 3. Обработка эмбеддингов (шум и усреднение)
    print("[Шаг 3/6] Обработка эмбеддингов (шум/усреднение)...")
    if clip_embed.shape[0] > 1:
        if combine_embeds == 'average':
            clip_embed = torch.mean(clip_embed, dim=0, keepdim=True)
        elif combine_embeds == 'norm average':
            clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=-1, keepdim=True), dim=0, keepdim=True)

    if noise > 0:
        seed = int(torch.sum(clip_embed).item()) % 1000000007
        torch.manual_seed(seed)
        clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        print(f"  -> Добавлен шум (noise={noise}), seed={seed}")
    else:
        clip_embed_zeroed = torch.zeros_like(clip_embed)
        print("  -> Шум не добавлен (noise=0)")

    # 4. Проекция эмбеддингов через Resampler
    print("[Шаг 4/6] Проекция эмбеддингов через Resampler...")
    instantid_model.to(device, dtype=dtype)
    print("  -> Запуск instantid_model.get_image_embeds()...")
    image_prompt_embeds, uncond_image_prompt_embeds = instantid_model.get_image_embeds(
        clip_embed, clip_embed_zeroed
    )
    print(f"  -> ✅ Resampler отработал успешно. Форма cond: {image_prompt_embeds.shape}, uncond: {uncond_image_prompt_embeds.shape}")
    
    image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)

    # 5. Патчинг UNet
    print("[Шаг 5/6] Патчинг слоев Cross-Attention в UNet...")
    work_model = unet_model.clone()
    print("  -> Модель клонирована.")

    if sigma_min is not None and sigma_max is not None:
        sigma_start = sigma_max + start_at * (sigma_min - sigma_max)
        sigma_end = sigma_max + end_at * (sigma_min - sigma_max)
        print(f"  -> Сигмы рассчитаны из переданных значений: start={sigma_start:.4f}, end={sigma_end:.4f}")
    else:
        sigma_start = 14.6146 * (1.0 - start_at)
        sigma_end = 14.6146 * (1.0 - end_at)
        print(f"  -> Сигмы рассчитаны по умолчанию (fallback): start={sigma_start:.4f}, end={sigma_end:.4f}")

    patch_kwargs = {
        "ipadapter": instantid_model,
        "weight": ip_weight,
        "cond": image_prompt_embeds,
        "uncond": uncond_image_prompt_embeds,
        "mask": None, 
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }

    print("  -> Применение патчей к input, output и middle блокам...")
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
    print(f"  -> ✅ Всего применено {number} патчей.")

    # 6. Подготовка данных для ControlNet (НЕ применяем, а передаём в хук)
    print("\n" + "="*60)
    print("[Шаг 6/6] Подготовка данных для ControlNet хука...")
    print("="*60)
    
    # Если ControlNet не передан явно - пытаемся загрузить автоматически
    if control_net is None:
        print("  -> ControlNet не передан. Пытаемся загрузить автоматически...")
        control_net = get_or_load_instantid_controlnet()
    
    if control_net is not None:
        print("  -> ✅ ControlNet загружен и готов к использованию в хуке")
        print(f"  -> Тип control_net: {type(control_net)}")
        print(f"  -> Форма face_kps: {face_kps.shape}")
        print(f"  -> Форма image_prompt_embeds: {image_prompt_embeds.shape}")
        print(f"  -> Форма uncond_image_prompt_embeds: {uncond_image_prompt_embeds.shape}")
        print(f"  -> cn_strength: {cn_strength}")
        print(f"  -> sigma_start: {sigma_start:.4f}, sigma_end: {sigma_end:.4f}")
        
        # НЕ применяем ControlNet здесь — это будет делать хук в process_diffusion
        # Вместо этого возвращаем все данные, необходимые для хука
        instantid_data = {
            'control_net': control_net,
            'face_kps': face_kps,
            'image_prompt_embeds': image_prompt_embeds,
            'uncond_image_prompt_embeds': uncond_image_prompt_embeds,
            'cn_strength': cn_strength,
            'sigma_start': sigma_start,
            'sigma_end': sigma_end
        }
        
        print("  -> ✅ Данные для хука подготовлены")
    else:
        print("  -> ⚠️ ControlNet недоступен. Хук не будет применён.")
        instantid_data = None
    
    print("="*60)
    print("[Pipeline] === apply_instantid_pipeline ЗАВЕРШЕН УСПЕШНО ===")
    print("="*60 + "\n")
    
    # Возвращаем instantid_data вместе с остальными данными
    return work_model, positive, negative, instantid_data

def apply(image_path, target_unet, positive_cond, negative_cond, sigma_min, sigma_max):
    print("\n" + "="*60)
    print("[InstantID Pipeline] === ЗАПУСК ПАПЛАЙНА ===")
    print("="*60)
    
    # 1. Скачивание
    print("[Шаг 1/5] Проверка и скачивание моделей...")
    download()
    print("[Шаг 1/5] ✅ Готово.")
    
    # 2. InsightFace
    print("[Шаг 2/5] Инициализация InsightFace...")
    insightface_app = FaceAnalysis(name='antelopev2', root='extentions/instant2', providers=['CPUExecutionProvider'])
    insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[Шаг 2/5] ✅ InsightFace инициализирован.")

    # 3. Загрузка InstantID
    print("[Шаг 3/5] Загрузка модели InstantID...")
    instantid_file = "extentions/instant2/checkpoints/ip-adapter.bin"
    
    if not os.path.exists(instantid_file):
        raise FileNotFoundError(f"Файл InstantID не найден: {instantid_file}")
    
    instantid_model = load_instantid_model(instantid_file)
    print("[Шаг 3/5] ✅ Модель InstantID загружена.")

    # 4. Запуск пайплайна
    print("[Шаг 4/5] Запуск apply_instantid_pipeline...")
    print(f"  - Путь к изображению: {image_path}")
    print(f"  - Sigma min/max: {sigma_min} / {sigma_max}")
    
    try:
        patched_unet, new_positive, new_negative, instantid_data = apply_instantid_pipeline(
            image_path=image_path,
            unet_model=target_unet,
            insightface=insightface_app,
            instantid_model=instantid_model,
            positive=positive_cond,
            negative=negative_cond,
            control_net=None, 
            weight=0.8,
            start_at=0.0,
            end_at=1.0,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        print("[Шаг 4/5] ✅ Пайплайн выполнен успешно.")
    except Exception as e:
        print(f"\n❌ ОШИБКА ВНУТРИ apply_instantid_pipeline:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("[Шаг 5/5] Возврат пропатченных моделей и кондишенов...")
    print("="*60)
    print("[InstantID Pipeline] === ЗАВЕРШЕНО УСПЕШНО ===")
    print("="*60 + "\n")
    
    # Возвращаем instantid_data вместе с остальными данными
    return patched_unet, new_positive, new_negative, instantid_data
