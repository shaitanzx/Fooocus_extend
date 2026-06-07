"""
InstantID Integration for Fooocus
Location: extensions/instant2/instantid.py
"""
import os
import cv2
import torch
import numpy as np
import copy
from huggingface_hub import hf_hub_download


# Импорт модулей Fooocus и ComfyUI
import modules.config
import ldm_patched.modules.model_management as model_management
import ldm_patched.modules.utils as comfy_utils
from insightface.app import FaceAnalysis

# Глобальный кэш для моделей, чтобы не загружать их каждый раз
_INSTANTID_CACHE = {
    "insightface": None,
    "ip_adapter": None,
    "controlnet": None
}

def get_models_dir():
    """Возвращает путь к папке с моделями InstantID"""
    # Можно хранить в папке расширения или в общей папке моделей Fooocus
    return os.path.join(os.path.dirname(__file__), 'models')

def load_instantid_models():
    """Загружает и кэширует модели InstantID"""
    global _INSTANTID_CACHE
    
    if _INSTANTID_CACHE["insightface"] is not None:
        return _INSTANTID_CACHE["insightface"], _INSTANTID_CACHE["ip_adapter"], _INSTANTID_CACHE["controlnet"]

    print("[InstantID] Загрузка моделей (первый запуск, это займет время)...")
    models_dir = get_models_dir()
    os.makedirs(models_dir, exist_ok=True)

    # 1. InsightFace
    print("[InstantID] Инициализация InsightFace...")
    insightface_app = FaceAnalysis(
        name='antelopev2',
        root=models_dir,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    _INSTANTID_CACHE["insightface"] = insightface_app

    # 2. IP-Adapter (InstantID)
    ip_path = os.path.join(models_dir, 'ip-adapter.bin') # или .safetensors
    if os.path.exists(ip_path):
        print(f"[InstantID] Загрузка IP-Adapter из {ip_path}")
        # Пока оставляем как заглушку, полную загрузку сделаем на Шаге 3
        _INSTANTID_CACHE["ip_adapter"] = {"path": ip_path, "loaded": False} 
    else:
        print(f"[InstantID] ВНИМАНИЕ: IP-Adapter не найден по пути {ip_path}")

    # 3. ControlNet
    cn_path = os.path.join(models_dir, 'diffusion_pytorch_model.safetensors')
    if os.path.exists(cn_path):
        print(f"[InstantID] Загрузка ControlNet из {cn_path}")
        # Пока оставляем как заглушку, полную загрузку сделаем на Шаге 3
        _INSTANTID_CACHE["controlnet"] = {"path": cn_path, "loaded": False}
    else:
        print(f"[InstantID] ВНИМАНИЕ: ControlNet не найден по пути {cn_path}")

    print("[InstantID] Модели успешно инициализированы и закэшированы!")
    return _INSTANTID_CACHE["insightface"], _INSTANTID_CACHE["ip_adapter"], _INSTANTID_CACHE["controlnet"]


def extract_face_features(insightface_app, face_image_np):
    """
    Извлекает эмбеддинги и ключевые точки лица.
    face_image_np: numpy array (H, W, 3) в диапазоне 0-255, RGB
    """
    # InsightFace ожидает BGR
    if face_image_np.shape[2] == 3:
        face_image_bgr = cv2.cvtColor(face_image_np, cv2.COLOR_RGB2BGR)
    else:
        face_image_bgr = face_image_np
    
    faces = insightface_app.get(face_image_bgr)
    
    if len(faces) == 0:
        raise ValueError("[InstantID] Лицо не обнаружено на референсном изображении!")
    
    # Берем первое обнаруженное лицо (можно добавить логику выбора самого крупного)
    face = faces[0]
    
    # Эмбеддинг [1, 512]
    face_embed = torch.from_numpy(face.embedding).float().unsqueeze(0)
    
    # Ключевые точки (landmarks) [5, 2]
    kps = face.kps 
    
    # Создаем изображение для ControlNet (черный фон, белые точки)
    h, w = face_image_bgr.shape[:2]
    kps_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for kp in kps:
        x, y = int(kp[0]), int(kp[1])
        # Рисуем точки и линии для лучшей детекции ControlNet (как в оригинальном InstantID)
        cv2.circle(kps_image, (x, y), 3, (255, 255, 255), -1)
        
    # Конвертируем в тензор [1, H, W, 3] в диапазоне 0.0 - 1.0
    kps_tensor = torch.from_numpy(kps_image).float() / 255.0
    kps_tensor = kps_tensor.unsqueeze(0)
    
    return face_embed, kps_tensor


def apply_instantid_to_unet(unet, ip_model, insightface, face_image_np, weight=0.8, start_at=0.0, end_at=1.0):
    """Патчит UNet для работы с IP-Adapter InstantID"""
    print(f"[InstantID] Патчинг UNet (weight={weight}, start={start_at}, end={end_at})...")
    
    face_embed, face_kps = extract_face_features(insightface, face_image_np)
    
    # Клонируем модель, чтобы не менять оригинал
    work_model = unet.clone()
    
    # TODO: ШАГ 4 - Реальная логика патчинга слоев внимания (Attention)
    # Здесь мы добавим хуки в cross-attention слои
    
    print("[InstantID] UNet пропатчен (заглушка, ждем Шага 4)!")
    return work_model


def apply_instantid_to_conditioning(positive_cond, negative_cond, cn_model, insightface, face_image_np, weight=0.8, start_at=0.0, end_at=1.0):
    """Добавляет ControlNet и эмбеддинги в conditioning"""
    print("[InstantID] Модификация conditioning...")
    
    face_embed, face_kps = extract_face_features(insightface, face_image_np)
    
    new_positive = copy.deepcopy(positive_cond)
    new_negative = copy.deepcopy(negative_cond)
    
    # TODO: ШАГ 5 - Реальная логика внедрения ControlNet в словари conditioning
    # Добавление ключей 'control', 'cross_attn_controlnet' и т.д.
    
    print("[InstantID] Conditioning модифицирован (заглушка, ждем Шага 5)!")
    return new_positive, new_negative

def download_models():
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/1k3d68.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/2d106det.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/genderage.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/glintr100.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="antelopev2/scrfd_10g_bnkps.onnx", local_dir="extentions/instantid/models")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="canny_small/config.json", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="canny_small/diffusion_pytorch_model.safetensors", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="depth_small/config.json", local_dir="extentions/instantid/checkpoints")
    hf_hub_download(repo_id="shaitanzx/FooocusExtend", filename="depth_small/diffusion_pytorch_model.safetensors", local_dir="extentions/instantid/checkpoints")

def apply_instantid(pipeline_model, positive_cond, negative_cond, face_image_np, weight=0.8, start_at=0.0, end_at=1.0):
    """Главная точка входа для интеграции InstantID"""
    download_model()
    try:
        insightface_app, ip_model, cn_model = load_instantid_models()
        
        patched_model = apply_instantid_to_unet(
            unet=pipeline_model,
            ip_model=ip_model,
            insightface=insightface_app,
            face_image_np=face_image_np,
            weight=weight,
            start_at=start_at,
            end_at=end_at
        )
        
        modified_positive, modified_negative = apply_instantid_to_conditioning(
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            cn_model=cn_model,
            insightface=insightface_app,
            face_image_np=face_image_np,
            weight=weight,
            start_at=start_at,
            end_at=end_at
        )
        
        return patched_model, modified_positive, modified_negative
        
    except Exception as e:
        print(f"[InstantID] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        # Возвращаем оригинальные данные, чтобы не сломать пайплайн полностью
        return pipeline_model, positive_cond, negative_cond