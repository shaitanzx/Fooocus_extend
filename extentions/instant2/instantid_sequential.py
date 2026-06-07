import os
import math
import torch
import numpy as np
import cv2
import PIL.Image
from safetensors.torch import load_file
from insightface.app import FaceAnalysis

# ─────────────────────────────────────────────────────────────────────────────
# Resampler (стандартный, без изменений)
# ─────────────────────────────────────────────────────────────────────────────
class Resampler(torch.nn.Module):
    def __init__(self, dim=1024, depth=8, dim_head=64, heads=16, num_queries=8,
                 embedding_dim=768, output_dim=1024, ff_mult=4):
        super().__init__()
        self.num_queries = num_queries
        self.latents = torch.nn.Parameter(torch.randn(1, num_queries, dim))
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                torch.nn.MultiheadAttention(dim, heads, batch_first=True),
                torch.nn.Sequential(
                    torch.nn.LayerNorm(dim),
                    torch.nn.Linear(dim, dim * ff_mult, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(dim * ff_mult, dim, bias=False),
                )
            ]))

    def forward(self, x):
        h = self.latents.repeat(x.shape[0], 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            h = attn(h, x, x)[0] + h
            h = ff(h) + h
        return self.norm_out(self.proj_out(h))

# ─────────────────────────────────────────────────────────────────────────────
# InstantIDAdapter (ИСПОЛЬЗУЕТ ТВОЙ To_KV)
# ─────────────────────────────────────────────────────────────────────────────
class InstantIDAdapter(torch.nn.Module):
    """
    Контейнер для весов адаптера (image_proj + ip_adapter).
    НЕ содержит базовую модель — только дополнения для неё.
    """
    def __init__(self, state_dict, cross_attention_dim=2048, output_cross_attention_dim=1024,
                 clip_embeddings_dim=1280, clip_extra_context_tokens=16):
        super().__init__()
        # Resampler: face embed (1280) -> UNet context (2048)
        self.image_proj_model = Resampler(
            dim=cross_attention_dim, depth=4, dim_head=64, heads=20,
            num_queries=clip_extra_context_tokens,
            embedding_dim=clip_embeddings_dim, output_dim=output_cross_attention_dim, ff_mult=4
        )
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        
        # 🔧 ИСПОЛЬЗУЕТ ТВОЙ СУЩЕСТВУЮЩИЙ КЛАСС To_KV
        # Он должен быть импортирован в область видимости перед использованием этого класса
        self.ip_layers = To_KV(cross_attention_dim)
        self.ip_layers.load_state_dict_ordered(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        """Генерирует cond/uncond токены для cross-attention"""
        return self.image_proj_model(clip_embed), self.image_proj_model(clip_embed_zeroed)

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная функция для отрисовки keypoints
# ─────────────────────────────────────────────────────────────────────────────
def _draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        x, y = kps[index][:, 0], kps[index][:, 1]
        length = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**0.5
        angle = math.degrees(math.atan2(y[0]-y[1], x[0]-x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))),
                                   (int(length/2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        out_img = cv2.circle(out_img.copy(), (int(kp[0]), int(kp[1])), 10, color, -1)
    return PIL.Image.fromarray(out_img.astype(np.uint8))

# ─────────────────────────────────────────────────────────────────────────────
# ПОСЛЕДОВАТЕЛЬНЫЕ ФУНКЦИИ (ШАГ ЗА ШАГОМ)
# ─────────────────────────────────────────────────────────────────────────────

def load_adapter(adapter_path: str, cross_attention_dim: int = 2048) -> InstantIDAdapter:
    """
    Загружает ТОЛЬКО веса адаптера (не базовую модель!).
    Возвращает InstantIDAdapter, готовый к использованию.
    """
    print("[InstantID] STEP 1: Loading adapter weights...")
    raw = load_file(adapter_path, device="cpu")
    
    # Разделение на image_proj и ip_adapter (как в оригинале)
    clean = {"image_proj": {}, "ip_adapter": {}}
    for k, v in raw.items():
        if k.startswith("image_proj."): clean["image_proj"][k.replace("image_proj.", "")] = v
        elif k.startswith("ip_adapter."): clean["ip_adapter"][k.replace("ip_adapter.", "")] = v
    
    # Динамическое определение output_dim из весов
    out_dim = clean["ip_adapter"]["0.to_k_ip.weight"].shape[1]
    
    # Создание адаптера (использует ТВОЙ To_KV)
    adapter = InstantIDAdapter(
        clean, 
        cross_attention_dim=cross_attention_dim, 
        output_cross_attention_dim=out_dim,
        clip_embeddings_dim=1280,  # CLIP ViT-L/14 face embedding dim
        clip_extra_context_tokens=16
    )
    print(f"[InstantID] STEP 1: Adapter loaded. Output dim: {out_dim}")
    return adapter

def init_face_analyzer(provider: str = "CUDA", det_size: int = 640) -> FaceAnalysis:
    """Инициализирует InsightFace детектор (один раз на сессию)"""
    print(f"[InstantID] STEP 2: Initializing FaceAnalysis (provider={provider})...")
    # Путь к папке с моделями insightface (относительно этого файла)
    face_dir = os.path.join(os.path.dirname(__file__), "models")
    analyzer = FaceAnalysis(name="antelopev2", root=face_dir, providers=[f'{provider}ExecutionProvider'])
    analyzer.prepare(ctx_id=0, det_size=(det_size, det_size))
    print("[InstantID] STEP 2: FaceAnalyzer ready.")
    return analyzer

def step3_extract_embedding(face_analyzer, image_pil) -> torch.Tensor:
    """Извлекает эмбеддинг самого крупного лица из изображения"""
    print("[InstantID] STEP 3: Extracting face embedding...")
    faces = face_analyzer.get(np.array(image_pil))
    if not faces:
        print("[InstantID] ⚠️ STEP 3: No face detected!")
        return None
    
    # Берём самое крупное лицо (по площади bbox), как в оригинале
    face = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    print(f"[InstantID] STEP 3: Face found. Score: {face['det_score']:.2f}")
    
    # Возвращаем тензор [1, 512] (или [1, 1280] в зависимости от модели antelope)
    return torch.from_numpy(face['embedding']).unsqueeze(0)

def step4_extract_kps(face_analyzer, image_pil) -> PIL.Image:
    """Генерирует изображение с ключевыми точками для ControlNet"""
    print("[InstantID] STEP 4: Extracting face keypoints...")
    faces = face_analyzer.get(np.array(image_pil))
    if not faces:
        print("[InstantID] ⚠️ STEP 4: No face for keypoints!")
        return None
    face = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    return _draw_kps(np.array(image_pil), face['kps'])

def step5_compute_embeds(adapter: InstantIDAdapter, face_embed: torch.Tensor, 
                         noise: float = 0.35) -> tuple[torch.Tensor, torch.Tensor]:
    """Вычисляет cond/uncond эмбеддинги через Resampler"""
    print("[InstantID] STEP 5: Computing cond/uncond embeddings...")
    
    # Определяем устройство и тип данных (как в основном коде Fooocus)
    from ldm_patched.modules import model_management as mm
    device = mm.get_torch_device()
    dtype = torch.float16 if mm.should_use_fp16() else torch.float32
    
    adapter.to(device, dtype=dtype)
    face_embed = face_embed.to(device, dtype=dtype)
    
    # Генерация "нулевого" эмбеддинга для unconditional генерации
    if noise > 0:
        torch.manual_seed(int(torch.sum(face_embed).item()) % 1_000_000_007)
        clip_embed_zeroed = noise * torch.rand_like(face_embed)
        print(f"[InstantID] STEP 5: Applied noise {noise}")
    else:
        clip_embed_zeroed = torch.zeros_like(face_embed)
        print("[InstantID] STEP 5: Uncond = zeros")

    # Проекция через image_proj_model
    cond, uncond = adapter.get_image_embeds(face_embed, clip_embed_zeroed)
    print(f"[InstantID] STEP 5: Embeds ready. Shape: {cond.shape}")
    return cond, uncond

def step6_patch_unet(model, adapter: InstantIDAdapter, cond: torch.Tensor, 
                     uncond: torch.Tensor, weight: float = 0.8, 
                     start_at: float = 0.0, end_at: float = 1.0) -> object:
    """
    Патчит cross-attention слои UNet (SDXL конфигурация).
    Использует существующий механизм: model_options["transformer_options"]["patches_replace"]["attn2"]
    """
    print("[InstantID] STEP 6: Patching UNet attention layers...")
    from ldm_patched.modules.attention import optimized_attention
    
    work_model = model.clone()  # Используем твой .clone()
    to = work_model.model_options.setdefault("transformer_options", {})
    patches = to.setdefault("patches_replace", {}).setdefault("attn2", {})
    
    # Конвертация процентов в sigma (как в твоём коде)
    ms = model.model.model_sampling
    sigma_start = ms.percent_to_sigma(start_at)
    sigma_end = ms.percent_to_sigma(end_at)
    
    # SDXL блоки с cross-attention (проверенная конфигурация)
    sdxl_blocks = [
        ("input", 2, 1), ("input", 5, 2), ("input", 8, 10),
        ("middle", 0, 10),
        ("output", 3, 1), ("output", 6, 2), ("output", 8, 10)
    ]
    
    number = 0
    for block_type, block_id, depth in sdxl_blocks:
        for idx in range(depth):
            key = (block_type, block_id, idx)
            
            # 🔧 Доступ к весам через ТВОЙ To_KV (плоский список: [K0, V0, K1, V1...])
            k_layer = adapter.ip_layers.to_kvs[number * 2]
            v_layer = adapter.ip_layers.to_kvs[number * 2 + 1]

            class InstantIDPatch:
                def __init__(self, k_w, v_w, c, u, w, ss, se):
                    self.k_w, self.v_w = k_w, v_w  # Веса линейных слоев
                    self.c, self.u = c, u          # Cond/uncond эмбеддинги
                    self.w = w                     # Сила применения
                    self.s_start, self.s_end = ss, se  # Диапазон шагов
                def __call__(self, q, context, value, extra_options):
                    org = q.dtype
                    c_or_u = extra_options["cond_or_uncond"]  # [0]=cond, [1]=uncond
                    sigma = extra_options.get("sigmas", [999.])[0].item()
                    
                    # Пропуск, если вне временного диапазона
                    if sigma > self.s_start or sigma < self.s_end:
                        return q
                    
                    # Базовое внимание
                    out = optimized_attention(q, context, value, extra_options["n_heads"])
                    
                    # Формируем батч из cond/uncond согласно c_or_u
                    embeds = torch.cat([self.c if i == 0 else self.u for i in c_or_u], dim=0)
                    
                    # Применяем IP-Adapter слои (K/V проекции)
                    k_ip = torch.nn.functional.linear(embeds, self.k_w) * self.w
                    v_ip = torch.nn.functional.linear(embeds, self.v_w) * self.w
                    
                    # Дополнительное внимание с face-эмбеддингами
                    out_ip = optimized_attention(q, k_ip.to(org), v_ip.to(org), extra_options["n_heads"])
                    return (out + out_ip).to(dtype=org)
            
            patches[key] = InstantIDPatch(k_layer.weight, v_layer.weight, cond, uncond, weight, sigma_start, sigma_end)
            number += 1
            
    print(f"[InstantID] STEP 6: Registered {number} attention patches.")
    return work_model

def step7_apply_controlnet(controlnet, positive, negative, kps_image, cond, uncond, 
                           strength: float = 0.8, start_at: float = 0.0, end_at: float = 1.0) -> tuple:
    """Внедряет ControlNet в conditioning (точно как в оригинале)"""
    print("[InstantID] STEP 7: Applying ControlNet to conditioning...")
    
    # Конвертация PIL -> Tensor [1, C, H, W] в диапазоне [0, 1]
    kps_np = np.array(kps_image).astype(np.float32) / 255.0
    kps_tensor = torch.from_numpy(kps_np).permute(2, 0, 1).unsqueeze(0)
    
    def _patch_cond(conditioning, cross_embed):
        new_c = []
        for t in conditioning:
            d = t[1].copy()  # Копируем словарь параметров
            # Копируем и настраиваем контролнет
            c_net = controlnet.copy().set_cond_hint(kps_tensor, strength, (start_at, end_at))
            d['control'] = c_net
            d['control_apply_to_uncond'] = False  # Не применять к negative
            # InstantID-специфика: передача IP-эмбеддингов в ControlNet
            d['cross_attn_controlnet'] = cross_embed
            new_c.append([t[0], d])
        return new_c

    pos_out = _patch_cond(positive, cond)
    neg_out = _patch_cond(negative, uncond)
    print("[InstantID] STEP 7: Conditioning modified.")
    return pos_out, neg_out