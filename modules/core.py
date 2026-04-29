import os
import einops
import torch
import numpy as np

import ldm_patched.modules.model_management
import ldm_patched.modules.model_detection
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils
import ldm_patched.modules.controlnet
import modules.sample_hijack
import ldm_patched.modules.samplers
import ldm_patched.modules.latent_formats

from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external import VAEDecode, EmptyLatentImage, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, \
    ControlNetApplyAdvanced
from ldm_patched.contrib.external_freelunch import FreeU_V2
from ldm_patched.modules.sample import prepare_mask
from modules.lora import match_lora
from modules.util import get_file_from_folder_list
from ldm_patched.modules.lora import model_lora_keys_unet, model_lora_keys_clip
from modules.config import path_embeddings
from ldm_patched.contrib.external_model_advanced import ModelSamplingDiscrete, ModelSamplingContinuousEDM



# ================= LoRA Block Weight (TE only) =================
import re
from typing import List, Dict, Any, Tuple

# def parse_te_bw(bw_str: str) -> List[float]:
#     try:
#         if not bw_str:
#             return [1.0] * 16
#         weights = [float(x.strip()) for x in bw_str.split(',') if x.strip()]
#         return (weights + [1.0] * 16)[:16]
#     except Exception:
#         return [1.0] * 16

# def apply_te_block_weights(patch_dict: Dict[str, Any], te_bw: List[float]) -> Dict[str, Any]:
#     for key, (lora_type, data) in patch_dict.items():
#         if not re.search(r'(?:clip_l|clip_g|text_model)\.transformer\.(?:text_model\.)?encoder\.layers\.(\d+)', key):
#             continue
#         match = re.search(r'encoder\.layers\.(\d+)', key)
#         if not match:
#             continue
#         layer_idx = int(match.group(1))
#         bw_index = min(layer_idx // 2, 15) if 'clip_g' in key else min(layer_idx, 15)
#         coeff = te_bw[bw_index]
#         if lora_type == 'lora':
#             A, B, alpha, mid = data
#             patch_dict[key] = ('lora', (A, B, alpha * coeff, mid))
#         elif lora_type == 'loha':
#             w1a, w1b, alpha, w2a, w2b, t1, t2 = data
#             patch_dict[key] = ('loha', (w1a, w1b, alpha * coeff, w2a, w2b, t1, t2))
#         elif lora_type == 'lokr':
#             w1, w2, alpha, w1a, w1b, w2a, w2b, t2 = data
#             patch_dict[key] = ('lokr', (w1, w2, alpha * coeff, w1a, w1b, w2a, w2b, t2))
#     return patch_dict
# ================= END =================

opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()
opFreeU = FreeU_V2()
opModelSamplingDiscrete = ModelSamplingDiscrete()
opModelSamplingContinuousEDM = ModelSamplingContinuousEDM()

def _apply_block_weights_sdxl(lora_dict, lbw_str, debug=False):
    """
    Применяет блочные веса (12 индексов) к патчам ldm_patched.
    Безопасно обрабатывает тензоры, списки кортежей и другие форматы.
    """
    if not lbw_str:
        return lora_dict

    ratios = [float(x.strip()) for x in lbw_str.split(",")]
    if len(ratios) != 12:
        print(f"[LBW Warning] Ожидалось 12 значений для SDXL, получено {len(ratios)}. Пропускаю патчинг.")
        return lora_dict

    modified = {}
    
    # Маппинг ключей → индексы SDXL (12 блоков)
    def get_sdxl_block_idx(key):
        if "input_blocks.4" in key: return 1
        if "input_blocks.5" in key: return 2
        if "input_blocks.7" in key: return 3
        if "input_blocks.8" in key: return 4
        if "middle_block" in key: return 5
        if "output_blocks.0" in key: return 6
        if "output_blocks.1" in key: return 7
        if "output_blocks.2" in key: return 8
        if "output_blocks.3" in key: return 9
        if "output_blocks.4" in key: return 10
        if "output_blocks.5" in key: return 11
        return 0  # BASE / CLIP / fallback

    for k, v in lora_dict.items():
        idx = get_sdxl_block_idx(k)
        ratio = ratios[idx]

        # 🔹 Безопасная обработка разных форматов патчей
        if isinstance(v, list):
            # Формат ComfyUI/Fooocus: [(strength, up_key/tensor, down_key/tensor, ...), ...]
            new_list = []
            for patch in v:
                if isinstance(patch, tuple) and len(patch) > 0:
                    strength = patch[0]
                    if isinstance(strength, (int, float)):
                        # Масштабируем силу патча на коэффициент блока
                        new_list.append((strength * ratio, *patch[1:]))
                    else:
                        new_list.append(patch)  # Если структура нестандартная, оставляем как есть
                else:
                    new_list.append(patch)
            modified[k] = new_list
            
        elif isinstance(v, tuple):
            # Прямой формат тензоров: (up_tensor, down_tensor)
            if isinstance(v[0], torch.Tensor):
                modified[k] = (v[0] * ratio, v[1])
            else:
                modified[k] = v  # Fallback
                
        elif isinstance(v, torch.Tensor):
            modified[k] = v * ratio
            
        else:
            if debug:
                print(f"[LBW Debug] Неизвестный тип для ключа {k}: {type(v)} | Значение: {v}")
            modified[k] = v

    if debug:
        applied_blocks = {i: ratios[i] for i in range(12) if ratios[i] != 1.0}
        if applied_blocks:
            print(f"[LBW Applied] Масштабированы блоки: {applied_blocks}")
            
    return modified


class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None, vae_filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip
        self.visited_loras = ''

        self.lora_key_map_unet = {}
        self.lora_key_map_clip = {}

        if self.unet is not None:
            self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})
        # 🔹 НОВОЕ: Индексы для расширенного формата (для читаемости кода)
        self.IDX_FILENAME = 0
        self.IDX_WEIGHT   = 1
        self.IDX_TE       = 2
        self.IDX_UNET     = 3
        self.IDX_LBW      = 4
        self.IDX_LBWE     = 5
        self.IDX_START    = 6
        self.IDX_STOP     = 7
        self.IDX_X        = 8
        self.IDX_EXTRA    = 9

        # 🔹 НОВОЕ: Хранилище полных конфигов для доступа извне (например, из ksampler)
        self.loras_config = [] 

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras(self, loras, te_bw=None):
        assert isinstance(loras, list)

        # Проверка кэша (оптимизация, чтобы не перегружать одно и то же)
        if self.visited_loras == str(loras):
            return
        
        self.visited_loras = str(loras)

        if self.unet is None:
            return

        print(f'Request to load LoRAs {str(loras)} for model [{self.filename}].')

        # 🔹 1. Подготовка: очищаем старое хранилище конфигов
        self.loras_config = []
        loras_to_load = []

        for item in loras:
            # 🔹 2. Универсальный парсинг: определяем формат (2 элемента или 10)
            cfg = {
                'filename': None, 'weight': 1.0,
                'te_mult': 1.0, 'unet_mult': 1.0,
                'lbw_str': None, 'lbwe_str': None,
                'start': None, 'stop': None, 'x': 0.0, 'extra': {}
            }

            if len(item) == 2:
                # 📦 Старый формат: (filename, weight)
                cfg['filename'], cfg['weight'] = item
            else:
                # 📦 Расширенный формат: [name, w, te, unet, lbw, lbwe, start, stop, x, extra]
                # Используем проверки на None, чтобы код не падал, если список неполный
                cfg['filename'] = item[self.IDX_FILENAME]
                cfg['weight']   = item[self.IDX_WEIGHT] if item[self.IDX_WEIGHT] is not None else 1.0
                cfg['te_mult']  = item[self.IDX_TE]     if len(item) > self.IDX_TE and item[self.IDX_TE] is not None else 1.0
                cfg['unet_mult']= item[self.IDX_UNET]   if len(item) > self.IDX_UNET and item[self.IDX_UNET] is not None else 1.0
                cfg['lbw_str']  = item[self.IDX_LBW]    if len(item) > self.IDX_LBW else None
                cfg['lbwe_str'] = item[self.IDX_LBWE]   if len(item) > self.IDX_LBWE else None
                cfg['start']    = item[self.IDX_START]  if len(item) > self.IDX_START else None
                cfg['stop']     = item[self.IDX_STOP]   if len(item) > self.IDX_STOP else None
                cfg['x']        = item[self.IDX_X]      if len(item) > self.IDX_X else 0.0
                cfg['extra']    = item[self.IDX_EXTRA]  if len(item) > self.IDX_EXTRA else {}

            if cfg['filename'] == 'None' or not cfg['filename']:
                continue

            # Разрешение пути к файлу (ваша оригинальная логика)
            if os.path.exists(cfg['filename']):
                lora_filename = cfg['filename']
            else:
                lora_filename = get_file_from_folder_list(cfg['filename'], modules.config.paths_loras)

            if not os.path.exists(lora_filename):
                print(f'Lora file not found: {lora_filename}')
                continue

            # Сохраняем полный конфиг и путь для загрузки
            cfg['filepath'] = lora_filename
            self.loras_config.append(cfg)
            loras_to_load.append(cfg)

        # Клонирование моделей (подготовка к применению патчей)
        self.unet_with_lora = self.unet.clone() if self.unet is not None else None
        self.clip_with_lora = self.clip.clone() if self.clip is not None else None

        # 🔹 3. Цикл загрузки и применения
        for cfg in loras_to_load:
            print(f"[LBW] Loading {cfg['filename']}: w={cfg['weight']}, te={cfg['te_mult']}, unet={cfg['unet_mult']}, start={cfg['start']}, stop={cfg['stop']}")
            
            # Загрузка файла и матчинг ключей (ваша оригинальная логика)
            lora_unmatch = ldm_patched.modules.utils.load_torch_file(cfg['filepath'], safe_load=False)
            lora_unet, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_unet)
            lora_clip, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_clip)



            if cfg['lbw_str']:
                # ✅ Патчим ТОЛЬКО UNet
                lora_unet = _apply_block_weights_sdxl(lora_unet, cfg['lbw_str'])
            # --- 🔹 4. Обработка Block Weights (LBW) ---
            # Если заданы локальные lbw/lbwe, здесь можно вызвать функцию применения весов
            # Пример (псевдокод):
            # if cfg['lbw_str'] or cfg['lbwe_str']:
            #     lora_unet = apply_lbw_to_unet(lora_unet, cfg['lbw_str'], cfg['lbwe_str'])
            #     lora_clip = apply_lbw_to_clip(lora_clip, cfg['lbw_str'], cfg['lbwe_str'])
            # -------------------------------------------

            # Проверка на несовпадение модели
            if len(lora_unmatch) > 12:
                continue

            if len(lora_unmatch) > 0:
                print(f'Loaded LoRA [{cfg["filename"]}] for model [{self.filename}] '
                      f'with unmatched keys {list(lora_unmatch.keys())}')

            # 🟦 Применение к UNet с учетом множителя unet_mult
            if self.unet_with_lora is not None and len(lora_unet) > 0:
                final_unet_weight = cfg['weight'] * cfg['unet_mult']
                loaded_keys = self.unet_with_lora.add_patches(lora_unet, final_unet_weight)
                print(f'Loaded LoRA [{cfg["filename"]}] for UNet [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {final_unet_weight:.4f}.')
                for item in lora_unet:
                    if item not in loaded_keys:
                        print("UNet LoRA key skipped: ", item)

            # 🟥 Применение к CLIP с учетом множителя te_mult
            if self.clip_with_lora is not None and len(lora_clip) > 0:
                final_te_weight = cfg['weight'] * cfg['te_mult']
                loaded_keys = self.clip_with_lora.add_patches(lora_clip, final_te_weight)
                print(f'Loaded LoRA [{cfg["filename"]}] for CLIP [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {final_te_weight:.4f}.')
                for item in lora_clip:
                    if item not in loaded_keys:
                        print("CLIP LoRA key skipped: ", item)


@torch.no_grad()
@torch.inference_mode()
def apply_freeu(model, b1, b2, s1, s2):
    return opFreeU.patch(model=model, b1=b1, b2=b2, s1=s1, s2=s2)[0]


@torch.no_grad()
@torch.inference_mode()
def load_controlnet(ckpt_filename):
    return ldm_patched.modules.controlnet.load_controlnet(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return opControlNetApplyAdvanced.apply_controlnet(positive=positive, negative=negative, control_net=control_net,
        image=image, strength=strength, start_percent=start_percent, end_percent=end_percent)


@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename, vae_filename=None):
    unet, clip, vae, vae_filename, clip_vision = load_checkpoint_guess_config(ckpt_filename, embedding_directory=path_embeddings,
                                                                vae_filename_param=vae_filename)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, filename=ckpt_filename, vae_filename=vae_filename)


@torch.no_grad()
@torch.inference_mode()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
    if tiled:
        return opVAEDecodeTiled.decode(samples=latent_image, vae=vae, tile_size=512)[0]
    else:
        return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels, tiled=False):
    if tiled:
        return opVAEEncodeTiled.encode(pixels=pixels, vae=vae, tile_size=512)[0]
    else:
        return opVAEEncode.encode(pixels=pixels, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae_inpaint(vae, pixels, mask):
    assert mask.ndim == 3 and pixels.ndim == 4
    assert mask.shape[-1] == pixels.shape[-2]
    assert mask.shape[-2] == pixels.shape[-3]

    w = mask.round()[..., None]
    pixels = pixels * (1 - w) + 0.5 * w

    latent = vae.encode(pixels)
    B, C, H, W = latent.shape

    latent_mask = mask[:, None, :, :]
    latent_mask = torch.nn.functional.interpolate(latent_mask, size=(H * 8, W * 8), mode="bilinear").round()
    latent_mask = torch.nn.functional.max_pool2d(latent_mask, (8, 8)).round().to(latent)

    return latent, latent_mask


class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x


VAE_approx_models = {}


@torch.no_grad()
@torch.inference_mode()
def get_previewer(model):
    global VAE_approx_models

    from modules.config import path_vae_approx
    is_sdxl = isinstance(model.model.latent_format, ldm_patched.modules.latent_formats.SDXL)
    vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth' if is_sdxl else 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
    else:
        sd = torch.load(vae_approx_filename, map_location='cpu', weights_only=True)
        VAE_approx_model = VAEApprox()
        VAE_approx_model.load_state_dict(sd)
        del sd
        VAE_approx_model.eval()

        if ldm_patched.modules.model_management.should_use_fp16():
            VAE_approx_model.half()
            VAE_approx_model.current_type = torch.float16
        else:
            VAE_approx_model.float()
            VAE_approx_model.current_type = torch.float32

        VAE_approx_model.to(ldm_patched.modules.model_management.get_torch_device())
        VAE_approx_models[vae_approx_filename] = VAE_approx_model

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function(x0, step, total_steps):
        with torch.no_grad():
            x_sample = x0.to(VAE_approx_model.current_type)
            x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample

    return preview_function


@torch.no_grad()
@torch.inference_mode()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None, refiner=None, refiner_switch=-1,
             previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False):

    if sigmas is not None:
        sigmas = sigmas.clone().to(ldm_patched.modules.model_management.get_torch_device())

    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = ldm_patched.modules.sample.prepare_noise(latent_image, seed, batch_inds)

    if isinstance(noise_mean, torch.Tensor):
        noise = noise + noise_mean - torch.mean(noise, dim=1, keepdim=True)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(model)

    if previewer_start is None:
        previewer_start = 0

    if previewer_end is None:
        previewer_end = steps

    def callback(step, x0, x, total_steps):
        ldm_patched.modules.model_management.throw_exception_if_processing_interrupted()
        y = None
        if previewer is not None and not disable_preview:
            y = previewer(x0, previewer_start + step, previewer_end)
        if callback_function is not None:
            callback_function(previewer_start + step, x0, x, previewer_end, y)

    disable_pbar = False
    modules.sample_hijack.current_refiner = refiner
    modules.sample_hijack.refiner_switch_step = refiner_switch
    ldm_patched.modules.samplers.sample = modules.sample_hijack.sample_hacked

    try:
        samples = ldm_patched.modules.sample.sample(model,
                                                    noise, steps, cfg, sampler_name, scheduler,
                                                    positive, negative, latent_image,
                                                    denoise=denoise, disable_noise=disable_noise,
                                                    start_step=start_step,
                                                    last_step=last_step,
                                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                                                    callback=callback,
                                                    disable_pbar=disable_pbar, seed=seed, sigmas=sigmas)

        out = latent.copy()
        out["samples"] = samples
    finally:
        modules.sample_hijack.current_refiner = None

    return out


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
