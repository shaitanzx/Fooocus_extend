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
import extentions.lbw.lbw as lbw

opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()
opFreeU = FreeU_V2()
opModelSamplingDiscrete = ModelSamplingDiscrete()
opModelSamplingContinuousEDM = ModelSamplingContinuousEDM()

BLOCK_NAMES = ["BASE", "IN04", "IN05", "IN07", "IN08", "M00", 
               "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"]

def _parse_block_range_to_indices(range_str: str) -> set:
    """Превращает 'IN05-OUT05' или 'M00,BASE' в set индексов [0..11]"""
    indices = set()
    parts = [p.strip() for p in range_str.split(',') if p.strip()]
    for part in parts:
        if '-' in part:
            start, end = [p.strip() for p in part.split('-', 1)]
            if start in BLOCK_NAMES and end in BLOCK_NAMES:
                s, e = BLOCK_NAMES.index(start), BLOCK_NAMES.index(end)
                step = 1 if e >= s else -1
                for i in range(s, e + step, step): indices.add(i)
        elif part in BLOCK_NAMES:
            indices.add(BLOCK_NAMES.index(part))
    return indices

def _load_elemental_presets():
    """
    Загружает elempresets.txt через lbw._load_presets и парсит в структуру:
    {NAME: {'blocks': set[int], 'layers': [str], 'ratio': float}}
    """
    # 1. Используем готовый парсер файлов (убираем дублирование кода чтения)
    raw_presets = lbw._load_presets("elempresets.txt")
    
    structured_presets = {}
    
    # 2. Преобразуем сырые строки "BLOCKS:LAYERS:RATIO" в словарь
    for name, raw_val in raw_presets.items():
        try:
            parts = raw_val.split(':')
            if len(parts) == 3:
                blocks_str, layers_str, ratio_str = parts
                structured_presets[name] = {
                    'blocks': _parse_block_range_to_indices(blocks_str.strip()),
                    'layers': [l.strip().lower() for l in layers_str.split(',')],
                    'ratio': float(ratio_str.strip())
                }
        except Exception:
            # Пропускаем некорректные строки, не ломая загрузку остальных
            pass
            
    return structured_presets

def _apply_elemental_sdxl(lora_dict, lbwe_str, elemental_presets=None, debug=True):
    if not lbwe_str: 
        return lora_dict
    
    rules = []
    
    # 1. Если строка совпадает с именем пресета → используем файл
    if elemental_presets and lbwe_str in elemental_presets:
        p = elemental_presets[lbwe_str]
        rules.append((p['blocks'], p['layers'], p['ratio']))
    else:
        # 2. Парсим inline-правила (поддержка нескольких через ;)
        for rule in lbwe_str.split(';'):
            rule = rule.strip()
            if not rule: continue
            parts = rule.split(':')
            if len(parts) == 3:
                blocks, layers, ratio = parts
                try:
                    rules.append((
                        _parse_block_range_to_indices(blocks.strip()),
                        [l.strip().lower() for l in layers.split(',') if l.strip()],
                        float(ratio.strip())
                    ))
                except Exception:
                    pass

    if not rules: 
        return lora_dict

    # 3. Маппинг ключ → индекс блока
    def get_block_idx(k):
        k_n = k.replace('.', '_').lower()
        if 'input_blocks_4_' in k_n: return 1
        if 'input_blocks_5_' in k_n: return 2
        if 'input_blocks_7_' in k_n: return 3
        if 'input_blocks_8_' in k_n: return 4
        if 'middle_block_' in k_n: return 5
        if 'output_blocks_0_' in k_n: return 6
        if 'output_blocks_1_' in k_n: return 7
        if 'output_blocks_2_' in k_n: return 8
        if 'output_blocks_3_' in k_n: return 9
        if 'output_blocks_4_' in k_n: return 10
        if 'output_blocks_5_' in k_n: return 11
        return 0

    modified = {}
    matched_log = []

    for k, v in lora_dict.items():
        idx = get_block_idx(k)
        applied_ratio = 1.0

        for block_set, layer_keywords, ratio in rules:
            if idx in block_set:
                k_lower = k.lower()
                if any(kw in k_lower for kw in layer_keywords):
                    applied_ratio *= ratio
                    if debug: 
                        # 🔧 ИСПРАВЛЕНИЕ: берём последние 3 части ключа (напр. "attn1.to_q.weight")
                        parts = k.split('.')
                        layer_name = '.'.join(parts[-3:]) if len(parts) >= 3 else k
                        matched_log.append((BLOCK_NAMES[idx] if idx < 12 else "BASE", layer_name, ratio))

        # 4. Применяем накопленный коэффициент к патчу
        if applied_ratio != 1.0:
            if isinstance(v, tuple) and len(v) >= 2 and v[0] == 'lora':
                inner = v[1]
                if isinstance(inner, (list, tuple)):
                    new_inner = []
                    for i, item in enumerate(inner):
                        if isinstance(item, torch.Tensor) and i < 2:
                            new_inner.append(item * applied_ratio)
                        else: 
                            new_inner.append(item)
                    modified[k] = ('lora', tuple(new_inner))
                else: 
                    modified[k] = v
            elif isinstance(v, torch.Tensor):
                modified[k] = v * applied_ratio
            else: 
                modified[k] = v
        else:
            modified[k] = v

    if debug:
        print("\n" + "="*70)
        print(f"🔍 [LBWE DEBUG] Elemental Rule Parser: '{lbwe_str}'")
        print("="*70)
        
        # 1. Вывод правил
        print("📜 Разобранные правила:")
        for r_idx, (blocks, layers, ratio) in enumerate(rules, 1):
            b_names = [BLOCK_NAMES[b] for b in sorted(blocks)]
            print(f"  {r_idx}. Блоки: {b_names}")
            print(f"     Слои : {layers}")
            print(f"     Коэфф: ×{ratio}")
            
        print(f"\n📊 Сканирование UNet патчей (всего {len(lora_dict)} ключа):")
        print("-"*70)
        print(f"{'БЛОК':<6} | {'Ключ':<35} | {'Тип':<6} | {'Правило':<8} | {'Итоговый Δ'}")
        print("-"*70)
        
        # 2. Таблица обработки
        matched_count = 0
        rule_hits = {i+1: 0 for i in range(len(rules))}
        
        for k, v in lora_dict.items():
            idx = get_block_idx(k)
            block_name = BLOCK_NAMES[idx] if idx < 12 else "BASE"
            layer_type = "other"
            matched_rule = "-"
            final_ratio = 1.0
            
            k_lower = k.lower()
            if 'attn' in k_lower: layer_type = "attn"
            elif 'ff' in k_lower or 'net' in k_lower: layer_type = "ff"
            elif 'proj' in k_lower or 'conv' in k_lower: layer_type = "proj"
            
            for r_idx, (block_set, keywords, ratio) in enumerate(rules, 1):
                if idx in block_set and any(kw in k_lower for kw in keywords):
                    final_ratio *= ratio
                    matched_rule = f"Rule {r_idx}"
                    rule_hits[r_idx] += 1
                    matched_count += 1
                    
            # Показываем только первые 30 ключей для читаемости, но считаем все
            if len(matched_log) < 30 or matched_rule != "-":
                key_short = k.split('_')[-1][:25] + '...'
                print(f"{block_name:<6} | {key_short:<35} | {layer_type:<6} | {matched_rule:<8} | ×{final_ratio:.3f}")
                
        print("="*70)
        
        # 3. Сводка
        print(f"\n📈 Итоговая статистика LBWE:")
        print(f"  • Всего ключей просканировано : {len(lora_dict)}")
        print(f"  • Ключей изменено (ratio≠1.0)  : {matched_count}")
        print(f"  • Ключей оставлено без изменений: {len(lora_dict) - matched_count}")
        for r_idx, hits in rule_hits.items():
            print(f"  • Правило {r_idx} сработало на: {hits} ключей")
        print("  ✅ Патчи готовы к передаче в add_patches()")
        print("="*70 + "\n")

    return modified


def _print_lbw_debug(lora_unet, lbw_str):
    block_means = {i: [] for i in range(12)}
    block_examples = {i: "" for i in range(12)}

    def get_idx(k):
        k_n = k.replace('.', '_').lower()
        if 'input_blocks_4_' in k_n: return 1
        if 'input_blocks_5_' in k_n: return 2
        if 'input_blocks_7_' in k_n: return 3
        if 'input_blocks_8_' in k_n: return 4
        if 'middle_block_' in k_n: return 5
        if 'output_blocks_0_' in k_n: return 6
        if 'output_blocks_1_' in k_n: return 7
        if 'output_blocks_2_' in k_n: return 8
        if 'output_blocks_3_' in k_n: return 9
        if 'output_blocks_4_' in k_n: return 10
        if 'output_blocks_5_' in k_n: return 11
        return 0

    for k, v in lora_unet.items():
        idx = get_idx(k)
        if not block_examples[idx]:
            block_examples[idx] = k[-50:]  # Сохраняем пример ключа для лога

        # 🔹 Извлекаем тензоры из формата Fooocus: ('lora', (up, down, alpha, None))
        tensors = []
        if isinstance(v, tuple) and len(v) >= 2 and v[0] == 'lora':
            tensors = [t for t in v[1] if isinstance(t, torch.Tensor)]
        elif isinstance(v, torch.Tensor):
            tensors = [v]

        for t in tensors:
            block_means[idx].append(t.abs().mean().item())

    # 🔹 Вывод полной таблицы по всем 12 блокам
    print("\n" + "="*75)
    print(f"[LBW VERIFY] Состояние весов после lbw='{lbw_str}':")
    print(f"{'Блок':<8} | {'Найдено':>8} | {'Ср. |Δ| (mean)':>15} | Пример ключа")
    print("-"*75)
    for i in range(12):
        count = len(block_means[i])
        mean_val = sum(block_means[i]) / count if count > 0 else 0.0
        status = "✅" if count > 0 else "⚪"
        print(f"{BLOCK_NAMES[i]:<8} | {count:>6} патч. | {mean_val:>15.8f} | {block_examples[i]}")
    print("="*75 + "\n")


def _apply_block_weights_sdxl(lora_dict, lbw_str, debug=True):
    """Применяет блочные веса к патчам Fooocus: {key: ('lora', (up, down, ...))}"""
    if not lbw_str:
        return lora_dict

    ratios = [float(x.strip()) for x in lbw_str.split(",")]
    if len(ratios) != 12:
        print(f"[LBW Warning] Ожидалось 12 значений для SDXL, получено {len(ratios)}.")
        return lora_dict

    block_names = ["BASE", "IN04", "IN05", "IN07", "IN08", "M00", 
                   "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"]
    stats = {i: {"count": 0, "examples": []} for i in range(12)}
    modified = {}

    def get_idx(k):
        if "input_blocks.4" in k: return 1
        if "input_blocks.5" in k: return 2
        if "input_blocks.7" in k: return 3
        if "input_blocks.8" in k: return 4
        if "middle_block" in k: return 5
        if "output_blocks.0" in k: return 6
        if "output_blocks.1" in k: return 7
        if "output_blocks.2" in k: return 8
        if "output_blocks.3" in k: return 9
        if "output_blocks.4" in k: return 10
        if "output_blocks.5" in k: return 11
        return 0

    for k, v in lora_dict.items():
        idx = get_idx(k)
        ratio = ratios[idx]
        stats[idx]["count"] += 1
        if len(stats[idx]["examples"]) < 3:
            stats[idx]["examples"].append(k.split(".")[-1])

        # 🔹 ОБРАБОТКА ФОРМАТА: ('lora', (up, down, ...))
        if isinstance(v, tuple) and len(v) >= 2 and v[0] == 'lora':
            inner = v[1]
            if isinstance(inner, (list, tuple)):
                # Умножаем все тензоры внутри вложенного кортежа
                new_inner = tuple(x * ratio if isinstance(x, torch.Tensor) else x for x in inner)
                modified[k] = ('lora', new_inner)
            elif isinstance(inner, torch.Tensor):
                modified[k] = ('lora', inner * ratio)
            else:
                modified[k] = v  # fallback
        elif isinstance(v, torch.Tensor):
            modified[k] = v * ratio
        else:
            modified[k] = v  # fallback для неизвестных форматов

    if debug:
        total = sum(s["count"] for s in stats.values())
        print("\n" + "="*70)
        print(f"[LBW DEBUG] Блочные веса применены к {total} патчам:")
        print(f"{'Блок':<8} | {'Коэфф.':>6} | {'Патчи':>8} | {'Примеры слоёв'}")
        print("-" * 70)
        for i in range(12):
            s = stats[i]
            ex_str = ", ".join(s["examples"]) if s["examples"] else "-"
            print(f"{block_names[i]:<8} | {ratios[i]:<6.4f} | {s['count']:>8} | {ex_str}")
        print("="*70 + "\n")

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

        # 🔒 Кэш: не перегружаем, если список не изменился
        if self.visited_loras == str(loras):
            return
        
        self.visited_loras = str(loras)
        if self.unet is None:
            return

        print(f'Request to load LoRAs {str(loras)} for model [{self.filename}].')

        # 🔹 1. Подготовка конфигов
        self.loras_config = []
        loras_to_load = []

        for item in loras:
            cfg = {
                'filename': None, 'weight': 1.0,
                'te_mult': 1.0, 'unet_mult': 1.0,
                'lbw_str': None, 'lbwe_str': None,
                'start': None, 'stop': None, 'x': 0.0, 'extra': {}
            }

            if len(item) == 2:
                cfg['filename'], cfg['weight'] = item
            else:
                cfg['filename'] = item[0]
                cfg['weight']   = item[1] if len(item) > 1 and item[1] is not None else 1.0
                cfg['te_mult']  = item[2] if len(item) > 2 and item[2] is not None else 1.0
                cfg['unet_mult']= item[3] if len(item) > 3 and item[3] is not None else 1.0
                cfg['lbw_str']  = item[4] if len(item) > 4 else None
                cfg['lbwe_str'] = item[5] if len(item) > 5 else None
                cfg['start']    = item[6] if len(item) > 6 else None
                cfg['stop']     = item[7] if len(item) > 7 else None
                cfg['x']        = item[8] if len(item) > 8 else 0.0
                cfg['extra']    = item[9] if len(item) > 9 else {}

            if not cfg['filename'] or cfg['filename'] == 'None':
                continue

            # Разрешение пути
            if os.path.exists(cfg['filename']):
                lora_filename = cfg['filename']
            else:
                lora_filename = get_file_from_folder_list(cfg['filename'], modules.config.paths_loras)

            if not os.path.exists(lora_filename):
                print(f'Lora file not found: {lora_filename}')
                continue

            cfg['filepath'] = lora_filename
            self.loras_config.append(cfg)
            loras_to_load.append(cfg)

        # 🔹 2. Клонирование моделей & Загрузка пресетов
        self.unet_with_lora = self.unet.clone() if self.unet is not None else None
        self.clip_with_lora = self.clip.clone() if self.clip is not None else None

        # Загружаем элементарные пресеты ОДИН РАЗ на все LoRA
        elemental_presets = _load_elemental_presets()

        # 🔧 Включите True для подробного логирования патчей
        DEBUG_LBW_VERBOSE = True  

        # 🔹 3. Цикл загрузки и применения
        for cfg in loras_to_load:
            print(f"[LBW] Loading {cfg['filename']}: w={cfg['weight']}, te={cfg['te_mult']}, unet={cfg['unet_mult']}, start={cfg['start']}, stop={cfg['stop']}")
            
            lora_unmatch = ldm_patched.modules.utils.load_torch_file(cfg['filepath'], safe_load=False)
            lora_unet, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_unet)
            lora_clip, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_clip)

            # ✅ 3.1 Применяем блочные веса (LBW)
            if cfg['lbw_str']:
                lora_unet = _apply_block_weights_sdxl(lora_unet, cfg['lbw_str'])

            # ✅ 3.2 Применяем послойные веса (LBWE)
            if cfg['lbwe_str']:
                lora_unet = _apply_elemental_sdxl(lora_unet, cfg['lbwe_str'], elemental_presets)

            # 🔍 Отладочный вывод (включается через DEBUG_LBW_VERBOSE)
            if DEBUG_LBW_VERBOSE and cfg['lbw_str']:
                _print_lbw_debug(lora_unet, cfg['lbw_str'])

            # Проверка на несовпадение модели
            if len(lora_unmatch) > 12:
                continue
            if len(lora_unmatch) > 0:
                print(f'Loaded LoRA [{cfg["filename"]}] for model [{self.filename}] '
                      f'with unmatched keys {list(lora_unmatch.keys())}')

            # 🟦 Применение к UNet
            if self.unet_with_lora is not None and len(lora_unet) > 0:
                final_unet_weight = cfg['weight'] * cfg['unet_mult']
                loaded_keys = self.unet_with_lora.add_patches(lora_unet, final_unet_weight)
                print(f'Loaded LoRA [{cfg["filename"]}] for UNet [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {final_unet_weight:.4f}.')
                for item in lora_unet:
                    if item not in loaded_keys:
                        print("UNet LoRA key skipped: ", item)

            # 🟥 Применение к CLIP (сохранена оригинальная логика te_bw)
            if self.clip_with_lora is not None and len(lora_clip) > 0:
                # Если задан глобальный te_bw, применяем его к CLIP-патчам                
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
