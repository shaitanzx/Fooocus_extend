from collections import OrderedDict

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.model_loading import load_state_dict,UnsupportedModel
from ldm_patched.pfn.model_loading import (
    ESRGAN, RealESRGANv2, SPSR, SwiftSRGAN, SwinIR, Swin2SR,
    HAT, DAT, OmniSR, SCUNet, GFPGANv1Clean, RestoreFormer, CodeFormer, LaMa
)
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model2






# 👇 ИМПОРТИРУЕМ ТВОЙ model_loading.py (должен лежать в модуле, где вызывается perform_upscale)
from .model_loading import load_state_dict, UnsupportedModel  # ← важно: не из ldm_patched!

# Импорты для isinstance — нужны ТОЛЬКО они, остальное не требуется
from .model_loading import (
    ESRGAN, RealESRGANv2, SPSR, SwiftSRGAN, SwinIR, Swin2SR,
    HAT, DAT, OmniSR, SCUNet, GFPGANv1Clean, RestoreFormer, CodeFormer, LaMa
)



opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None
upscale_model_glob=None

def get_model_architecture_safe(model_path: str) -> str:
    """Определяет архитектуру через фактическую загрузку модели на CPU (лёгкая, без весов в GPU)."""
    try:
        sd = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Загружаем модель — но держим её на CPU, без .to('cuda')
        model = load_state_dict(sd)
        
        # Определяем по типу
        if isinstance(model, ESRGAN):
            return "ESRGAN"
        elif isinstance(model, RealESRGANv2):
            return "RealESRGANv2"
        elif isinstance(model, SPSR):
            return "SPSR"
        elif isinstance(model, SwiftSRGAN):
            return "SwiftSRGAN"
        elif isinstance(model, SwinIR):
            return "SwinIR"
        elif isinstance(model, Swin2SR):
            return "Swin2SR"
        elif isinstance(model, HAT):
            return "HAT"
        elif isinstance(model, DAT):
            return "DAT"
        elif isinstance(model, OmniSR):
            return "OmniSR"
        elif isinstance(model, SCUNet):
            return "SCUNet"
        elif isinstance(model, GFPGANv1Clean):
            return "GFPGAN"
        elif isinstance(model, RestoreFormer):
            return "RestoreFormer"
        elif isinstance(model, CodeFormer):
            return "CodeFormer"
        elif isinstance(model, LaMa):
            return "LaMa"
        else:
            return "Unknown"
            
    except Exception as e:
        return f"Error: {type(e).__name__}"


def perform_upscale(img,upscale_model):
    global model, upscale_model_glob

    print(f'Upscaling image with shape {str(img.shape)} ...')
    h_in, w_in = img.shape[:2]  # img — numpy, HWC
    print(f"📥 Input image: {w_in} × {h_in}")
    if  model is None or upscale_model != upscale_model_glob:        
        model_filename = downloading_upscale_model2(upscale_model)
        upscale_model_glob = model_filename

        arch = get_model_architecture_safe(model_filename)
        print(f"✅ Model '{upscale_model}' → {arch}"
        
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        print(f"✅ Loaded model '{upscale_model_glob}': scale = {model.scale}x, "
              f"blocks = {model.num_blocks}, arch = {model.model_arch}")
        model.cpu()
        model.eval()
    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = core.pytorch_to_numpy(img)[0]
    h_out, w_out = img.shape[:2]
    print(f"📏 Input: {w_in}×{h_in} → Output: {w_out}×{h_out} (×{w_out / w_in:.2f})")
    return img
