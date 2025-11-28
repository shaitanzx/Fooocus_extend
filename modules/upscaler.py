from collections import OrderedDict
import os
import modules.core as core
import torch
from safetensors.torch import load_file as load_safetensors
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.model_loading import load_state_dict,UnsupportedModel
from ldm_patched.pfn.model_loading import (
    ESRGAN, RealESRGANv2, SPSR, SwiftSRGAN, SwinIR, Swin2SR,
    HAT, DAT, OmniSR, SCUNet, GFPGANv1Clean, RestoreFormer, CodeFormer, LaMa
)
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model2

def load_state_dict_robust(path: str):
    _, ext = os.path.splitext(path)
    ext = ext.lower().lstrip('.')

    if ext == "safetensors":
        return load_safetensors(path, device="cpu")

    else:
        # Сначала weights_only=True (безопасно), fallback на False при ошибке
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)

opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None
upscale_model_glob=None

def get_model_architecture_safe(model_path: str) -> str:
    try:
        sd = load_state_dict_robust(model_path)
        
        model = load_state_dict(sd)
        
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

        sd = load_state_dict_robust(model_filename)

        arch = get_model_architecture_safe(model_filename)


        # 🔹 Шаг 3: создаём модель в зависимости от архитектуры
        if arch == "RealESRGANv2":
            model = RealESRGANv2(sd)
        elif arch == "ESRGAN":
            # Для ESRGAN делаем совместимость с legacy-ключами (ваш sdo)
            sdo = OrderedDict()
            for k, v in sd.items():
                sdo[k.replace('residual_block_', 'RDB')] = v
            model = ESRGAN(sdo)
        elif arch == "SPSR":
            model = SPSR(sd)
        elif arch == "SwiftSRGAN":
            model = SwiftSRGAN(sd)
        elif arch == "SwinIR":
            model = SwinIR(sd)
        elif arch == "Swin2SR":
            model = Swin2SR(sd)
        elif arch == "HAT":
            model = HAT(sd)
        elif arch == "DAT":
            model = DAT(sd)
        elif arch == "OmniSR":
            model = OmniSR(sd)
        elif arch == "SCUNet":
            model = SCUNet(sd)
        elif arch == "GFPGAN":
            model = GFPGANv1Clean(sd)
        elif arch == "RestoreFormer":
            model = RestoreFormer(sd)
        elif arch == "CodeFormer":
            model = CodeFormer(sd)
        elif arch == "LaMa":
            model = LaMa(sd)
        else:
            sdo = OrderedDict()
            for k, v in sd.items():
                sdo[k.replace('residual_block_', 'RDB')] = v
            try:
                model = ESRGAN(sdo)

            except Exception as e:
                raise RuntimeError(f"Error model '{upscale_model}': {e}")


        scale = getattr(model, 'scale', '?')
        blocks = getattr(model, 'num_blocks', '?')
        arch_name = getattr(model, 'model_arch', arch)
        print(f"✅ Model '{upscale_model_glob}' → {arch}")
        print(f"✅ scale = {scale}x, blocks = {blocks}, arch = {arch_name}")
        del sd
        model.cpu()
        model.eval()
    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = core.pytorch_to_numpy(img)[0]
    h_out, w_out = img.shape[:2]
    print(f"📏 Input: {w_in}×{h_in} → Output: {w_out}×{h_out} (×{w_out / w_in:.2f})")
    return img
