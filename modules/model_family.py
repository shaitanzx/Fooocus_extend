import json
import struct

SDXL, SD15, SD2, SDXL_REFINER, UNSUPPORTED, UNKNOWN = (
    'sdxl', 'sd15', 'sd2', 'sdxl_refiner', 'unsupported', 'unknown')

_cache = {}

def detect_family_from_file(path):
    if path in _cache:
        return _cache[path]
    family = UNKNOWN
    try:
        if str(path).endswith('.safetensors'):
            with open(path, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(n))
            in_ch = header.get('model.diffusion_model.input_blocks.0.0.weight', {}).get('shape', [0, 4])[1]
            ctx = None
            for k, v in header.items():
                if k.startswith('model.diffusion_model.') and k.endswith('attn2.to_k.weight'):
                    ctx = v['shape'][1]
                    break
            family = {768: SD15, 1024: SD2, 2048: SDXL, 1280: SDXL_REFINER}.get(ctx, UNKNOWN)

            # SD 1.5 может иметь 4 входных канала (обычный UNet)
            # или 9 входных каналов (native inpaint UNet).
            if family == SD15 and in_ch not in (4, 9):
                family = UNSUPPORTED
    except Exception as e:
        print(f'[model_family] header detect failed: {e}')
    _cache[path] = family
    return family

def detect_family_from_model(unet_patcher):
    from ldm_patched.modules.model_base import SDXL as _SDXL, SDXLRefiner as _Ref

    m = unet_patcher.model
    if isinstance(m, _SDXL):
        return SDXL
    if isinstance(m, _Ref):
        return SDXL_REFINER

    ucfg = getattr(
        getattr(m, "model_config", None),
        "unet_config",
        None
    ) or {}

    if (
        ucfg.get("context_dim") == 768
        and ucfg.get("in_channels", 4) in (4, 9)
    ):
        return SD15
    return UNSUPPORTED

def is_sd15_inpaint_file(path):
    try:
        if not str(path).endswith(".safetensors"):
            return False

        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))

        in_channels = header.get(
            "model.diffusion_model.input_blocks.0.0.weight", {}
        ).get("shape", [0, 4])[1]

        return in_channels == 9 and detect_family_from_file(path) == SD15
    except Exception as e:
        print(f"[model_family] SD1.5 inpaint detect failed: {e}")
        return False
