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
            if family == SD15 and in_ch != 4:
                family = UNSUPPORTED  # 9-канальные inpaint-чекпоинты SD1.5
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
    ucfg = getattr(getattr(m, 'model_config', None), 'unet_config', None) or {}
    if ucfg.get('context_dim') == 768 and ucfg.get('in_channels', 4) == 4:
        return SD15
    return UNSUPPORTED