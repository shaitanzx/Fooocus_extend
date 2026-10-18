import json
import os
import re
import struct
import sys

SD15, SDXL, SD2, UNKNOWN = 'sd15', 'sdxl', 'sd2', 'unknown'
_CTX = {768: SD15, 1024: SD2, 2048: SDXL}
_ATTN2_K = re.compile(r'attn2[._]to[._]k')
_cache = {}


def _read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


def describe_lora(path):
    info = {'formats': [], 'family': UNKNOWN, 'dora': False}
    try:
        if not str(path).endswith('.safetensors'):
            return info
        ck = (path, os.path.getmtime(path))
        if ck in _cache:
            return _cache[ck]
        header = _read_header(path)
        keys = [k for k in header if k != '__metadata__']

        def has(token):
            return any(token in k for k in keys)

        def ends(*suffixes):
            return any(k.endswith(suffixes) for k in keys)

        fmts = []
        if has('.lora_down.') or has('.lora_up.') or has('lora_A') or has('lora_B'):
            fmts.append('lora')
        if has('.hada_w1_a'):
            fmts.append('loha')
        if has('.lokr_w'):
            fmts.append('lokr')
        if ends('.a1.weight') and ends('.b1.weight'):
            fmts.append('glora')
        if ends('.diff', '.diff_b'):
            fmts.append('diff')
        info['formats'] = fmts
        info['dora'] = has('.dora_scale') or has('lora_magnitude_vector')

        family = UNKNOWN
        if any('lora_te2_' in k or 'text_encoder_2' in k for k in keys):
            family = SDXL
        if family == UNKNOWN:
            for k in keys:
                if _ATTN2_K.search(k) and k.endswith(
                        ('.lora_down.weight', '.lora_A.weight', '.lora_A.default.weight', '.hada_w1_b')):
                    shape = header[k].get('shape', [])
                    if len(shape) >= 2:
                        family = _CTX.get(shape[1], UNKNOWN)
                        break
        if family == UNKNOWN:
            # у SDXL нет четвёртого уровня блоков (down/up_blocks_3, input/output_blocks 9-11)
            if any(re.search(r'(down|up)_blocks_3', k) or re.search(r'(input|output)_blocks_(9|10|11)_', k)
                   for k in keys):
                family = SD15
        if family == UNKNOWN and any(k.startswith('lora_te_') for k in keys):
            family = SD15
        info['family'] = family
        _cache[ck] = info
    except Exception as e:
        print(f'[lora_family] failed for {path}: {e}')
    return info


def check_lora_for_family(path, model_family):
    """model_family: 'sd15' или 'sdxl'. Возвращает (ok, message)."""
    info = describe_lora(path)
    fam = info['family']
    if fam != UNKNOWN and model_family in (SD15, SDXL) and fam != model_family:
        kinds = '/'.join(info['formats']) or '?'
        return False, (f'{os.path.basename(path)}: looks like {fam} ({kinds}), '
                       f'model is {model_family} -> skipped')
    return True, ''
