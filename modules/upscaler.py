from collections import OrderedDict

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model2

opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None
upscale_model_glob=None


def perform_upscale(img,upscale_model):
    global model, upscale_model_glob

    print(f'Upscaling image with shape {str(img.shape)} ...')
    h_in, w_in = img.shape[:2]  # img — numpy, HWC
    print(f"📥 Input image: {w_in} × {h_in}")
    if  model is None or upscale_model != upscale_model_glob:
        model_filename = downloading_upscale_model2(upscale_model)
        upscale_model_glob = model_filename
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
