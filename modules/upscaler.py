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
    print('33333333333333333',upscale_model_glob)
    if  model is None or upscale_model != upscale_model_glob:
    #!if model is None:
        model_filename = downloading_upscale_model2(upscale_model)
        upscale_model_glob = model_filename
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()
    print('444444444444444444444444',upscale_model_glob)
    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = core.pytorch_to_numpy(img)[0]

    return img
