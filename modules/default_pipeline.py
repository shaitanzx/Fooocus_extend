import modules.core as core
import os
import torch
import modules.patch
import modules.config
import modules.flags
import ldm_patched.modules.model_management
import ldm_patched.modules.latent_formats
import modules.inpaint_worker
import extras.vae_interpose as vae_interpose
from extras.expansion import FooocusExpansion

from ldm_patched.modules.model_base import SDXL, SDXLRefiner
from modules.sample_hijack import clip_separate
from modules.util import get_file_from_folder_list, get_enabled_loras


import copy
from typing import Optional
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair



from enum import Enum
from extentions.layerdiffuse.lib_layerdiffusion.enums import ResizeMode
from extentions.layerdiffuse.lib_layerdiffusion.utils import rgba2rgbfp32, to255unit8, crop_and_resize_image, forge_clip_encode
import numpy as np
from modules.model_loader import load_file_from_url
from extentions.layerdiffuse import layer as layer_module
from extentions.layerdiffuse.lib_layerdiffusion.models import TransparentVAEDecoder, TransparentVAEEncoder


model_base = core.StableDiffusionModel()
model_refiner = core.StableDiffusionModel()

final_expansion = None
final_unet = None
final_clip = None
final_vae = None
final_refiner_unet = None
final_refiner_vae = None

loaded_ControlNets = {}


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    global loaded_ControlNets
    cache = {}
    for p in model_paths:
        if p is not None:
            if p in loaded_ControlNets:
                cache[p] = loaded_ControlNets[p]
            else:
                cache[p] = core.load_controlnet(p)
    loaded_ControlNets = cache
    return


@torch.no_grad()
@torch.inference_mode()
def assert_model_integrity():
    error_message = None

    if not isinstance(model_base.unet_with_lora.model, SDXL):
        error_message = 'You have selected base model other than SDXL. This is not supported yet.'

    if error_message is not None:
        raise NotImplementedError(error_message)

    return True


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name, vae_name=None):
    global model_base

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    vae_filename = None
    if vae_name is not None and vae_name != modules.flags.default_vae:
        vae_filename = get_file_from_folder_list(vae_name, modules.config.path_vae)

    if model_base.filename == filename and model_base.vae_filename == vae_filename:
        return

    model_base = core.load_model(filename, vae_filename)
    print(f'Base model loaded: {model_base.filename}')
    print(f'VAE loaded: {model_base.vae_filename}')
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_refiner_model(name):
    global model_refiner

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    if model_refiner.filename == filename:
        return

    model_refiner = core.StableDiffusionModel()

    if name == 'None':
        print(f'Refiner unloaded.')
        return

    model_refiner = core.load_model(filename)
    print(f'Refiner model loaded: {model_refiner.filename}')

    if isinstance(model_refiner.unet.model, SDXL):
        model_refiner.clip = None
        model_refiner.vae = None
    elif isinstance(model_refiner.unet.model, SDXLRefiner):
        model_refiner.clip = None
        model_refiner.vae = None
    else:
        model_refiner.clip = None

    return


@torch.no_grad()
@torch.inference_mode()
def synthesize_refiner_model():
    global model_base, model_refiner

    print('Synthetic Refiner Activated')
    model_refiner = core.StableDiffusionModel(
        unet=model_base.unet,
        vae=model_base.vae,
        clip=model_base.clip,
        clip_vision=model_base.clip_vision,
        filename=model_base.filename
    )
    model_refiner.vae = None
    model_refiner.clip = None
    model_refiner.clip_vision = None

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_loras(loras, base_model_additional_loras=None):
    global model_base, model_refiner

    if not isinstance(base_model_additional_loras, list):
        base_model_additional_loras = []

    model_base.refresh_loras(loras + base_model_additional_loras)
    model_refiner.refresh_loras(loras)

    return


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False):
    cached = clip.fcs_cond_cache.get(text, None)
    if cached is not None:
        if verbose:
            print(f'[CLIP Cached] {text}')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    clip.fcs_cond_cache[text] = result
    if verbose:
        print(f'[CLIP Encoded] {text}')
    return result


@torch.no_grad()
@torch.inference_mode()
def clone_cond(conds):
    results = []

    for c, p in conds:
        p = p["pooled_output"]

        if isinstance(c, torch.Tensor):
            c = c.clone()

        if isinstance(p, torch.Tensor):
            p = p.clone()

        results.append([c, {"pooled_output": p}])

    return results


@torch.no_grad()
@torch.inference_mode()
def clip_encode(texts, pool_top_k=1):
    global final_clip

    if final_clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(final_clip, text)
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]


@torch.no_grad()
@torch.inference_mode()
def set_clip_skip(clip_skip: int):
    global final_clip

    if final_clip is None:
        return

    final_clip.clip_layer(-abs(clip_skip))
    return

@torch.no_grad()
@torch.inference_mode()
def clear_all_caches():
    final_clip.fcs_cond_cache = {}


@torch.no_grad()
@torch.inference_mode()
def prepare_text_encoder(async_call=True):
    if async_call:
        # TODO: make sure that this is always called in an async way so that users cannot feel it.
        pass
    assert_model_integrity()
    ldm_patched.modules.model_management.load_models_gpu([final_clip.patcher, final_expansion.patcher])
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras,
                       base_model_additional_loras=None, use_synthetic_refiner=False, vae_name=None):
    global final_unet, final_clip, final_vae, final_refiner_unet, final_refiner_vae, final_expansion

    final_unet = None
    final_clip = None
    final_vae = None
    final_refiner_unet = None
    final_refiner_vae = None

    if use_synthetic_refiner and refiner_model_name == 'None':
        print('Synthetic Refiner Activated')
        refresh_base_model(base_model_name, vae_name)
        synthesize_refiner_model()
    else:
        refresh_refiner_model(refiner_model_name)
        refresh_base_model(base_model_name, vae_name)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    final_refiner_unet = model_refiner.unet_with_lora
    final_refiner_vae = model_refiner.vae

    if final_expansion is None:
        final_expansion = FooocusExpansion()

    prepare_text_encoder(async_call=True)
    clear_all_caches()
    return


refresh_everything(
    refiner_model_name=modules.config.default_refiner_model_name,
    base_model_name=modules.config.default_base_model_name,
    loras=get_enabled_loras(modules.config.default_loras),
    vae_name=modules.config.default_vae,
)


@torch.no_grad()
@torch.inference_mode()
def vae_parse(latent):
    if final_refiner_vae is None:
        return latent

    result = vae_interpose.parse(latent["samples"])
    return {'samples': result}


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_all(sampler, model, scheduler, steps):
    from ldm_patched.modules.samplers import calculate_sigmas_scheduler

    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral']:
        steps += 1
        discard_penultimate_sigma = True

    sigmas = calculate_sigmas_scheduler(model, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas(sampler, model, scheduler, steps, denoise):
    if denoise is None or denoise > 0.9999:
        sigmas = calculate_sigmas_all(sampler, model, scheduler, steps)
    else:
        new_steps = int(steps / denoise)
        sigmas = calculate_sigmas_all(sampler, model, scheduler, new_steps)
        sigmas = sigmas[-(steps + 1):]
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def get_candidate_vae(steps, switch, denoise=1.0, refiner_swap_method='joint'):
    assert refiner_swap_method in ['joint', 'separate', 'vae']

    if final_refiner_vae is not None and final_refiner_unet is not None:
        if denoise > 0.9:
            return final_vae, final_refiner_vae
        else:
            if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                return final_vae, None
            else:
                return final_refiner_vae, None

    return final_vae, final_refiner_vae

class LayerMethod(Enum):
    FG_ONLY_ATTN = "(SDXL) Only Generate Transparent Image (Attention Injection)"
    FG_ONLY_CONV = "(SDXL) Only Generate Transparent Image (Conv Injection)"
    FG_TO_BLEND = "(SDXL) From Foreground to Blending"
    FG_BLEND_TO_BG = "(SDXL) From Foreground and Blending to Background"
    BG_TO_BLEND = "(SDXL) From Background to Blending"
    BG_BLEND_TO_FG = "(SDXL) From Background and Blending to Foreground"
layer_model_root = os.path.join(os.path.dirname(modules.config.path_vae), 'layer_model')
os.makedirs(layer_model_root, exist_ok=True)
@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, 
        scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint', 
        disable_preview=False,tile_x=False,tile_y=False,layer_diff=['False']):

    target_unet, target_vae, target_refiner_unet, target_refiner_vae, target_clip \
        = final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip

    assert refiner_swap_method in ['joint', 'separate', 'vae']

    if final_refiner_vae is not None and final_refiner_unet is not None:
        # Refiner Use Different VAE (then it is SD15)
        if denoise > 0.9:
            refiner_swap_method = 'vae'
        else:
            refiner_swap_method = 'joint'
            if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                    = final_unet, final_vae, None, None
                print(f'[Sampler] only use Base because of partial denoise.')
            else:
                positive_cond = clip_separate(positive_cond, target_model=final_refiner_unet.model, target_clip=final_clip)
                negative_cond = clip_separate(negative_cond, target_model=final_refiner_unet.model, target_clip=final_clip)
                target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                    = final_refiner_unet, final_refiner_vae, None, None
                print(f'[Sampler] only use Refiner because of partial denoise.')

    print(f'[Sampler] refiner_swap_method = {refiner_swap_method}')

    if latent is None:
        initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        initial_latent = latent

    minmax_sigmas = calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=final_unet.model, steps=steps, denoise=denoise)
    sigma_min, sigma_max = minmax_sigmas[minmax_sigmas > 0].min(), minmax_sigmas.max()
    sigma_min = float(sigma_min.cpu().numpy())
    sigma_max = float(sigma_max.cpu().numpy())
    print(f'[Sampler] sigma_min = {sigma_min}, sigma_max = {sigma_max}')

    modules.patch.BrownianTreeNoiseSamplerPatched.global_init(
        initial_latent['samples'].to(ldm_patched.modules.model_management.get_torch_device()),
        sigma_min, sigma_max, seed=image_seed, cpu=False)

    decoded_latent = None



    if len(layer_diff)>1:
        method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt = layer_diff


        B, C, H, W = initial_latent['samples'].shape  # latent_shape
        height = H * 8
        width = W * 8
        batch_size = 1

        method = LayerMethod(method)
        print(f'[LayerDiffuse] {method}')

        resize_mode = ResizeMode(resize_mode)
        fg_image = crop_and_resize_image(rgba2rgbfp32(fg_image), resize_mode, height, width) if fg_image is not None else None
        bg_image = crop_and_resize_image(rgba2rgbfp32(bg_image), resize_mode, height, width) if bg_image is not None else None
        blend_image = crop_and_resize_image(rgba2rgbfp32(blend_image), resize_mode, height, width) if blend_image is not None else None

        original_unet = target_unet
        unet = target_unet.clone()
        vae = target_vae
        clip = target_clip

        if method in [LayerMethod.FG_TO_BLEND, LayerMethod.FG_BLEND_TO_BG, LayerMethod.BG_TO_BLEND, LayerMethod.BG_BLEND_TO_FG]:
            if fg_image is not None:
                fg_image = vae.encode(torch.from_numpy(np.ascontiguousarray(fg_image[None].copy())))
                fg_image = vae.first_stage_model.process_in(fg_image)

            if bg_image is not None:
                bg_image = vae.encode(torch.from_numpy(np.ascontiguousarray(bg_image[None].copy())))
                bg_image = vae.first_stage_model.process_in(bg_image)

            if blend_image is not None:
                blend_image = vae.encode(torch.from_numpy(np.ascontiguousarray(blend_image[None].copy())))
                blend_image = vae.first_stage_model.process_in(blend_image)

        if method == LayerMethod.FG_ONLY_ATTN:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_attn.safetensors'
            )
            layer_lora_model = layer_module.load_layer_model_state_dict(model_path)
            #unet.load_frozen_patcher('layer_xl_transparent_attn.safetensors', layer_lora_model, weight)
            unet.load_frozen_patcher(os.path.basename(model_dir), layer_lora_model, weight)

        if method == LayerMethod.FG_ONLY_CONV:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_conv.safetensors'
            )
            layer_lora_model = layer_module.load_layer_model_state_dict(model_path)
            #unet.load_frozen_patcher('layer_xl_transparent_conv.safetensors', layer_lora_model, weight)
            unet.load_frozen_patcher(os.path.basename(model_dir), layer_lora_model, weight)

        if method == LayerMethod.BG_TO_BLEND:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_bg2ble.safetensors'
            )
            unet.extra_concat_condition = bg_image
            layer_lora_model = layer_module.load_layer_model_state_dict(model_path)
            #unet.load_frozen_patcher('layer_xl_bg2ble.safetensors', layer_lora_model, weight)
            unet.load_frozen_patcher(os.path.basename(model_dir), layer_lora_model, weight)

        if method == LayerMethod.FG_TO_BLEND:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_fg2ble.safetensors'
            )
            unet.extra_concat_condition = fg_image
            layer_lora_model = layer_module.load_layer_model_state_dict(model_path)
            #unet.load_frozen_patcher('layer_xl_fg2ble.safetensors', layer_lora_model, weight)
            unet.load_frozen_patcher(os.path.basename(model_dir), layer_lora_model, weight)

        if method == LayerMethod.BG_BLEND_TO_FG:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bgble2fg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_bgble2fg.safetensors'
            )
            unet.extra_concat_condition = torch.cat([bg_image, blend_image], dim=1)
            layer_lora_model = layer_module.load_layer_model_state_dict(model_path)
            #unet.load_frozen_patcher('layer_xl_bgble2fg.safetensors', layer_lora_model, weight)
            unet.load_frozen_patcher(os.path.basename(model_dir), layer_lora_model, weight)

        if method == LayerMethod.FG_BLEND_TO_BG:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fgble2bg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_fgble2bg.safetensors'
            )
            unet.extra_concat_condition = torch.cat([fg_image, blend_image], dim=1)
            layer_lora_model = layer_module.load_layer_model_state_dict(model_path)
            #unet.load_frozen_patcher('layer_xl_fgble2bg.safetensors', layer_lora_model, weight)
            unet.load_frozen_patcher(os.path.basename(model_dir), layer_lora_model, weight)
        #sigma_end = unet.model.predictor.percent_to_sigma(ending_step)
        step_index = int((len(minmax_sigmas) - 1) * ending_step)
        sigma_end = minmax_sigmas[step_index].item()
        print(f'[LayerDiffusion] Ending at step {step_index}/{len(minmax_sigmas)-1}, sigma = {sigma_end}')
        
        def remove_concat(cond):
            cond = copy.deepcopy(cond)
            for i in range(len(cond)):
                try:
                    del cond[i]['model_conds']['c_concat']
                except:
                    pass
            return cond

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            if timestep[0].item() < sigma_end:
                #if not layer_module.is_model_loaded(original_unet):
                #    sampling_prepare(original_unet, x)
                target_model = original_unet.model
                cond = remove_concat(cond)
                uncond = remove_concat(uncond)
            else:
                target_model = model

            return target_model, x, timestep, uncond, cond, cond_scale, model_options, seed
        
        unet.add_conditioning_modifier(conditioning_modifier)

        target_unet = unet








    def make_circular_asymm(model, tileX: bool, tileY: bool):
        
        for layer in [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            layer._conv_forward = __replacementConv2DConvForward.__get__(layer, Conv2d)
        return model
    def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        working = F.pad(input, self.paddingX, mode=self.padding_modeX)
        working = F.pad(working, self.paddingY, mode=self.padding_modeY)
        return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
    
    if tile_x or tile_y:
        make_circular_asymm(target_unet.model, tile_x, tile_y)
        make_circular_asymm(target_vae.first_stage_model, tile_x, tile_y)

    if refiner_swap_method == 'joint':
        sampled_latent = core.ksampler(
            model=target_unet,
            refiner=target_refiner_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            refiner_switch=switch,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview
        )
        decoded_latent = core.decode_vae(vae=target_vae, latent_image=sampled_latent, tiled=tiled)

    if refiner_swap_method == 'separate':
        sampled_latent = core.ksampler(
            model=target_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=False,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview
        )
        print('Refiner swapped by changing ksampler. Noise preserved.')

        target_model = target_refiner_unet
        if target_model is None:
            target_model = target_unet
            print('Use base model to refine itself - this may because of developer mode.')

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
            latent=sampled_latent,
            steps=steps, start_step=switch, last_step=steps, disable_noise=True, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
            disable_preview=disable_preview
        )

        target_model = target_refiner_vae
        if target_model is None:
            target_model = target_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)

    if refiner_swap_method == 'vae':
        modules.patch.patch_settings[os.getpid()].eps_record = 'vae'

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.unswap()

        sampled_latent = core.ksampler(
            model=target_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview
        )
        print('Fooocus VAE-based swap.')

        target_model = target_refiner_unet
        if target_model is None:
            target_model = target_unet
            print('Use base model to refine itself - this may because of developer mode.')

        sampled_latent = vae_parse(sampled_latent)

        k_sigmas = 1.4
        sigmas = calculate_sigmas(sampler=sampler_name,
                                  scheduler=scheduler_name,
                                  model=target_model.model,
                                  steps=steps,
                                  denoise=denoise)[switch:] * k_sigmas
        len_sigmas = len(sigmas) - 1

        noise_mean = torch.mean(modules.patch.patch_settings[os.getpid()].eps_record, dim=1, keepdim=True)

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.swap()

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
            latent=sampled_latent,
            steps=len_sigmas, start_step=0, last_step=len_sigmas, disable_noise=False, force_full_denoise=True,
            seed=image_seed+1,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
            sigmas=sigmas,
            noise_mean=noise_mean,
            disable_preview=disable_preview
        )

        target_model = target_refiner_vae
        if target_model is None:
            target_model = target_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)

    images = core.pytorch_to_numpy(decoded_latent)
    modules.patch.patch_settings[os.getpid()].eps_record = None
    if tile_x or tile_y:
        for layer in [l for l in target_unet.model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, Conv2d)
        for layer in [l for l in target_vae.first_stage_model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, Conv2d)
    


    if len(layer_diff) > 1:
        mod_number = 1
        method = LayerMethod(method)
        need_process = False

        if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
                need_process = True
            #if vae_transparent_decoder is None:
                model_path = load_file_from_url(
                    url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='vae_transparent_decoder.safetensors'
                )
                vae_transparent_decoder = TransparentVAEDecoder(ldm_patched.modules.utils.load_torch_file(vae_decoder))
                #vae_transparent_decoder = TransparentVAEDecoder(utils.load_torch_file(model_path))


        if need_process:

            #i = pp.index

            #if i % mod_number == 0:
            latent = sampled_latent['samples'][0]
            pixel = images[0]

            lC, lH, lW = latent.shape
            if lH != pixel.height // 8 or lW != pixel.width // 8:
                    print('[LayerDiffuse] VAE zero latent mode.')
                    latent = torch.zeros((lC, pixel.height // 8, pixel.width // 8)).to(latent)

            png, vis = vae_transparent_decoder.decode(latent, pixel)
            images.append(png)
            images.append(vis)
            target_unet = original_unet
    return images
