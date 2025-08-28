from collections import namedtuple
import copy
from itertools import permutations, chain
from logging import info
import random
import csv
import os.path
from io import StringIO
from PIL import Image
import numpy as np
import cv2

###import modules.scripts as scripts
import gradio as gr
###from modules import images, sd_samplers, processing, sd_models, sd_vae, sd_schedulers, errors
###from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
###from modules.shared import opts, state
###from modules.sd_models import model_data, select_checkpoint
###import modules.shared as shared
###import modules.sd_samplers
###import modules.sd_models
###import modules.sd_vae
import re

##from modules.ui_components import ToolButton
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.config
import modules.flags
from modules.util import get_enabled_loras
from modules.sdxl_styles import legal_style_names
from modules.private_logger import log
import html

fill_values_symbol = "\U0001f4d2"  # рџ“’

AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun

def apply_bool(field):
    def fun(p, x, xs):
        if x.lower=="enable":
            setattr(p, field, True)
        else:
            setattr(p, field, False)

    return fun

def apply_lora(field):
    def set_lora(p,x,xs):
        p.lora_list[field] = (p.lora_list[field][0], p.lora_list[field][1], x)
        p.loras = get_enabled_loras(p.lora_list)
    return set_lora


def apply_lora_name(field):
    def set_lora(p,x,xs):
        p.lora_list[field] = (p.lora_list[field][0], x,p.lora_list[field][2])
        p.loras = get_enabled_loras(p.lora_list)
    return set_lora



def apply_ctrlnet(field1,field2):
    def set_ctrl(p,x,sx):
      p.cn_tasks[field1][0][field2]=x
    return set_ctrl
def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)

def apply_order(p, x, xs):
    token_order = []

    # Initially grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def confirm_samplers(p, xs):
    for x in xs:
        p.sampler_name_name=x


def apply_checkpoint(p, x, xs):
    """
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    # skip if the checkpoint was last override
    if info.name == p.override_settings.get('sd_model_checkpoint', None):
        return
    org_cp = getattr(opts, 'sd_model_checkpoint', None)
    p.override_settings['sd_model_checkpoint'] = info.name
    opts.set('sd_model_checkpoint', info.name)
    refresh_loading_params_for_xyz_grid()
    # This saves part of the reload
    opts.set('sd_model_checkpoint', org_cp)
    """

    for x in xs:
        p.base_model_name=x

def refresh_loading_params_for_xyz_grid():
    """
    Refreshes the loading parameters for the model, 
    prompts a reload in sd_models.forge_model_reload()
    """
    checkpoint_info = select_checkpoint()

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        #unet_storage_dtype=shared.opts.forge_unet_storage_dtype
        unet_storage_dtype=model_data.forge_loading_parameters.get('unet_storage_dtype', None)
    )


def confirm_checkpoints(p, xs):
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def confirm_checkpoints_or_none(p, xs):
    for x in xs:
        if x in (None, "", "None", "none"):
            continue

        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def confirm_range(min_val, max_val, axis_label):
    """Generates a AxisOption.confirm() function that checks all values are within the specified range."""

    def confirm_range_fun(p, xs):
        for x in xs:
            if not (max_val >= x >= min_val):
                raise ValueError(f'{axis_label} value "{x}" out of range [{min_val}, {max_val}]')

    return confirm_range_fun


def apply_size(p, x: str, xs) -> None:
    try:
        width, _, height = x.partition('x')
        width = int(width.strip())
        height = int(height.strip())
        p.width = width
        p.height = height
    except ValueError:
        print(f"Invalid size in XYZ plot: {x}")


def find_vae(name: str):
    if (name := name.strip().lower()) in ('auto', 'automatic'):
        return 'Automatic'
    elif name == 'none':
        return 'None'
    return next((k for k in modules.sd_vae.vae_dict if k.lower() == name), print(f'No VAE found for {name}; using Automatic') or 'Automatic')


def apply_vae(p, x, xs):
    p.override_settings['sd_vae'] = find_vae(x)


def apply_styles(p, x: str, _):
    p.style_selections.extend(x.split(','))


def apply_uni_pc_order(p, x, xs):
    p.override_settings['uni_pc_order'] = min(x, p.steps - 1)


def apply_face_restore(p, opt, x):
    opt = opt.lower()
    if opt == 'codeformer':
        is_active = True
        p.face_restoration_model = 'CodeFormer'
    elif opt == 'gfpgan':
        is_active = True
        p.face_restoration_model = 'GFPGAN'
    else:
        is_active = opt in ('true', 'yes', 'y', '1')

    p.restore_faces = is_active


def apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        p.override_settings[field] = x

    return fun


def boolean_choice(reverse: bool = False):
    def choice():
        return ["False", "True"] if reverse else ["True", "False"]

    return choice


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def format_remove_path(p, opt, x):
    return os.path.basename(x)


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


def list_to_csv_string(data_list):
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()


def csv_string_to_list_strip(data_str):
    return list(map(str.strip, chain.from_iterable(csv.reader(StringIO(data_str), skipinitialspace=True))))


class AxisOption:
    def __init__(self, label, type, apply, format_value=format_value_add_label, confirm=None, cost=0.0, choices=None, prepare=None):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value
        self.confirm = confirm
        self.cost = cost
        self.prepare = prepare
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True



class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False

axis_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
    AxisOption("Prompt order", str_permutations, apply_order, format_value=format_value_join_list),	
	AxisOption("Styles", str, apply_styles, choices=lambda: list(legal_style_names)),
	AxisOption("Steps", int, apply_field("steps")),
#	AxisOption("Aspect Ratio", str, apply_size),
	AxisOption("Seed", int, apply_field("seed")),
	AxisOption("Sharpness", int, apply_field("sharpness")),
	AxisOption("CFG (Guidance) Scale", float, apply_field("cfg_scale")),
	AxisOption("Checkpoint name", str, apply_field('base_model_name'), format_value=format_remove_path, confirm=None, cost=1.0, choices=lambda: sorted(modules.config.model_filenames, key=str.casefold)),
    AxisOption("LoRA 1 name", str, apply_lora_name(0), format_value=format_remove_path, confirm=None, cost=0.6, choices=lambda: ['None'] + sorted(modules.config.lora_filenames, key=str.casefold)),
	AxisOption("LoRA 1 weight", float, apply_lora(0), cost=0.6),
    AxisOption("LoRA 2 name", str, apply_lora_name(1), format_value=format_remove_path, confirm=None, cost=0.6, choices=lambda: ['None'] + sorted(modules.config.lora_filenames, key=str.casefold)),
    AxisOption("LoRA 2 weight", float, apply_lora(1), cost=0.6),
    AxisOption("LoRA 3 name", str, apply_lora_name(2), format_value=format_remove_path, confirm=None, cost=0.6, choices=lambda: ['None'] + sorted(modules.config.lora_filenames, key=str.casefold)),
    AxisOption("LoRA 3 weight", float, apply_lora(2), cost=0.6),
    AxisOption("LoRA 4 name", str, apply_lora_name(3), format_value=format_remove_path, confirm=None, cost=0.6, choices=lambda: ['None'] + sorted(modules.config.lora_filenames, key=str.casefold)),
    AxisOption("LoRA 4 weight", float, apply_lora(3), cost=0.6),
    AxisOption("LoRA 5 name", str, apply_lora_name(4), format_value=format_remove_path, confirm=None, cost=0.6, choices=lambda: ['None'] + sorted(modules.config.lora_filenames, key=str.casefold)),
    AxisOption("LoRA 5 weight", float, apply_lora(4), cost=0.6),

#	  AxisOption("Refiner checkpoint", str, apply_field('refiner_model_name'), format_value=format_remove_path, confirm=None, cost=1.0, choices=lambda: ['None'] + sorted(modules.config.model_filenames, key=str.casefold)),
#	  AxisOption("Refiner switch at", float, apply_field('refiner_switch_at')),
	AxisOption("Clip skip", int, apply_field('clip_skip')),
	AxisOption("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: sorted(modules.flags.sampler_list, key=str.casefold)),
	AxisOption("Scheduler", str, apply_field("scheduler_name"), choices=lambda: sorted(modules.flags.scheduler_list, key=str.casefold)),
	AxisOption("VAE", str, apply_field("vae_name"), cost=0.7, choices=lambda: ['Default (model)'] + list(modules.config.vae_filenames)),
#	  AxisOption("Refiner swap method", str, apply_field("refiner_swap_method"), format_value=format_value, choices=lambda: sorted(['joint', 'separate', 'vae'], key=str.casefold))
	AxisOption("Softness of ControlNet", float, apply_field("controlnet_softness")),
    AxisOption("ImagePrompt stop at", float, apply_ctrlnet("ImagePrompt",1)),
    AxisOption("ImagePrompt weight", float, apply_ctrlnet("ImagePrompt",2)),
    AxisOption("PyraCanny stop at", float, apply_ctrlnet("PyraCanny",1)),
    AxisOption("PyraCanny weight", float, apply_ctrlnet("PyraCanny",2)),
    AxisOption("CPDS stop at", float, apply_ctrlnet("CPDS",1)),
    AxisOption("CPDS weight", float, apply_ctrlnet("CPDS",2)),
    AxisOption("FaceSwap stop at", float, apply_ctrlnet("FaceSwap",1)),
    AxisOption("FaceSwap weight", float, apply_ctrlnet("FaceSwap",2)),
    AxisOption("OpenPose stop at", float, apply_ctrlnet("OpenPose",1)),
    AxisOption("OpenPose weight", float, apply_ctrlnet("OpenPose",2)),
    AxisOption("Recolor stop at", float, apply_ctrlnet("Recolor",1)),
    AxisOption("Recolor weight", float, apply_ctrlnet("Recolor",2)),
    AxisOption("Scribble stop at", float, apply_ctrlnet("Scribble",1)),
    AxisOption("Scribble weight", float, apply_ctrlnet("Scribble",2)),
    AxisOption("Manga stop at", float, apply_ctrlnet("Manga",1)),
    AxisOption("Manga weight", float, apply_ctrlnet("Manga",2)),
    AxisOption("Codeformer", str, apply_bool('codeformer_gen_enabled'), choices=lambda: ['Enable','Disable'])
]

def draw_grid(x_labels,y_labels,z_labels,list_size,ix,iy,iz,xs,ys,zs,currentTask,xyz_results):
    
    results = []
    for img in currentTask.results:
        if isinstance(img, str) and os.path.exists(img):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not isinstance(img, np.ndarray):
            return
        if img.ndim != 3:
            return
        results.append(img)

    H, W, C = results[0].shape

    for img in results:
        Hn, Wn, Cn = img.shape
        if H != Hn:
            return
        if W != Wn:
            return
        if C != Cn:
            return
    x_coord=len(xs)
    y_coord=len(ys)
    z_coord=len(zs)
    for z in range(z_coord):
        if currentTask.grid_theme:
            grid_color=(255,255,255)
            wall = np.ones(shape=((H+currentTask.margin_size) * y_coord, (W+currentTask.margin_size) * x_coord, C), dtype=np.uint8)*255
            text_color=(0,0,0)
        else:
            grid_color=(0,0,0)
            wall = np.zeros(shape=((H+currentTask.margin_size) * y_coord, (W+currentTask.margin_size) * x_coord, C), dtype=np.uint8)
            text_color=(255,255,255)
        for y in range(y_coord):
            for x in range(x_coord):
                index_list=[x,y,z]
                index = xyz_results.index(index_list)
                img = results[index]
                wall[y * (H + currentTask.margin_size):y * (H + currentTask.margin_size) + H, x * (W + currentTask.margin_size):x * (W + currentTask.margin_size) + W, :] = img
        hor_text = [x.replace(".safetensor", "") for x in x_labels]
        vert_text = [y.replace(".safetensor", "") for y in y_labels]
        title_text = [z.replace(".safetensor", "") for z in z_labels]
        
        if currentTask.draw_legend:
                font=cv2.FONT_HERSHEY_COMPLEX
                font_scale=1
                thickness=3

                if hor_text[0]:
                  extend_h=wall.shape[0] + 100
                  new_shape = (extend_h, wall.shape[1], wall.shape[2])
                  image_extended = np.full(new_shape, grid_color, dtype=wall.dtype)
                  image_extended[:wall.shape[0], :] = wall
                  for i in range(len(hor_text)):
                      cv2.putText(image_extended, hor_text[i], (i*(W+currentTask.margin_size),wall.shape[0]+50), font,  font_scale, text_color, thickness)
                  wall=image_extended
                if vert_text[0]:
                  y_text_max = max(vert_text, key=len)
                  (y_text_width, y_text_height), _ = cv2.getTextSize(y_text_max, font, font_scale, thickness)
                  extend_width=wall.shape[1] + y_text_width + 100
                  new_shape = (wall.shape[0], extend_width, wall.shape[2])
                  image_extended = np.full(new_shape, grid_color, dtype=wall.dtype)
                  image_extended[:, y_text_width+100:] = wall
                  for i in range(len(vert_text)):
                    cv2.putText(image_extended, vert_text[i], (50,int((H+currentTask.margin_size) * i + ((H+currentTask.margin_size)/2))), font, font_scale, text_color, thickness)
                  wall=image_extended

                if title_text[z]:
                  extend_h=wall.shape[0] + 100
                  new_shape = (extend_h, wall.shape[1], wall.shape[2])
                  image_extended = np.full(new_shape, grid_color, dtype=wall.dtype)
                  image_extended[100:, :] = wall
                  (title_text_width, title_text_height), _ = cv2.getTextSize(title_text[z], font, font_scale, thickness)
                  cv2.putText(image_extended, title_text[z], (int((image_extended.shape[1]-title_text_width)/2),20+title_text_height), font, font_scale, text_color, thickness)
                  wall=image_extended
        meta_xyz=[('Base prompt','prompt',html.escape(currentTask.args[1], quote=True))]
        if hor_text[0]:
            for i in range(len(hor_text)):
              meta_xyz.append((f'X axis {i+1}:', f'X axis {i+1}:', html.escape(hor_text[i], quote=True)))
        if vert_text[0]:
            for i in range(len(vert_text)):
              meta_xyz.append((f'Y axis {i+1}:', f'Y axis {i+1}:', html.escape(vert_text[i], quote=True)))
        if title_text[z]:
            meta_xyz.append((f'Z axis:', 'Z axis', html.escape(title_text[z], quote=True)))

        log(wall, metadata=meta_xyz, metadata_parser=None, output_format=None, task=None, persist_image=True,name_prefix=currentTask.name_prefix)

re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*])?\s*")


###class Script():
def title():
    return "X/Y/Z plot"

def ui():
    current_axis_options = [x for x in axis_options if type(x) == AxisOption]

    with gr.Row():
        with gr.Column(scale=19):
            with gr.Row():
                x_type = gr.Dropdown(scale=4,label="X type", choices=[x.label for x in current_axis_options], value=current_axis_options[0].label, type="index", elem_id="x_type")
                x_values = gr.Textbox(scale=4,label="X values", lines=1, elem_id="x_values")
                x_values_dropdown = gr.Dropdown(scale=4,label="X values", visible=False, multiselect=True, interactive=True)
                fill_x_button = gr.Button(scale=1,value=fill_values_symbol, elem_id="xyz_grid_fill_x_tool_button", visible=False)

            with gr.Row():
                y_type = gr.Dropdown(scale=4,label="Y type", choices=[x.label for x in current_axis_options], value=current_axis_options[0].label, type="index", elem_id="y_type")
                y_values = gr.Textbox(scale=4,label="Y values", lines=1, elem_id="y_values")
                y_values_dropdown = gr.Dropdown(scale=4,label="Y values", visible=False, multiselect=True, interactive=True)
                fill_y_button = gr.Button(scale=1,value=fill_values_symbol, elem_id="xyz_grid_fill_y_tool_button", visible=False)

            with gr.Row():
                z_type = gr.Dropdown(scale=4,label="Z type", choices=[x.label for x in current_axis_options], value=current_axis_options[0].label, type="index", elem_id="z_type")
                z_values = gr.Textbox(scale=4,label="Z values", lines=1, elem_id="z_values")
                z_values_dropdown = gr.Dropdown(scale=4,label="Z values", visible=False, multiselect=True, interactive=True)
                fill_z_button = gr.Button(scale=1,value=fill_values_symbol, elem_id="xyz_grid_fill_z_tool_button", visible=False)

    with gr.Row(variant="compact", elem_id="axis_options"):
        with gr.Column():
            draw_legend = gr.Checkbox(label='Draw legend', value=True, elem_id="draw_legend")
            always_random = gr.Checkbox(label='Always random seed', value=False, elem_id="always_random")
            no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False, elem_id="no_fixed_seeds", visible=False)
            with gr.Row(visible=False):
                vary_seeds_x = gr.Checkbox(label='Vary seeds for X', value=False, min_width=80, elem_id="vary_seeds_x", info="Use different seeds for images along X axis.")
                vary_seeds_y = gr.Checkbox(label='Vary seeds for Y', value=False, min_width=80, elem_id="vary_seeds_y", info="Use different seeds for images along Y axis.")
                vary_seeds_z = gr.Checkbox(label='Vary seeds for Z', value=False, min_width=80, elem_id="vary_seeds_z", info="Use different seeds for images along Z axis.")
        with gr.Column(visible=False):
            include_lone_images = gr.Checkbox(label='Include Sub Images', value=False, elem_id="include_lone_images")
            include_sub_grids = gr.Checkbox(label='Include Sub Grids', value=False, elem_id="include_sub_grids")
            csv_mode = gr.Checkbox(label='Use text inputs instead of dropdowns', value=False, elem_id="csv_mode")
        with gr.Column():
            margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=500, value=0, step=2, elem_id="margin_size")
            grid_theme= gr.Checkbox(label='White theme of grid', value=False)
    with gr.Row(variant="compact", elem_id="swap_axes"):
        swap_xy_axes_button = gr.Button(value="Swap X/Y axes", elem_id="xy_grid_swap_axes_button")
        swap_yz_axes_button = gr.Button(value="Swap Y/Z axes", elem_id="yz_grid_swap_axes_button")
        swap_xz_axes_button = gr.Button(value="Swap X/Z axes", elem_id="xz_grid_swap_axes_button")

    def swap_axes(axis1_type, axis1_values, axis1_values_dropdown, axis2_type, axis2_values, axis2_values_dropdown):
        return current_axis_options[axis2_type].label, axis2_values, axis2_values_dropdown, current_axis_options[axis1_type].label, axis1_values, axis1_values_dropdown

    xy_swap_args = [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown]
    swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)
    yz_swap_args = [y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown]
    swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)
    xz_swap_args = [x_type, x_values, x_values_dropdown, z_type, z_values, z_values_dropdown]
    swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

    def fill(axis_type, csv_mode):
        axis = current_axis_options[axis_type]
        if axis.choices:
            if csv_mode:
                return list_to_csv_string(axis.choices()), gr.update()
            else:
                return gr.update(), axis.choices()
        else:
            return gr.update(), gr.update()

    fill_x_button.click(fn=fill, inputs=[x_type, csv_mode], outputs=[x_values, x_values_dropdown])
    fill_y_button.click(fn=fill, inputs=[y_type, csv_mode], outputs=[y_values, y_values_dropdown])
    fill_z_button.click(fn=fill, inputs=[z_type, csv_mode], outputs=[z_values, z_values_dropdown])

    def select_axis(axis_type, axis_values, axis_values_dropdown, csv_mode):
        axis_type = axis_type or 0  # if axle type is None set to 0

        choices = current_axis_options[axis_type].choices
        has_choices = choices is not None

        if has_choices:
            choices = choices()
            if csv_mode:
                if axis_values_dropdown:
                    axis_values = list_to_csv_string(list(filter(lambda x: x in choices, axis_values_dropdown)))
                    axis_values_dropdown = []
            else:
                if axis_values:
                   axis_values_dropdown = list(filter(lambda x: x in choices, csv_string_to_list_strip(axis_values)))
                   axis_values = ""
        return (gr.Button.update(visible=has_choices), gr.Textbox.update(visible=not has_choices or csv_mode, value=axis_values),
                gr.update(choices=choices if has_choices else None, visible=has_choices and not csv_mode, value=axis_values_dropdown))

    x_type.change(fn=select_axis, inputs=[x_type, x_values, x_values_dropdown, csv_mode], outputs=[fill_x_button, x_values, x_values_dropdown])
    y_type.change(fn=select_axis, inputs=[y_type, y_values, y_values_dropdown, csv_mode], outputs=[fill_y_button, y_values, y_values_dropdown])
    z_type.change(fn=select_axis, inputs=[z_type, z_values, z_values_dropdown, csv_mode], outputs=[fill_z_button, z_values, z_values_dropdown])

    def change_choice_mode(csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown):
        _fill_x_button, _x_values, _x_values_dropdown = select_axis(x_type, x_values, x_values_dropdown, csv_mode)
        _fill_y_button, _y_values, _y_values_dropdown = select_axis(y_type, y_values, y_values_dropdown, csv_mode)
        _fill_z_button, _z_values, _z_values_dropdown = select_axis(z_type, z_values, z_values_dropdown, csv_mode)
        return _fill_x_button, _x_values, _x_values_dropdown, _fill_y_button, _y_values, _y_values_dropdown, _fill_z_button, _z_values, _z_values_dropdown

    csv_mode.change(fn=change_choice_mode, inputs=[csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown], outputs=[fill_x_button, x_values, x_values_dropdown, fill_y_button, y_values, y_values_dropdown, fill_z_button, z_values, z_values_dropdown])

    def get_dropdown_update_from_params(axis, params):
        val_key = f"{axis} Values"
        vals = params.get(val_key, "")
        valslist = csv_string_to_list_strip(vals)
        return gr.update(value=valslist)

    infotext_fields = (
        (x_type, "X Type"),
        (x_values, "X Values"),
        (x_values_dropdown, lambda params: get_dropdown_update_from_params("X", params)),
        (y_type, "Y Type"),
        (y_values, "Y Values"),
        (y_values_dropdown, lambda params: get_dropdown_update_from_params("Y", params)),
        (z_type, "Z Type"),
        (z_values, "Z Values"),
        (z_values_dropdown, lambda params: get_dropdown_update_from_params("Z", params)),
    )

    return [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode,grid_theme,always_random]

def run(p):
    grid_theme = p.grid_theme
    csv_mode = p.csv_mode
    margin_size = p.margin_size
    vary_seeds_z = p.vary_seeds_z
    vary_seeds_y = p.vary_seeds_y
    vary_seeds_x = p.vary_seeds_x
    no_fixed_seeds = p.no_fixed_seeds
    include_sub_grids = p.include_sub_grids
    include_lone_images = p.include_lone_images
    draw_legend = p.draw_legend
    z_values_dropdown = p.z_values_dropdown
    z_values = p.z_values
    z_type = p.z_type
    y_values_dropdown = p.y_values_dropdown
    y_values = p.y_values
    y_type = p.y_type
    x_values_dropdown = p.x_values_dropdown
    x_values = p.x_values
    x_type = p.x_type
    
    x_type, y_type, z_type = x_type or 0, y_type or 0, z_type or 0  # if axle type is None set to 0
    current_axis_options = [x for x in axis_options if type(x) == AxisOption]
    def process_axis(opt, vals, vals_dropdown):
        if opt.label == 'Nothing':
            return [0]
        if opt.choices is not None and not csv_mode:
            valslist = vals_dropdown
        elif opt.prepare is not None:
            valslist = opt.prepare(vals)
        else:
            valslist = csv_string_to_list_strip(vals)

        if opt.type == int:
            valslist_ext = []

            for val in valslist:
                if val.strip() == '':
                    continue
                m = re_range.fullmatch(val)
                mc = re_range_count.fullmatch(val)
                if m is not None:
                    start = int(m.group(1))
                    end = int(m.group(2)) + 1
                    step = int(m.group(3)) if m.group(3) is not None else 1

                    valslist_ext += list(range(start, end, step))
                elif mc is not None:
                    start = int(mc.group(1))
                    end = int(mc.group(2))
                    num = int(mc.group(3)) if mc.group(3) is not None else 1

                    valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                else:
                    valslist_ext.append(val)

            valslist = valslist_ext
        elif opt.type == float:
            valslist_ext = []

            for val in valslist:
                if val.strip() == '':
                    continue
                m = re_range_float.fullmatch(val)
                mc = re_range_count_float.fullmatch(val)
                if m is not None:
                    start = float(m.group(1))
                    end = float(m.group(2))
                    step = float(m.group(3)) if m.group(3) is not None else 1

                    valslist_ext += np.arange(start, end + step, step).tolist()
                elif mc is not None:
                    start = float(mc.group(1))
                    end = float(mc.group(2))
                    num = int(mc.group(3)) if mc.group(3) is not None else 1

                    valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                else:
                    valslist_ext.append(val)

            valslist = valslist_ext
        elif opt.type == str_permutations:
            valslist = list(permutations(valslist))

        valslist = [opt.type(x) for x in valslist]

            # Confirm options are valid before starting
        if opt.confirm:
            opt.confirm(p, valslist)

        return valslist

    x_opt = current_axis_options[x_type]
    if x_opt.choices is not None and not csv_mode:
        x_values = list_to_csv_string(x_values_dropdown)
    xs = process_axis(x_opt, x_values, x_values_dropdown)


    y_opt = current_axis_options[y_type]
    if y_opt.choices is not None and not csv_mode:
        y_values = list_to_csv_string(y_values_dropdown)
    ys = process_axis(y_opt, y_values, y_values_dropdown)

    z_opt = current_axis_options[z_type]
    if z_opt.choices is not None and not csv_mode:
        z_values = list_to_csv_string(z_values_dropdown)
    zs = process_axis(z_opt, z_values, z_values_dropdown)

    def fix_axis_seeds(axis_opt, axis_list):
        if axis_opt.label in ['Seed', 'Var. seed']:
            return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
        else:
            return axis_list

    if x_opt.label == 'Steps':
        total_steps = sum(xs) * len(ys) * len(zs)
    elif y_opt.label == 'Steps':
        total_steps = sum(ys) * len(xs) * len(zs)
    elif z_opt.label == 'Steps':
        total_steps = sum(zs) * len(xs) * len(ys)
    else:
        total_steps = p.steps * len(xs) * len(ys) * len(zs)

    image_cell_count = 1
	
    cell_console_text = f"; {image_cell_count} images per cell" if image_cell_count > 1 else ""
    plural_s = 's' if len(zs) > 1 else ''
    print(f"X/Y/Z plot will create {len(xs) * len(ys) * len(zs) * image_cell_count} images on {len(zs)} {len(xs)}x{len(ys)} grid{plural_s}{cell_console_text}. (Total steps to process: {total_steps})")


    xyz_plot_x = AxisInfo(x_opt, xs)

    xyz_plot_y = AxisInfo(y_opt, ys)

    xyz_plot_z = AxisInfo(z_opt, zs)

        # If one of the axes is very slow to change between (like SD model
        # checkpoint), then make sure it is in the outer iteration of the nested
        # `for` loop.
    first_axes_processed = 'z'
    second_axes_processed = 'y'
    if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
        first_axes_processed = 'x'
        if y_opt.cost > z_opt.cost:
            second_axes_processed = 'y'
        else:
            second_axes_processed = 'z'
    elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
        first_axes_processed = 'y'
        if x_opt.cost > z_opt.cost:
            second_axes_processed = 'x'
        else:
            second_axes_processed = 'z'
    elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
        first_axes_processed = 'z'
        if x_opt.cost > y_opt.cost:
            second_axes_processed = 'x'
        else:
            second_axes_processed = 'y'
    list_size = (len(xs) * len(ys) * len(zs))
 
    def cell(x, y, z, ix, iy, iz,xyz_task,xyz_results):

        pc = copy.deepcopy(p)
        x_opt.apply(pc, x, xs)
        y_opt.apply(pc, y, ys)
        z_opt.apply(pc, z, zs)        
        new_copy = copy.deepcopy(pc)
        xyz_task.append(new_copy)
        cell_list=[ix,iy,iz]
        xyz_results.append(cell_list)
        return xyz_task,xyz_results
    xyz_task=[]
    xyz_results=[]
    grid_infotext = [None] * (1 + len(zs))
    if first_axes_processed == 'x':
        for ix, x in enumerate(xs):
            if second_axes_processed == 'y':
                for iy, y in enumerate(ys):
                    for iz, z in enumerate(zs):
                        cell(x, y, z, ix, iy, iz,xyz_task,xyz_results)
            else:
                for iz, z in enumerate(zs):
                    for iy, y in enumerate(ys):
                        cell(x, y, z, ix, iy, iz,xyz_task,xyz_results)
    elif first_axes_processed == 'y':
        for iy, y in enumerate(ys):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iz, z in enumerate(zs):
                        cell(x, y, z, ix, iy, iz,xyz_task,xyz_results)
            else:
                for iz, z in enumerate(zs):
                    for ix, x in enumerate(xs):
                        cell(x, y, z, ix, iy, iz,xyz_task,xyz_results)
    elif first_axes_processed == 'z':
        for iz, z in enumerate(zs):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        cell(x, y, z, ix, iy, iz,xyz_task,xyz_results)
            else:
                for iy, y in enumerate(ys):
                    for ix, x in enumerate(xs):
                        cell(x, y, z, ix, iy, iz,xyz_task,xyz_results)
    x_labels=[x_opt.format_value(p, x_opt, x) for x in xs]
    y_labels=[y_opt.format_value(p, y_opt, y) for y in ys]
    z_labels=[z_opt.format_value(p, z_opt, z) for z in zs]
    return xyz_results,xyz_task,x_labels,y_labels,z_labels,list_size,ix,iy,iz,xs,ys,zs
