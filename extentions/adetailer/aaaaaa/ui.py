from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import chain
from types import SimpleNamespace
from typing import Any
from extentions.tutorial import tutorial
import gradio as gr
import modules

from extentions.adetailer.adetailer.args import ALL_ARGS, MASK_MERGE_INVERT



class Widgets(SimpleNamespace):
    def tolist(self):
        return [getattr(self, attr) for attr in ALL_ARGS.attrs]


@dataclass
class WebuiInfo:
    ad_model_list: list[str]
    sampler_names: list[str]
    scheduler_names: list[str]
    #!t2i_button: gr.Button
    #!i2i_button: gr.Button
    checkpoints_list: list[str]
    vae_list: list[str]


def gr_interactive(value: bool = True):
    return gr.update(interactive=value)


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


def tab_default(n: int, attr: str, stock: Any, choices: list | None = None) -> Any:
    # value of the n-th tab from the config/preset key default_adetailer_tabs
    tabs = modules.config.default_adetailer_tabs
    value = tabs[n].get(attr) if n < len(tabs) else None
    return stock if value is None or choices is not None and value not in choices else value


def on_widget_change(state: dict, value: Any, *, attr: str):
    if "is_api" in state:
        state = state.copy()
        state.pop("is_api")
    state[attr] = value
    return state


def on_generate_click(state: dict, *values: Any):
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value  # noqa: PERF403
    state["is_api"] = ()
    return state


def on_ad_model_update(model: str):
    if "-world" in model:
        return gr.update(
            visible=True,
            placeholder="Comma separated class names to detect, ex: 'person,cat'. default: COCO 80 classes",
        )
    return gr.update(visible=False, placeholder="")


def on_cn_model_update(cn_model_name: str):
    cn_model_name = cn_model_name.replace("inpaint_depth", "depth")
    for t in cn_module_choices:
        if t in cn_model_name:
            choices = cn_module_choices[t]
            return gr.update(visible=True, choices=choices, value=choices[0])
    return gr.update(visible=False, choices=["None"], value="None")


def elem_id(item_id: str, n: int, is_img2img: bool) -> str:
    tab = "img2img" if is_img2img else "txt2img"
    suf = suffix(n, "_")
    return f"script_{tab}_adetailer_{item_id}{suf}"


def state_init(w: Widgets) -> dict[str, Any]:
    return {attr: getattr(w, attr).value for attr in ALL_ARGS.attrs}


def adui(
    num_models: int,
    is_img2img: bool,
    webui_info: WebuiInfo,
):
        states = []

        ad_model_dropdowns = []
        ad_ckpt_dropdown =[]
        ad_widgets = []

        eid = partial(elem_id, n=0, is_img2img=is_img2img)
        with gr.Group():
            with gr.Row():
                tutorial("adetailer/readme.md")
            with gr.Row():
                only_detect = gr.Checkbox(
                    label=f"Only detect",
                    value=False,
                    visible=True),

        with gr.Group():
            with gr.Tabs():
                with gr.Tab(label='Upscale or Variation'):
                    with gr.Row():
                        with gr.Column():
                            adetail_uov_method = gr.Radio(label='Upscale or Variation:', choices=modules.flags.uov_list,
                                                              value=modules.config.default_adetail_uov_method)
                            adetail_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                        info='Use before to enhance small details and after to enhance large areas.',
                                                                        choices=modules.flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_adetail_uov_processing_order)
                            adetail_uov_prompt_type = gr.Radio(label='Prompt',
                                                                   info='Choose which prompt to use for Upscale or Variation.',
                                                                   choices=modules.flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_adetail_uov_prompt_type,
                                                                   visible=modules.config.default_adetail_uov_processing_order == modules.flags.enhancement_uov_after)

                            adetail_uov_processing_order.change(lambda x: gr.update(visible=x == modules.flags.enhancement_uov_after),
                                                                    inputs=adetail_uov_processing_order,
                                                                    outputs=adetail_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                for n in range(num_models):
                    with gr.Tab(ordinal(n + 1)):
                        state,ad_model_dd, ad_ckpt_dd, widgets = one_ui_group(
                            n=n,
                            is_img2img=is_img2img,
                            webui_info=webui_info,
                        )
                    ad_model_dropdowns.append(ad_model_dd)
                    ad_ckpt_dropdown.append(ad_ckpt_dd)
                    ad_widgets.append(widgets)
                    states.append(state)
                    #!infotext_fields.extend(infofields)

    # components: [bool, bool, dict, dict, ...]
        components = [*states]
        with gr.Row():
            gr.HTML('* \"ADetailer\" is powered by Bing-su. <a href="https://github.com/Bing-su/adetailer" target="_blank">\U0001F4D4 Document</a>')
        return  adetail_uov_method, adetail_uov_processing_order, adetail_uov_prompt_type, only_detect, components,ad_model_dropdowns,ad_ckpt_dropdown,ad_widgets


def one_ui_group(n: int, is_img2img: bool, webui_info: WebuiInfo):
    w = Widgets()
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    model_choices = [*webui_info.ad_model_list]

    with gr.Group():
        with gr.Row(variant="compact"):
            w.ad_tab_enable = gr.Checkbox(
                label=f"Enable this tab ({ordinal(n + 1)})",
                value=tab_default(n, "ad_tab_enable", n == 0),
                visible=True,
                elem_id=eid("ad_tab_enable"),
            )

        with gr.Row():
            w.ad_model = gr.Dropdown(
                label="ADetailer detector" + suffix(n),
                choices=model_choices,
                value=tab_default(n, "ad_model", model_choices[0], model_choices),
                visible=True,
                type="value",
                elem_id=eid("ad_model"),
                info="Select a model to use for detection.",
            )

        with gr.Row():
            w.ad_model_classes = gr.Textbox(
                label="ADetailer detector classes" + suffix(n),
                value=tab_default(n, "ad_model_classes", ""),
                visible="-world" in w.ad_model.value,
                elem_id=eid("ad_model_classes"),
            )

            w.ad_model.change(
                on_ad_model_update,
                inputs=w.ad_model,
                outputs=w.ad_model_classes,
                queue=False,
            )

    gr.HTML("<br>")

    with gr.Group():
        with gr.Row(elem_id=eid("ad_toprow_prompt")):
            w.ad_prompt = gr.Textbox(
                value=tab_default(n, "ad_prompt", ""),
                label="ad_prompt" + suffix(n),
                show_label=False,
                lines=3,
                placeholder="ADetailer prompt"
                + suffix(n)
                + "\nIf blank, the main prompt is used.",
                elem_id=eid("ad_prompt"),
            )

        with gr.Row(elem_id=eid("ad_toprow_negative_prompt")):
            w.ad_negative_prompt = gr.Textbox(
                value=tab_default(n, "ad_negative_prompt", ""),
                label="ad_negative_prompt" + suffix(n),
                show_label=False,
                lines=2,
                placeholder="ADetailer negative prompt"
                + suffix(n)
                + "\nIf blank, the main negative prompt is used.",
                elem_id=eid("ad_negative_prompt"),
            )

    with gr.Group():
        with gr.Accordion(
            "Detection", open=False, elem_id=eid("ad_detection_accordion")
        ):
            detection(w, n, is_img2img)

        with gr.Accordion(
            "Mask Preprocessing",
            open=False,
            elem_id=eid("ad_mask_preprocessing_accordion"),
        ):
            mask_preprocessing(w, n, is_img2img)

        with gr.Accordion(
            "Inpainting", open=False, elem_id=eid("ad_inpainting_accordion")
        ):
            pass
            w.ad_checkpoint = inpainting(w, n, is_img2img, webui_info)



    state = gr.State(lambda: state_init(w))

    for attr in ALL_ARGS.attrs:
        widget = getattr(w, attr)
        on_change = partial(on_widget_change, attr=attr)
        widget.change(fn=on_change, inputs=[state, widget], outputs=state, queue=False)

    all_inputs = [state, *w.tolist()]


    return state, w.ad_model, w.ad_checkpoint, {attr: getattr(w, attr) for attr in (*ALL_ARGS.attrs, "inpaint_mode")}


def detection(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Row():
        with gr.Column(variant="compact"):
            w.ad_confidence = gr.Slider(
                label="Detection model confidence threshold" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=tab_default(n, "ad_confidence", 0.3),
                visible=True,
                elem_id=eid("ad_confidence"),
            )
            w.ad_mask_filter_method = gr.Radio(
                choices=["Area", "Confidence"],
                value=tab_default(n, "ad_mask_filter_method", "Area", ["Area", "Confidence"]),
                label="Method to filter top k masks by (confidence or area)"
                + suffix(n),
                visible=True,
                elem_id=eid("ad_mask_filter_method"),
            )
            w.ad_mask_k = gr.Slider(
                label="Mask only the top k (0 to disable)" + suffix(n),
                minimum=0,
                maximum=10,
                step=1,
                value=tab_default(n, "ad_mask_k", 0),
                visible=True,
                elem_id=eid("ad_mask_k"),
            )

        with gr.Column(variant="compact"):
            w.ad_mask_min_ratio = gr.Slider(
                label="Mask min area ratio" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=tab_default(n, "ad_mask_min_ratio", 0.0),
                visible=True,
                elem_id=eid("ad_mask_min_ratio"),
            )
            w.ad_mask_max_ratio = gr.Slider(
                label="Mask max area ratio" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=tab_default(n, "ad_mask_max_ratio", 1.0),
                visible=True,
                elem_id=eid("ad_mask_max_ratio"),
            )


def mask_preprocessing(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Group():
        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_sam_mask = gr.Checkbox(
                    label="SAM mask (face shape instead of box)" + suffix(n),
                    value=tab_default(n, "ad_sam_mask", False),
                    visible=True,
                    elem_id=eid("ad_sam_mask"),
                    info="Refine each detected box with SAM, so the mask follows the object shape.",
                )
            with gr.Column(variant="compact"):
                sam_choices = list(modules.flags.inpaint_mask_sam_model)
                w.ad_sam_model = gr.Dropdown(
                    label="SAM model" + suffix(n),
                    choices=sam_choices,
                    value=tab_default(n, "ad_sam_model", "vit_b", sam_choices),
                    visible=True,
                    interactive=w.ad_sam_mask.value,
                    elem_id=eid("ad_sam_model"),
                    info="vit_b - faster, vit_h - more accurate",
                )
                w.ad_sam_mask.change(
                    lambda value: gr_interactive(value),
                    inputs=w.ad_sam_mask,
                    outputs=w.ad_sam_model,
                    queue=False,
                )

    with gr.Group():
        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_x_offset = gr.Slider(
                    label="Mask x(→) offset" + suffix(n),
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=tab_default(n, "ad_x_offset", 0),
                    visible=True,
                    elem_id=eid("ad_x_offset"),
                )
                w.ad_y_offset = gr.Slider(
                    label="Mask y(↑) offset" + suffix(n),
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=tab_default(n, "ad_y_offset", 0),
                    visible=True,
                    elem_id=eid("ad_y_offset"),
                )

            with gr.Column(variant="compact"):
                w.ad_dilate_erode = gr.Slider(
                    label="Mask erosion (-) / dilation (+)" + suffix(n),
                    minimum=-128,
                    maximum=128,
                    step=4,
                    value=tab_default(n, "ad_dilate_erode", 4),
                    visible=True,
                    elem_id=eid("ad_dilate_erode"),
                )

        with gr.Row():
            w.ad_mask_merge_invert = gr.Radio(
                label="Mask merge mode" + suffix(n),
                choices=MASK_MERGE_INVERT,
                value=tab_default(n, "ad_mask_merge_invert", "None", MASK_MERGE_INVERT),
                elem_id=eid("ad_mask_merge_invert"),
                info="None: do nothing, Merge: merge masks, Merge and Invert: merge all masks and invert",
            )


def inpainting(w: Widgets, n: int, is_img2img: bool, webui_info: WebuiInfo):  # noqa: PLR0915
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    with gr.Group():
        with gr.Column(variant="compact"):
            with gr.Row():
                w.inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                    value=tab_default(n, "inpaint_mode", modules.config.default_inpaint_method, modules.flags.inpaint_options),
                                    label='Method'+ suffix(n), interactive=True)
            with gr.Row():
                w.ad_disable_latent = gr.Checkbox(
                                    label="ADetailer Disable initial latent" + suffix(n),
                                    value=tab_default(n, "ad_disable_latent", False),
                                    visible=True,
                                    elem_id=eid("ad_disable_latent"))

            with gr.Row():
        
                w.ad_inpaint_engine = gr.Dropdown(
                    label="Inpaint Engine" + suffix(n),
                    choices=modules.flags.inpaint_engine_versions,
                    value=tab_default(n, "ad_inpaint_engine", modules.config.default_inpaint_engine_version, modules.flags.inpaint_engine_versions),
                    visible=True,
                    interactive=True,
                    elem_id=eid("ad_inpaint_engine"),
                    info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.'
                )
        
            with gr.Row():

                w.ad_denoising_strength = gr.Slider(
                    label="Inpaint denoising strength" + suffix(n),
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=tab_default(n, "ad_denoising_strength", 1),
                    visible=True,
                    interactive=True,
                    elem_id=eid("ad_denoising_strength"),
                    info='Same as the denoising strength in A1111 inpaint. '
                         'Only used in inpaint, not used in outpaint. '
                         '(Outpaint always use 1.0)'
                )

            with gr.Row():


                w.ad_resp_field = gr.Slider(
                    label="Inpaint Respective Field" + suffix(n),
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=tab_default(n, "ad_resp_field", 0.618),
                    visible=True,
                    interactive=True,
                    elem_id=eid("ad_resp_field"),
                    info='The area to inpaint. '
                         'Value 0 is same as "Only Masked" in A1111. '
                         'Value 1 is same as "Whole Image" in A1111. '
                         'Only used in inpaint, not used in outpaint. '
                         '(Outpaint always use 1.0)'
                )

    def inpaint_mode_change(mode, inpaint_engine_version):


        if mode == modules.flags.inpaint_option_detail:
            return [                
                False, 'None', 0.5, 0.0
            ]


        if mode == modules.flags.inpaint_option_modify:
            return [
                True, inpaint_engine_version, 1.0, 0.0
            ]

        return [
            False, inpaint_engine_version, 1.0, 0.618
        ]
    # .input fires only on user choice, so a preset that sets Method keeps its own denoise/respective field
    w.inpaint_mode.input(inpaint_mode_change, inputs=[w.inpaint_mode, w.ad_inpaint_engine], outputs=[
                            w.ad_disable_latent, w.ad_inpaint_engine,
                            w.ad_denoising_strength, w.ad_resp_field
                        ], show_progress=False, queue=False)            
    with gr.Group():
        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_cfg_scale = gr.Checkbox(
                    label="ADetailer use separate CFG scale" + suffix(n),
                    value=tab_default(n, "ad_use_cfg_scale", False),
                    visible=True,
                    elem_id=eid("ad_use_cfg_scale"),
                )
                w.ad_cfg_scale =  gr.Slider(
                    label="ADetailer CFG scale" + suffix(n),
                    minimum=0.0,
                    maximum=30.0,
                    step=0.5,
                    value=tab_default(n, "ad_cfg_scale", 7.0),
                    visible=True,
                    interactive=w.ad_use_cfg_scale.value,
                    elem_id=eid("ad_cfg_scale"),
                )
                w.ad_use_cfg_scale.change(
                    lambda value: (gr_interactive(value)),
                    inputs=w.ad_use_cfg_scale,
                    outputs=w.ad_cfg_scale,
                    queue=False,
                )



            with gr.Column(variant="compact"):
                w.ad_use_clip_skip = gr.Checkbox(
                    label="Use separate CLIP skip" + suffix(n),
                    value=tab_default(n, "ad_use_clip_skip", False),
                    visible=True,
                    interactive=True,
                    elem_id=eid("ad_use_clip_skip"),
                )

                w.ad_clip_skip = gr.Slider(
                    label="ADetailer CLIP skip" + suffix(n),
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=tab_default(n, "ad_clip_skip", 1),
                    visible=True,
                    elem_id=eid("ad_clip_skip"),
                    interactive=w.ad_use_clip_skip.value
                )

                w.ad_use_clip_skip.change(
                    lambda value: (gr_interactive(value)),
                    inputs=w.ad_use_clip_skip,
                    outputs=w.ad_clip_skip,
                    queue=False,
                )



    with gr.Group():   
        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_checkpoint = gr.Checkbox(
                    label="Use separate checkpoint" + suffix(n),
                    value=tab_default(n, "ad_use_checkpoint", False),
                    visible=True,
                    elem_id=eid("ad_use_checkpoint"),
                )

                ckpts = ["Use same checkpoint", *webui_info.checkpoints_list]

                w.ad_checkpoint = gr.Dropdown(
                    label="ADetailer checkpoint" + suffix(n),
                    choices=ckpts,
                    value=tab_default(n, "ad_checkpoint", ckpts[0], ckpts),
                    visible=True,
                    elem_id=eid("ad_checkpoint"),
                    interactive=w.ad_use_checkpoint.value
                )
                w.ad_use_checkpoint.change(
                    lambda value: (gr_interactive(value)),
                    inputs=w.ad_use_checkpoint,
                    outputs=w.ad_checkpoint,
                    queue=False,
                )
        with gr.Row(), gr.Column(variant="compact"):
            w.ad_use_sampler = gr.Checkbox(
                label="Use separate sampler" + suffix(n),
                value=tab_default(n, "ad_use_sampler", False),
                visible=True,
                elem_id=eid("ad_use_sampler"),
            )

            sampler_names = [
                "Use same sampler",
                *webui_info.sampler_names,
            ]

            with gr.Row():
                w.ad_sampler = gr.Dropdown(
                    label="ADetailer sampler" + suffix(n),
                    choices=sampler_names,
                    value=tab_default(n, "ad_sampler", sampler_names[0], sampler_names),
                    visible=True,
                    elem_id=eid("ad_sampler"),
                    interactive=w.ad_use_sampler.value
                )

                scheduler_names = [
                    "Use same scheduler",
                    *webui_info.scheduler_names,
                ]
                w.ad_scheduler = gr.Dropdown(
                    label="ADetailer scheduler" + suffix(n),
                    choices=scheduler_names,
                    value=tab_default(n, "ad_scheduler", scheduler_names[0], scheduler_names),
                    visible=len(scheduler_names) > 1,
                    elem_id=eid("ad_scheduler"),
                    interactive=w.ad_use_sampler.value
                )

                w.ad_use_sampler.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_sampler,
                    outputs=[w.ad_sampler, w.ad_scheduler],
                    queue=False,
                )

    return w.ad_checkpoint
