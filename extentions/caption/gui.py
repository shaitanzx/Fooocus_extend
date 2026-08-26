import argparse
import json
import os
import time

import gradio as gr
from PIL import Image

from . import caption
from .utils import inference
from .utils.logger import print_title

WD_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_wd.json")
JOY_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_joy.json")
LLAMA_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_llama_3.2V.json")
QWEN_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_qwen2_vl.json")
MINICPM_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_minicpm.json")
FLORENCE_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_florence.json")

SKIP_DOWNLOAD = True

IS_MODEL_LOAD = False
ARGS = None
CAPTION_FN = None


def read_json(config_file):
    with open(config_file, 'r', encoding='utf-8') as config_json:
        datas = json.load(config_json)
        return list(datas.keys())


def gui_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str, default="base", choices=["base", "ocean", "origin"],
                        help="set themes")
    parser.add_argument('--port', type=int, default="8282", help="port, default is `8282`")
    parser.add_argument('--listen', action='store_true', help="allow remote connections")
    parser.add_argument('--share', action='store_true', help="allow gradio share")
    parser.add_argument('--inbrowser', action='store_true', help="auto open in browser")
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="set log level, default is `INFO`")
    parser.add_argument('--models_save_path', type=str, default=caption.DEFAULT_MODELS_SAVE_PATH,
                        help='path to save models, default is `models`.')

    return parser.parse_args()


def gui():
    # print_title()
    # get_gui_args = gui_setup_args()
    # if get_gui_args.theme == "ocean":
    #     theme = gr.themes.Ocean()
    # elif get_gui_args.theme == "origin":
    #     theme = gr.themes.Origin()
    # else:
    #     theme = gr.themes.Base()

    with gr.Row():
        # with gr.Row(equal_height=True):
        #     with gr.Column(scale=6):
        #         gr.Markdown("## Caption images with WD and LLM models (By DukeG)")
        #     close_gradio_server_button = gr.Button(value="Close Gradio Server", variant="primary")

        with gr.Row():
            with gr.Column():
                with gr.Column(min_width=240):
                    model_site = gr.Radio(label="Model Site", choices=["huggingface", "modelscope"],
                                          value="huggingface")
                    huggingface_token = gr.Textbox(label="Hugging Face TOKEN", type="password",
                                                   placeholder="Enter your Hugging Face TOKEN(READ-PERMISSION)")

                # with gr.Column(min_width=240):
                #     log_level = gr.Dropdown(label="Log level",
                #                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                #                             value="INFO")

                with gr.Row(equal_height=True) as models_settings:
                    with gr.Column(min_width=240):
                        with gr.Column(min_width=240):
                            caption_method = gr.Radio(label="Caption method", choices=["WD+LLM", "WD", "LLM"],
                                                      value="WD+LLM")
                            llm_choice = gr.Radio(label="Choice LLM",
                                                  choices=["Llama", "Joy", "Qwen", "MiniCPM", "Florence"],
                                                  value="Llama")

                            def llm_choice_visibility(caption_method_radio):
                                return gr.update(visible=True if "LLM" in caption_method_radio else False)

                            caption_method.select(fn=llm_choice_visibility, inputs=caption_method, outputs=llm_choice)

                        with gr.Column(min_width=240):
                            wd_models = gr.Dropdown(label="WD models", choices=read_json(WD_CONFIG),
                                                    value=read_json(WD_CONFIG)[0])
                            joy_models = gr.Dropdown(label="Joy models", choices=read_json(JOY_CONFIG),
                                                     value=read_json(JOY_CONFIG)[0], visible=False)
                            llama_models = gr.Dropdown(label="Llama models", choices=read_json(LLAMA_CONFIG),
                                                       value=read_json(LLAMA_CONFIG)[0])
                            qwen_models = gr.Dropdown(label="Qwen models", choices=read_json(QWEN_CONFIG),
                                                      value=read_json(QWEN_CONFIG)[0], visible=False)
                            minicpm_models = gr.Dropdown(label="MiniCPM models", choices=read_json(MINICPM_CONFIG),
                                                         value=read_json(MINICPM_CONFIG)[0], visible=False)
                            florence_models = gr.Dropdown(label="Florence models", choices=read_json(FLORENCE_CONFIG),
                                                          value=read_json(FLORENCE_CONFIG)[0], visible=False)

                    with gr.Column(min_width=240):
                        with gr.Column(min_width=240):
                            wd_force_use_cpu = gr.Checkbox(label="Force use CPU for WD inference")
                            llm_use_cpu = gr.Checkbox(label="Use cpu for LLM inference")

                        llm_use_patch = gr.Checkbox(label="Use LLM LoRA to avoid censored")

                        with gr.Column(min_width=240) as llm_load_settings:
                            llm_dtype = gr.Radio(label="LLM dtype", choices=["fp16", "bf16", "fp32"],
                                                 value="fp16",
                                                 interactive=True)
                            llm_qnt = gr.Radio(label="LLM Quantization", choices=["none", "4bit", "8bit"], value="none",
                                               interactive=True)

                with gr.Row():
                    load_model_button = gr.Button(value="Load Models", variant='primary')
                    unload_model_button = gr.Button(value="Unload Models")

                with gr.Row():
                    with gr.Column(min_width=240) as wd_settings:
                        with gr.Group():
                            gr.Markdown("<center>WD Settings</center>")
                        wd_remove_underscore = gr.Checkbox(label="Replace underscores with spaces",
                                                           value=True)

                        # wd_tags_frequency = gr.Checkbox(label="Show frequency of tags for images", value=True,
                        #                                 interactive=True)
                        wd_threshold = gr.Slider(label="Threshold", minimum=0.01, maximum=1.00, value=0.35, step=0.01)
                        wd_general_threshold = gr.Slider(label="General threshold",
                                                         minimum=0.01, maximum=1.00, value=0.35, step=0.01)
                        wd_character_threshold = gr.Slider(label="Character threshold",
                                                           minimum=0.01, maximum=1.00, value=0.85, step=0.01)

                        wd_add_rating_tags_to_first = gr.Checkbox(label="Adds rating tags to the first")
                        wd_add_rating_tags_to_last = gr.Checkbox(label="Adds rating tags to the last")
                        wd_character_tags_first = gr.Checkbox(label="Always put character tags before the general tags")
                        wd_character_tag_expand = gr.Checkbox(
                            label="Expand tag tail parenthesis to another tag for character tags")

                        wd_undesired_tags = gr.Textbox(label="undesired tags to remove",
                                                       placeholder="comma-separated list of tags")
                        wd_always_first_tags = gr.Textbox(label="Tags always put at the beginning",
                                                          placeholder="comma-separated list of tags")

                        wd_caption_extension = gr.Textbox(label="extension for wd captions files", value=".wdcaption")
                        wd_caption_separator = gr.Textbox(label="Separator for tags", value=", ")
                        wd_tag_replacement = gr.Textbox(label="Tag replacement",
                                                        placeholder="in the format of `source1,target1;source2,target2;...`")

                    with gr.Column(min_width=240) as llm_settings:
                        with gr.Group():
                            gr.Markdown("<center>LLM Settings</center>")
                        llm_caption_extension = gr.Textbox(label="extension of LLM caption file", value=".llmcaption")
                        llm_read_wd_caption = gr.Checkbox(label="llm will read wd caption for inference")
                        llm_caption_without_wd = gr.Checkbox(label="llm will not read wd caption for inference")

                        with gr.Accordion(label="Joy Formated Prompts", open=False) as joy_formated_prompts:
                            caption_type = gr.Dropdown(
                                label="Caption Type",
                                choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney",
                                         "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing",
                                         "Social Media Post"],
                                value="Descriptive")

                            caption_length = gr.Dropdown(
                                label="Caption Length",
                                choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                                        [str(i) for i in range(20, 261, 10)],
                                value="long",
                            )
                            with gr.Column(min_width=240) as extra_options_column:
                                extra_options = gr.CheckboxGroup(
                                    label="Extra Options",
                                    choices=[
                                        "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
                                        "Include information about lighting.",
                                        "Include information about camera angle.",
                                        "Include information about whether there is a watermark or not.",
                                        "Include information about whether there are JPEG artifacts or not.",
                                        "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
                                        "Do NOT include anything sexual; keep it PG.",
                                        "Do NOT mention the image's resolution.",
                                        "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
                                        "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
                                        "Do NOT mention any text that is in the image.",
                                        "Specify the depth of field and whether the background is in focus or blurred.",
                                        "If applicable, mention the likely use of artificial or natural lighting sources.",
                                        "Do NOT use any ambiguous language.",
                                        "Include whether the image is sfw, suggestive, or nsfw.",
                                        "ONLY describe the most important elements of the image.",
                                        "If there is a person/character in the image you must refer to them as {name}."
                                    ])

                                name_input = gr.Textbox(label="Person/Character Name (if applicable)")

                            generate_prompt_button = gr.Button(value="Generate prompts", variant="primary")

                        llm_system_prompt = gr.Textbox(label="system prompt for llm caption", lines=7, max_lines=7,
                                                       value=caption.DEFAULT_SYSTEM_PROMPT)
                        llm_user_prompt = gr.Textbox(label="user prompt for llm caption", lines=7, max_lines=7,
                                                     value=caption.DEFAULT_USER_PROMPT_WITH_WD)

                        with gr.Accordion(label="Advanced Options", open=False):
                            llm_temperature = gr.Slider(label="temperature for LLM model",
                                                        minimum=0, maximum=1.0, value=0, step=0.1)
                            llm_max_tokens = gr.Slider(label="max token for LLM model",
                                                       minimum=0, maximum=2048, value=0, step=1)
                            image_size = gr.Slider(label="Resize image for inference",
                                                   minimum=256, maximum=2048, value=1024, step=1)
                            auto_unload = gr.Checkbox(label="Auto Unload Models after inference.")

            with gr.Column():
                with gr.Tab("Single mode"):
                    with gr.Column():
                        input_image = gr.Image(elem_id="input_image", type='filepath', label="Upload Image",
                                               sources=["upload", "clipboard"])
                        single_image_submit_button = gr.Button(elem_id="single_image_submit_button",
                                                               value="Inference", variant='primary')

                    with gr.Column():
                        # current_image = gr.Image(label='Current Image', interactive=False)
                        # wd_tags_rating = gr.Label(label="WD Ratings")
                        # wd_tags_text = gr.Label(label="WD Tags")
                        wd_tags_output = gr.Text(label='WD Tags Output', lines=10,
                                                 interactive=False, show_label=True)
                        llm_caption_output = gr.Text(label='LLM Caption Output', lines=10,
                                                     interactive=False, show_label=True)

                with gr.Tab("Batch mode") as bs_mode:
                    with gr.Column(min_width=240):
                        with gr.Row():
                            input_dir = gr.Textbox(label="Batch Directory",
                                                   placeholder="Enter the directory path for batch processing",
                                                   scale=4)
                            is_recursive = gr.Checkbox(label="recursive subfolder", scale=1)
                        custom_caption_save_path = gr.Textbox(label="Custom caption save directory",
                                                              placeholder="Enter custom caption save directory path "
                                                                          "for batch processing")
                        with gr.Row(equal_height=True):
                            run_method = gr.Radio(label="Run method", choices=['sync', 'queue'],
                                                  value="sync", interactive=True)

                            with gr.Column(min_width=240):
                                skip_exists = gr.Checkbox(label="Will not caption if caption file exists")
                                not_overwrite = gr.Checkbox(label="Will not overwrite caption file if exists")
                        with gr.Column(min_width=240):
                            caption_extension = gr.Textbox(label="Caption file extension", value=".txt")
                            save_caption_together = gr.Checkbox(label="Save WD and LLM captions in one file",
                                                                value=True)
                            save_caption_together_seperator = gr.Textbox(
                                label="Seperator between WD tags and LLM captions", value="|")

                        batch_process_submit_button = gr.Button(elem_id="batch_process_submit_button",
                                                                value="Batch Process", variant='primary')

        def huggingface_token_update_visibility(model_site_radio):
            return gr.Textbox(visible=True if model_site_radio == "huggingface" else False)

        model_site.change(fn=huggingface_token_update_visibility,
                          inputs=model_site, outputs=huggingface_token)

        def caption_method_update_visibility(caption_method_radio):
            run_method_visible = gr.update(visible=True if caption_method_radio == "WD+LLM" else False)
            wd_force_use_cpu_visible = wd_model_visible = gr.update(
                visible=True if "WD" in caption_method_radio else False)
            llm_load_settings_visible = llm_use_cpu_visible = gr.update(
                visible=True if "LLM" in caption_method_radio else False)
            wd_settings_visible = gr.update(visible=True if "WD" in caption_method_radio else False)
            llm_settings_visible = gr.update(visible=True if "LLM" in caption_method_radio else False)
            return run_method_visible, wd_model_visible, wd_force_use_cpu_visible, \
                llm_use_cpu_visible, wd_settings_visible, llm_load_settings_visible, llm_settings_visible

        def llm_choice_update_visibility(caption_method_radio, llm_choice_radio, joy_models_dropdown):
            joy_model_visible = gr.update(
                visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Joy" else False)
            llama_model_visible = gr.update(
                visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Llama" else False)
            llama_use_patch_visible = gr.update(
                visible=True if "LLM" in caption_method_radio and (llm_choice_radio == "Llama" or (
                        llm_choice_radio == "Joy" and joy_models_dropdown == "Joy-Caption-Pre-Alpha")) else False)
            qwen_model_visible = gr.update(
                visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Qwen" else False)
            minicpm_model_visible = gr.update(
                visible=True if "LLM" in caption_method_radio and llm_choice_radio == "MiniCPM" else False)
            florence_model_visible = gr.update(
                visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Florence" else False)

            return joy_model_visible, llama_model_visible, llama_use_patch_visible, \
                qwen_model_visible, minicpm_model_visible, florence_model_visible

        def joy_formated_prompts_visibility(llm_choice_radio, joy_models_dropdown):
            joy_formated_prompts_visible = gr.update(
                visible=True if llm_choice_radio == "Joy" and joy_models_dropdown != "Joy-Caption-Pre-Alpha" else False)
            extra_options_visible = gr.update(
                visible=True if llm_choice_radio == "Joy" and joy_models_dropdown in ["Joy-Caption-Alpha-Two-Llava",
                                                                                      "Joy-Caption-Alpha-Two"] else False)
            return joy_formated_prompts_visible, extra_options_visible

        caption_method.change(fn=caption_method_update_visibility, inputs=caption_method,
                              outputs=[run_method, wd_models, wd_force_use_cpu, llm_use_cpu,
                                       wd_settings, llm_load_settings, llm_settings])
        caption_method.change(fn=llm_choice_update_visibility, inputs=[caption_method, llm_choice, joy_models],
                              outputs=[joy_models, llama_models, llm_use_patch,
                                       qwen_models, minicpm_models, florence_models])
        llm_choice.change(fn=llm_choice_update_visibility, inputs=[caption_method, llm_choice, joy_models],
                          outputs=[joy_models, llama_models, llm_use_patch,
                                   qwen_models, minicpm_models, florence_models])
        llm_choice.change(fn=joy_formated_prompts_visibility, inputs=[llm_choice, joy_models],
                          outputs=[joy_formated_prompts, extra_options_column])
        joy_models.change(fn=llm_choice_update_visibility, inputs=[caption_method, llm_choice, joy_models],
                          outputs=[joy_models, llama_models, llm_use_patch,
                                   qwen_models, minicpm_models, florence_models])
        joy_models.change(fn=joy_formated_prompts_visibility, inputs=[llm_choice, joy_models],
                          outputs=[joy_formated_prompts, extra_options_column])

        def llm_use_patch_visibility(llama_model_dropdown):
            return gr.update(
                visible=True if llama_model_dropdown == "Llama-3.2-11B-Vision-Instruct" else False)

        llama_models.select(fn=llm_use_patch_visibility, inputs=llama_models, outputs=llm_use_patch)

        def llm_user_prompt_default(caption_method_radio,
                                    llm_read_wd_caption_select,
                                    llm_user_prompt_textbox):
            if caption_method_radio != "WD+LLM" and llm_user_prompt_textbox == inference.DEFAULT_USER_PROMPT_WITH_WD:
                llm_user_prompt_change = gr.update(value=inference.DEFAULT_USER_PROMPT_WITHOUT_WD)
            elif caption_method_radio == "WD+LLM" and llm_user_prompt_textbox == inference.DEFAULT_USER_PROMPT_WITHOUT_WD:
                llm_user_prompt_change = gr.update(value=inference.DEFAULT_USER_PROMPT_WITH_WD)
            elif caption_method_radio == "LLM" and llm_read_wd_caption_select:
                llm_user_prompt_change = gr.update(value=inference.DEFAULT_USER_PROMPT_WITHOUT_WD)
            else:
                llm_user_prompt_change = gr.update(value=llm_user_prompt_textbox)
            return llm_user_prompt_change

        def build_joy_user_prompt(caption_method_value: str, joy_models_value: str, llm_read_wd_caption_value: bool,
                                  caption_type_value: str, caption_length_value: str,
                                  extra_options_value: list[str], name_input_value: str):
            caption_type_map = {
                "Descriptive": [
                    "Write a descriptive caption for this image in a formal tone.",
                    "Write a descriptive caption for this image in a formal tone within {word_count} words.",
                    "Write a {length} descriptive caption for this image in a formal tone.",
                ],
                "Descriptive (Informal)": [
                    "Write a descriptive caption for this image in a casual tone.",
                    "Write a descriptive caption for this image in a casual tone within {word_count} words.",
                    "Write a {length} descriptive caption for this image in a casual tone.",
                ],
                "Training Prompt": [
                    "Write a stable diffusion prompt for this image.",
                    "Write a stable diffusion prompt for this image within {word_count} words.",
                    "Write a {length} stable diffusion prompt for this image.",
                ],
                "MidJourney": [
                    "Write a MidJourney prompt for this image.",
                    "Write a MidJourney prompt for this image within {word_count} words.",
                    "Write a {length} MidJourney prompt for this image.",
                ],
                "Booru tag list": [
                    "Write a list of Booru tags for this image.",
                    "Write a list of Booru tags for this image within {word_count} words.",
                    "Write a {length} list of Booru tags for this image.",
                ],
                "Booru-like tag list": [
                    "Write a list of Booru-like tags for this image.",
                    "Write a list of Booru-like tags for this image within {word_count} words.",
                    "Write a {length} list of Booru-like tags for this image.",
                ],
                "Art Critic": [
                    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
                    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
                    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
                ],
                "Product Listing": [
                    "Write a caption for this image as though it were a product listing.",
                    "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
                    "Write a {length} caption for this image as though it were a product listing.",
                ],
                "Social Media Post": [
                    "Write a caption for this image as if it were being used for a social media post.",
                    "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
                    "Write a {length} caption for this image as if it were being used for a social media post.",
                ],
            }

            length = None if caption_length_value == "any" else caption_length_value

            if isinstance(length, str):
                try:
                    length = int(length)
                except ValueError:
                    pass

            # Build prompt
            if length is None:
                map_idx = 0
            elif isinstance(length, int):
                map_idx = 1
            elif isinstance(length, str):
                map_idx = 2
            else:
                raise ValueError(f"Invalid caption length: {length}")

            prompt_str = ""
            if caption_method_value == "WD+LLM" or llm_read_wd_caption_value:
                prompt_str = "Refer to the following tags: {wd_tags}. "

            prompt_str = prompt_str + caption_type_map[caption_type_value][map_idx]
            # Add extra options
            if joy_models_value in ["Joy-Caption-Alpha-Two-Llava", "Joy-Caption-Alpha-Two"] and \
                    len(extra_options_value) > 0:
                prompt_str += " " + " ".join(extra_options_value)
            # Add name, length, word_count
            system_prompt = "You are a helpful image captioner."
            user_prompt_str = prompt_str.format(wd_tags="{wd_tags}", name=name_input_value, length=caption_length_value,
                                                word_count=caption_length_value)
            return system_prompt, user_prompt_str.strip()

        caption_method.change(fn=llm_user_prompt_default,
                              inputs=[caption_method, llm_read_wd_caption, llm_user_prompt],
                              outputs=llm_user_prompt)

        generate_prompt_button.click(fn=build_joy_user_prompt,
                                     inputs=[caption_method, joy_models, llm_read_wd_caption,
                                             caption_type, caption_length, extra_options, name_input],
                                     outputs=[llm_system_prompt, llm_user_prompt])

        def use_wd(check_caption_method):
            return True if check_caption_method in ["wd", "wd+llm"] else False

        def use_joy(check_caption_method, check_llm_choice):
            return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "joy" else False

        def use_llama(check_caption_method, check_llm_choice):
            return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "llama" else False

        def use_qwen(check_caption_method, check_llm_choice):
            return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "qwen" else False

        def use_minicpm(check_caption_method, check_llm_choice):
            return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "minicpm" else False

        def use_florence(check_caption_method, check_llm_choice):
            return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "florence" else False

        def load_models_interactive_group():
            return [
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(variant='secondary'),
                gr.update(variant='primary')
            ]

        def unloads_models_interactive_group():
            return [
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(variant='primary'),
                gr.update(variant='secondary')
            ]

        single_inference_input_args = [wd_remove_underscore,
                                       wd_threshold,
                                       wd_general_threshold,
                                       wd_character_threshold,
                                       wd_add_rating_tags_to_first,
                                       wd_character_tags_first,
                                       wd_add_rating_tags_to_last,
                                       wd_character_tag_expand,
                                       wd_undesired_tags,
                                       wd_always_first_tags,
                                       wd_caption_extension,
                                       wd_caption_separator,
                                       wd_tag_replacement,
                                       llm_caption_extension,
                                       llm_read_wd_caption,
                                       llm_caption_without_wd,
                                       llm_system_prompt,
                                       llm_user_prompt,
                                       llm_temperature,
                                       llm_max_tokens,
                                       image_size,
                                       auto_unload,
                                       input_image]

        batch_inference_input_args = [batch_process_submit_button,
                                      run_method,
                                      wd_remove_underscore,
                                      wd_threshold,
                                      wd_general_threshold,
                                      wd_character_threshold,
                                      wd_add_rating_tags_to_first,
                                      wd_character_tags_first,
                                      wd_add_rating_tags_to_last,
                                      wd_character_tag_expand,
                                      wd_undesired_tags,
                                      wd_always_first_tags,
                                      wd_caption_extension,
                                      wd_caption_separator,
                                      wd_tag_replacement,
                                      llm_caption_extension,
                                      llm_read_wd_caption,
                                      llm_caption_without_wd,
                                      llm_system_prompt,
                                      llm_user_prompt,
                                      llm_temperature,
                                      llm_max_tokens,
                                      image_size,
                                      auto_unload,
                                      input_dir,
                                      is_recursive,
                                      custom_caption_save_path,
                                      skip_exists,
                                      not_overwrite,
                                      caption_extension,
                                      save_caption_together,
                                      save_caption_together_seperator]

        def caption_models_load(
                model_site_value,
                huggingface_token_value,
                caption_method_value,
                llm_choice_value,
                wd_model_value,
                joy_model_value,
                llama_model_value,
                qwen_model_value,
                minicpm_model_value,
                florence_model_value,
                wd_force_use_cpu_value,
                llm_use_cpu_value,
                llm_use_patch_value,
                llm_dtype_value,
                llm_qnt_value
        ):
            global IS_MODEL_LOAD, ARGS, CAPTION_FN

            if not IS_MODEL_LOAD:
                start_time = time.monotonic()

                if ARGS is None:
                    ARGS = caption.setup_args()

                args = ARGS
                args.model_site = model_site_value
                if huggingface_token_value != "" and str(huggingface_token_value).startswith("hf"):
                    os.environ["HF_TOKEN"] = str(huggingface_token_value)

                get_gradio_args = gui_setup_args()
                args.models_save_path = str(get_gradio_args.models_save_path)
                args.log_level = str(get_gradio_args.log_level)
                args.caption_method = str(caption_method_value).lower()
                args.llm_choice = str(llm_choice_value).lower()

                if use_wd(args.caption_method):
                    args.wd_config = WD_CONFIG
                    args.wd_model_name = str(wd_model_value)
                if use_joy(args.caption_method, args.llm_choice):
                    args.llm_config = JOY_CONFIG
                    args.llm_model_name = str(joy_model_value)
                elif use_llama(args.caption_method, args.llm_choice):
                    args.llm_config = LLAMA_CONFIG
                    args.llm_model_name = str(llama_model_value)
                elif use_qwen(args.caption_method, args.llm_choice):
                    args.llm_config = QWEN_CONFIG
                    args.llm_model_name = str(qwen_model_value)
                elif use_minicpm(args.caption_method, args.llm_choice):
                    args.llm_config = MINICPM_CONFIG
                    args.llm_model_name = str(minicpm_model_value)
                elif use_florence(args.caption_method, args.llm_choice):
                    args.llm_config = FLORENCE_CONFIG
                    args.llm_model_name = str(florence_model_value)

                if CAPTION_FN is None:
                    CAPTION_FN = caption.Caption()
                    CAPTION_FN.set_logger(args)

                caption_init = CAPTION_FN
                args.wd_force_use_cpu = bool(wd_force_use_cpu_value)

                args.llm_use_cpu = bool(llm_use_cpu_value)
                args.llm_patch = bool(llm_use_patch_value)
                args.llm_dtype = str(llm_dtype_value)
                args.llm_qnt = str(llm_qnt_value)
                # SKIP DOWNLOAD
                args.skip_download = SKIP_DOWNLOAD

                caption_init.download_models(args)
                caption_init.load_models(args)

                IS_MODEL_LOAD = True
                gr.Info(f"Models loaded in {time.monotonic() - start_time:.1f}s.")
                return load_models_interactive_group()
            else:
                args = ARGS
                if args.wd_model_name and not args.llm_model_name:
                    warning = f"{args.wd_model_name}"
                elif not args.wd_model_name and args.llm_model_name:
                    warning = f"{args.llm_model_name}"
                else:
                    warning = f"{args.wd_model_name} & {args.llm_model_name}"
                gr.Warning(f"{warning} already loaded!")
                return unloads_models_interactive_group()

        def caption_single_inference(wd_remove_underscore_value,
                                     wd_threshold_value,
                                     wd_general_threshold_value,
                                     wd_character_threshold_value,
                                     wd_add_rating_tags_to_first_value,
                                     wd_character_tags_first_value,
                                     wd_add_rating_tags_to_last_value,
                                     wd_character_tag_expand_value,
                                     wd_undesired_tags_value,
                                     wd_always_first_tags_value,
                                     wd_caption_extension_value,
                                     wd_caption_separator_value,
                                     wd_tag_replacement_value,
                                     llm_caption_extension_value,
                                     llm_read_wd_caption_value,
                                     llm_caption_without_wd_value,
                                     llm_system_prompt_value,
                                     llm_user_prompt_value,
                                     llm_temperature_value,
                                     llm_max_tokens_value,
                                     image_size_value,
                                     auto_unload_value,
                                     input_image_value):
            if not IS_MODEL_LOAD:
                raise gr.Error("Models not loaded!")
            args, get_caption_fn = ARGS, CAPTION_FN
            # Read args
            args.wd_remove_underscore = bool(wd_remove_underscore_value)
            args.wd_threshold = float(wd_threshold_value)
            args.wd_general_threshold = float(wd_general_threshold_value)
            args.wd_character_threshold = float(wd_character_threshold_value)
            args.wd_add_rating_tags_to_first = bool(wd_add_rating_tags_to_first_value)
            args.wd_add_rating_tags_to_last = bool(wd_add_rating_tags_to_last_value)
            args.wd_character_tags_first = bool(wd_character_tags_first_value)
            args.wd_character_tag_expand = bool(wd_character_tag_expand_value)
            args.wd_undesired_tags = str(wd_undesired_tags_value)
            args.wd_always_first_tags = str(wd_always_first_tags_value)
            args.wd_caption_extension = str(wd_caption_extension_value)
            args.wd_caption_separator = str(wd_caption_separator_value)
            args.wd_tag_replacement = str(wd_tag_replacement_value)

            args.llm_caption_extension = str(llm_caption_extension_value)
            args.llm_read_wd_caption = bool(llm_read_wd_caption_value)
            args.llm_caption_without_wd = bool(llm_caption_without_wd_value)
            args.llm_system_prompt = str(llm_system_prompt_value)
            args.llm_user_prompt = str(llm_user_prompt_value)
            args.llm_temperature = float(llm_temperature_value)
            args.llm_max_tokens = int(llm_max_tokens_value)

            args.image_size = int(image_size_value)

            args.data_path = str(input_image_value)
            start_time = time.monotonic()
            image = Image.open(input_image_value)
            tag_text = ""
            caption_text = ""
            get_caption_fn.my_logger.debug(f"Input image: {args.data_path}.")
            if use_wd(args.caption_method):
                get_caption_fn.my_logger.debug(f"Tagging with WD: {args.wd_model_name}.")
                # WD tag
                tag_text, rating_tag_text, character_tag_text, general_tag_text = get_caption_fn.my_tagger.get_tags(
                    image=image)
                if rating_tag_text:
                    get_caption_fn.my_logger.debug(f"WD Rating tags: {rating_tag_text}")
                if character_tag_text:
                    get_caption_fn.my_logger.debug(f"WD Character tags: {character_tag_text}")
                get_caption_fn.my_logger.debug(f"WD General tags: {general_tag_text}")
                get_caption_fn.my_logger.info(f"WD tags content: {tag_text}")

            if use_joy(args.caption_method, args.llm_choice) \
                    or use_llama(args.caption_method, args.llm_choice) \
                    or use_qwen(args.caption_method, args.llm_choice) \
                    or use_minicpm(args.caption_method, args.llm_choice) \
                    or use_florence(args.caption_method, args.llm_choice):
                get_caption_fn.my_logger.debug(f"Caption with LLM: {args.llm_model_name}.")
                # LLM Caption
                caption_text = get_caption_fn.my_llm.get_caption(
                    image=image,
                    system_prompt=str(args.llm_system_prompt),
                    user_prompt=str(args.llm_user_prompt).format(wd_tags=tag_text) if tag_text else \
                        str(args.llm_user_prompt),
                    temperature=args.llm_temperature,
                    max_new_tokens=args.llm_max_tokens
                )
                get_caption_fn.my_logger.info(f"LLM Caption content: {caption_text}")
            gr.Info(f"Inference end in {time.monotonic() - start_time:.1f}s.")
            get_caption_fn.my_logger.info(f"Inference end in {time.monotonic() - start_time:.1f}s.")
            if auto_unload_value:
                caption_unload_models()
            return tag_text, caption_text

        def caption_batch_inference(batch_process_submit_button_value,
                                    run_method_value,
                                    wd_remove_underscore_value,
                                    wd_threshold_value,
                                    wd_general_threshold_value,
                                    wd_character_threshold_value,
                                    wd_add_rating_tags_to_first_value,
                                    wd_character_tags_first_value,
                                    wd_add_rating_tags_to_last_value,
                                    wd_character_tag_expand_value,
                                    wd_undesired_tags_value,
                                    wd_always_first_tags_value,
                                    wd_caption_extension_value,
                                    wd_caption_separator_value,
                                    wd_tag_replacement_value,
                                    llm_caption_extension_value,
                                    llm_read_wd_caption_value,
                                    llm_caption_without_wd_value,
                                    llm_system_prompt_value,
                                    llm_user_prompt_value,
                                    llm_temperature_value,
                                    llm_max_tokens_value,
                                    image_size_value,
                                    auto_unload_value,
                                    input_dir_value,
                                    recursive_value,
                                    custom_caption_save_path_value,
                                    skip_exists_value,
                                    not_overwrite_value,
                                    caption_extension_value,
                                    save_caption_together_value,
                                    save_caption_together_seperator_value):
            if batch_process_submit_button_value == "Batch Process":
                if not IS_MODEL_LOAD:
                    raise gr.Error("Models not loaded!")
                args, get_caption_fn = ARGS, CAPTION_FN

                if not input_dir_value:
                    raise gr.Error("None input image/dir!")

                args.data_path = str()

                args.wd_remove_underscore = bool(wd_remove_underscore_value)
                args.wd_threshold = float(wd_threshold_value)
                args.wd_general_threshold = float(wd_general_threshold_value)
                args.wd_character_threshold = float(wd_character_threshold_value)
                args.wd_add_rating_tags_to_first = bool(wd_add_rating_tags_to_first_value)
                args.wd_add_rating_tags_to_last = bool(wd_add_rating_tags_to_last_value)
                args.wd_character_tags_first = bool(wd_character_tags_first_value)
                args.wd_character_tag_expand = bool(wd_character_tag_expand_value)
                args.wd_undesired_tags = str(wd_undesired_tags_value)
                args.wd_always_first_tags = str(wd_always_first_tags_value)
                args.wd_caption_extension = str(wd_caption_extension_value)
                args.wd_caption_separator = str(wd_caption_separator_value)
                args.wd_tag_replacement = str(wd_tag_replacement_value)

                args.llm_caption_extension = str(llm_caption_extension_value)
                args.llm_read_wd_caption = bool(llm_read_wd_caption_value)
                args.llm_caption_without_wd = bool(llm_caption_without_wd_value)
                args.llm_system_prompt = str(llm_system_prompt_value)
                args.llm_user_prompt = str(llm_user_prompt_value)
                args.llm_temperature = float(llm_temperature_value)
                args.llm_max_tokens = int(llm_max_tokens_value)

                args.image_size = int(image_size_value)

                args.data_path = str(input_dir_value)
                args.run_method = str(run_method_value)
                args.recursive = bool(recursive_value)
                args.custom_caption_save_path = str(custom_caption_save_path_value)
                args.skip_exists = bool(skip_exists_value)
                args.not_overwrite = bool(not_overwrite_value)
                args.caption_extension = str(caption_extension_value)
                args.save_caption_together = bool(save_caption_together_value)
                args.save_caption_together_seperator = str(save_caption_together_seperator_value)

                if args.data_path and not os.path.exists(args.data_path):
                    raise gr.Error(f"{args.data_path} NOT FOUND!!!")
                if args.custom_caption_save_path and not os.path.exists(args.custom_caption_save_path):
                    raise gr.Error(f"{args.data_path} NOT FOUND!!!")

                start_time = time.monotonic()
                get_caption_fn.run_inference(args)
                gr.Info(f"Inference end in {time.monotonic() - start_time:.1f}s.")
                if auto_unload_value:
                    caption_unload_models()

                return gr.update(value="Done!", variant='stop')
            else:
                return gr.update(value="Batch Process", variant='primary')

        def caption_unload_models():
            global IS_MODEL_LOAD
            if IS_MODEL_LOAD:
                get_caption_args, get_caption_fn = ARGS, CAPTION_FN
                get_caption_fn.unload_models()

                IS_MODEL_LOAD = False

                gr.Info("Models unloaded successfully.")
            else:
                gr.Warning("Models not loaded!")
            return unloads_models_interactive_group()

        # Button Listener
        load_model_button.click(fn=caption_models_load,
                                inputs=[model_site, huggingface_token,
                                        caption_method, llm_choice,
                                        wd_models, joy_models, llama_models,
                                        qwen_models, minicpm_models, florence_models,
                                        wd_force_use_cpu,
                                        llm_use_cpu, llm_use_patch, llm_dtype, llm_qnt],
                                outputs=[model_site, huggingface_token,
                                         caption_method, llm_choice,
                                         wd_models, joy_models, llama_models,
                                         qwen_models, minicpm_models, florence_models,
                                         wd_force_use_cpu,
                                         llm_use_cpu, llm_use_patch, llm_dtype, llm_qnt,
                                         load_model_button, unload_model_button])

        unload_model_button.click(fn=caption_unload_models,
                                  outputs=[model_site, huggingface_token,
                                           caption_method, llm_choice,
                                           wd_models, joy_models, llama_models,
                                           qwen_models, minicpm_models, florence_models,
                                           wd_force_use_cpu,
                                           llm_use_cpu, llm_use_patch, llm_dtype, llm_qnt,
                                           load_model_button, unload_model_button])

        single_image_submit_button.click(fn=caption_single_inference,
                                         inputs=single_inference_input_args,
                                         outputs=[wd_tags_output, llm_caption_output])

        batch_process_submit_button.click(fn=caption_batch_inference,
                                          inputs=batch_inference_input_args,
                                          outputs=batch_process_submit_button)

#         def close_gradio_server():
#             demo.close()

#         close_gradio_server_button.click(fn=close_gradio_server)

#     demo.launch(
#         server_name="0.0.0.0" if get_gui_args.listen else None,
#         server_port=get_gui_args.port,
#         share=True,
#         inbrowser=False
#     )


# if __name__ == "__main__":
#     gui()
