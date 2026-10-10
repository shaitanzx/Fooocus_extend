import json
import os
import time

import gradio as gr
from PIL import Image
import ldm_patched.modules.model_management as mm
import modules.default_pipeline as pipeline
import modules.core as core
import modules.config
import modules.util 
from modules.launch_util import delete_folder_content
import gc
import torch
import zipfile
from extentions.tutorial import tutorial

from . import caption
from .utils import inference

SINGLE_TASKS = (
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
    "Object Detection",
    "Dense Region Caption",
    "Region Proposal",
    "Caption to Phrase Grounding",
    "Referring Expression Segmentation",
    "Region to Segmentation",
    "Open Vocabulary Detection",
    "Region to Category",
    "Region to Description",
    "OCR",
    "OCR with Region",
)
WD_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_wd.json")
JOY_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_joy.json")
QWEN_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_qwen2_vl.json")
FLORENCE_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default_florence.json")

with open(WD_CONFIG, "r", encoding="utf-8") as f:
    TAGGER_MODEL_CONFIG = json.load(f)

SKIP_DOWNLOAD = True

IS_MODEL_LOAD = False
ARGS = None
CAPTION_FN = None


temp_dir=modules.config.temp_path+os.path.sep
def clear_dirs(ext_dir):
    result=delete_folder_content(f"{temp_dir}{ext_dir}", '')
    result=delete_folder_content(f"{temp_dir}batch_temp", '')
    return

def output_zip():
    directory=f"{temp_dir}batch_caption"
    _, _, filename = modules.util.generate_temp_filename(folder=temp_dir)
    name, ext = os.path.splitext(filename)
    zip_file = os.path.join(temp_dir, f"output_{name[:-5]}.zip")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, directory))
    zipf.close()
    return zip_file   
def zip_enable(enable):
    if enable:
        return gr.update(visible=True),gr.update(visible=False)
    else:
        return gr.update(visible=False),gr.update(visible=True)

def unzip_file(zip_file_obj,files_single,enable_zip):
    extract_folder = f"{temp_dir}batch_temp"
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    if enable_zip:
        zip_ref=zipfile.ZipFile(zip_file_obj.name, 'r')
        zip_ref.extractall(extract_folder)
        zip_ref.close()
    else:
        for file in files_single:
            original_name = os.path.basename(getattr(file, 'orig_name', file.name))
            save_path = os.path.join(extract_folder, original_name)
            try:
                with open(file.name, 'rb') as src:
                    with open(save_path, 'wb') as dst:
                        while True:
                            chunk = src.read(8192)  # Читаем по 8KB за раз
                            if not chunk:
                                break
                            dst.write(chunk)
            except Exception as e:
                print(f"copy error {original_name}: {str(e)}")
    return

def read_json(config_file):
    with open(config_file, 'r', encoding='utf-8') as config_json:
        datas = json.load(config_json)
        return list(datas.keys())

def gui():
    with gr.Row():
        tutorial("https://github.com/shaitanzx/Fooocus_extend/blob/dev/extentions/caption/Readme_eng.md")
    with gr.Row():
        with gr.Column():

            with gr.Row(equal_height=True) as models_settings:
                with gr.Column(min_width=240):
                    with gr.Column(min_width=240):
                        caption_method = gr.Radio(label="Caption method", choices=["WD+LLM", "WD", "LLM"], value="WD+LLM")
                        llm_choice = gr.Radio(label="Choice LLM", choices=["Joy", "Qwen", "Florence"], value="Joy")

                        def llm_choice_visibility(caption_method_radio):
                            return gr.update(visible=True if "LLM" in caption_method_radio else False)

                        caption_method.change(fn=llm_choice_visibility, inputs=caption_method, outputs=llm_choice)

                    with gr.Column(min_width=240):
                        wd_model_names = list(TAGGER_MODEL_CONFIG.keys())
                        wd_models = gr.Dropdown(label="Tagger models",choices=wd_model_names,value=wd_model_names[0],)
                        joy_models = gr.Dropdown(label="Joy models", choices=read_json(JOY_CONFIG), value=read_json(JOY_CONFIG)[0], visible=True)
                        qwen_models = gr.Dropdown(label="Qwen models", choices=read_json(QWEN_CONFIG), value=read_json(QWEN_CONFIG)[0], visible=False)
                        florence_models = gr.Dropdown(label="Florence models", choices=read_json(FLORENCE_CONFIG), value=read_json(FLORENCE_CONFIG)[0], visible=False)

                with gr.Column(min_width=240):
                    with gr.Column(min_width=240):
                        wd_force_use_cpu = gr.Checkbox(label="Force use CPU for tagger inference", value=True)
                        llm_use_cpu = gr.Checkbox(label="Use cpu for LLM inference")

                    llm_use_patch = gr.Checkbox(label="Use LLM LoRA to avoid censored")

                    with gr.Column(min_width=240) as llm_load_settings:
                        llm_dtype = gr.Radio(label="LLM dtype", choices=["fp16", "bf16", "fp32"], value="fp16", interactive=True)
                        llm_qnt = gr.Radio(label="LLM Quantization", choices=["none", "4bit", "8bit"], value="none", interactive=True)

            with gr.Row():
                load_model_button = gr.Button(value="Load Models", variant='primary')
                unload_model_button = gr.Button(value="Unload Models",interactive=False)

            with gr.Row():
                with gr.Column(min_width=240) as tagger_settings:
                    with gr.Column(min_width=240) as wd_settings:
                        with gr.Group():
                            gr.Markdown("<center>WD Settings</center>")

                        wd_remove_underscore = gr.Checkbox(label="Replace underscores with spaces",value=True)
                        wd_threshold = gr.Slider(label="Threshold",minimum=0.01,maximum=1.00,value=0.35,step=0.01)
                        wd_general_threshold = gr.Slider(label="General threshold",minimum=0.01,maximum=1.00,value=0.35,step=0.01)
                        wd_character_threshold = gr.Slider(label="Character threshold",minimum=0.01,maximum=1.00,value=0.85,step=0.01)
                        wd_add_rating_tags_to_first = gr.Checkbox(label="Adds rating tags to the first")
                        wd_add_rating_tags_to_last = gr.Checkbox(label="Adds rating tags to the last")
                        wd_character_tags_first = gr.Checkbox(label="Always put character tags before the general tags")
                        wd_character_tag_expand = gr.Checkbox(label="Expand tag tail parenthesis to another tag for character tags")
                        wd_undesired_tags = gr.Textbox(label="undesired tags to remove",placeholder="comma-separated list of tags")
                        wd_always_first_tags = gr.Textbox(label="Tags always put at the beginning",placeholder="comma-separated list of tags")
                        wd_tag_replacement = gr.Textbox(label="Tag replacement",placeholder="in the format of `source1,target1;source2,target2;...`")

                    with gr.Column(min_width=240, visible=False) as pixai_settings:
                        with gr.Group():
                            gr.Markdown("<center>PixAI Tagger Settings</center>")
                        with gr.Column(visible=False) as pixai_advanced_settings:
                            pixai_general_threshold = gr.Slider(label="General threshold",minimum=0.0,maximum=1.0,value=0.17,step=0.01,interactive=True)
                            pixai_style_threshold = gr.Slider(label="Style threshold",minimum=0.0,maximum=1.0,value=0.15,step=0.01,interactive=True)
                            pixai_copyright_threshold = gr.Slider(label="Copyright threshold",minimum=0.0,maximum=1.0,value=0.24,step=0.01,interactive=True)
                            pixai_meta_threshold = gr.Slider(label="Meta threshold",minimum=0.0,maximum=1.0,value=0.17,step=0.01,interactive=True)
                            pixai_rating_threshold = gr.Slider(label="Rating threshold",minimum=0.0,maximum=1.0,value=0.41,step=0.01,interactive=True)
                        with gr.Column(visible=False) as pixai_simple_settings:
                            pixai_threshold = gr.Slider(label="General / Style threshold",minimum=0.0,maximum=1.0,value=0.17,step=0.01, interactive=True)
                            pixai_trailing_comma = gr.Checkbox(label="Add trailing comma",value=False,interactive=True)

                        pixai_character_threshold = gr.Slider(label="Character threshold",minimum=0.0,maximum=1.0,value=0.27,step=0.01,interactive=True)
                        pixai_replace_underscore = gr.Checkbox(label="Replace underscores with spaces",value=False,interactive=True)
                        pixai_exclude_tags = gr.Textbox(label="Tags to exclude",placeholder="comma-separated list of tags",interactive=True)

                    with gr.Column(min_width=240,visible=True,) as tagger_common_settings:
                        with gr.Group():
                            gr.Markdown("<center>Common Tagger Settings</center>")
                        wd_caption_extension = gr.Textbox(label="Extension for tag captions files",value=".wdcaption")
                        wd_caption_separator = gr.Textbox(label="Separator for tags",value=", ")

                with gr.Column(min_width=240) as llm_settings:
                    with gr.Group():
                        gr.Markdown("<center>LLM Settings</center>")

                    llm_caption_extension = gr.Textbox(label="extension of LLM caption file",value=".llmcaption")
                    llm_read_wd_caption = gr.Checkbox(label="llm will read wd caption for inference")
                    llm_caption_without_wd = gr.Checkbox(label="llm will not read wd caption for inference")

                    with gr.Accordion(label="Joy Formated Prompts",open=False) as joy_formated_prompts:
                        caption_type = gr.Dropdown(label="Caption Type",
                            choices=[
                                "Descriptive",
                                "Descriptive (Informal)",
                                "Training Prompt",
                                "MidJourney",
                                "Booru tag list",
                                "Booru-like tag list",
                                "Art Critic",
                                "Product Listing",
                                "Social Media Post",
                            ],
                            value="Descriptive"
                        )

                        caption_length = gr.Dropdown(label="Caption Length",
                            choices=[
                                "any",
                                "very short",
                                "short",
                                "medium-length",
                                "long",
                                "very long",
                            ] + [str(i) for i in range(20, 261, 10)],
                            value="long"
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
                                    "If there is a person/character in the image you must refer to them as {name}.",
                                ]
                            )
                            name_input = gr.Textbox(label="Person/Character Name (if applicable)")

                        generate_prompt_button = gr.Button(value="Generate prompts",variant="primary")

                    florence_system_prompt = gr.Dropdown(choices=list(SINGLE_TASKS),value="More Detailed Caption",label="Florence2 system prompt",visible=False,interactive=True)
                    florence_user_prompt = gr.Textbox(label="Florence user prompt",placeholder="Required for grounding/region tasks",visible=False,interactive=True)
                    llm_system_prompt = gr.Textbox(label="system prompt for llm caption",lines=7,max_lines=7,value=caption.DEFAULT_SYSTEM_PROMPT)
                    llm_user_prompt = gr.Textbox(label="user prompt for llm caption",lines=7,max_lines=7,value=caption.DEFAULT_USER_PROMPT_WITH_WD)

                    with gr.Accordion(label="Advanced Options", open=False):
                        llm_temperature = gr.Slider(label="temperature for LLM model",minimum=0,maximum=1.0,value=0,step=0.1)
                        llm_max_tokens = gr.Slider(label="max token for LLM model",minimum=0,maximum=2048,value=0,step=1)
                        image_size = gr.Slider(label="Resize image for inference",minimum=256,maximum=2048,value=1024,step=1)
                        auto_unload = gr.Checkbox(label="Auto Unload Models after inference.",visible=False)


        with gr.Column():
            with gr.Tab("Single mode"):
                with gr.Column():
                    input_image = gr.Image(elem_id="input_image", type='filepath', label="Upload Image")
                    single_image_submit_button = gr.Button(elem_id="single_image_submit_button", value="Inference", variant='primary',interactive=False)

                with gr.Column():
                    wd_tags_output = gr.Text(label='WD Tags Output', lines=10, interactive=False, show_label=True)
                    llm_caption_output = gr.Text(label='LLM Caption Output', lines=10, interactive=False, show_label=True)
                    florence_image = gr.Image(label="Florence Visualization", type="pil",visible=False)
            with gr.Tab("Batch mode") as bs_mode:
                ext_dir=gr.Textbox(value='batch_caption',visible=False)
                with gr.Column(min_width=240):
                    with gr.Row():
                        file_zip=gr.File(label="Upload a ZIP file",file_count='single',file_types=['.zip'],visible=False,height=260,interactive=True)
                        files_single = gr.Files(label="Drag (Select) 1 or more reference images",file_count="multiple",
                                            file_types=["image"],visible=True,interactive=True,height=260)
                        preview=gr.Image(label="Process preview",visible=False,height=260,interactive=False)
                        file_out=gr.File(label="Download a ZIP file", file_count='single',height=260,visible=True)
                        enable_zip = gr.Checkbox(label="Upload ZIP-file", value=False)
                        input_dir = gr.Textbox(value=f"{temp_dir}batch_temp", visible=False)
                        is_recursive = gr.Checkbox(visible=False)
                    custom_caption_save_path = gr.Textbox(value=f"{temp_dir}batch_caption",visible=False)
                    with gr.Row(equal_height=True):
                        run_method = gr.Radio(label="Run method", choices=['sync', 'queue'], value="sync", interactive=True)

                        with gr.Column(min_width=240, visible=False):
                            skip_exists = gr.Checkbox()
                            not_overwrite = gr.Checkbox()
                    with gr.Column(min_width=240):
                        caption_extension = gr.Textbox(label="Caption file extension", value=".txt")
                        save_caption_together = gr.Checkbox(label="Save WD and LLM captions in one file", value=True)
                        save_caption_together_seperator = gr.Textbox(label="Seperator between WD tags and LLM captions", value="|")

                    enable_zip.change(fn=zip_enable,inputs=[enable_zip],outputs=[file_zip,files_single],show_progress=False)
                    batch_process_submit_button = gr.Button(elem_id="batch_process_submit_button", value="Batch Process", variant='primary',interactive=False)
    with gr.Row():
        gr.HTML('* Based by fireicewolf <a href="https://github.com/fireicewolf/wd-llm-caption-cli" target="_blank">\U0001F4D4 Document</a> and gokaygokay. <a href="https://huggingface.co/spaces/gokaygokay/Florence-2" target="_blank">\U0001F4D4 Document</a>')
    def caption_method_update_visibility(caption_method_radio, llm_choice):
        caption_method_radio = caption_method_radio or ""
        tagger_enabled = "WD" in caption_method_radio
        llm_enabled = "LLM" in caption_method_radio

        run_method_visible = gr.update(visible=caption_method_radio == "WD+LLM")
        wd_model_visible = gr.update(visible=tagger_enabled)
        wd_force_use_cpu_visible = gr.update(visible=tagger_enabled)
        llm_use_cpu_visible = gr.update(visible=llm_enabled)
        llm_load_settings_visible = gr.update(visible=llm_enabled)
        llm_settings_visible = gr.update(visible=llm_enabled)
        wd_tags_output_visible = gr.update(visible=tagger_enabled)
        llm_caption_output_visible = gr.update(visible=llm_enabled)
        florence_image_visible = gr.update(
            visible=llm_enabled and llm_choice == "Florence"
        )

        return (
            run_method_visible,
            wd_model_visible,
            wd_force_use_cpu_visible,
            llm_use_cpu_visible,
            llm_load_settings_visible,
            llm_settings_visible,
            wd_tags_output_visible,
            llm_caption_output_visible,
            florence_image_visible,
        )

    def selected_tagger_visibility(caption_method_value, model_name):
        caption_method_value = caption_method_value or ""
        tagger_enabled = "WD" in caption_method_value

        profile = TAGGER_MODEL_CONFIG.get(str(model_name), {})
        is_pixai = profile.get("tagger_type") == "pixai"
        mode = str(profile.get("pixai_mode", "simple")).lower()
        defaults = profile.get("pixai_defaults", {})

        return (
            gr.update(visible=tagger_enabled),
            gr.update(visible=tagger_enabled and not is_pixai),
            gr.update(visible=tagger_enabled and is_pixai),
            gr.update(visible=tagger_enabled and is_pixai and mode == "simple"),
            gr.update(visible=tagger_enabled and is_pixai and mode == "advanced"),
            gr.update(visible=tagger_enabled)
        )

    tagger_profile_outputs = [
        tagger_settings,
        wd_settings,
        pixai_settings,
        pixai_simple_settings,
        pixai_advanced_settings,
        tagger_common_settings
    ]

    def llm_choice_update_visibility(caption_method_radio, llm_choice_radio, joy_models_dropdown):
        joy_model_visible = gr.update(visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Joy" else False)
        llama_use_patch_visible = gr.update(visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Joy" and joy_models_dropdown == "Joy-Caption-Alpha-Two" or joy_models_dropdown == "Joy-Caption-Pre-Alpha" else False)
        qwen_model_visible = gr.update(visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Qwen" else False)
        florence_model_visible = gr.update(visible=True if "LLM" in caption_method_radio and llm_choice_radio == "Florence" else False)
        prompt_visible = gr.update(visible=False if "LLM" in caption_method_radio and llm_choice_radio == "Florence" else True)
        return joy_model_visible, llama_use_patch_visible, qwen_model_visible, florence_model_visible, florence_model_visible, florence_model_visible, prompt_visible, prompt_visible,florence_model_visible
        
    def joy_formated_prompts_visibility(llm_choice_radio, joy_models_dropdown):
        joy_formated_prompts_visible = gr.update(visible=True if llm_choice_radio == "Joy" and joy_models_dropdown != "Joy-Caption-Pre-Alpha" else False)
        extra_options_visible = gr.update(visible=True if llm_choice_radio == "Joy" and joy_models_dropdown in ["Joy-Caption-Alpha-Two-Llava", "Joy-Caption-Alpha-Two"] else False)
        return joy_formated_prompts_visible, extra_options_visible

    caption_method.change(fn=caption_method_update_visibility,inputs=[caption_method, llm_choice],
        outputs=[run_method,wd_models,wd_force_use_cpu,llm_use_cpu,llm_load_settings,llm_settings,wd_tags_output,llm_caption_output,florence_image,])
    caption_method.change(fn=selected_tagger_visibility,inputs=[caption_method, wd_models],outputs=tagger_profile_outputs)
    wd_models.change(fn=selected_tagger_visibility,inputs=[caption_method, wd_models],outputs=tagger_profile_outputs)
    caption_method.change(fn=llm_choice_update_visibility, inputs=[caption_method, llm_choice, joy_models], outputs=[joy_models, llm_use_patch, qwen_models, florence_models, florence_system_prompt, florence_user_prompt, llm_system_prompt, llm_user_prompt, florence_image])

    llm_choice.change(fn=llm_choice_update_visibility, inputs=[caption_method, llm_choice, joy_models], outputs=[joy_models, llm_use_patch, qwen_models, florence_models, florence_system_prompt, florence_user_prompt, llm_system_prompt, llm_user_prompt, florence_image])    
    llm_choice.change(fn=joy_formated_prompts_visibility, inputs=[llm_choice, joy_models], outputs=[joy_formated_prompts, extra_options_column])
    joy_models.change(fn=llm_choice_update_visibility, inputs=[caption_method, llm_choice, joy_models], outputs=[joy_models, llm_use_patch, qwen_models, florence_models, florence_system_prompt, florence_user_prompt, llm_system_prompt, llm_user_prompt, florence_image])
    joy_models.change(fn=joy_formated_prompts_visibility, inputs=[llm_choice, joy_models], outputs=[joy_formated_prompts, extra_options_column])

    def llm_user_prompt_default(caption_method_radio, llm_read_wd_caption_select, llm_user_prompt_textbox):
        if caption_method_radio != "WD+LLM" and llm_user_prompt_textbox == inference.DEFAULT_USER_PROMPT_WITH_WD:
            llm_user_prompt_change = gr.update(value=inference.DEFAULT_USER_PROMPT_WITHOUT_WD)
        elif caption_method_radio == "WD+LLM" and llm_user_prompt_textbox == inference.DEFAULT_USER_PROMPT_WITHOUT_WD:
            llm_user_prompt_change = gr.update(value=inference.DEFAULT_USER_PROMPT_WITH_WD)
        elif caption_method_radio == "LLM" and llm_read_wd_caption_select:
            llm_user_prompt_change = gr.update(value=inference.DEFAULT_USER_PROMPT_WITHOUT_WD)
        else:
            llm_user_prompt_change = gr.update(value=llm_user_prompt_textbox)
        return llm_user_prompt_change

    def build_joy_user_prompt(caption_method_value: str, joy_models_value: str, llm_read_wd_caption_value: bool, caption_type_value: str, caption_length_value: str, extra_options_value: list[str], name_input_value: str):
        caption_type_map = {
            "Descriptive": ["Write a descriptive caption for this image in a formal tone.", "Write a descriptive caption for this image in a formal tone within {word_count} words.", "Write a {length} descriptive caption for this image in a formal tone."],
            "Descriptive (Informal)": ["Write a descriptive caption for this image in a casual tone.", "Write a descriptive caption for this image in a casual tone within {word_count} words.", "Write a {length} descriptive caption for this image in a casual tone."],
            "Training Prompt": ["Write a stable diffusion prompt for this image.", "Write a stable diffusion prompt for this image within {word_count} words.", "Write a {length} stable diffusion prompt for this image."],
            "MidJourney": ["Write a MidJourney prompt for this image.", "Write a MidJourney prompt for this image within {word_count} words.", "Write a {length} MidJourney prompt for this image."],
            "Booru tag list": ["Write a list of Booru tags for this image.", "Write a list of Booru tags for this image within {word_count} words.", "Write a {length} list of Booru tags for this image."],
            "Booru-like tag list": ["Write a list of Booru-like tags for this image.", "Write a list of Booru-like tags for this image within {word_count} words.", "Write a {length} list of Booru-like tags for this image."],
            "Art Critic": ["Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.", "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.", "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}."],
            "Product Listing": ["Write a caption for this image as though it were a product listing.", "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.", "Write a {length} caption for this image as though it were a product listing."],
            "Social Media Post": ["Write a caption for this image as if it were being used for a social media post.", "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.", "Write a {length} caption for this image as if it were being used for a social media post."],
        }

        length = None if caption_length_value == "any" else caption_length_value
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

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
        if joy_models_value in ["Joy-Caption-Alpha-Two-Llava", "Joy-Caption-Alpha-Two"] and len(extra_options_value) > 0:
            prompt_str += " " + " ".join(extra_options_value)
            
        system_prompt = "You are a helpful image captioner."
        user_prompt_str = prompt_str.format(wd_tags="{wd_tags}", name=name_input_value, length=caption_length_value, word_count=caption_length_value)
        return system_prompt, user_prompt_str.strip()

    caption_method.change(fn=llm_user_prompt_default, inputs=[caption_method, llm_read_wd_caption, llm_user_prompt], outputs=llm_user_prompt)
    generate_prompt_button.click(fn=build_joy_user_prompt, inputs=[caption_method, joy_models, llm_read_wd_caption, caption_type, caption_length, extra_options, name_input], outputs=[llm_system_prompt, llm_user_prompt])

    def use_wd(check_caption_method):
        return True if check_caption_method in ["wd", "wd+llm"] else False

    def use_joy(check_caption_method, check_llm_choice):
        return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "joy" else False

    def use_qwen(check_caption_method, check_llm_choice):
        return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "qwen" else False

    def use_florence(check_caption_method, check_llm_choice):
        return True if check_caption_method in ["llm", "wd+llm"] and check_llm_choice == "florence" else False

    def load_models_interactive_group():
        return [gr.update(interactive=False)] * 11 + [gr.update(variant='secondary'), gr.update(variant='primary')]

    def unloads_models_interactive_group():
        return [gr.update(interactive=True)] * 11 + [gr.update(variant='primary'), gr.update(variant='secondary')]
        
    single_inference_input_args = [
        wd_remove_underscore, wd_threshold, wd_general_threshold, wd_character_threshold,
        wd_add_rating_tags_to_first, wd_character_tags_first, wd_add_rating_tags_to_last, wd_character_tag_expand,
        wd_undesired_tags, wd_always_first_tags, wd_caption_extension, wd_caption_separator, wd_tag_replacement,
        llm_caption_extension, llm_read_wd_caption, llm_caption_without_wd,
        pixai_threshold,pixai_character_threshold,pixai_general_threshold,pixai_style_threshold,pixai_copyright_threshold,
        pixai_meta_threshold,pixai_rating_threshold,pixai_replace_underscore,pixai_trailing_comma,pixai_exclude_tags,
        florence_system_prompt, florence_user_prompt,
        llm_system_prompt, llm_user_prompt,
        llm_temperature, llm_max_tokens, image_size, auto_unload, input_image,
    ]
    batch_inference_input_args = [
        batch_process_submit_button, run_method, wd_remove_underscore, wd_threshold, wd_general_threshold, wd_character_threshold,
        wd_add_rating_tags_to_first, wd_character_tags_first, wd_add_rating_tags_to_last, wd_character_tag_expand,
        wd_undesired_tags, wd_always_first_tags, wd_caption_extension, wd_caption_separator, wd_tag_replacement,
        llm_caption_extension, llm_read_wd_caption, llm_caption_without_wd, 
        pixai_threshold,pixai_character_threshold,pixai_general_threshold,pixai_style_threshold,pixai_copyright_threshold,
        pixai_meta_threshold,pixai_rating_threshold,pixai_replace_underscore,pixai_trailing_comma,pixai_exclude_tags,
        florence_system_prompt, florence_user_prompt,
        llm_system_prompt, llm_user_prompt,
        llm_temperature, llm_max_tokens, image_size, auto_unload, input_dir, is_recursive, custom_caption_save_path,
        skip_exists, not_overwrite, caption_extension, save_caption_together, save_caption_together_seperator
    ]


    def defragment_vram():
        if not torch.cuda.is_available():
            return    
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"\n[Caption] Warning during VRAM defragmentation: {e}")

    def unload_fooocus_completely():
        print(f"\n[Caption] === Unloading ALL Fooocus models ===")
        try:
            mm.unload_all_models()
            print(f"[Caption] ✓ Models properly unloaded from GPU")
        except Exception as e:
            print(f"[Caption] Warning during unload_all_models: {e}")

        if len(mm.current_loaded_models) > 0:
            for i in range(len(mm.current_loaded_models) - 1, -1, -1):
                try:
                    m = mm.current_loaded_models.pop(i)
                    m.model_unload()
                    del m
                except Exception as e:
                    print(f"[Caption] Warning: {e}")
        try:
            pipeline.final_unet = None
            pipeline.final_clip = None
            pipeline.final_vae = None
            pipeline.final_refiner_unet = None
            pipeline.final_refiner_vae = None
            pipeline.final_expansion = None
            pipeline.loaded_ControlNets = {}
        except Exception as e:
            print(f"[Caption] Warning: {e}")

        try:
            pipeline.model_base = core.StableDiffusionModel()
            pipeline.model_refiner = core.StableDiffusionModel()
        except Exception as e:
            print(f"[Omost] Warning: {e}")

        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            torch.cuda.synchronize()

        defragment_vram()

    def caption_models_load(
            caption_method_value, llm_choice_value,
            wd_model_value, joy_model_value, qwen_model_value, florence_model_value,
            wd_force_use_cpu_value, llm_use_cpu_value, llm_use_patch_value, llm_dtype_value, llm_qnt_value
    ):

        if (("LLM" in caption_method_value and not llm_use_cpu_value) or 
            ("WD" in caption_method_value and not wd_force_use_cpu_value)):
            unload_fooocus_completely()

        global IS_MODEL_LOAD, ARGS, CAPTION_FN

        if not IS_MODEL_LOAD:
            start_time = time.monotonic()

            if ARGS is None:
                ARGS = caption.CaptionConfig()
            
            config = ARGS
            config.models_save_path = str(os.path.join("models", "caption"))
            config.log_level = "INFO"
            config.caption_method = str(caption_method_value).lower()
            config.llm_choice = str(llm_choice_value).lower()

            if use_wd(config.caption_method):
                config.wd_config = WD_CONFIG
                config.wd_model_name = str(wd_model_value)
            if use_joy(config.caption_method, config.llm_choice):
                config.llm_config = JOY_CONFIG
                config.llm_model_name = str(joy_model_value)
            elif use_qwen(config.caption_method, config.llm_choice):
                config.llm_config = QWEN_CONFIG
                config.llm_model_name = str(qwen_model_value)
            elif use_florence(config.caption_method, config.llm_choice):
                config.llm_config = FLORENCE_CONFIG
                config.llm_model_name = str(florence_model_value)

            if CAPTION_FN is None:
                CAPTION_FN = caption.Caption()
                CAPTION_FN.set_logger(config)

            caption_init = CAPTION_FN
            config.wd_force_use_cpu = bool(wd_force_use_cpu_value)
            config.llm_use_cpu = bool(llm_use_cpu_value)
            config.llm_patch = bool(llm_use_patch_value)
            config.llm_dtype = str(llm_dtype_value)
            config.llm_qnt = str(llm_qnt_value)
            config.skip_download = SKIP_DOWNLOAD
            caption_init.download_models(config)
            caption_init.load_models(config)

            IS_MODEL_LOAD = True
            gr.Info(f"Models loaded in {time.monotonic() - start_time:.1f}s.")
            return load_models_interactive_group()
        else:
            config = ARGS
            if config.wd_model_name and not config.llm_model_name:
                warning = f"{config.wd_model_name}"
            elif not config.wd_model_name and config.llm_model_name:
                warning = f"{config.llm_model_name}"
            else:
                warning = f"{config.wd_model_name} & {config.llm_model_name}"
            gr.Warning(f"{warning} already loaded!")
            return unloads_models_interactive_group()

    def caption_single_inference(
            wd_remove_underscore_value, wd_threshold_value, wd_general_threshold_value, wd_character_threshold_value,
            wd_add_rating_tags_to_first_value, wd_character_tags_first_value, wd_add_rating_tags_to_last_value, wd_character_tag_expand_value,
            wd_undesired_tags_value, wd_always_first_tags_value, wd_caption_extension_value, wd_caption_separator_value, wd_tag_replacement_value,
            llm_caption_extension_value, llm_read_wd_caption_value, llm_caption_without_wd_value, 
            pixai_threshold_value,pixai_character_threshold_value,pixai_general_threshold_value,pixai_style_threshold_value,pixai_copyright_threshold_value,
            pixai_meta_threshold_value,pixai_rating_threshold_value,pixai_replace_underscore_value,pixai_trailing_comma_value,pixai_exclude_tags_value,            
            florence_system_prompt, florence_user_prompt,llm_system_prompt_value, llm_user_prompt_value,
            llm_temperature_value, llm_max_tokens_value, image_size_value, auto_unload_value, input_image_value
    ):

        florence_visualization = None

        if not IS_MODEL_LOAD:
            raise gr.Error("Models not loaded!")
            
        config = ARGS
        config.wd_remove_underscore = bool(wd_remove_underscore_value)
        config.wd_threshold = float(wd_threshold_value)
        config.wd_general_threshold = float(wd_general_threshold_value)
        config.wd_character_threshold = float(wd_character_threshold_value)
        config.wd_add_rating_tags_to_first = bool(wd_add_rating_tags_to_first_value)
        config.wd_add_rating_tags_to_last = bool(wd_add_rating_tags_to_last_value)
        config.wd_character_tags_first = bool(wd_character_tags_first_value)
        config.wd_character_tag_expand = bool(wd_character_tag_expand_value)
        config.wd_undesired_tags = str(wd_undesired_tags_value)
        config.wd_always_first_tags = str(wd_always_first_tags_value)
        config.wd_caption_extension = str(wd_caption_extension_value)
        config.wd_caption_separator = str(wd_caption_separator_value)
        config.wd_tag_replacement = str(wd_tag_replacement_value)

        config.pixai_threshold = float(pixai_threshold_value)
        config.pixai_character_threshold = float(pixai_character_threshold_value)
        config.pixai_general_threshold = float(pixai_general_threshold_value)
        config.pixai_style_threshold = float(pixai_style_threshold_value)
        config.pixai_copyright_threshold = float(pixai_copyright_threshold_value)
        config.pixai_meta_threshold = float(pixai_meta_threshold_value)
        config.pixai_rating_threshold = float(pixai_rating_threshold_value)
        config.pixai_replace_underscore = bool(pixai_replace_underscore_value)
        config.pixai_trailing_comma = bool(pixai_trailing_comma_value)
        config.pixai_exclude_tags = str(pixai_exclude_tags_value)

        config.llm_caption_extension = str(llm_caption_extension_value)
        config.llm_read_wd_caption = bool(llm_read_wd_caption_value)
        config.llm_caption_without_wd = bool(llm_caption_without_wd_value)
        config.llm_system_prompt = str(llm_system_prompt_value)
        config.llm_user_prompt = str(llm_user_prompt_value)
        config.llm_temperature = float(llm_temperature_value)
        config.llm_max_tokens = int(llm_max_tokens_value)
        config.image_size = int(image_size_value)
        config.data_path = str(input_image_value)

        config.florence_task = str(florence_system_prompt)
        config.florence_text_input = str(florence_user_prompt)

        start_time = time.monotonic()
        image = Image.open(input_image_value)
        tag_text = ""
        caption_text = ""
        CAPTION_FN.my_logger.debug(f"Input image: {config.data_path}.")
        
        if use_wd(config.caption_method):
            CAPTION_FN.my_logger.debug(f"Tagging with WD: {config.wd_model_name}.")
            tag_text, rating_tag_text, character_tag_text, general_tag_text = CAPTION_FN.my_tagger.get_tags(image=image)
            if rating_tag_text: CAPTION_FN.my_logger.debug(f"WD Rating tags: {rating_tag_text}")
            if character_tag_text: CAPTION_FN.my_logger.debug(f"WD Character tags: {character_tag_text}")
            CAPTION_FN.my_logger.debug(f"WD General tags: {general_tag_text}")
            CAPTION_FN.my_logger.info(f"WD tags content: {tag_text}")

        if use_florence(config.caption_method, config.llm_choice):
            caption_text, florence_visualization = CAPTION_FN.my_llm.get_florence_result(
                image=image,
                task_name=config.florence_task,
                text_input=config.florence_text_input,
                max_token=config.llm_max_tokens
            )
        elif use_joy(config.caption_method, config.llm_choice) or use_qwen(
            config.caption_method, config.llm_choice
        ):
            CAPTION_FN.my_logger.debug(f"Caption with LLM: {config.llm_model_name}.")
            caption_text = CAPTION_FN.my_llm.get_caption(
                image=image, system_prompt=str(config.llm_system_prompt),
                user_prompt=str(config.llm_user_prompt).format(wd_tags=tag_text) if tag_text else str(config.llm_user_prompt),
                temperature=config.llm_temperature, max_new_tokens=config.llm_max_tokens
            )
            CAPTION_FN.my_logger.info(f"LLM Caption content: {caption_text}")


        gr.Info(f"Inference end in {time.monotonic() - start_time:.1f}s.")
        CAPTION_FN.my_logger.info(f"Inference end in {time.monotonic() - start_time:.1f}s.")
        
        if auto_unload_value:
            caption_unload_models()
  
        return tag_text, caption_text, florence_visualization

    def caption_batch_inference(
            batch_process_submit_button_value, run_method_value, wd_remove_underscore_value, wd_threshold_value, wd_general_threshold_value,
            wd_character_threshold_value, wd_add_rating_tags_to_first_value, wd_character_tags_first_value, wd_add_rating_tags_to_last_value,
            wd_character_tag_expand_value, wd_undesired_tags_value, wd_always_first_tags_value, wd_caption_extension_value, wd_caption_separator_value,
            wd_tag_replacement_value, llm_caption_extension_value, llm_read_wd_caption_value, llm_caption_without_wd_value, 
            pixai_threshold_value,pixai_character_threshold_value,pixai_general_threshold_value,pixai_style_threshold_value,pixai_copyright_threshold_value,
            pixai_meta_threshold_value,pixai_rating_threshold_value,pixai_replace_underscore_value,pixai_trailing_comma_value,pixai_exclude_tags_value,
            florence_system_prompt_value,florence_user_prompt_value,
            llm_system_prompt_value,
            llm_user_prompt_value, llm_temperature_value, llm_max_tokens_value, image_size_value, auto_unload_value, input_dir_value,
            recursive_value, custom_caption_save_path_value, skip_exists_value, not_overwrite_value, caption_extension_value,
            save_caption_together_value, save_caption_together_seperator_value
    ):
        
        if not IS_MODEL_LOAD:
            raise gr.Error("Models not loaded!")
            
        config = ARGS
        if not input_dir_value:
            raise gr.Error("None input image/dir!")

        config.wd_remove_underscore = bool(wd_remove_underscore_value)
        config.wd_threshold = float(wd_threshold_value)
        config.wd_general_threshold = float(wd_general_threshold_value)
        config.wd_character_threshold = float(wd_character_threshold_value)
        config.wd_add_rating_tags_to_first = bool(wd_add_rating_tags_to_first_value)
        config.wd_add_rating_tags_to_last = bool(wd_add_rating_tags_to_last_value)
        config.wd_character_tags_first = bool(wd_character_tags_first_value)
        config.wd_character_tag_expand = bool(wd_character_tag_expand_value)
        config.wd_undesired_tags = str(wd_undesired_tags_value)
        config.wd_always_first_tags = str(wd_always_first_tags_value)
        config.wd_caption_extension = str(wd_caption_extension_value)
        config.wd_caption_separator = str(wd_caption_separator_value)
        config.wd_tag_replacement = str(wd_tag_replacement_value)

        config.pixai_threshold = float(pixai_threshold_value)
        config.pixai_character_threshold = float(pixai_character_threshold_value)
        config.pixai_general_threshold = float(pixai_general_threshold_value)
        config.pixai_style_threshold = float(pixai_style_threshold_value)
        config.pixai_copyright_threshold = float(pixai_copyright_threshold_value)
        config.pixai_meta_threshold = float(pixai_meta_threshold_value)
        config.pixai_rating_threshold = float(pixai_rating_threshold_value)
        config.pixai_replace_underscore = bool(pixai_replace_underscore_value)
        config.pixai_trailing_comma = bool(pixai_trailing_comma_value)
        config.pixai_exclude_tags = str(pixai_exclude_tags_value)

        config.llm_caption_extension = str(llm_caption_extension_value)
        config.llm_read_wd_caption = bool(llm_read_wd_caption_value)
        config.llm_caption_without_wd = bool(llm_caption_without_wd_value)
        config.llm_system_prompt = str(llm_system_prompt_value)
        config.llm_user_prompt = str(llm_user_prompt_value)
        config.llm_temperature = float(llm_temperature_value)
        config.llm_max_tokens = int(llm_max_tokens_value)
        config.image_size = int(image_size_value)

        config.florence_task = str(florence_system_prompt_value)
        config.florence_text_input = str(florence_user_prompt_value)

        config.data_path = str(input_dir_value)
        config.run_method = str(run_method_value)
        config.recursive = bool(recursive_value)
        config.custom_caption_save_path = str(custom_caption_save_path_value)
        config.skip_exists = bool(skip_exists_value)
        config.not_overwrite = bool(not_overwrite_value)
        config.caption_extension = str(caption_extension_value)
        config.save_caption_together = bool(save_caption_together_value)
        config.save_caption_together_seperator = str(save_caption_together_seperator_value)

        if config.data_path and not os.path.exists(config.data_path):
            raise gr.Error(f"{config.data_path} NOT FOUND!!!")
        if config.custom_caption_save_path and not os.path.exists(config.custom_caption_save_path):
            raise gr.Error(f"{config.custom_caption_save_path} NOT FOUND!!!")

        start_time = time.monotonic()

        for image_number, total_images, image_path in CAPTION_FN.iter_inference(config):
            filename = os.path.basename(os.fspath(image_path))

            gr.Info(
                f"Caption Batch: start element generation "
                f"{image_number}/{total_images}. "
                f"Filename: {filename}"
            )

            yield gr.update(value=image_path)

        if auto_unload_value:
            caption_unload_models()

        gr.Info(
            f"Caption Batch: completed in "
            f"{time.monotonic() - start_time:.1f}s."
        )

        yield gr.update(value=None)


    def caption_unload_models():
        global IS_MODEL_LOAD

        if IS_MODEL_LOAD:
            CAPTION_FN.unload_models()
            IS_MODEL_LOAD = False
            gr.Info("Models unloaded successfully.")
        else:
            gr.Warning("Models not loaded!")


        return unloads_models_interactive_group()

    load_model_button.click(
        fn=caption_models_load,
        inputs=[caption_method, llm_choice, wd_models, joy_models, qwen_models, florence_models, wd_force_use_cpu, llm_use_cpu, llm_use_patch, llm_dtype, llm_qnt],
        outputs=[caption_method, llm_choice, wd_models, joy_models, qwen_models, florence_models, wd_force_use_cpu, llm_use_cpu, llm_use_patch, llm_dtype, llm_qnt, load_model_button, unload_model_button]) \
        .then(lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),gr.update(interactive=False)),
        outputs=[unload_model_button, single_image_submit_button, batch_process_submit_button, load_model_button])

    unload_model_button.click(
        fn=caption_unload_models,
        outputs=[caption_method, llm_choice, wd_models, joy_models, qwen_models, florence_models, wd_force_use_cpu, llm_use_cpu, llm_use_patch, llm_dtype, llm_qnt, load_model_button, unload_model_button]) \
        .then(lambda: (gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False),gr.update(interactive=False)),
        outputs=[load_model_button, single_image_submit_button, batch_process_submit_button, unload_model_button]
    )

    single_image_submit_button.click(lambda: (gr.update(interactive=False)),outputs=[single_image_submit_button]) \
        .then(fn=caption_single_inference,inputs=single_inference_input_args,outputs=[wd_tags_output, llm_caption_output, florence_image]) \
        .then(lambda: (gr.update(interactive=True)),outputs=[single_image_submit_button])

    batch_process_submit_button.click(lambda: (gr.update(interactive=False),gr.update(visible=False), gr.update(visible=True)),outputs=[batch_process_submit_button,file_out,preview]) \
        .then(fn=clear_dirs,inputs=ext_dir) \
        .then(fn=unzip_file,inputs=[file_zip,files_single,enable_zip]) \
        .then(fn=caption_batch_inference,inputs=batch_inference_input_args,outputs=preview,show_progress=False) \
        .then(fn=output_zip,outputs=file_out) \
        .then(lambda: (gr.update(interactive=True),gr.update(visible=True), gr.update(visible=False)),outputs=[batch_process_submit_button,file_out,preview])
