import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
import requests
import launch
import re
import urllib.request
import zipfile
import threading
import math
import numpy as np
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from extentions.module_translate import translate, GoogleTranslator
from urllib.parse import urlparse, parse_qs, unquote
from modules.model_loader import load_file_from_url
from rembg import remove
from PIL import Image
from gradio.components import label
from modules.util import is_json

from extentions.md_lib import civitai_helper
from extentions.md_lib import md_config

from extentions import wildcards

from extentions.obp.scripts import onebuttonprompt as ob_prompt

from extentions.op_edit.body import Body
from pathlib import Path
import io
import cv2

obp_prompt=[]


choices_ar1=["Any", "1:1", "3:2", "4:3", "4:5", "16:9"]
choices_ar2=["Any", "1:1", "2:3", "3:4", "5:4", "9:16"]
modules.config.available_aspect_ratios_labels=[item.replace("<span style=\"color: grey;\">", "").replace("</span>", "") for item in modules.config.available_aspect_ratios_labels]
modules.config.available_aspect_ratios_labels+=[f'512×512  ∣ 1:1']
modules.config.default_aspect_ratio=modules.config.default_aspect_ratio.replace("<span style=\"color: grey;\">", "").replace("</span>", "")


ar_def=[1,1]
swap_def=False
cell_index='0'

def queue_obp(*args):
    global finished_batch
    finished_batch=False
    args = list(args)
    seed_random = args.pop()
    presetsuffix=args.pop()
    presetprefix=args.pop()
    promptenhancer=args.pop()
    amountoffluff=args.pop()
    OBP_preset=args.pop()

    base_model_obp=args.pop()
    autonegativepromptenhance=args.pop()
    autonegativepromptstrength=args.pop()
    autonegativeprompt=args.pop()
    givenoutfit=args.pop()
    promptvariantinsanitylevel=args.pop()
    chosensubjectsubtypeconcept=args.pop()
    chosensubjectsubtypehumanoid=args.pop()
    chosensubjectsubtypeobject=args.pop()
    gender=args.pop()
    imagemodechance=args.pop()
    giventypeofimage=args.pop()

    smartsubject=args.pop()
    givensubject=args.pop()
    seperator=args.pop()
    promptcompounderlevel=args.pop()
    negative_prompt=args.pop()
    suffixprompt=args.pop()
    prefixprompt=args.pop()
    antistring=args.pop()
    workprompt=args.pop()
    silentmode=args.pop()
    imagetype=args.pop()
    artist=args.pop()
    subject=args.pop()
    insanitylevel=args.pop()
    amountofimages=args.pop()
    aspect_ratios_selection=args.pop()
    size=args.pop()
    base_model=args.pop()
    model=args.pop()
    
    sect_models=ob_prompt.modellist
    cur_sect_models=model
    model_list=modules.config.model_filenames
    cur_model=base_model

    sect_ratios=ob_prompt.sizelist
    cur_sect_ratios=size
    ratio_list=['768×1344 ∣ 4:7','896×1152 ∣ 7:9','1024×1024 ∣ 1:1','1152×896 ∣ 9:7','1344×768 ∣ 7:4']
    cur_ratio=aspect_ratios_selection

    originalnegativeprompt = negative_prompt
    prompts=amountofimages

    if(silentmode==True and workprompt != ""):
        randomprompt = ob_prompt.createpromptvariant(workprompt, promptvariantinsanitylevel)
        print("Using provided workflow prompt")
        print(randomprompt)
    else:    
            randompromptlist = ob_prompt.build_dynamic_prompt(insanitylevel,subject,artist,imagetype, False,antistring,prefixprompt,suffixprompt,promptcompounderlevel, seperator,givensubject,smartsubject,giventypeofimage,imagemodechance, gender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept,True,False,-1,givenoutfit, prompt_g_and_l=True, base_model_obp=base_model_obp, OBP_preset=OBP_preset, prompt_enhancer=promptenhancer, preset_prefix=presetprefix, preset_suffix=presetsuffix)
            randomprompt = randompromptlist[0]
            randomsubject = randompromptlist[1]
    negativeprompt=negative_prompt
    if(autonegativeprompt):
            negativeprompt = ob_prompt.build_dynamic_negative(positive_prompt=randomprompt, insanitylevel=autonegativepromptstrength,enhance=autonegativepromptenhance, existing_negative_prompt=originalnegativeprompt, base_model_obp=base_model_obp)
    randomprompt = ob_prompt.flufferizer(prompt=randomprompt, amountoffluff=amountoffluff)




    for generation in range(int(prompts)):
      if not finished_batch:
        if(silentmode==True and workprompt != ""):
            randomprompt = ob_prompt.createpromptvariant(workprompt, promptvariantinsanitylevel)
            print("Using provided workflow prompt")
            print(randomprompt)
        else:    
                randompromptlist = ob_prompt.build_dynamic_prompt(insanitylevel,subject,artist,imagetype, False,antistring,prefixprompt,suffixprompt,promptcompounderlevel, seperator,givensubject,smartsubject,giventypeofimage,imagemodechance, gender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept,True,False,-1,givenoutfit, prompt_g_and_l=True, base_model_obp=base_model_obp, OBP_preset=OBP_preset, prompt_enhancer=promptenhancer, preset_prefix=presetprefix, preset_suffix=presetsuffix)
                randomprompt = randompromptlist[0]
                randomsubject = randompromptlist[1]
        negativeprompt=negative_prompt
        if(autonegativeprompt):
                negativeprompt = ob_prompt.build_dynamic_negative(positive_prompt=randomprompt, insanitylevel=autonegativepromptstrength,enhance=autonegativepromptenhance, existing_negative_prompt=originalnegativeprompt, base_model_obp=base_model_obp)
        randomprompt = ob_prompt.flufferizer(prompt=randomprompt, amountoffluff=amountoffluff)

        if cur_sect_models=='all':
          model_list_gen=len(model_list)
        else:
          model_list_gen=1
        for gen_models in range(model_list_gen):
          if cur_sect_models=='random model':
              randnum=random.randint(0, len(model_list)-1)
              name_model=model_list[randnum]
          elif cur_sect_models=='all':
              name_model=model_list[gen_models]
          else:
              name_model=cur_model
          if cur_sect_ratios=='all':
              ratio_lis_gen=len(ratio_list)
          else:
              ratio_lis_gen=1
          for gen_ratio in range(ratio_lis_gen):
              if cur_sect_ratios=='random ratio':
                  randnum=random.randint(0, len(ratio_list)-1)
                  name_ratio=ratio_list[randnum]
              elif cur_sect_ratios=='all':
                  name_ratio=ratio_list[gen_ratio]
              elif cur_sect_ratios=='current ratio':
                  name_ratio=cur_ratio
              else:
                  name_ratio=ratio_list[(sect_ratios.index(cur_sect_ratios)-2)]
              args[2]=randomprompt
              args[3]=negativeprompt
              args[13]=name_model
              args[6]=name_ratio
              currentTask=get_task_batch(args)
              yield from generate_clicked(currentTask)
              if seed_random:
                  args[9]=int (random.randint(constants.MIN_SEED, constants.MAX_SEED))



def civitai_helper_nsfw(black_out_nsfw):
  md_config.ch_nsfw_threshold=black_out_nsfw
  return
civitai_helper_nsfw(modules.config.default_black_out_nsfw)
def get_task(*args):
    argsList = list(args)
    toT = argsList.pop() 
    srT = argsList.pop() 
    trans_automate = argsList.pop() 
    trans_enable = argsList.pop() 

    if trans_enable:      
        if trans_automate:
            positive, negative = translate(argsList[2], argsList[3], srT, toT)            
            argsList[2] = positive
            argsList[3] = negative
            
    args = tuple(argsList)
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)
finished_batch=False
batch_path='./batch_images'

def unzip_file(zip_file_obj):
    extract_folder = "./batch_images"
    if not os.path.exists(extract_folder):
      os.makedirs(extract_folder)    
    zip_ref=zipfile.ZipFile(zip_file_obj.name, 'r')
    zip_ref.extractall(extract_folder)
    zip_ref.close()
    return
def output_zip():
  directory=modules.config.path_outputs
  zip_file='outputs.zip'
  with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, directory))
  zipf.close()
  current_dir = os.getcwd()
  file_path = os.path.join(current_dir, "outputs.zip")
  return file_path

def stop_clicked_batch():
    global finished_batch
    finished_batch=True
    return
 
def delete_out(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_out(file_path)
                os.rmdir(file_path)
        except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    return
def clear_outputs():
  directory=modules.config.path_outputs
  delete_out(directory)
  return 
def clearer():
  directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batch_images')
  delete_out(directory)
  return      
def queue_new(*args):
    global finished_batch
    global cell_index
    index = int(cell_index)*4
    finished_batch=False 
    args = list(args)
    seed_random = args.pop()
    scale=args.pop()    
    lora_args=3*(int(modules.config.default_max_lora_number))
    ip_cell=66+lora_args+index
    
    batch_files=sorted([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    batch_all=len(batch_files)
    passed=1
    for f_name in batch_files:
      if not finished_batch:  
        copy = args[:]
        img = Image.open('./batch_images/'+f_name)
        if args[(16+lora_args)]:
            if (args[(17+lora_args)] == 'uov'): 
              args[(19+lora_args)]=np.array(img)
            if (args[(17+lora_args)] == 'ip'):
                  width, height = img.size
                  if scale=="to ORIGINAL":
                      aspect = math.gcd(width, height)
                      args[6]=f'{width}×{height} <span style="color: grey;"> ∣ {width // aspect}:{height // aspect}</span>'
                  if scale=="to OUTPUT":
                      new_width, new_height = args[6].replace('×', ' ').split(' ')[:2]
                      new_width = int(new_width)
                      new_height = int(new_height)
                      ratio = min(float(new_width) / width, float(new_height) / height)
                      w = int(width * ratio)
                      h = int(height * ratio)
                      img = img.resize((w, h), Image.LANCZOS)
                  args[ip_cell]=np.array(img)
        print (f"[Images QUEUE] {passed} / {batch_all}. Filename: {f_name}")
        passed+=1
        currentTask=get_task_batch(args)
        yield from generate_clicked(currentTask)
        args=copy[:]
        if seed_random:
          args[9]=int (random.randint(constants.MIN_SEED, constants.MAX_SEED))
    clearer()
    return
def queue_new_prompt(*args):
  global finished_batch
  finished_batch=False
  args = list(args)
  base_neg_token = args.pop()
  base_pos_token = args.pop()
  seed_random = args.pop()
  batch_args = args.pop()
  batch_args.reverse()
  copy = args[:]
  passed=1
  while batch_args and not finished_batch:
      print (f"[Prompts QUEUE] Element #{passed}")
      one_batch_args=batch_args.pop()
      if base_pos_token=='Prefix':
        args[2]= args[2] + one_batch_args[0]
      elif base_pos_token=='Suffix':
        args[2]= one_batch_args[0] + args[2]
      else:
        args[2]=one_batch_args[0]
      if base_neg_token=='Prefix':
        args[3]= args[3] + one_batch_args[1]
      elif base_neg_token=='Suffix':
        args[3]= one_batch_args[1] + args[3]
      else:
        args[3]=one_batch_args[1]
      currentTask=get_task_batch(args)
      yield from generate_clicked(currentTask)
      args=copy[:]
      if seed_random:
        args[9]=int (random.randint(constants.MIN_SEED, constants.MAX_SEED))
      passed+=1
  return 
def prompt_clearer(batch_prompt):
  batch_prompt=[{'prompt': '', 'negative prompt': ''}]
  return batch_prompt

def get_task_batch(*args):
    argsList = list(args[0])
    toT = argsList.pop() 
    srT = argsList.pop() 
    trans_automate = argsList.pop() 
    trans_enable = argsList.pop() 
    if trans_enable:      
        if trans_automate:
            positive, negative = translate(argsList[2], argsList[3], srT, toT)            
            argsList[2] = positive
            argsList[3] = negative            
    args = tuple(argsList)
    args = list(args)
    args.pop(0)
    return worker.AsyncTask(args=args)

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]


reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title).queue()

with shared.gradio_root:
    state_topbar = gr.State({})
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=768,
                                            elem_classes=['main_view'])
                progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                              height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery')
            with gr.Row():
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        autofocus=True, lines=3)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    reset_button = gr.Button(label="Reconnect", value="Reconnect", elem_classes='type_row', elem_id='reset_button', visible=False)
                    load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                    stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'stop'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    def skip_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'skip'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
                    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
            with gr.Accordion(label='Wildcards', visible=True, open=False) as prompt_wildcards:
                wildcards_list = gr.Dataset(components=[prompt], label='Wildcards:', samples=wildcards.get_wildcards_samples(), visible=True, samples_per_page=14)
                with gr.Accordion(label='Words/phrases of wildcard', visible=True, open=False) as words_in_wildcard:
                    wildcard_tag_name_selection = gr.Dataset(components=[prompt], label='Words/phrases:', samples=wildcards.get_words_of_wildcard_samples(), visible=True, samples_per_page=30, type='index')
                wildcards_list.click(wildcards.add_wildcards_and_array_to_prompt, inputs=[wildcards_list, prompt, state_topbar], outputs=[prompt, wildcard_tag_name_selection, words_in_wildcard], show_progress=False, queue=False)
                wildcard_tag_name_selection.click(wildcards.add_word_to_prompt, inputs=[wildcards_list, wildcard_tag_name_selection, prompt], outputs=prompt, show_progress=False, queue=False)
                wildcards_array = [prompt_wildcards, words_in_wildcard, wildcards_list, wildcard_tag_name_selection]
                wildcards_array_show =lambda x: [gr.update(visible=True)] * 2 + [gr.Dataset.update(visible=True, samples=wildcards.get_wildcards_samples()), gr.Dataset.update(visible=True, samples=wildcards.get_words_of_wildcard_samples(x["wildcard_in_wildcards"]))]
                wildcards_array_hidden = [gr.update(visible=False)] * 2 + [gr.Dataset.update(visible=False, samples=wildcards.get_wildcards_samples()), gr.Dataset.update(visible=False, samples=wildcards.get_words_of_wildcard_samples())]
                gr.HTML('* \"Wildcards\" is powered by SimpleSDXL. <a href="https://github.com/metercai/SimpleSDXL" target="_blank">\U0001F4D4 Document</a>')

            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')
                    with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='Advanced', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
                                inpaint_advanced_masking_checkbox = gr.Checkbox(label='Enable Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas')
                                invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox)
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_dino_prompt_text = gr.Textbox(label='Detection prompt', value='', visible=False, info='Use singular whenever possible', placeholder='Describe what you want to detect.')
                                example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                    samples=modules.config.example_enhance_detection_prompts,
                                    label='Detection Prompt Quick List',
                                    components=[inpaint_mask_dino_prompt_text],
                                    visible=modules.config.default_inpaint_mask_model == 'sam')
                                example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                            inputs=example_inpaint_mask_dino_prompt_text,
                                                                            outputs=inpaint_mask_dino_prompt_text,
                                                                            show_progress=False, queue=False)

                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                    inpaint_mask_sam_max_detections = gr.Slider(label="Maximum number of detections", info="Set to 0 to detect all", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                generate_mask_button = gr.Button(value='Generate mask from image')

                                def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    sam_options = None
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        sam_options = SAMOptions(
                                            dino_prompt=dino_prompt_text,
                                            dino_box_threshold=box_threshold,
                                            dino_text_threshold=text_threshold,
                                            dino_erode_or_dilate=dino_erode_or_dilate,
                                            dino_debug=dino_debug,
                                            max_detections=sam_max_detections,
                                            model_type=sam_model
                                        )

                                    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

                                    return mask


                                inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                    [gr.update(visible=x == 'sam')] * 2 +
                                                                    [gr.Dataset.update(visible=x == 'sam',
                                                                                       samples=modules.config.example_enhance_detection_prompts)],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category,
                                                                   inpaint_mask_dino_prompt_text,
                                                                   inpaint_mask_advanced_options,
                                                                   example_inpaint_mask_dino_prompt_text],
                                                          queue=False, show_progress=False)

                    with gr.Tab(label='Describe', id='describe_tab') as describe_tab:
                        with gr.Row():
                            with gr.Column():
                                describe_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                describe_methods = gr.CheckboxGroup(
                                    label='Content Type',
                                    choices=flags.describe_types,
                                    value=modules.config.default_describe_content_type)
                                describe_apply_styles = gr.Checkbox(label='Apply Styles', value=modules.config.default_describe_apply_prompts_checkbox)
                                describe_btn = gr.Button(value='Describe this Image into Prompt')
                                describe_image_size = gr.Textbox(label='Image Size and Recommended Size', elem_id='describe_image_size', visible=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Documentation</a>')

                                def trigger_show_image_properties(image):
                                    value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
                                    return gr.update(value=value, visible=True)

                                describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image,
                                                            outputs=describe_image_size, show_progress=False, queue=False)

                    with gr.Tab(label='Enhance', id='enhance_tab') as enhance_tab:
                        with gr.Row():
                            with gr.Column():
                                enhance_input_image = grh.Image(label='Use with Enhance, skips image generation', source='upload', type='numpy')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='For images created by Fooocus', source='upload', type='pil')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                with gr.Tabs():
                    with gr.Tab(label='Upscale or Variation'):
                        with gr.Row():
                            with gr.Column():
                                enhance_uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                              value=modules.config.default_enhance_uov_method)
                                enhance_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                        info='Use before to enhance small details and after to enhance large areas.',
                                                                        choices=flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_enhance_uov_processing_order)
                                enhance_uov_prompt_type = gr.Radio(label='Prompt',
                                                                   info='Choose which prompt to use for Upscale or Variation.',
                                                                   choices=flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_enhance_uov_prompt_type,
                                                                   visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)

                                enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                    inputs=enhance_uov_processing_order,
                                                                    outputs=enhance_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                    enhance_ctrls = []
                    enhance_inpaint_mode_ctrls = []
                    enhance_inpaint_engine_ctrls = []
                    enhance_inpaint_update_ctrls = []
                    for index in range(modules.config.default_enhance_tabs):
                        with gr.Tab(label=f'#{index + 1}') as enhance_tab_item:
                            enhance_enabled = gr.Checkbox(label='Enable', value=False, elem_classes='min_check',
                                                          container=False)

                            enhance_mask_dino_prompt_text = gr.Textbox(label='Detection prompt',
                                                                       info='Use singular whenever possible',
                                                                       placeholder='Describe what you want to detect.',
                                                                       interactive=True,
                                                                       visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text = gr.Dataset(
                                samples=modules.config.example_enhance_detection_prompts,
                                label='Detection Prompt Quick List',
                                components=[enhance_mask_dino_prompt_text],
                                visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                        inputs=example_enhance_mask_dino_prompt_text,
                                                                        outputs=enhance_mask_dino_prompt_text,
                                                                        show_progress=False, queue=False)

                            enhance_prompt = gr.Textbox(label="Enhancement positive prompt",
                                                        placeholder="Uses original prompt instead if empty.",
                                                        elem_id='enhance_prompt')
                            enhance_negative_prompt = gr.Textbox(label="Enhancement negative prompt",
                                                                 placeholder="Uses original negative prompt instead if empty.",
                                                                 elem_id='enhance_negative_prompt')

                            with gr.Accordion("Detection", open=False):
                                enhance_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_enhance_inpaint_mask_model)
                                enhance_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                          choices=flags.inpaint_mask_cloth_category,
                                                                          value=modules.config.default_inpaint_mask_cloth_category,
                                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                          interactive=True)

                                with gr.Accordion("SAM Options",
                                                  visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                  open=False) as sam_options:
                                    enhance_mask_sam_model = gr.Dropdown(label='SAM model',
                                                                         choices=flags.inpaint_mask_sam_model,
                                                                         value=modules.config.default_inpaint_mask_sam_model,
                                                                         interactive=True)
                                    enhance_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0,
                                                                           maximum=1.0, value=0.3, step=0.05,
                                                                           interactive=True)
                                    enhance_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0,
                                                                            maximum=1.0, value=0.25, step=0.05,
                                                                            interactive=True)
                                    enhance_mask_sam_max_detections = gr.Slider(label="Maximum number of detections",
                                                                                info="Set to 0 to detect all",
                                                                                minimum=0, maximum=10,
                                                                                value=modules.config.default_sam_max_detections,
                                                                                step=1, interactive=True)

                            with gr.Accordion("Inpaint", visible=True, open=False):
                                enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                   value=modules.config.default_inpaint_method,
                                                                   label='Method', interactive=True)
                                enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                    label='Disable initial latent in inpaint', value=False)
                                enhance_inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                                     value=modules.config.default_inpaint_engine_version,
                                                                     choices=flags.inpaint_engine_versions,
                                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                enhance_inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                                     minimum=0.0, maximum=1.0, step=0.001,
                                                                     value=1.0,
                                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                                          'Only used in inpaint, not used in outpaint. '
                                                                          '(Outpaint always use 1.0)')
                                enhance_inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                             minimum=0.0, maximum=1.0, step=0.001,
                                                                             value=0.618,
                                                                             info='The area to inpaint. '
                                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                                  'Only used in inpaint, not used in outpaint. '
                                                                                  '(Outpaint always use 1.0)')
                                enhance_inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                            minimum=-64, maximum=64, step=1, value=0,
                                                                            info='Positive value will make white area in the mask larger, '
                                                                                 'negative value will make white area smaller. '
                                                                                 '(default is 0, always processed before any mask invert)')
                                enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)

                            gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                        enhance_ctrls += [
                            enhance_enabled,
                            enhance_mask_dino_prompt_text,
                            enhance_prompt,
                            enhance_negative_prompt,
                            enhance_mask_model,
                            enhance_mask_cloth_category,
                            enhance_mask_sam_model,
                            enhance_mask_text_threshold,
                            enhance_mask_box_threshold,
                            enhance_mask_sam_max_detections,
                            enhance_inpaint_disable_initial_latent,
                            enhance_inpaint_engine,
                            enhance_inpaint_strength,
                            enhance_inpaint_respective_field,
                            enhance_inpaint_erode_or_dilate,
                            enhance_mask_invert
                        ]

                        enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                        enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

                        enhance_inpaint_update_ctrls += [[
                            enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ]]

                        enhance_inpaint_mode.change(inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state], outputs=[
                            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                            enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ], show_progress=False, queue=False)

                        enhance_mask_model.change(
                            lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                      [gr.update(visible=x == 'sam')] * 2 +
                                      [gr.Dataset.update(visible=x == 'sam',
                                                         samples=modules.config.example_enhance_detection_prompts)],
                            inputs=enhance_mask_model,
                            outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                     example_enhance_mask_dino_prompt_text],
                            queue=False, show_progress=False)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)
            with gr.Row(elem_classes='advanced_check_row'):
                batch_checkbox = gr.Checkbox(label='Images Batch', value=False, container=False, elem_classes='min_check')
                prompt_checkbox = gr.Checkbox(label='Prompts Batch', value=False, container=False, elem_classes='min_check')

            with gr.Row(visible=False) as batch_panel:

                with gr.Row():
                  file_in=gr.File(label="Upload a ZIP file",file_count='single',file_types=['.zip'])                 
                  with gr.Column():
                    def update_radio(value):
                      return gr.update(value=value)
                    ratio = gr.Radio(label='Scale method:', choices=['NOT scale','to ORIGINAL','to OUTPUT'], value='NOT scale', interactive=True)
                    gr.HTML('* "Images Batch Mode" is powered by Shahmatist^RMDA')
                with gr.Row():
                  with gr.Column():
                    def cell_index_change(index):
                      global cell_index
                      cell_index = index
                      return
                    add_to_queue = gr.Button(label="Add to queue", value='Add to queue ({}'.format(len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))]))+')', elem_id='add_to_queue', visible=True)
                    batch_start = gr.Button(value='Start queue', visible=True)
                    batch_stop = gr.Button(value='Stop queue', visible=False)
                    batch_clear = gr.Button(value='Clear queue')
                    select_target_batch = gr.Dropdown([str(i) for i in range(modules.config.default_controlnet_image_count)], label="Use cell", value=cell_index, interactive=True, visible=(modules.config.default_controlnet_image_count > 1))
                    status_batch = gr.Textbox(show_label=False, value = '', container=False, visible=False, interactive=False)
                    select_target_batch.change(cell_index_change,inputs=select_target_batch)

                with gr.Row():
                  with gr.Column():
                    file_out=gr.File(label="Download a ZIP file", file_count='single')
                    save_output = gr.Button(value='Output --> ZIP')
                    clear_output = gr.Button(value='Clear Output')
            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)
            batch_checkbox.change(lambda x: gr.update(visible=x), inputs=batch_checkbox,
                                        outputs=batch_panel, queue=False, show_progress=False, _js=switch_js)
            with gr.Row(visible=False) as prompt_panel:


                def prompts_delete(batch_prompt):
                  if len(batch_prompt) > 1:
                      removed=batch_prompt.pop()
                  return batch_prompt


                with gr.Column():
                    batch_prompt=gr.Dataframe(
                      headers=["prompt", "negative prompt"],
                      datatype=["str", "str"],
                      row_count=1, wrap=True,
                      col_count=(2, "fixed"), type="array", interactive=True)
                    with gr.Row():
                      positive_batch = gr.Radio(label='Base positive prompt:', choices=['None','Prefix','Suffix'], value='None', interactive=True)
                      negative_batch = gr.Radio(label='Base negative prompt:', choices=['None','Prefix','Suffix'], value='None', interactive=True)
                    with gr.Row():
                      prompt_delete=gr.Button(value="Delete last row")
                      prompt_clear=gr.Button(value="Clear Batch")
                      prompt_start=gr.Button(value="Start batch", visible=True)
                      prompt_stop=gr.Button(value="Stop batch", visible=False)
                    with gr.Row():
                      gr.HTML('* "Prompt Batch Mode" is powered by Shahmatist^RMDA')
                
                prompt_delete.click(prompts_delete,inputs=batch_prompt,outputs=batch_prompt)
                prompt_clear.click(prompt_clearer,inputs=batch_prompt,outputs=batch_prompt)
            prompt_checkbox.change(lambda x: gr.update(visible=x), inputs=prompt_checkbox,
                                        outputs=prompt_panel, queue=False, show_progress=False, _js=switch_js)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
			
            with gr.Row(elem_classes='extend_row'):
                with gr.Accordion('Extention', open=False):
                  with gr.TabItem(label='OpenPoseEditor', elem_id='op_edit_tab') as op_edit_tab:
                    body_estimation = None
                    presets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extentions", "op_edit","presets.json")
                    presets = {}

                    try: 
                      with open(presets_file) as file:
                        presets = json.load(file)
                    except FileNotFoundError:
                      pass

                    def pil2cv(in_image):
                      out_image = np.array(in_image, dtype=np.uint8)

                      if out_image.shape[2] == 3:
                          out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
                      return out_image

                    def candidate2li(li):
                      res = []
                      for x, y, *_ in li:
                        res.append([x, y])
                      return res

                    def subset2li(li):
                      res = []
                      for r in li:
                        for c in r:
                          res.append(c)
                      return res                    
                    with gr.Row():
                      with gr.Column():
                        width_ope = gr.Slider(label="width", minimum=64, maximum=2048, value=512, step=64, interactive=True)
                        height_ope = gr.Slider(label="height", minimum=64, maximum=2048, value=512, step=64, interactive=True)
                        with gr.Row():
                          add_ope = gr.Button(value="Add", variant="primary")
                          # delete = gr.Button(value="Delete")
                        with gr.Row():
                          reset_btn_ope = gr.Button(value="Reset")
                          json_input_ope = gr.UploadButton(label="Load from JSON", file_types=[".json"], elem_id="openpose_json_button")
                          png_input_ope = gr.UploadButton(label="Detect from Image", file_types=["image"], type="bytes", elem_id="openpose_detect_button")
                          bg_input_ope = gr.UploadButton(label="Add Background Image", file_types=["image"], elem_id="openpose_bg_button")
                        with gr.Row():
                          preset_list = gr.Dropdown(label="Presets", choices=sorted(presets.keys()), interactive=True)
                          preset_load = gr.Button(value="Load Preset")
                          preset_save = gr.Button(value="Save Preset")
                        with gr.Row():
                          json_output_ope = gr.Button(value="Save JSON")
                          png_output_ope = gr.Button(value="Save PNG")
                      with gr.Column():
        # gradioooooo...
                        canvas_ope = gr.HTML('<canvas id="openpose_editor_canvas" width="512" height="512" style="margin: 0.25rem; border-radius: 0.25rem; border: 0.5px solid"></canvas>')
                        jsonbox_ope = gr.Text(label="json", elem_id="jsonbox", visible=False)
                    def estimate(file):
                      global body_estimation

                      if body_estimation is None:
                        model_path_ope = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "controlnet")
                        model_file_ope = os.path.join(model_path_ope, "body_pose_model.pth")
                        if not os.path.isfile(model_path_ope):

                          load_file_from_url(
                            url="https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/body_pose_model.pth",
                            model_dir=model_path_ope,
                            file_name='body_pose_model.pth')


                        body_estimation = Body(model_file_ope)
        
                      stream = io.BytesIO(file)
                      img = Image.open(stream)
                      candidate, subset = body_estimation(pil2cv(img))

                      result = {
                        "candidate": candidate2li(candidate),
                        "subset": subset2li(subset),
                      }
      
                      return str(result).replace("'", '"')

                    def savePreset(name, data):
                      if name:
                        presets[name] = json.loads(data)
                        with open(presets_file, "w") as file:
                          json.dump(presets, file)
                        return gr.update(choices=sorted(presets.keys()), value=name), json.dumps(data)
                      return gr.update(), gr.update()

                    dummy_component = gr.Label(visible=False)

                    preset_ope = gr.Text(visible=False)
                    width_ope.change(None, [width_ope, height_ope], None, _js="(w, h) => {resizeCanvas(w, h)}")
                    height_ope.change(None, [width_ope, height_ope], None, _js="(w, h) => {resizeCanvas(w, h)}")
                    png_output_ope.click(None, [], None, _js="savePNG")
                    bg_input_ope.upload(None, [], [width_ope, height_ope], _js="() => {addBackground('openpose_bg_button')}")
                    png_input_ope.upload(estimate, png_input_ope, jsonbox_ope)
                    png_input_ope.upload(None, [], [width_ope, height_ope], _js="() => {addBackground('openpose_detect_button')}")
                    add_ope.click(None, [], None, _js="addPose")
                    reset_btn_ope.click(None, [], None, _js="resetCanvas")
                    json_input_ope.upload(None, json_input_ope, [width_ope, height_ope], _js="() => {loadJSON('openpose_json_button')}")
                    json_output_ope.click(None, None, None, _js="saveJSON")
                    preset_save.click(savePreset, [dummy_component, dummy_component], [preset_list, preset_ope], _js="savePreset")
                    preset_load.click(None, preset_ope, [width_ope, height_ope], _js="loadPreset")
                    preset_list.change(lambda selected: json.dumps(presets[selected]), preset_list, preset_ope)


                  with gr.TabItem(label='OBP') as obp_tab:
                        with gr.Tab("Main"):
                            with gr.Row(variant="compact"):
                                gr.Markdown(ob_prompt.main_mark1)
                # Part of presets
                            with gr.Row():
                                    OBP_preset = gr.Dropdown(
                                        label="One Button Preset",
                                        choices=ob_prompt.obp_preset_choices,
                                        value="Standard",interactive=True)
                            with gr.Group(visible=True) as presetgroup:
                                with gr.Row():
                                    gr.Markdown('<font size="2">These prefix and suffix are run on top the of preset. Can be used for LoRAs and other general stylings.</font>')
                                with gr.Row():
                                    presetprefix = gr.Textbox(label="Preset prefix: ", value="")
                                    presetsuffix = gr.Textbox(label="Preset suffix: ", value="")
                                
                            with gr.Group(visible=False) as maingroup:
                                gr.Markdown('<font size="2">Type a name and press "Save as Preset" to store the current generation settings.</font>')
                                with gr.Row():
                                        obp_preset_name = gr.Textbox(
                                            show_label=False,
                                            placeholder="Name of new preset",
                                            interactive=True,
                                            visible=True)
                                        obp_preset_save = gr.Button(
                                            value="Save as preset",
                                            visible=True)
                                gr.Markdown('<font size="4">Generation settings:</font>')
            
            # End of this part of presets
                
                                with gr.Row(variant="compact"):
                                    insanitylevel = gr.Slider(1, 10, value=5, step=1, label="Higher levels increases complexity and randomness of generated prompt",interactive=True)
                                with gr.Row(variant="compact"):
                                    with gr.Column(variant="compact"):
                                        subject = gr.Dropdown(
                                            ob_prompt.subjects, label="Subject Types", value="all",interactive=True)                   
                                    with gr.Column(variant="compact"):
                                        artist = gr.Dropdown(
                                            ob_prompt.artists, label="Artists", value="all",interactive=True)
                                with gr.Row(variant="compact"):
                                    chosensubjectsubtypeobject = gr.Dropdown(
                                        ob_prompt.subjectsubtypesobject, label="Type of object", value="all", visible=False)
                                    chosensubjectsubtypehumanoid = gr.Dropdown(
                                        ob_prompt.subjectsubtypeshumanoid, label="Type of humanoids", value="all", visible=False)
                                    chosensubjectsubtypeconcept = gr.Dropdown(
                                        ob_prompt.subjectsubtypesconcept, label="Type of concept", value="all", visible=False)
                                    chosengender = gr.Dropdown(
                                        ob_prompt.genders, label="gender", value="all", visible=False)
                                with gr.Row(variant="compact"):
                                    with gr.Column(variant="compact"):
                                        imagetype = gr.Dropdown(
                                            ob_prompt.imagetypes, label="type of image", value="all",interactive=True)
                                    with gr.Column(variant="compact"):
                                        imagemodechance = gr.Slider(
                                            1, 100, value="20", step=1, label="One in X chance to use special image type mode",interactive=True)
                                with gr.Row(variant="compact"):
                                    gr.Markdown('<font size="2">Override options (choose the related subject type first for better results)</font>')
                                with gr.Row(variant="compact"):
                                    givensubject = gr.Textbox(label="Overwrite subject: ", value="")
                                    smartsubject = gr.Checkbox(label="Smart subject", value = True)
                                with gr.Row(variant="compact"):
                                    givenoutfit = gr.Textbox(label="Overwrite outfit: ", value="")
                                with gr.Row(variant="compact"):
                                    gr.Markdown('<font size="2">Prompt fields</font>')
                                with gr.Row(variant="compact"):
                                    with gr.Column(variant="compact"):
                                        prefixprompt = gr.Textbox(label="Place this in front of generated prompt (prefix)",value="")
                                        suffixprompt = gr.Textbox(label="Place this at back of generated prompt (suffix)",value="")
                                with gr.Row(variant="compact"):
                                    gr.Markdown('<font size="2">Additional options</font>')
                                with gr.Row(variant="compact"):
                                    giventypeofimage = gr.Textbox(label="Overwrite type of image: ", value="")
                                with gr.Row(variant="compact"):
                                    with gr.Column(variant="compact"):
                                        antistring = gr.Textbox(label="Filter out following properties (comma seperated). Example ""film grain, purple, cat"" ")
                                with gr.Accordion("Help", open=False):
                                        gr.Markdown(ob_prompt.main_mark2)

                        with gr.Tab("Workflow assist"):
                            with gr.Row(variant="compact"):
                                    silentmode = gr.Checkbox(
                                        label="Workflow mode, turns off prompt generation and uses below Workflow prompt instead.")
                            with gr.Row(variant="compact"):
                                workprompt = gr.Textbox(label="Workflow prompt")
                            with gr.Row(variant="compact"):
                                promptvariantinsanitylevel = gr.Slider(0, 10, value=0, step=1, label="Prompt variant. Strength of variation of workflow prompt. 0 = no variance.")
                            with gr.Accordion("Help", open=False):
                                gr.Markdown(ob_prompt.wf_mark1)
                            with gr.Row(variant="compact"):
                                genprom = gr.Button("Generate me some prompts!")
                            with gr.Row(variant="compact"):
                                    with gr.Column(scale=4, variant="compact"):
                                        prompt1 = gr.Textbox(label="prompt 1")
                                    with gr.Column(variant="compact"):
                                        prompt1toworkflow = gr.Button("prompt1->Workflow Prompt")
                                        prompt1toprompt = gr.Button("prompt1->Prompt")
                            with gr.Row(variant="compact"):
                                    with gr.Column(scale=4, variant="compact"):
                                        prompt2 = gr.Textbox(label="prompt 2")
                                    with gr.Column(variant="compact"):
                                        prompt2toworkflow = gr.Button("prompt2->Workflow Prompt")
                                        prompt2toprompt = gr.Button("prompt2->Prompt")
                            with gr.Row(variant="compact"):
                                    with gr.Column(scale=4, variant="compact"):
                                        prompt3 = gr.Textbox(label="prompt 3")
                                    with gr.Column(variant="compact"):
                                        prompt3toworkflow = gr.Button("prompt3->Workflow Prompt")
                                        prompt3toprompt = gr.Button("prompt3->Prompt")
                            with gr.Row(variant="compact"):
                                    with gr.Column(scale=4, variant="compact"):
                                        prompt4 = gr.Textbox(label="prompt 4")
                                    with gr.Column(variant="compact"):
                                        prompt4toworkflow = gr.Button("prompt4->Workflow Prompt")
                                        prompt4toprompt = gr.Button("prompt4->Prompt")
                            with gr.Row(variant="compact"):
                                    with gr.Column(scale=4, variant="compact"):
                                        prompt5 = gr.Textbox(label="prompt 5")
                                    with gr.Column(variant="compact"):
                                        prompt5toworkflow = gr.Button("prompt5->Workflow Prompt")
                                        prompt5toprompt = gr.Button("prompt5->Prompt")
                            
                        with gr.Tab("Advanced"):
                            with gr.Row(variant="compact"):
                                gr.Markdown(ob_prompt.adv_mark1)
                            with gr.Row(variant="compact"):
                                base_model_obp = gr.Dropdown(
                                    ob_prompt.basemodelslist, label="Base model", value="SDXL")
                            with gr.Row(variant="compact"):
                                amountoffluff = gr.Dropdown(
                                    ob_prompt.amountofflufflist, label="Flufferizer", value="dynamic")
                            with gr.Row(variant="compact"):
                                promptenhancer = gr.Dropdown(
                                    ob_prompt.prompt_enhancers, label="Prompt enhancer", value="none")
                            with gr.Row(variant="compact"):
                                gr.Markdown('<font size="2">Other options</font>')
                            with gr.Row(variant="compact"):
                                with gr.Column(variant="compact"):
                                    promptcompounderlevel = gr.Dropdown(
                                        ob_prompt.promptcompounder, label="Prompt compounder", value="1")
                            with gr.Row(variant="compact"):
                                with gr.Column(variant="compact"):
                                    seperator = gr.Dropdown(
                                        ob_prompt.seperatorlist, label="Prompt seperator", value="comma")    
                                with gr.Column(variant="compact"):
                                    ANDtoggle = gr.Dropdown(
                                        ob_prompt.ANDtogglemode, label="Prompt seperator mode", value="none",interactive=True)
                            with gr.Accordion("Help", open=False):
                                gr.Markdown(ob_prompt.adv_mark2)
                        with gr.Tab("Negative prompt"):
                            gr.Markdown("### Negative prompt settings")
                            with gr.Column(variant="compact"):
                                with gr.Row(variant="compact"): 
                                    autonegativeprompt = gr.Checkbox(label="Auto generate negative prompt", value=True,interactive=True) 
                                    autonegativepromptenhance = gr.Checkbox(label="Enable base enhancement prompt", value=False)
                            with gr.Row(variant="compact"): 
                                autonegativepromptstrength = gr.Slider(0, 10, value="0", step=1, label="Randomness of negative prompt (lower is more consistency)",interactive=True)
                            
                 
                        with gr.Tab("One Button Run"):
                            with gr.Row(variant="compact"):
                                with gr.Column(variant="compact"):
                                    size = gr.Dropdown(ob_prompt.sizelist, label="Size to generate", value="all",interactive=True)
                                with gr.Column(variant="compact"):
                                    amountofimages = gr.Slider(1, 50, value="20", step=1, label="Generations of prompts",interactive=True)
                                with gr.Column(variant="compact"):
                                    model = gr.Dropdown(
                                        choices=ob_prompt.modellist, label="model to use", value="current model",interactive=True)
                            with gr.Row(variant="compact"):
                                    with gr.Column(variant="compact"):
                                        start_obp = gr.Button("Start generating")
                                        stop_obp = gr.Button("Stop generating")
                        gr.HTML('* \"OneButtonPrompt\" is powered by AIrjen. <a href="https://github.com/AIrjen/OneButtonPrompt" target="_blank">\U0001F4D4 Document</a>')
                        gr.HTML('* Adaptation for Fooocus is powered by Shahmatist^RMDA')   
                        OBP_preset.change(ob_prompt.obppreset_changed,inputs=[OBP_preset],
                              outputs=[obp_preset_name,maingroup,presetgroup,presetprefix,presetsuffix])
                        OBP_preset.change(ob_prompt.OBPPreset_changed_update_custom,inputs=[OBP_preset],
                              outputs=[insanitylevel,subject,artist,chosensubjectsubtypeobject,chosensubjectsubtypehumanoid,
                                      chosensubjectsubtypeconcept,chosengender,imagetype,imagemodechance,givensubject,
                                      smartsubject,givenoutfit,prefixprompt,suffixprompt,giventypeofimage,antistring])
                        obp_outputs = [obp_preset_name,obp_preset_save,insanitylevel,subject,artist,chosensubjectsubtypeobject,
                              chosensubjectsubtypehumanoid,chosensubjectsubtypeconcept,chosengender,imagetype,imagemodechance,givensubject,
                              smartsubject,givenoutfit,prefixprompt,suffixprompt,giventypeofimage,antistring]
                        obp_preset_save.click(ob_prompt.act_obp_preset_save,inputs=obp_outputs,outputs=[OBP_preset])
                        subject.change(ob_prompt.subjectsvalue,[subject],[chosengender])
                        subject.change(ob_prompt.subjectsvalueforsubtypeobject,[subject],[chosensubjectsubtypeobject])
                        subject.change(ob_prompt.subjectsvalueforsubtypeconcept,[subject],[chosensubjectsubtypeconcept])
                        subject.change(ob_prompt.subjectsvalueforsubtypeobject,[subject],[chosensubjectsubtypehumanoid])
                        genprom.click(ob_prompt.gen_prompt, inputs=[insanitylevel,subject, artist, imagetype, antistring,prefixprompt, suffixprompt,promptcompounderlevel, seperator, givensubject,smartsubject,giventypeofimage,imagemodechance, chosengender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept, givenoutfit, base_model_obp, OBP_preset, amountoffluff, promptenhancer, presetprefix, presetsuffix,silentmode,workprompt,promptvariantinsanitylevel], 
                                outputs=[prompt1, prompt2, prompt3,prompt4,prompt5])
                        prompt1toworkflow.click(ob_prompt.prompttoworkflowprompt, inputs=prompt1, outputs=workprompt)
                        prompt1toprompt.click(ob_prompt.prompttoworkflowprompt, inputs=prompt1, outputs=prompt)
                        prompt2toworkflow.click(ob_prompt.prompttoworkflowprompt, inputs=prompt2, outputs=workprompt)
                        prompt2toprompt.click(ob_prompt.prompttoworkflowprompt, inputs=prompt2, outputs=prompt)
                        prompt3toworkflow.click(ob_prompt.prompttoworkflowprompt, inputs=prompt3, outputs=workprompt)
                        prompt3toprompt.click(ob_prompt.prompttoworkflowprompt, inputs=prompt3, outputs=prompt)
                        prompt4toworkflow.click(ob_prompt.prompttoworkflowprompt, inputs=prompt4, outputs=workprompt)
                        prompt4toprompt.click(ob_prompt.prompttoworkflowprompt, inputs=prompt4, outputs=prompt)
                        prompt5toworkflow.click(ob_prompt.prompttoworkflowprompt, inputs=prompt5, outputs=workprompt)
                        prompt5toprompt.click(ob_prompt.prompttoworkflowprompt, inputs=prompt5, outputs=prompt)

                  with gr.TabItem(label='Civitai_helper') as download_tab:
                        civitai_helper.civitai_help()
                  with gr.TabItem(label='Prompt Translate') as promp_tr_tab:       
                    langs_sup = GoogleTranslator().get_supported_languages(as_dict=True)
                    langs_sup = list(langs_sup.values())

                    def change_lang(src, dest):
                            if src != 'auto' and src != dest:
                                return [src, dest]
                            return ['en','auto']
                        
                    def show_viewtrans(checkbox):
                        return {viewstrans: gr.update(visible=checkbox)} 
                                       
                    with gr.Row():
                            translate_enabled = gr.Checkbox(label='Enable translate', value=False, elem_id='translate_enabled_el')
                            translate_automate = gr.Checkbox(label='Auto translate "Prompt and Negative prompt" before Generate', value=True, interactive=True, elem_id='translate_enabled_el')
                            
                    with gr.Row():
                            gtrans = gr.Button(value="Translate")        

                            srcTrans = gr.Dropdown(['auto']+langs_sup, value='auto', label='From', interactive=True)
                            toTrans = gr.Dropdown(langs_sup, value='en', label='To', interactive=True)
                            change_src_to = gr.Button(value="🔃")
                            
                    with gr.Row():
                            adv_trans = gr.Checkbox(label='See translated prompts after click Generate', value=False)          
                            
                    with gr.Box(visible=False) as viewstrans:
                            gr.Markdown('Tranlsated prompt & negative prompt')
                            with gr.Row():
                                p_tr = gr.Textbox(label='Prompt translate', show_label=False, value='', lines=2, placeholder='Translated text prompt')

                            with gr.Row():            
                                p_n_tr = gr.Textbox(label='Negative Translate', show_label=False, value='', lines=2, placeholder='Translated negative text prompt')             
                    gr.HTML('* \"Prompt Translate\" is powered by AlekPet. <a href="https://github.com/AlekPet/Fooocus_Extensions_AlekPet" target="_blank">\U0001F4D4 Document</a>')

                  with gr.TabItem(label='Photopea') as photopea_tab:
                    PHOTOPEA_MAIN_URL = 'https://www.photopea.com/'
                    PHOTOPEA_IFRAME_ID = 'webui-photopea-iframe'
                    PHOTOPEA_IFRAME_HEIGHT = '800px'
                    PHOTOPEA_IFRAME_WIDTH = '100%'
                    PHOTOPEA_IFRAME_LOADED_EVENT = 'onPhotopeaLoaded'

                    def get_photopea_url_params():
                      return '#%7B%22resources%22:%5B%22data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIAAQMAAADOtka5AAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAANQTFRF////p8QbyAAAADZJREFUeJztwQEBAAAAgiD/r25IQAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfBuCAAAB0niJ8AAAAABJRU5ErkJggg==%22%5D%7D'

                    with gr.Row():
                          photopea = gr.HTML(
                            f'''
                            <iframe id='{PHOTOPEA_IFRAME_ID}' 
                            src = '{PHOTOPEA_MAIN_URL}{get_photopea_url_params()}' 
                            width = '{PHOTOPEA_IFRAME_WIDTH}' 
                            height = '{PHOTOPEA_IFRAME_HEIGHT}'
                            onload = '{PHOTOPEA_IFRAME_LOADED_EVENT}(this)'>'''
                          )
                    with gr.Row():
                          gr.HTML('* \"Photopea\" is powered by Photopea API. <a href="https://www.photopea.com/api" target="_blank">\U0001F4D4 Document</a>')


                  with gr.TabItem(label='Remove Background') as rembg_tab:
                        def rembg_run(path, progress=gr.Progress(track_tqdm=True)):
                          input = Image.open(path)
                          output = remove(input)
                          return output
                        with gr.Row():
                            with gr.Column():
                                rembg_input = gr.Image(label='Drag above image to here', source='upload', type='filepath', height=380)
                            with gr.Column():
                                rembg_output = gr.Image(label='rembg Output', interactive=False, height=380)
                        with gr.Row():
                          rembg_button = gr.Button(value='Remove Background', interactive=True)
                        with gr.Row():
                          gr.Markdown('Powered by [🪄 rembg 2.0.53](https://github.com/danielgatis/rembg/releases/tag/v2.0.53)')
                        rembg_button.click(rembg_run, inputs=rembg_input, outputs=rembg_output, show_progress='full')

            enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                                        outputs=enhance_input_panel, queue=False, show_progress=False, _js=switch_js)

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='Settings'):
                if not args_manager.args.disable_preset_selection:
                    preset_selection = gr.Dropdown(label='Preset',
                                                   choices=modules.config.available_presets,
                                                   value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                   interactive=True)

                performance_selection = gr.Radio(label='Performance',
                                                 choices=flags.Performance.values(),
                                                 value=modules.config.default_performance,
                                                 elem_classes=['performance_selection'])

                with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                    aspect_ratios_selection = gr.Radio(label='Aspect Ratios', show_label=False,
                                                       choices=modules.config.available_aspect_ratios_labels,
                                                       value=modules.config.default_aspect_ratio,
                                                       info='width × height',
                                                       elem_classes='aspect_ratios_news')
                    

                    aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                    shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                    width_ar = gr.Slider(minimum=64,maximum=2048,step=8,value=512,label='Width', interactive=True)
                    height_ar = gr.Slider(minimum=64,maximum=2048,step=8,value=512,label='Height',interactive=True)
                    lock_ar = gr.Dropdown(choices=choices_ar1,value="Any",label="AspectRatio", show_label=True,interactive=True)
                    swap = gr.Button(value='Portrait', visible=True)
                    set_ar = gr.Button(value='Set', visible=True)

                    def locker(lock,width,height):
                        global ar_def, swap_def
                        if lock == "Any":
                          interact1, interact2 = True, True
                        else:
                          new_width, new_height = lock.replace(':', ' ').split(' ')[:2]
                          ar_def=[int (new_width), int (new_height)]
                          if swap_def:
                            width = round (height / ar_def[1] * ar_def[0])
                            interact1, interact2 = False, True
                          else:
                            height=round (width / ar_def[0] * ar_def[1])
                            interact1, interact2 = True, False
                        return gr.update (interactive=interact1, value=width), gr.update (interactive=interact2, value=height)
                    def swap_ar(lock,width,height):
                        global swap_def, ar_def
                        swap_def=not swap_def
                        ar_def[0], ar_def[1] = ar_def[1], ar_def[0]
                        width, height = height, width                       
                        if swap_def:
                          choices=choices_ar2
                          interact1, interact2 = False, True
                          name='Landscape'
                        else: 
                          choices=choices_ar1
                          interact1, interact2 = True, False
                          name = 'Portrait'
                        if lock != "Any":
                          ratio_x, ratio_y = lock.replace(':', ' ').split(' ')[:2]
                          lock=str(ratio_y)+":"+str(ratio_x) 
                          interact1, interact2 = True, False
                        if lock == "Any":
                          interact1, interact2 = True, True
                        return gr.update (choices=choices, value=lock),gr.update (value=width, interactive=interact1),gr.update (value=height, interactive=interact2),gr.update (value=name)
                    def w_slide(lock,width,height):
                        global ar_def
                        if lock != "Any":
                          height=width / ar_def[0] * ar_def[1]
                        return gr.update (value=height)
                    def h_slide(lock,width,height):
                        global ar_def
                        if lock != "Any":
                          width=height / ar_def[1] * ar_def[0]
                        return gr.update (value=width)
                    def set_to_ar(aspect_ratios_selection,width,height): 
                        g = math.gcd(width, height)
                        selector=f'{width}×{height}  \U00002223 {width // g}:{height // g}'
                        if aspect_ratios_selection==modules.config.available_aspect_ratios_labels[-1]:
                          previos_aspect=selector
                        else:
                          previos_aspect=aspect_ratios_selection
                        modules.config.available_aspect_ratios_labels[-1]=selector
                        return gr.update (choices=modules.config.available_aspect_ratios_labels, value=previos_aspect)
                swap.click(swap_ar,inputs=[lock_ar,width_ar,height_ar],outputs=[lock_ar,width_ar,height_ar,swap],show_progress=False)
                lock_ar.change(locker, inputs=[lock_ar,width_ar,height_ar],outputs=[width_ar, height_ar],show_progress=False)
                width_ar.release(w_slide,inputs=[lock_ar,width_ar,height_ar],outputs=[height_ar],show_progress=False)
                height_ar.release(h_slide,inputs=[lock_ar,width_ar,height_ar],outputs=[width_ar],show_progress=False)
                set_ar.click(set_to_ar,inputs=[aspect_ratios_selection,width_ar,height_ar],outputs=aspect_ratios_selection,show_progress=False)

                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)

                output_format = gr.Radio(label='Output Format',
                                         choices=flags.OutputFormat.list(),
                                         value=modules.config.default_output_format)

                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                             info='Describing what you do not want to see.', lines=2,
                                             elem_id='negative_prompt',
                                             value=modules.config.default_prompt_negative)
                seed_random = gr.Checkbox(label='Random', value=True)
                image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')

                history_link = gr.HTML()
                shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)

            with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=modules.config.default_styles)

                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="\U0001F50E Type here to search styles ...",
                                              value="",
                                              label='Search Styles')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                        outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

            with gr.Tab(label='Models'):
                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                        refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Group():
                    lora_ctrls = []

                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'], scale=1)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=filename,
                                                     elem_classes='lora_model', scale=5)
                            lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                                    elem_classes='lora_weight', scale=5)
                            lora_ctrls += [lora_enabled, lora_model, lora_weight]

                with gr.Row():
                    refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
            with gr.Tab(label='Advanced'):
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Documentation</a>')
                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=modules.config.default_developer_debug_mode_checkbox, container=False)

                with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                 value=modules.config.default_clip_skip,
                                                 info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).')
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)
                        vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                     value=modules.config.default_vae, show_label=True)

                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                               minimum=-1, maximum=1.0, step=0.001,
                                                               value=modules.config.default_overwrite_upscale,
                                                               info='Set as negative number to disable. For developer debugging.')

                        disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw,
                                                      info='Disable preview during generation.')
                        disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                      value=flags.Performance.has_restricted_features(modules.config.default_performance),
                                                      info='Disable intermediate results during generation, only show final gallery.')

                        disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                             info='Disable automatic seed increment when image number is > 1.',
                                                             value=False)
                        read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False)

                        black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='Use black image if NSFW is detected.')

                        black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                              inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                                         show_progress=False)
                        black_out_nsfw.change(civitai_helper_nsfw,inputs=black_out_nsfw)


                        if not args_manager.args.disable_image_log:
                            save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image',
                                                                         value=modules.config.default_save_only_final_enhanced_image)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme],
                                                           queue=False, show_progress=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)
                        

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                        debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False,
                                                                       info='Show enhance masks in preview and final results')
                        debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False,
                                                     info='Use GroundingDINO boxes instead of more detailed SAM masks')
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller. '
                                                                 '(default is 0, always processed before any mask invert)')
                        dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                                         minimum=-64, maximum=64, step=1, value=0,
                                                         info='Positive value will make white area in the mask larger, '
                                                              'negative value will make white area smaller. '
                                                              '(default is 0, processed before SAM)')

                        inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                 inputs=inpaint_advanced_masking_checkbox,
                                                                 outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                 queue=False, show_progress=False)

                        inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                                  outputs=inpaint_input_image,
                                                  queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                def dev_mode_checked(r):
                    return gr.update(visible=r)

                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def refresh_files_clicked():
                    modules.config.update_files()
                    results = [gr.update(choices=modules.config.model_filenames)]
                    results += [gr.update(choices=['None'] + modules.config.model_filenames)]
                    results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(modules.config.default_max_lora_number):
                        results += [gr.update(interactive=True),
                                    gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                    return results

                refresh_files_output = [base_model, refiner_model, vae_name]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                             base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                             seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                             inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                             load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def preset_selection_change(preset, is_generating, inpaint_mode):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                default_model = preset_prepared.get('base_model')
                previous_default_models = preset_prepared.get('previous_default_models', [])
                checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                lora_downloads = preset_prepared.get('lora_downloads', {})
                vae_downloads = preset_prepared.get('vae_downloads', {})

                preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = launch.download_models(
                    default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
                    vae_downloads)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating, inpaint_mode)


            def inpaint_engine_state_change(inpaint_engine_version, *args):
                if inpaint_engine_version == 'empty':
                    inpaint_engine_version = modules.config.default_inpaint_engine_version

                result = []
                for inpaint_mode in args:
                    if inpaint_mode != modules.flags.inpaint_option_detail:
                        result.append(gr.update(value=inpaint_engine_version))
                    else:
                        result.append(gr.update())

                return result

            preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}') \
                .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)

        performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                               [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                               [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)

        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)
        def seeTranlateAfterClick(adv_trans, prompt, negative_prompt="", srcTrans="auto", toTrans="en"):
            if(adv_trans):
                positive, negative = translate(prompt, negative_prompt, srcTrans, toTrans)
                return [positive, negative]   
            return ["", ""]
        
        gtrans.click(translate, inputs=[prompt, negative_prompt, srcTrans, toTrans], outputs=[prompt, negative_prompt])
        gtrans.click(translate, inputs=[prompt, negative_prompt, srcTrans, toTrans], outputs=[p_tr, p_n_tr])
        
        change_src_to.click(change_lang, inputs=[srcTrans,toTrans], outputs=[toTrans,srcTrans])
        adv_trans.change(show_viewtrans, inputs=adv_trans, outputs=[viewstrans])

        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        # load configured default_inpaint_method
        default_inpaint_ctrls = [inpaint_mode, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
        for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
            shared.gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
                inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
                engine, strength, respective_field
            ], show_progress=False, queue=False)

        generate_mask_button.click(fn=generate_mask,
                                   inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                           inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                           inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                           inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                                   outputs=inpaint_mask_image, show_progress=True, queue=True)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
            read_wildcards_in_order, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
        ctrls += [sampler_name, scheduler_name, vae_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        if not args_manager.args.disable_image_log:
            ctrls += [save_final_enhanced_image_only]

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls
        ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                  enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
                  enhance_uov_prompt_type]
        ctrls += enhance_ctrls
        ctrls += [translate_enabled, translate_automate, srcTrans, toTrans]
		

        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            if is_json(raw_prompt_txt):
                loaded_json = json.loads(raw_prompt_txt)

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(file, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                print('Could not find metadata in the image!')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.to_json(parameters)

            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating, inpaint_mode)

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)
        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(fn=seeTranlateAfterClick, inputs=[adv_trans, prompt, negative_prompt, srcTrans, toTrans], outputs=[p_tr, p_n_tr]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')
        ctrls_batch = ctrls[:]
        ctrls_batch.append(ratio)
        ctrls_batch.append(seed_random)
        add_to_queue.click(lambda: (gr.update(interactive=False), gr.update(visible=True,value='File unZipping')),
                                    outputs=[add_to_queue, status_batch]) \
              .then(fn=unzip_file,inputs=file_in) \
              .then(lambda: (gr.update(visible=False)),outputs=[status_batch]) \
              .then(lambda: (gr.update(value=f'Add to queue ({len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])})')), outputs=[add_to_queue]) \
              .then(lambda: (gr.update(interactive=True)),outputs=[add_to_queue])
        clear_output.click(lambda: (gr.update(interactive=False)),outputs=[clear_output]) \
              .then(clear_outputs) \
              .then(lambda: (gr.update(interactive=True)),outputs=[clear_output])
        save_output.click(lambda: (gr.update(interactive=False)),outputs=[save_output]) \
            .then(fn=output_zip, outputs=file_out) \
            .then(lambda: (gr.update(interactive=True)),outputs=[save_output])
        batch_clear.click(lambda: (gr.update(interactive=False),  gr.update(visible=True,value='Queue is clearing')),
                        outputs=[batch_clear,status_batch]) \
              .then(fn=clearer) \
              .then(lambda: (gr.update(value=f'Add to queue ({len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])})')), outputs=[add_to_queue]) \
              .then(lambda: (gr.update(interactive=True),gr.update(visible=False)),outputs=[batch_clear,status_batch])
        batch_start.click(lambda: (gr.update(visible=False),gr.update(visible=False), gr.update(visible=True, interactive=True),gr.update(visible=True,value='Queue in progress')),
                          outputs=[generate_button,batch_start, batch_stop, status_batch]) \
              .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
              .then(fn=queue_new, inputs=ctrls_batch, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
              .then(lambda: (gr.update(visible=True),gr.update(visible=False), gr.update(visible=True),gr.update(visible=False)),
                          outputs=[generate_button,batch_stop, batch_start,status_batch]) \
              .then(lambda: (gr.update(value=f'Add to queue ({len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])})')), outputs=[add_to_queue])
        batch_stop.click(stop_clicked_batch, queue=False, show_progress=False, _js='cancelGenerateForever')
        ctrls_prompt = ctrls[:]
        ctrls_prompt.append(batch_prompt)
        ctrls_prompt.append(seed_random)
        ctrls_prompt.append(positive_batch)
        ctrls_prompt.append(negative_batch)
        prompt_start.click(lambda: (gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(visible=False),gr.update(visible=False), gr.update(visible=True, interactive=True)),
                              outputs=[batch_prompt,prompt_delete,prompt_clear,generate_button,prompt_start, prompt_stop]) \
              .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
              .then(fn=queue_new_prompt,inputs=ctrls_prompt, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
              .then(lambda: (gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(visible=True),gr.update(visible=False), gr.update(visible=True)),
                          outputs=[batch_prompt,prompt_delete,prompt_clear,generate_button,prompt_stop, prompt_start]) \
              .then(fn=update_history_link, outputs=history_link) \
              .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')
        prompt_stop.click(stop_clicked_batch, queue=False, show_progress=False, _js='cancelGenerateForever')
        ctrls_obp = ctrls[:]
        obp_ctrl=[model,base_model,size,aspect_ratios_selection,amountofimages,insanitylevel,subject, artist, imagetype, silentmode, workprompt, antistring, prefixprompt, suffixprompt,negative_prompt,promptcompounderlevel, seperator, givensubject,smartsubject,giventypeofimage,imagemodechance, chosengender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept, promptvariantinsanitylevel, givenoutfit, autonegativeprompt, autonegativepromptstrength, autonegativepromptenhance, base_model_obp, OBP_preset, amountoffluff, promptenhancer, presetprefix, presetsuffix,seed_random]
        ctrls_obp.extend(obp_ctrl)
        start_obp.click(lambda: (gr.update(visible=False),gr.update(interactive=False)),
                              outputs=[generate_button,start_obp]) \
              .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
              .then(fn=queue_obp,inputs=ctrls_obp, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
              .then(lambda: (gr.update(visible=True),gr.update(interactive=True),gr.update(interactive=True)),
                          outputs=[generate_button,start_obp,stop_obp]) \
              .then(fn=update_history_link, outputs=history_link) \
              .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')
        stop_obp.click(stop_clicked_batch, queue=False, show_progress=False, _js='cancelGenerateForever')

        reset_button.click(lambda: [worker.AsyncTask(args=[]), False, gr.update(visible=True, interactive=True)] +
                                   [gr.update(visible=False)] * 6 +
                                   [gr.update(visible=True, value=[])],
                           outputs=[currentTask, state_is_generating, generate_button,
                                    reset_button, stop_button, skip_button,
                                    progress_html, progress_window, progress_gallery, gallery],
                           queue=False)

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(modes, img, apply_styles):
            describe_prompts = []
            styles = set()

            if flags.describe_type_photo in modes:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                describe_prompts.append(default_interrogator_photo(img))
                styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

            if flags.describe_type_anime in modes:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                describe_prompts.append(default_interrogator_anime(img))
                styles.update(["Fooocus V2", "Fooocus Masterpiece"])

            if len(styles) == 0 or not apply_styles:
                styles = gr.update()
            else:
                styles = list(styles)

            if len(describe_prompts) == 0:
                describe_prompt = gr.update()
            else:
                describe_prompt = ', '.join(describe_prompts)

            return describe_prompt, styles

        describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                           outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        if args_manager.args.enable_auto_describe_image:
            def trigger_auto_describe(mode, img, prompt, apply_styles):
                # keep prompt if not empty
                if prompt == '':
                    return trigger_describe(mode, img, apply_styles)
                return gr.update(), gr.update()

            uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                                   outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                      outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')
    gr.HTML("<div><p style='text-align:center;'>We are in <a href='https://t.me/+xlhhGmrz9SlmYzg6' target='_blank'>Telegram</a></p> </div>")
  
def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
