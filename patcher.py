# Script for patching file webui.py for Fooocus
# Author: AlekPet & Shahmatist/RMDA

import os
import datetime
import shutil

DIR_FOOOCUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fooocus")
PATH_TO_WEBUI = os.path.join(DIR_FOOOCUS, "webui.py")

PATH_OBJ_DATA_PATCHER = [
    ["import copy\n","import requests\n"],
    ["import launch\n","""import re
import urllib.request
import zipfile
import threading
import math
import numpy as np\n"""],


    ["from modules.auth import auth_enabled, check_auth\n","""from modules.module_translate import translate, GoogleTranslator
from urllib.parse import urlparse, parse_qs, unquote
from modules.model_loader import load_file_from_url
from rembg import remove
from PIL import Image
from gradio.components import label\n"""],

    ["from modules.util import is_json\n","""from md_lib import civitai_helper
from md_lib import md_config

def civitai_helper_nsfw(black_out_nsfw):
  md_config.ch_nsfw_threshold=black_out_nsfw
  return
civitai_helper_nsfw(modules.config.default_black_out_nsfw)\n"""],

    
    ["def get_task(*args):\n", """    argsList = list(args)
    toT = argsList.pop() 
    srT = argsList.pop() 
    trans_automate = argsList.pop() 
    trans_enable = argsList.pop() 

    if trans_enable:      
        if trans_automate:
            positive, negative = translate(argsList[2], argsList[3], srT, toT)            
            argsList[2] = positive
            argsList[3] = negative
            
    args = tuple(argsList)\n"""],

    
    [
        "    return worker.AsyncTask(args=args)\n",
        """finished_batch=False
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
    finished_batch=False 
    args = list(args)
    scale=args.pop()    
    lora_args=3*(int(modules.config.default_max_lora_number))
    batch_all=len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    passed=1
    for f_name in os.listdir('./batch_images'):
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
                  args[(65+lora_args)]=np.array(img)
        print (f"[QUEUE] {passed} / {batch_all}")
        passed+=1
        currentTask=get_task_batch(args)
        yield from generate_clicked(currentTask)
        args=copy[:]
    clearer()
    return

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
    return worker.AsyncTask(args=args)\n"""],

    
    ["                                        outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)\n","""
            batch_checkbox = gr.Checkbox(label='Batch', value=False, container=False, elem_classes='min_check')
            with gr.Row(visible=False) as batch_panel:

                with gr.Row():
                  file_in=gr.File(label="Upload a ZIP file",file_count='single',file_types=['.zip'])                 
                  with gr.Column():
                    def update_radio(value):
                      return gr.update(value=value)
                    ratio = gr.Radio(label='Scale method:', choices=['NOT scale','to ORIGINAL','to OUTPUT'], value='NOT scale', interactive=True)
                    gr.HTML('* "Batch Mode" is powered by Shahmatist^RMDA')
                with gr.Row():
                  with gr.Column():
                    add_to_queue = gr.Button(label="Add to queue", value='Add to queue ({}'.format(len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))]))+')', elem_id='add_to_queue', visible=True)
                    batch_start = gr.Button(value='Start queue', visible=True)
                    batch_stop = gr.Button(value='Stop queue', visible=False)
                    batch_clear = gr.Button(value='Clear queue')
                    status_batch = gr.Textbox(show_label=False, value = '', container=False, visible=False, interactive=False)


                with gr.Row():
                  with gr.Column():
                    file_out=gr.File(label="Download a ZIP file", file_count='single')
                    save_output = gr.Button(value='Output --> ZIP')
                    clear_output = gr.Button(value='Clear Output')                                        
            batch_checkbox.change(lambda x: gr.update(visible=x), inputs=batch_checkbox,
                                        outputs=batch_panel, queue=False, show_progress=False, _js=switch_js)\n"""],
										
	["            describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)\n", """
            with gr.Row(elem_classes='extend_row'):
               with gr.Accordion('Extention', open=False):
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
                        rembg_button.click(rembg_run, inputs=rembg_input, outputs=rembg_output, show_progress='full')\n"""],
    
	["                                                         show_progress=False)\n","                                            black_out_nsfw.change(civitai_helper_nsfw,inputs=black_out_nsfw)\n"],
	
	["            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)\n","""
        def seeTranlateAfterClick(adv_trans, prompt, negative_prompt="", srcTrans="auto", toTrans="en"):
            if(adv_trans):
                positive, negative = translate(prompt, negative_prompt, srcTrans, toTrans)
                return [positive, negative]   
            return ["", ""]
        
        gtrans.click(translate, inputs=[prompt, negative_prompt, srcTrans, toTrans], outputs=[prompt, negative_prompt])
        gtrans.click(translate, inputs=[prompt, negative_prompt, srcTrans, toTrans], outputs=[p_tr, p_n_tr])
        
        change_src_to.click(change_lang, inputs=[srcTrans,toTrans], outputs=[toTrans,srcTrans])
        adv_trans.change(show_viewtrans, inputs=adv_trans, outputs=[viewstrans])\n"""],

	["        ctrls += enhance_ctrls\n","        ctrls += [translate_enabled, translate_automate, srcTrans, toTrans]\n"],

	["            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \\\n","""            .then(fn=seeTranlateAfterClick, inputs=[adv_trans, prompt, negative_prompt, srcTrans, toTrans], outputs=[p_tr, p_n_tr]) \\\n"""],

	["            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')\n","""
        ctrls_batch = ctrls[:]
        ctrls_batch.append(ratio)
        add_to_queue.click(lambda: (gr.update(interactive=False), gr.update(visible=True,value='File unZipping')),
                                    outputs=[add_to_queue, status_batch]) \\
              .then(fn=unzip_file,inputs=file_in) \\
              .then(lambda: (gr.update(visible=False)),outputs=[status_batch]) \\
              .then(lambda: (gr.update(value=f'Add to queue ({len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])})')), outputs=[add_to_queue]) \\
              .then(lambda: (gr.update(interactive=True)),outputs=[add_to_queue])
        download_start.click(lambda: (gr.update(value='Downloading...', interactive=False)),outputs=download_start) \\
            .then(downloader, inputs=[civitai_api_key,downloader_checkpoint,downloader_loras,downloader_embd,downloader_vae],outputs=civitai_api_key) \\
            .then(refresh_files_clicked, [], refresh_files_output + lora_ctrls, queue=False, show_progress=False) \\
            .then(lambda: (gr.update(value='Start Download', interactive=True)),outputs=download_start)
        clear_output.click(lambda: (gr.update(interactive=False)),outputs=[clear_output]) \\
              .then(clear_outputs) \\
              .then(lambda: (gr.update(interactive=True)),outputs=[clear_output])
        save_output.click(lambda: (gr.update(interactive=False)),outputs=[save_output]) \\
            .then(fn=output_zip, outputs=file_out) \\
            .then(lambda: (gr.update(interactive=True)),outputs=[save_output])
        batch_clear.click(lambda: (gr.update(interactive=False),  gr.update(visible=True,value='Queue is clearing')),
                        outputs=[batch_clear,status_batch]) \\
              .then(fn=clearer) \\
              .then(lambda: (gr.update(value=f'Add to queue ({len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])})')), outputs=[add_to_queue]) \\
              .then(lambda: (gr.update(interactive=True),gr.update(visible=False)),outputs=[batch_clear,status_batch])
        batch_start.click(lambda: (gr.update(visible=False),gr.update(visible=False), gr.update(visible=True, interactive=True),gr.update(visible=True,value='Queue in progress')),
                          outputs=[generate_button,batch_start, batch_stop, status_batch]) \\
              .then(fn=queue_new, inputs=ctrls_batch, outputs=[progress_html, progress_window, progress_gallery, gallery]) \\
              .then(lambda: (gr.update(visible=True),gr.update(visible=False), gr.update(visible=True),gr.update(visible=False)),
                          outputs=[generate_button,batch_stop, batch_start,status_batch]) \\
              .then(lambda: (gr.update(value=f'Add to queue ({len([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])})')), outputs=[add_to_queue])
        batch_stop.click(stop_clicked_batch, queue=False, show_progress=False, _js='cancelGenerateForever')\n"""]
]    

def search_and_path():
    isOk = 0
    pathesLen = len(PATH_OBJ_DATA_PATCHER)
    patchedFileName = os.path.join(DIR_FOOOCUS, "webui_patched.py")

    with open(PATH_TO_WEBUI, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        len_lines = len(lines)

        if not len_lines:
            print(f"File '{PATH_TO_WEBUI}' is empty!\n")
            return

        if PATH_OBJ_DATA_PATCHER[0][1] in lines:
            return "Already"

        pathed = 0
        pathSteps = 100 / pathesLen

        patchedFile = open(patchedFileName, 'w+', encoding='utf-8')

        for line in lines:
            for linepath in PATH_OBJ_DATA_PATCHER:
                if line.startswith(linepath[0]):
                    line = line + linepath[1]
                    isOk = isOk + 1

                    pathed += pathSteps
                    print('Patches applied to file {0} of {1} [{2:1.1f}%)]'.format(isOk, pathesLen, pathed), end='\r',
                          flush=True)

            patchedFile.write(line)

        patchedFile.close()

        pathResult = isOk == pathesLen

        if not pathResult:
            # Remove tmp file
            os.remove(patchedFileName)
        else:
            # Rename to webui.py and backup original
            if not os.path.exists(os.path.join(DIR_FOOOCUS, "webui_original.py")):
                shutil.copy(PATH_TO_WEBUI, os.path.join(DIR_FOOOCUS, "webui_original.py"))
            else:
                currentDateTime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                shutil.copy(PATH_TO_WEBUI, os.path.join(DIR_FOOOCUS, f"webui_original_{currentDateTime}.py"))

            shutil.move(patchedFileName, PATH_TO_WEBUI)

    return "Ok" if pathResult else "Error"


def start_path():
    print("""=== Script for patching file webui.py for Fooocus ===
> Extension: 'Extention Panel'
> Author: Shahmatist/RMDA
=== ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ===""")

    isOk = search_and_path()
    if isOk == "Ok":
        print("\nPatched successfully!")

    elif isOk == "Already":
        print("\nPath already appied!")

    else:
        print("\nError path data incorrect!")


start_path()
