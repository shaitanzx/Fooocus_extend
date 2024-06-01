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
import urllib.request\n"""],


    ["from modules.auth import auth_enabled, check_auth\n","""from modules.module_translate import translate, GoogleTranslator
from urllib.parse import urlparse, parse_qs, unquote
from modules.model_loader import load_file_from_url
from rembg import remove
from PIL import Image\n"""],
    ["def get_task(*args):\n", """    # Prompt translate AlekPet
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
    # end -Prompt translate AlekPet\n"""],
    [
        "            desc_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)\n",
        """            def downloader(civitai_api_key,downloader_checkpoint,downloader_loras,downloader_embd):
              if not civitai_api_key:
                return
              model_dir='/content/Fooocus/models/checkpoints/'
              urls_download = downloader_checkpoint
              download_files (model_dir,urls_download,civitai_api_key)
              model_dir='/content/Fooocus/models/loras/'
              urls_download = downloader_loras
              download_files (model_dir,urls_download,civitai_api_key)
              model_dir='/content/Fooocus/models/embeddings/'
              urls_download = downloader_embd
              download_files (model_dir,urls_download,civitai_api_key)
              return civitai_api_key
            def download_files (model_dir,urls_download,civitai_api_key):
              if not urls_download:
                return
              URLs_paths = urls_download.split(',')
              for main_url in URLs_paths:
                URL=main_url.split('?')
                url_down = URL[0] + '?token='+civitai_api_key
                USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                headers = {
                  'Authorization': f'Bearer {civitai_api_key}',
                  'User-Agent': USER_AGENT,
                  }
                class NoRedirection(urllib.request.HTTPErrorProcessor):
                    def http_response(self, request, response):
                        return response
                    https_response = http_response
                request = urllib.request.Request(url_down, headers=headers)
                opener = urllib.request.build_opener(NoRedirection)
                response = opener.open(request)
                if response.status in [301, 302, 303, 307, 308]:
                    redirect_url = response.getheader('Location')
                    parsed_url = urlparse(redirect_url)
                    query_params = parse_qs(parsed_url.query)
                    content_disposition = query_params.get('response-content-disposition', [None])[0]
                    if content_disposition:
                        filename = unquote(content_disposition.split('filename=')[1].strip('"'))
                    else:
                        raise Exception('Unable to determine filename')
                    response = urllib.request.urlopen(redirect_url)
                elif response.status == 404:
                    raise Exception('File not found')
                else:
                    raise Exception('No redirect found, something went wrong')
                load_file_from_url(url=url_down,model_dir=model_dir,file_name=filename)
              return


            with gr.Row(elem_classes='extend_row'):
               with gr.Accordion('Extention', open=False):
                  with gr.TabItem(label='Model Dowloader') as download_tab:
                        with gr.Row():
                            civitai_api_key=downloader_checkpoint=gr.Textbox(label='Civitai API Key', show_label=True, interactive=True, value='')
                        with gr.Row():
                            downloader_checkpoint=gr.Textbox(label='Checkpoint Link', show_label=True, interactive=True, value='')
                        with gr.Row():
                            downloader_loras=gr.Textbox(label='Lora Link', show_label=True, interactive=True)
                        with gr.Row():
                            downloader_embd=gr.Textbox(label='Embedding Link', show_label=True, interactive=True)
                        with gr.Row():
                            download_start = gr.Button(value='Start Download')
                        download_start.click(downloader, inputs=[civitai_api_key,downloader_checkpoint,downloader_loras,downloader_embd],outputs=civitai_api_key)
                        gr.HTML('For apply emmbeding, in the prompt field use a record like (embedding:file_name:1.1)')
                        gr.HTML('* \"Model Downloader\" is powered by Shahmatist^RMDA')
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
                        with gr.Column(scale=1):
                            rembg_input = grh.Image(label='Drag above image to here', source='upload', type='filepath', scale=20)
                            rembg_button = gr.Button(value='Remove Background', interactive=True, scale=1)
                        with gr.Column(scale=3):
                            rembg_output = grh.Image(label='rembg Output', interactive=False, height=380)
                        gr.Markdown('Powered by [🪄 rembg 2.0.53](https://github.com/danielgatis/rembg/releases/tag/v2.0.53)')
                        rembg_button.click(rembg_run, inputs=rembg_input, outputs=rembg_output, show_progress='full')
                  gr.Markdown('* \"Extention panel\" is powered by Shahmatist^RMDA')\n"""],
    ["            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)\n","""
        # [start] Prompt translate AlekPet
        def seeTranlateAfterClick(adv_trans, prompt, negative_prompt="", srcTrans="auto", toTrans="en"):
            if(adv_trans):
                positive, negative = translate(prompt, negative_prompt, srcTrans, toTrans)
                return [positive, negative]   
            return ["", ""]
        
        gtrans.click(translate, inputs=[prompt, negative_prompt, srcTrans, toTrans], outputs=[prompt, negative_prompt])
        gtrans.click(translate, inputs=[prompt, negative_prompt, srcTrans, toTrans], outputs=[p_tr, p_n_tr])
        
        change_src_to.click(change_lang, inputs=[srcTrans,toTrans], outputs=[toTrans,srcTrans])
        adv_trans.change(show_viewtrans, inputs=adv_trans, outputs=[viewstrans])
        # [end] Prompt translate AlekPet\n"""],
["        ctrls += ip_ctrls\n", "        ctrls += [translate_enabled, translate_automate, srcTrans, toTrans]\n"],
["            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \\\n","""            .then(fn=seeTranlateAfterClick, inputs=[adv_trans, prompt, negative_prompt, srcTrans, toTrans], outputs=[p_tr, p_n_tr]) \\\n"""]
    
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
