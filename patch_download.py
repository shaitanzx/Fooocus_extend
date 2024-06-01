# Script for patching file webui.py for Fooocus
# Author: AlekPet & Shahmatist/RMDA

import os
import datetime
import shutil

DIR_FOOOCUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fooocus")
PATH_TO_WEBUI = os.path.join(DIR_FOOOCUS, "webui.py")

PATH_OBJ_DATA_DOWNLOAD_MODEL = [
    ["import copy\n","import requests\n"],
    ["import launch\n","""import re
import urllib.request\n"""],


    ["from modules.auth import auth_enabled, check_auth\n","""from urllib.parse import urlparse, parse_qs, unquote
from modules.model_loader import load_file_from_url\n"""],   
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

            with gr.Row(elem_classes='downloader_row'):
                 with gr.Accordion('Model Dowloader', open=False):
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
                        download_start.click(downloader, inputs=[civitai_api_key,downloader_checkpoint,downloader_loras,downloader_embd],outputs=civitai_api_key)\n"""],
]


def search_and_path():
    isOk = 0
    pathesLen = len(PATH_OBJ_DATA_DOWNLOAD_MODEL)
    patchedFileName = os.path.join(DIR_FOOOCUS, "webui_patched.py")

    with open(PATH_TO_WEBUI, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        len_lines = len(lines)

        if not len_lines:
            print(f"File '{PATH_TO_WEBUI}' is empty!\n")
            return

        if PATH_OBJ_DATA_DOWNLOAD_MODEL[0][1] in lines:
            return "Already"

        pathed = 0
        pathSteps = 100 / pathesLen

        patchedFile = open(patchedFileName, 'w+', encoding='utf-8')

        for line in lines:
            for linepath in PATH_OBJ_DATA_DOWNLOAD_MODEL:
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
> Extension: 'Download Model'
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
