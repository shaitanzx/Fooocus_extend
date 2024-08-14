# Script for patching file webui.py for Fooocus
# Author: AlekPet & Shahmatist/RMDA

import os
import datetime
import shutil

DIR_FOOOCUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fooocus")
PATH_TO_WEBUI = os.path.join(DIR_FOOOCUS, "modules/ui_gradio_extentions.py")

PATH_OBJ_DATA_PATCHER = [
    ["# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py\n","# patched\n"],
    ["""    head += f'<script type="text/javascript" src="{image_viewer_js_path}"></script>\\n'\n""","""
    civitai_js_path = webpath(\'md_lib/civitai_helper.js\')
    head += f\'<script type="text/javascript\" src=\"{civitai_js_path}\"></script>\\n\'\n"""]
]    

def search_and_path():
    isOk = 0
    pathesLen = len(PATH_OBJ_DATA_PATCHER)
    patchedFileName = os.path.join(DIR_FOOOCUS, "modules/ui_gradio_extentions_patched.py")

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
            if not os.path.exists(os.path.join(DIR_FOOOCUS, "modules/ui_gradio_extentions_original.py")):
                shutil.copy(PATH_TO_WEBUI, os.path.join(DIR_FOOOCUS, "modules/ui_gradio_extentions_original.py"))
            else:
                currentDateTime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                shutil.copy(PATH_TO_WEBUI, os.path.join(DIR_FOOOCUS, f"modules/ui_gradio_extentions_original_{currentDateTime}.py"))

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
