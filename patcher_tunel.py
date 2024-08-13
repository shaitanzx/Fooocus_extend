# Script for patching file webui.py for Fooocus
# Author: AlekPet & Shahmatist/RMDA

import os
import datetime
import shutil


DIR_FOOOCUS = os.path.dirname(os.path.abspath(__file__))
PATH_TO_WEBUI = os.path.join(DIR_FOOOCUS, "webui.py")

PATH_OBJ_DATA_PATCHER = [
    ["# dump_default_english_config()\n","""import subprocess
import threading
import time
import socket

def iframe_thread(port):
    while True:
        time.sleep(0.5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        if result == 0:
            break
        sock.close()
    print(\"\\nFooocus finished loading, trying to launch cloudflared (if it gets stuck here cloudflared is having issues)\\n\")
    p = subprocess.Popen([\"cloudflared\", \"tunnel\", \"--url\", \"http://127.0.0.1:{}\".format(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in p.stderr:
        l = line.decode()
        if \"trycloudflare.com\" in l:
            print(\"This is the URL to access Fooocus:\", l[l.find(\"https\"):], end='')

port = 7865 # Replace with the port number used by Fooocus
threading.Thread(target=iframe_thread, daemon=True, args=(port,)).start()\n"""],

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
> Extension: 'Tunnel Patch'
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
