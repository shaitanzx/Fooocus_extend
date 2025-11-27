import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Теперь можно использовать абсолютный импорт
import modules

def get_civitai_api_key():
    file_path = 'civitai_api_key.txt'
    
    try:
        with open(file_path, 'r') as file:
            key = file.read()
    except FileNotFoundError:
        # Создаем пустой файл, если он не существует
        with open(file_path, 'w') as file:
            pass
        key = ''
    
    return key
BUTTONS = {
    "replace_preview_button": False,
    "open_url_button": False,
    "add_trigger_words_button": False,
    "add_preview_prompt_button": False,
    "rename_model_button": False,
    "remove_model_button": False,
}
ch_civiai_api_key=get_civitai_api_key()
ch_dl_lyco_to_lora=False
ch_open_url_with_js=True
ch_hide_buttons=[x for x, y in BUTTONS.items() if y]
ch_always_display=False
ch_max_size_preview=True
ch_download_examples=False
ch_nsfw_threshold=False
ch_dl_webui_metadata=True
ch_proxy=""
path_ckp=modules.config.paths_checkpoints[0]
path_lora=modules.config.paths_loras[0]
path_emb=modules.config.path_embeddings
path_vae=modules.config.path_vae
path_detect = str(modules.config.path_yolo)
path_upscale = modules.config.path_upscale_models
