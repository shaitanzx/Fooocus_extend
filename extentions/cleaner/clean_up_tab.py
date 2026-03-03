import gradio as gr
#from extentions.cleaner import lama
from PIL import Image
from litelama import LiteLama
from litelama.model import download_file
import os
EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "cleaner","models")
print ('--------------------',EXTENSION_PATH)
print ('--------------------',MODEL_PATH)
def clean_object_init_img_with_mask(init_img_with_mask):
    return clean_object(init_img_with_mask['image'],init_img_with_mask['mask'])


def clean_object(image,mask):
    
    Lama = LiteLama2()
    
    init_image = image
    mask_image = mask

    init_image = init_image.convert("RGB")
    mask_image = mask_image.convert("RGB")

    #device_used = opts.data.get("cleaner_use_gpu",True)

    device = "cuda"
    #if not device_used:
    #    device = "cpu"

    result = None
    try:
        Lama.to(device)
        result = Lama.predict(init_image, mask_image)
    except:
        pass
    finally:
        Lama.to("cpu")
    
    return [result]


class LiteLama2(LiteLama):
    
    _instance = None
    
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
        
    def __init__(self, checkpoint_path=None, config_path=None):
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._model = None
        
        if self._checkpoint_path is None:
            os.makedirs(MODEL_PATH, exist_ok=True)
            checkpoint_path = os.path.join(MODEL_PATH, "big-lama.safetensors")
            if  os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
                pass
            else:
                download_file("https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors", checkpoint_path)
                
            self._checkpoint_path = checkpoint_path
        
        self.load(location="cpu")

def send_to_cleaner(result):
    image = Image.open(result[0]["name"])
    return image

def ui(init_img_with_mask):
    clean_up_init_img = None
    clean_up_init_mask = None
    with gr.Row():
        clean_button = gr.Button("Clean Up", height=100)
    with gr.Row():    
        result_gallery = gr.Gallery(show_fullscreen_button=True, label='Output', show_label=False, elem_id=f"cleanup_gallery", preview=False, height=512)
    with gr.Row():
        send_to_cleaner_button = gr.Button("Send back To clean up", height=100)

    clean_button.click(fn=clean_object_init_img_with_mask,inputs=[init_img_with_mask],outputs=[result_gallery])

    send_to_cleaner_button.click(fn=send_to_cleaner,inputs=[result_gallery],outputs=[init_img_with_mask])
    return

