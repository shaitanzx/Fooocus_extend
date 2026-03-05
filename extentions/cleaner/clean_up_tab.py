import gradio as gr
import modules.config
from PIL import Image
from litelama import LiteLama
from litelama.model import download_file
import os
EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "cleaner","model")
def clean_object_init_img_with_mask(init_img_with_mask):
    if init_img_with_mask:
        return clean_object(init_img_with_mask['image'],init_img_with_mask['mask']), gr.update(visible=True), gr.update(visible=True)
    else:
        return None, gr.update(visible=False), gr.update(visible=False)
def clean_object_video(frame,mask):
        return clean_object(frame,mask)

def clean_object(image,mask):
    
    Lama = LiteLama2()
    
    init_image = image
    mask_image = mask

    init_image = init_image.convert("RGB")
    mask_image = mask_image.convert("RGB")

    device = "cuda"
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
                
            self._checkpoint_path = modules.config.downloading_cleaner(MODEL_PATH)
        
        self.load(location="cpu")

def send_to_cleaner(result):
    image = Image.open(result[0]["name"])
    return image
