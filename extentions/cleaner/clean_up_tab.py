import gradio as gr
import modules.config
from PIL import Image
from litelama import LiteLama
from litelama.model import download_file
import os
import cv2
import numpy as np

EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "cleaner","model")
def clean_object_init_img_with_mask(init_img_with_mask):
    if init_img_with_mask:
        return clean_object(init_img_with_mask['image'],init_img_with_mask['mask']), gr.update(visible=True), gr.update(visible=True)
    else:
        return None, gr.update(visible=False), gr.update(visible=False)
def clean_object_video(frame,mask):
        return clean_object(frame,mask)
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        return None


def video_clean_process(video,frame):
    temp_dir_clean=modules.config.temp_path+os.path.sep
    print('================================',temp_dir_clean)
    mask=frame['mask'].convert("RGB")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    batch_path_clean=f"{temp_dir_clean}cleaner"+ os.path.sep
    Lama = LiteLama2()
    Lama.to(device)
    device = "cuda"
    for i in range(total_frames):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        print('111111111111111111111',frame)
        frame=Image.fromarray(frame)
        print('222222222222222222222',frame)
        #frame = frmae.convert("RGB")
        frame=Lama.predict(frame, mask)
        #image=clean_object_video(frame,mask)
        print('333333333333333333333',frame)
        cv2.imwrite(f'{batch_path_clean}frame_{i:06d}.png', frame)
    cap.release()
    
    
    
    init_image = image


    


    
    result = None
    try:
        
        
    except:
        pass
    finally:
        Lama.to("cpu")
    
    return [result]

#    images = [img for img in os.listdir('frames') 
#              if img.endswith(f".png")]
#    images.sort()
#    first_image = cv2.imread(os.path.join('frames', images[0]))    
#    height, width = first_image.shape[:2]
    
#    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#    video_name=f"{output_path}/video_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"

#    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    
#    for i, image_name in enumerate(images):
#        image_path = os.path.join('frames', image_name)
#        frame = cv2.imread(image_path)
#        out.write(frame)
#        yield f'Saved color frame number {i} of {total_frames} to videofile'
#    out.release()
#    yield 'Video colorization complete'
#    delete_input('frames')
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
