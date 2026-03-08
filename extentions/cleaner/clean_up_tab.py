import gradio as gr
import modules.config
from PIL import Image
from litelama import LiteLama
from litelama.model import download_file
import os
import cv2
import numpy as np
import time
from modules.launch_util import delete_folder_content


EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "cleaner","model")
def clean_object_init_img_with_mask(init_img_with_mask):
    if init_img_with_mask:
        return clean_object(init_img_with_mask['image'],init_img_with_mask['mask']), gr.update(visible=True), gr.update(visible=True)
    else:
        return None, gr.update(visible=False), gr.update(visible=False)
def clean_object_video(frame,mask):
        return clean_object(frame,mask)
def get_first_frame(video_files):
    video_path = video_files[0].name
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return video_path,frame_rgb
    else:
        return None,None


def video_clean_process(video,mask):
    temp_dir_clean=modules.config.temp_path+os.path.sep
    batch_path_clean=f"{temp_dir_clean}cleaner"+ os.path.sep
    result=delete_folder_content(batch_path_clean, '')
    os.makedirs(batch_path_clean, exist_ok=True)
    mask=mask['mask'].convert("RGB")
    video_files = [f.name for f in video]
    Lama = LiteLama2()
    device = "cuda"
    Lama.to(device)
    gallery_files=[]
    for file_index, filename in enumerate(video_files):
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for i in range(total_frames):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame=Image.fromarray(frame)
            frame=Lama.predict(frame, mask)
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            yield f'Processed {i} of {total_frames} ({file_index+1} of {len(video_files)})',gallery_files
            #print(f'Processed {i} of {total_frames}')
            cv2.imwrite(f'{batch_path_clean}frame_{i:06d}.png', frame)
        cap.release()
        


        images = [img for img in os.listdir(batch_path_clean) 
                    if img.endswith(f".png")]
        images.sort()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_base_name = os.path.splitext(os.path.basename(filename))[0]

        video_name=os.path.join(modules.config.path_outputs, f'{video_base_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4')

        out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    
        for i, image_name in enumerate(images):
            image_path = os.path.join(batch_path_clean, image_name)
            frame = cv2.imread(image_path)
            out.write(frame)
        out.release()
        gallery_files += video_name
        yield None,gallery_files
        result=delete_folder_content(batch_path_clean, '')
        os.makedirs(batch_path_clean, exist_ok=True)
    Lama.to("cpu")
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
