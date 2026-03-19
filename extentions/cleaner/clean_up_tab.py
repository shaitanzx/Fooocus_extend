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
import zipfile
import io



EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "cleaner","model")
temp_dir=modules.config.temp_path+os.path.sep

def get_first_image(image_files):
    image_path = image_files[0].name
    return image_path
def get_first_image_zip(zip_file):
    zip_path = zip_file.name
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        first_file_name = file_list[0]
        file_data = z.read(first_file_name)
        image = Image.open(io.BytesIO(file_data))
        return image
def process_image(mask,mask_check,mask_load):
    batch_path=f"{temp_dir}batch_cleaner"
    batch_temp=f"{temp_dir}batch_temp"
    batch_files=sorted([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    batch_all=len(batch_files)
    if mask_check and mask_load is not None:
        m1 = np.array(mask['mask'].convert("RGB"))
        m2 = np.array(mask_load.convert("RGB"))
        mask = Image.fromarray(np.maximum(m1, m2), mode="RGB")
    else:
        mask = mask['mask'].convert("RGB")
    Lama = LiteLama2()
    Lama.to("cuda")
    gallery_names=[]
    passed=1
    for f_name in batch_files:
        print (f"\033[91m[Cleaner QUEUE] {passed} / {batch_all}. Filename: {f_name} \033[0m")
        gr.Info(f"Cleaner Batch: start element generation {passed}/{batch_all}. Filename: {f_name}") 
        img = Image.open(batch_path+os.path.sep+f_name)
        yield gr.update(value=img,visible=True),gr.update(visible=False),gr.update(visible=False)
        source_image=Lama.predict(img, mask)
        name, ext = os.path.splitext(f_name)
        filename =  batch_temp + os.path.sep + name +'_clean'+ext
        source_image.save(filename)
        passed+=1
        gallery_names +=[filename]
    Lama.to("cpu")
    yield gr.update(value=None,visible=False),gallery_names,gr.update(visible=True)

def video_clean_process(video, mask, mask_check, mask_load):
    batch_path = f"{temp_dir}batch_cleaner_video"
    os.makedirs(batch_path, exist_ok=True)
    
    # Обработка маски
    if mask_check and mask_load is not None:
        m1 = np.array(mask['mask'].convert("RGB"))
        m2 = np.array(mask_load.convert("RGB"))
        mask_combined = Image.fromarray(np.maximum(m1, m2), mode="RGB")
    else:
        mask_combined = mask['mask'].convert("RGB")
    
    video_files = [f.name for f in video]
    Lama = LiteLama2()
    device = "cuda"
    Lama.to(device)
    output_video_names = []
    
    for file_index, input_path in enumerate(video_files):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Подготовка имени выходного файла
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(batch_path, f"{base_name}_clean.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(total_frames):
            ret, frame = cap.read()

            # Обработка кадра
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = Lama.predict(frame, mask_combined)
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
            # ПРЯМАЯ ЗАПИСЬ В ВИДЕО (без сохранения на диск)
            out.write(frame)
            
            yield f'Processed {i + 1} of {total_frames} ({file_index + 1} of {len(video_files)})', None
            
        cap.release()
        out.release()
        output_video_names.append(output_path)
        
    Lama.to("cpu")
    yield "Processing complete.", output_video_names







def clean_zip(filenames):
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    zip_filename = f"cleaned_images_{timestamp}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)
    valid_files = [item.get('name') for item in filenames if item.get('name')]
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in valid_files:
            arcname = os.path.basename(file_path)
            zipf.write(file_path, arcname=arcname)   
    return zip_path


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
def get_first_image(image_files):
    image_path = image_files[0].name
    return image_path
    

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
