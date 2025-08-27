import sys
import gradio as gr
import modules.gradio_hijack as grh
from PIL import Image
import cv2
import numpy as np
from typing import List, Union, Dict, Set, Tuple
import modules.config
from extentions.inswapper.swapper import process
import extentions.batch as batch
temp_dir=modules.config.temp_path+os.path.sep

def inswapper_gui():
  with gr.Row():
    with gr.Column():
      inswapper_enabled = gr.Checkbox(label="Enabled", value=False)
      inswapper_source_image_indicies = gr.Text(label="Source Image Index", info="-1 will swap all faces, otherwise provide the 0-based index of the face (0, 1, etc)", value="0")
      inswapper_target_image_indicies = gr.Text(label = "Target Image Index", info="-1 will swap all faces, otherwise provide the 0-based index of the face (0, 1, etc)", value="0")
    with gr.Column():
      inswapper_source_image = grh.Image(label='Source Face Image', source='upload', type='numpy')
  with gr.Row():
    gr.HTML('* \"inswapper\" is powered by haofanwang. <a href="https://github.com/haofanwang/inswapper" target="_blank">\U0001F4D4 Document</a>')
  return inswapper_enabled,inswapper_source_image_indicies,inswapper_target_image_indicies,inswapper_source_image

def get_image(input_data: Union[list, np.ndarray]) -> np.ndarray:
    if isinstance(input_data, (list, tuple)) and len(input_data) > 0:
        return input_data[0],True
    elif isinstance(input_data, np.ndarray):
        return input_data,False

def perform_face_swap(images, inswapper_source_image, inswapper_source_image_indicies, inswapper_target_image_indicies):
  modules.config.downloading_inswapper()
  swapped_images = []
  item,generator=get_image(images)
  source_image = Image.fromarray(inswapper_source_image)
  print(f"Inswapper: Source indicies: {inswapper_source_image_indicies}")
  print(f"Inswapper: Target indicies: {inswapper_target_image_indicies}") 
  
  result_image = process([source_image], item, inswapper_source_image_indicies, inswapper_target_image_indicies, f"{modules.config.path_clip_vision}/inswapper_128.onnx")
  restored_img = np.array(result_image)
  swapped_images.append(restored_img)
  if generator:
    return swapped_images
  else:
    return np.array(restored_img)

def inswapper_gui2():
    def zip_enable(enable,single_file):
        if enable:
            return gr.update(visible=True),gr.update(visible=False),gr.update(visible=False)
        else:
            if single_file and len(single_file)==1:
                return gr.update(visible=False),gr.update(visible=False),gr.update(visible=True)
            else:
                return gr.update(visible=False),gr.update(visible=True),gr.update(visible=False)
    def clear_single(image):
        return gr.update(value=None,visible=False),gr.update(value=None,visible=True)
    def single_image(single_upload):
        if len(single_upload) == 1:
            return gr.update (value=single_upload[0].name,visible=True),gr.update(visible=False)
        else:
            return gr.update (visible=False),gr.update(visible=True)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                file_in_face=gr.File(label="Upload a ZIP file of Source Face",file_count='single',file_types=['.zip'],visible=False,height=260)
                files_single_face = gr.Files(label="Drag (Select) 1 or more Source Face images",file_count="multiple",
                                            file_types=["image"],visible=True,interactive=True,height=260)
                image_single_face=gr.Image(label="Source Face",visible=False,height=260,interactive=True,type="filepath")
            with gr.Row():
                enable_zip_face = gr.Checkbox(label="Upload ZIP-file", value=False)
        with gr.Column():
            with gr.Row():
                file_in_image=gr.File(label="Upload a ZIP file of Source Image",file_count='single',file_types=['.zip'],visible=False,height=260)
                files_single_image = gr.Files(label="Drag (Select) 1 or more Source images",file_count="multiple",
                                            file_types=["image"],visible=True,interactive=True,height=260)
                image_single_image=gr.Image(label="Source Image",visible=False,height=260,interactive=True,type="filepath")
            with gr.Row():
                enable_zip_image = gr.Checkbox(label="Upload ZIP-file", value=False)
        with gr.Column():
            with gr.Row():
                file_out=gr.File(label="Download a ZIP file", file_count='single',height=260,visible=True)
                preview=gr.Image(label="Process preview",visible=False,height=260,interactive=False)
                image_out=gr.Image(label="Output image",visible=False,height=260,interactive=False)
    with gr.Row():
        with gr.Column():
            inswap_source_image_indicies = gr.Text(label="Source Image Index", info="-1 will swap all faces, otherwise provide the 0-based index of the face (0, 1, etc)", value="0")
        with gr.Column():        
            inswap_target_image_indicies = gr.Text(label = "Target Image Index", info="-1 will swap all faces, otherwise provide the 0-based index of the face (0, 1, etc)", value="0")
    with gr.Row():
            inswap_start=gr.Button(value='Start inswapper')
    with gr.Row():
        gr.HTML('* \"inswapper\" is powered by haofanwang. <a href="https://github.com/haofanwang/inswapper" target="_blank">\U0001F4D4 Document</a>')
    with gr.Row(visible=False):
        ext_dir_face=gr.Textbox(value='batch_insw_face',visible=False)
        ext_dir_image=gr.Textbox(value='batch_insw_image',visible=False)
    enable_zip_face.change(fn=zip_enable,inputs=[enable_zip_face,files_single_face],outputs=[file_in_face,files_single_face,image_single_face],show_progress=False)
    enable_zip_image.change(fn=zip_enable,inputs=[enable_zip_image,files_single_image],outputs=[file_in_image,files_single_image,image_single_image],show_progress=False)
    image_single_face.clear(fn=clear_single,inputs=image_single_face,outputs=[image_single_face,files_single_face],show_progress=False)
    image_single_image.clear(fn=clear_single,inputs=image_single_image,outputs=[image_single_image,files_single_image],show_progress=False)
    files_single_face.upload(fn=single_image,inputs=files_single_face,outputs=[image_single_face,files_single_face],show_progress=False)
    files_single_image.upload(fn=single_image,inputs=files_single_image,outputs=[image_single_image,files_single_image],show_progress=False)
    

    start.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False)),outputs=[start,file_out]) \
              .then(fn=batch.clear_dirs,inputs=ext_dir) \
              .then(fn=batch.unzip_file,inputs=[file_in,files_single,enable_zip,ext_dir]) \
              .then(fn=process, inputs=[poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poTransPNGQuant],
                        outputs=[preview,file_out],show_progress=False) \
              .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out,preview],show_progress=False) \
              .then(fn=batch.output_zip_one, outputs=file_out) \
              .then(lambda: (gr.update(visible=True, interactive=True)),outputs=start)
    
    
    
    
    inswap_start.click(perform_face_swap,inputs=[inswap_original_image, inswap_source_image, inswap_source_image_indicies, inswap_target_image_indicies],outputs=inswap_output)
