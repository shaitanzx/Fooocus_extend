import sys
import gradio as gr
import modules.gradio_hijack as grh
from PIL import Image
import cv2
import numpy as np
from typing import List, Union, Dict, Set, Tuple
import modules.config
from extentions.inswapper.swapper import process
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

