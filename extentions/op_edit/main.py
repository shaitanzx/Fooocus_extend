import gradio as gr
import os
import json
import numpy as np
import cv2
from PIL import Image
import sys
import io
from .body import Body
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.model_loader import load_file_from_url
def title():
    return "OpenPose Editor"

#  def show(self, is_img2img):
#    return scripts.AlwaysVisible

def ui():
      global body_estimation
      body_estimation = None
      presets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"presets.json")
      presets = {}

      try: 
        with open(presets_file) as file:
          presets = json.load(file)
      except FileNotFoundError:
        pass

      def pil2cv(in_image):
        out_image = np.array(in_image, dtype=np.uint8)

        if out_image.shape[2] == 3:
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        return out_image

      def candidate2li(li):
        res = []
        for x, y, *_ in li:
          res.append([x, y])
        return res

      def subset2li(li):
        res = []
        for r in li:
          for c in r:
            res.append(c)
        return res                    
      with gr.Row():
        with gr.Column():
          width_ope = gr.Slider(label="width", minimum=64, maximum=2048, value=512, step=64, interactive=True)
          height_ope = gr.Slider(label="height", minimum=64, maximum=2048, value=512, step=64, interactive=True)
          with gr.Row():
            add_ope = gr.Button(value="Add", variant="primary")
                          # delete = gr.Button(value="Delete")
          with gr.Row():
            reset_btn_ope = gr.Button(value="Reset")
            json_input_ope = gr.UploadButton(label="Load from JSON", file_types=[".json"], elem_id="openpose_json_button")
            png_input_ope = gr.UploadButton(label="Detect from Image", file_types=["image"], type="bytes", elem_id="openpose_detect_button")
            bg_input_ope = gr.UploadButton(label="Add Background Image", file_types=["image"], elem_id="openpose_bg_button")
          with gr.Row():
            preset_list = gr.Dropdown(label="Presets", choices=sorted(presets.keys()), interactive=True)
            preset_load = gr.Button(value="Load Preset")
            preset_save = gr.Button(value="Save Preset")
          with gr.Row():
            json_output_ope = gr.Button(value="Save JSON")
            png_output_ope = gr.Button(value="Save PNG")
        with gr.Column():
        # gradioooooo...
          canvas_ope = gr.HTML('<canvas id="openpose_editor_canvas" width="512" height="512" style="margin: 0.25rem; border-radius: 0.25rem; border: 0.5px solid"></canvas>')
          jsonbox_ope = gr.Text(label="json", elem_id="jsonbox", visible=False)
      with gr.Row():
            gr.HTML('* \"OpenPose Editor\" is powered by fkunn1326. <a href="https://github.com/fkunn1326/openpose-editor" target="_blank">\U0001F4D4 Document</a>')
      with gr.Row():
            gr.HTML('* Modification and adaptation for Fooocus is powered by Shahmatist^RMDA')
      def estimate(file):
        global body_estimation

        if body_estimation is None:
          model_path_ope = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), "models", "controlnet")
          model_file_ope = os.path.join(model_path_ope, "body_pose_model.pth")
          if not os.path.isfile(model_path_ope):

            load_file_from_url(
              url="https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/body_pose_model.pth",
              model_dir=model_path_ope,
              file_name='body_pose_model.pth')


          body_estimation = Body(model_file_ope)
        
        stream = io.BytesIO(file)
        img = Image.open(stream)
        candidate, subset = body_estimation(pil2cv(img))

        result = {
          "candidate": candidate2li(candidate),
          "subset": subset2li(subset),
        }
      
        return str(result).replace("'", '"')

      def savePreset(name, data):
        if name:
          presets[name] = json.loads(data)
          with open(presets_file, "w") as file:
            json.dump(presets, file)
          return gr.update(choices=sorted(presets.keys()), value=name), json.dumps(data)
        return gr.update(), gr.update()

      dummy_component = gr.Label(visible=False)

      preset_ope = gr.Text(visible=False)
      width_ope.change(None, [width_ope, height_ope], None, _js="(w, h) => {resizeCanvas(w, h)}")
      height_ope.change(None, [width_ope, height_ope], None, _js="(w, h) => {resizeCanvas(w, h)}")
      png_output_ope.click(None, [], None, _js="savePNG")
      bg_input_ope.upload(None, [], [width_ope, height_ope], _js="() => {addBackground('openpose_bg_button')}")
      png_input_ope.upload(estimate, png_input_ope, jsonbox_ope)
      png_input_ope.upload(None, [], [width_ope, height_ope], _js="() => {addBackground('openpose_detect_button')}")
      add_ope.click(None, [], None, _js="addPose")
      reset_btn_ope.click(None, [], None, _js="resetCanvas")
      json_input_ope.upload(None, json_input_ope, [width_ope, height_ope], _js="() => {loadJSON('openpose_json_button')}")
      json_output_ope.click(None, None, None, _js="saveJSON")
      preset_save.click(savePreset, [dummy_component, dummy_component], [preset_list, preset_ope], _js="savePreset")
      preset_load.click(None, preset_ope, [width_ope, height_ope], _js="loadPreset")
      preset_list.change(lambda selected: json.dumps(presets[selected]), preset_list, preset_ope)
