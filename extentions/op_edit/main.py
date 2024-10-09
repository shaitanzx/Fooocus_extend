





import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks
from modules.shared import opts
from modules.paths import models_path

from basicsr.utils.download_util import load_file_from_url



from PIL import Image



class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "OpenPose Editor"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

def on_ui_tabs():
  with gr.Blocks(analytics_enabled=False) as openpose_editor:

  return [(openpose_editor, "OpenPose Editor", "openpose_editor")]

script_callbacks.on_ui_tabs(on_ui_tabs)
