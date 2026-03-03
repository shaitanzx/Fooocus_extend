#import modules.scripts as scripts
import gradio as gr
import modules.gradio_hijack as grh

#from modules.shared import opts,OptionInfo

#from modules import script_callbacks
#from modules.ui_components import ToolButton, ResizeHandleRow
#import modules.generation_parameters_copypaste as parameters_copypaste
#from modules.ui_common import save_files

from extentions.cleaner import lama
from PIL import Image

#def on_ui_settings():
#    section = ('cleaner', "Cleaner")
#    opts.add_option("cleaner_use_gpu", OptionInfo(True, "Is Use GPU", gr.Checkbox, {"interactive": True}, section=section))


def send_to_cleaner(result):
    image = Image.open(result[0]["name"])

    print(image)

    return image

def ui():

    init_img_with_mask = None
    clean_up_init_img = None
    clean_up_init_mask = None
    with gr.Row():
        init_img_with_mask = grh.Image(label='Image', source='upload', type='pil', show_label=False)
        
        # gr.Image(label="Image for clean up with mask", show_label=False, elem_id="cleanup_img2maskimg", source="upload",
        #                interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=650, brush_color="#FFFFFF")
    with gr.Row():
        clean_button = gr.Button("Clean Up", height=100)
    with gr.Row():    
        result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"cleanup_gallery", preview=True, height=512)
    with gr.Row():
        send_to_cleaner_button = gr.Button("Send back To clean up", height=100)

    clean_button.click(fn=lama.clean_object_init_img_with_mask,inputs=[init_img_with_mask],outputs=[result_gallery])

    send_to_cleaner_button.click(fn=send_to_cleaner,inputs=[result_gallery],outputs=[init_img_with_mask])
    return 


#script_callbacks.on_ui_tabs(on_ui_tabs)
#script_callbacks.on_ui_settings(on_ui_settings)

