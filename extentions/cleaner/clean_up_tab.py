import gradio as gr
import modules.gradio_hijack as grh
from extentions.cleaner import lama
from PIL import Image
def send_to_cleaner(result):
    image = Image.open(result[0]["name"])
    return image

def ui(init_img_with_mask):
    clean_up_init_img = None
    clean_up_init_mask = None
    with gr.Row():
        clean_button = gr.Button("Clean Up", height=100)
    with gr.Row():    
        result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"cleanup_gallery", preview=True, height=512)
    with gr.Row():
        send_to_cleaner_button = gr.Button("Send back To clean up", height=100)

    clean_button.click(fn=lama.clean_object_init_img_with_mask,inputs=[init_img_with_mask],outputs=[result_gallery])

    send_to_cleaner_button.click(fn=send_to_cleaner,inputs=[result_gallery],outputs=[init_img_with_mask])
    return

