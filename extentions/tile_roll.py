import gradio as gr
import cv2


def start():
    with gr.Column():
        with gr.Row():
            tile_load = gr.Image(label="Upload file of tile", type="numpy")
            tile_copy = gr.Image(label="Upload file of tile", type="numpy",visible=False)
    with gr.Column():
        with gr.Row():
            y_shift = gr.Slider(label="Y Shift",minimum=0,maximum=1.5,step=1,value=0,interactive=True,visible=False)
        with gr.Row():
            x_shift = gr.Slider(label="X Shift",minimum=0,maximum=1.5,step=1,value=0,interactive=True,visible=False)
        with gr.Row():
            save_tile = gr.Button(value='Save to OUTPUT FOLDER', visible=True)
    def copy_tile(image):
        height, width = image.shape[:2]
        x_shift = gr.update(minimum=-(width/2),maximum=width/2,value=0,visible=True)
        y_shift = gr.update(minimum=-(height/2),maximum=height/2,value=0,visible=True)
        return image,x_shift,ysift


    tile_load.upload(fn=copy_tile, inputs=tile_load, outputs=[tile_copy,x_shift,y_shift])
