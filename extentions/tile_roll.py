import gradio as gr
import cv2
import numpy as np
import modules.util
import modules.config
from PIL import Image


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
            save_tile = gr.Button(value='Save to OUTPUT FOLDER', visible=False)
    def copy_tile(image):
        height, width = image.shape[:2]
        x_shift = gr.update(minimum=-(width/2),maximum=width/2,value=0,visible=True)
        y_shift = gr.update(minimum=-(height/2),maximum=height/2,value=0,visible=True)
        return image,x_shift,y_shift,gr.update(visible=True)
    def shifting(y_shift,x_shift,image):
        image = np.roll(image, (-y_shift,x_shift), axis=(0,1))
        return image
    def clear_tile(image):
        if image is None:
            return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)
        else:
            return gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)
    def save_image(image):

        _, filename, _ = modules.util.generate_temp_filename(folder=modules.config.path_outputs,
                                                                extension='png',name_prefix='roll')
        os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)
        img = Image.fromarray(image)
        print('---------------------------',filename)
        img.save(filename)


    save_tile(save_image,inputs=tile_load)
    tile_load.upload(copy_tile, inputs=tile_load, outputs=[tile_copy,x_shift,y_shift,save_tile],show_progress=False)
    tile_load.change(clear_tile, inputs=tile_load, outputs=[x_shift,y_shift,save_tile],show_progress=False)
    x_shift.release(shifting, inputs=[y_shift,x_shift,tile_copy],outputs=tile_load,show_progress=False)
    y_shift.release(shifting, inputs=[y_shift,x_shift,tile_copy],outputs=tile_load,show_progress=False)
