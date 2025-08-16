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
            save_tile = gr.Button(value='Save to OUTPUT FOLDER', visible=False)
    def copy_tile(image):
        height, width = image.shape[:2]
        x_shift = gr.update(minimum=-(width/2),maximum=width/2,value=0,visible=True)
        y_shift = gr.update(minimum=-(height/2),maximum=height/2,value=0,visible=True)
        return image,x_shift,y_shift,gr.update(visible=True)
    def shifting_x(shift,image):
        image = np.roll(image, shift, axis=1)
        return image
    def shifting_y(shift,image):
        image = np.roll(image, shift, axis=0)
        return image



    tile_load.upload(fcopy_tile, inputs=tile_load, outputs=[tile_copy,x_shift,y_shift,save_tile])
    x_shift.release(shifting_x, inputs=[x_shift,tile_copy],outputs=tile_load,show_progress=False)
    y_shift.release(shifting_y, inputs=[y_shift,tile_copy],outputs=tile_load,show_progress=False)
