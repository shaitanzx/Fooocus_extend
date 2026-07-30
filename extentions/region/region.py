import gradio as gr
import os




def gui():
    file_l = os.path.join(os.path.dirname(__file__),'mask.png')
    #file_r = os.path.join(os.path.dirname(__file__),'mask_right.png')
    with gr.Row():
        enable_region = gr.Checkbox(label="Enabled", value=True)
    with gr.Row():
        prompt_region= gr.Textbox(label='Prompt left', show_label=True, value='cute orange cat, sitting BREAK happy golden dog, running', lines=2)
        #prompt_r= gr.Textbox(label='Prompt right', show_label=True, value='happy golden dog, running', lines=2)
    with gr.Row():
        mask_region= gr.Image(value=file_l,label='Mask', source='upload', type='numpy',height=260, show_label=True,visible=True,interactive=True)
        #mask_r= gr.Image(value=file_r,label='Mask right', source='upload', type='numpy',height=260, show_label=True,visible=True,interactive=True)
    return enable_region, prompt_region, mask_region