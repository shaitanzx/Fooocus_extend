import gradio as gr
import os




def gui():
    file_l = os.path.join(os.path.dirname(__file__),'mask_left.png')
    file_r = os.path.join(os.path.dirname(__file__),'mask_right.png')
    with gr.Row():
        enable_region = gr.Checkbox(label="Enabled", value=False)
    with gr.Row():
        prompt_l= gr.Textbox(value=file_l,label='Prompt left', show_label=True, value='cute orange cat, sitting', lines=2)
        prompt_r= gr.Textbox(value=file_r,label='Prompt right', show_label=True, value='happy golden dog, running', lines=2)
    with gr.Row():
        mask_l= gr.Image(label='Mask left', source='upload', type='numpy',height=260, show_label=True,visible=True,interactive=True)
        mask_r= gr.Image(label='Mask right', source='upload', type='numpy',height=260, show_label=True,visible=True,interactive=True)
    return enable_region, [prompt_l,prompt_r], [mask_l,mask_r]