import gradio as gr


BASE_PROMPT=",(((lineart))),((low detail)),(simple),high contrast,sharp,2 bit"
BASE_NEGPROMPT="(((text))),((color)),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error,Illustration, watermark"

StyleDict = {
    "Illustration":BASE_PROMPT+",(((vector graphic))),medium detail",
    "Logo":BASE_PROMPT+",(((centered vector graphic logo))),negative space,stencil,trending on dribbble",
    "Drawing":BASE_PROMPT+",(((cartoon graphic))),childrens book,lineart,negative space",
    "Artistic":BASE_PROMPT+",(((artistic monochrome painting))),precise lineart,negative space",
    "Tattoo":BASE_PROMPT+",(((tattoo template, ink on paper))),uniform lighting,lineart,negative space",
    "Gothic":BASE_PROMPT+",(((gothic ink on paper))),H.P. Lovecraft,Arthur Rackham",
    "Anime":BASE_PROMPT+",(((clean ink anime illustration))),Studio Ghibli,Makoto Shinkai,Hayao Miyazaki,Audrey Kawasaki",
    "Cartoon":BASE_PROMPT+",(((clean ink funny comic cartoon illustration)))",
    "Sticker":",(Die-cut sticker, kawaii sticker,contrasting background, illustration minimalism, vector, pastel colors)",
    "Gold Pendant": ",gold dia de los muertos pendant, intricate 2d vector geometric, cutout shape pendant, blueprint frame lines sharp edges, svg vector style, product studio shoot",
    "None - prompt only":""
}

def ui():
    def prompt(style):
        return StyleDict[style]
    vector_prompt=prompt("Illustration")
    with gr.Row():
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        with gr.Row():
                            poDoVector = gr.Checkbox(label="Enable Vectorizing", value=True,interactive=True)
                            poOpaque = gr.Checkbox(label="White is Opaque", value=True,interactive=True)
                            poTight = gr.Checkbox(label="Cut white margin from input", value=True,interactive=True)
                        with gr.Row():
                            poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
                            poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5,interactive=True)

            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        poTransPNG = gr.Checkbox(label="Transparent PNG",value=False)
                        poTransPNGEps = gr.Slider(label="Noise Tolerance",minimum=0,maximum=128,value=16,interactive=True)
                        poTransPNGQuant = gr.Slider(label="Quantize",minimum=2,maximum=255,value=16,interactive=True)
        poUseColor = gr.Radio(list(StyleDict.keys()), label="Visual style", value="Illustration",interactive=True)
        prompt_box=gr.Textbox(value=vector_prompt,visible=False)
    poUseColor.change(prompt,inputs=poUseColor,outputs=prompt_box)    
    return prompt_box, poOpaque, poTight, poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poDoVector,poTransPNGQuant
