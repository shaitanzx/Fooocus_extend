import gradio as gr


def ui():

    with gr.Row():
        with gr.Row():
            poDoVector = gr.Checkbox(label="Enabled", value=False)

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Group():

                        with gr.Row():
                            poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
                            poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5,interactive=True)

            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        poTransPNG = gr.Checkbox(label="Transparent PNG",value=False)
                        poTransPNGEps = gr.Slider(label="Noise Tolerance",minimum=0,maximum=128,value=16,interactive=True)
                        poTransPNGQuant = gr.Slider(label="Quantize",minimum=2,maximum=255,value=16,interactive=True)   
    return poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poDoVector,poTransPNGQuant
