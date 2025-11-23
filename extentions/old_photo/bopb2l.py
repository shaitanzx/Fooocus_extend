import gradio as gr

#!from modules import scripts_postprocessing
#!from modules.ui_components import InputAccordion
from extentions.old_photo.bopb2l_main import main
from modules.model_loader import load_file_from_url
from pathlib import Path


def load_models():
    
    load_file_from_url(
        url='https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/old_photo/global/checkpoints/detection/FT_Epoch_latest.pt',
        model_dir=Path(__file__).parent / "lib_bopb2l" / "Global" / "checkpoints" / "detection",
        file_name='FT_Epoch_latest.pt'
    )
    load_file_from_url(
        url='https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/old_photo/global/checkpoints/restoration/VAE_A_quality/latest_net_D.pth',
        model_dir=Path(__file__).parent / "lib_bopb2l" / "Global" / "checkpoints" / "restoration" / "VAE_A_quality",
        file_name='latest_net_D.pth'
    )
    load_file_from_url(
        url='https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/old_photo/global/checkpoints/restoration/VAE_A_quality/latest_net_G.pth',
        model_dir=Path(__file__).parent / "lib_bopb2l" / "Global" / "checkpoints" / "restoration" / "VAE_A_quality",
        file_name='latest_net_G.pth'
    )










    
def ui():
    enable=True
    with gr.Row():
        files_input = gr.Files(label="",visible=True,interactive=True,height=260)
        files_output = gr.Files(label="",visible=True,interactive=True,height=260)  
    with gr.Row():
        proc_order = gr.Radio(
            choices=("Restoration First", "Upscale First"),
            value="Restoration First",
            label="Processing Order",interactive=True
            )

    with gr.Row():
        do_scratch = gr.Checkbox(False, label="Process Scratch")
        do_face_res = gr.Checkbox(False, label="Face Restore")
    with gr.Row():
        is_hr = gr.Checkbox(False, label="High Resolution")
        use_cpu = gr.Checkbox(True, label="Use CPU",interactive=True)
    with gr.Row():
        start=gr.Button(value='Start')
    args = {
        "enable": enable,
        "proc_order": proc_order,
        "do_scratch": do_scratch,
        "do_face_res": do_face_res,
        "is_hr": is_hr,
        "use_cpu": use_cpu,
        }
    start.click(lambda: (gr.update(interactive=False)),outputs=[start]) \
        .then (load_models) 
    return args




class OldPhotoRestoration():
    name = "BOP"
    order = 200409484
    """
    def ui(self):
        with InputAccordion(False, label="Old Photo Restoration") as enable:
            proc_order = gr.Radio(
                choices=("Restoration First", "Upscale First"),
                value="Restoration First",
                label="Processing Order",
            )

            with gr.Row():
                do_scratch = gr.Checkbox(False, label="Process Scratch")
                do_face_res = gr.Checkbox(False, label="Face Restore")
            with gr.Row():
                is_hr = gr.Checkbox(False, label="High Resolution")
                use_cpu = gr.Checkbox(True, label="Use CPU")

        args = {
            "enable": enable,
            "proc_order": proc_order,
            "do_scratch": do_scratch,
            "do_face_res": do_face_res,
            "is_hr": is_hr,
            "use_cpu": use_cpu,
        }

        return args
    """    
    def process_firstpass(self, pp, **args):

        if args["enable"] and args["proc_order"] == "Restoration First":

            do_scratch: bool = args["do_scratch"]
            do_face_res: bool = args["do_face_res"]
            is_hr: bool = args["is_hr"]
            use_cpu: bool = args["use_cpu"]

            img = pp.image
            pp.image = main(img, do_scratch, is_hr, do_face_res, use_cpu)

    def process(self, pp, **args):

        if args["enable"] and args["proc_order"] == "Upscale First":

            do_scratch: bool = args["do_scratch"]
            do_face_res: bool = args["do_face_res"]
            is_hr: bool = args["is_hr"]
            use_cpu: bool = args["use_cpu"]

            img = pp.image
            pp.image = main(img, do_scratch, is_hr, do_face_res, use_cpu)
