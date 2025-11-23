import gradio as gr

#!from modules import scripts_postprocessing
#!from modules.ui_components import InputAccordion
from extentions.old_photo.bopb2l_main import main
from modules.model_loader import load_file_from_url
from pathlib import Path


def load_models():
    
    hf_root = "https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/old_photo"

    # Группы загрузок: (base_local_dir, hf_subdir, [(rel_path, filename), ...])
    groups = [
        # Группа 1: Global
        (
            Path(__file__).parent / "lib_bopb2l" / "Global" / "checkpoints",
            "global/checkpoints",
            [
                ("detection", "FT_Epoch_latest.pt"),
                ("restoration/VAE_A_quality", "latest_net_D.pth"),
                ("restoration/VAE_A_quality", "latest_net_featD.pth"),
                ("restoration/VAE_A_quality", "latest_net_G.pth"),
                ("restoration/VAE_A_quality", "latest_optimizer_D.pth"),
                ("restoration/VAE_A_quality", "latest_optimizer_featD.pth"),
                ("restoration/VAE_A_quality", "latest_optimizer_G.pth"),

                ("restoration/VAE_B_quality", "latest_net_D.pth"),
                ("restoration/VAE_B_quality", "latest_net_G.pth"),
                ("restoration/VAE_B_quality", "latest_optimizer_D.pth"),
                ("restoration/VAE_B_quality", "latest_optimizer_G.pth"),


                ("restoration/VAE_B_scratch", "latest_net_D.pth"),
                ("restoration/VAE_B_scratch", "latest_net_G.pth"),
                ("restoration/VAE_B_scratch", "latest_optimizer_D.pth"),
                ("restoration/VAE_B_scratch", "latest_optimizer_G.pth"),



                ("restoration/mapping_Patch_Attention", "latest_net_D.pth"),
                ("restoration/mapping_Patch_Attention", "latest_net_mapping_net.pth"),



                ("restoration/mapping_quality", "latest_net_D.pth"),
                ("restoration/mapping_quality", "latest_net_mapping_net.pth"),
                ("restoration/mapping_quality", "latest_optimizer_D.pth"),
                ("restoration/mapping_quality", "latest_optimizer_mapping_net.pth"),



                ("restoration/mapping_scratch", "latest_net_D.pth"),
                ("restoration/mapping_scratch", "latest_net_mapping_net.pth"),
                ("restoration/mapping_scratch", "latest_optimizer_D.pth"),
                ("restoration/mapping_scratch", "latest_optimizer_mapping_net.pth"),
                
            ]
        ),
        # Группа 2: Face
        (
            Path(__file__).parent / "lib_bopb2l" / "Face_Enhancement" / "checkpoints",
            "face/checkpoints",
            [
                ("FaceSR_512", "latest_net_G.pth"),
                ("Setting_9_epoch_100", "latest_net_G.pth"),
                
            ]
        ),
        
    ]

    for base_dir, hf_subdir, files in groups:
        repo_prefix = f"{hf_root}/{hf_subdir}"
        for rel_path, filename in files:
            load_file_from_url(
                url=f"{repo_prefix}/{rel_path}/{filename}",
                model_dir=base_dir / rel_path,
                file_name=filename
            )
    load_file_from_url(
                url=f"https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/old_photo/shape_predictor_68_face_landmarks.dat",
                model_dir=Path(__file__).parent / "lib_bopb2l" / "Face_Detection",
                file_name='shape_predictor_68_face_landmarks.dat'
            )
def process_firstpass(proc_order,do_scratch,do_face_res,is_hr,use_cpu,img):

        if proc_order == "Restoration First":

            #!do_scratch: bool = args["do_scratch"]
            #!do_face_res: bool = args["do_face_res"]
            #!is_hr: bool = args["is_hr"]
            #!use_cpu: bool = args["use_cpu"]

            #!img = pp.image
            img = main(img, do_scratch, is_hr, do_face_res, use_cpu)
        return img







    
def ui():
    with gr.Row():
        image_input=gr.Image(label="Input image",visible=True,height=260,interactive=True,type="pil")
        image_output=gr.Image(label="Output image",visible=False,height=260,interactive=True,type="pil") 
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
    #!args = {enable,proc_order,do_scratch,do_face_res,is_hr,use_cpu}
    start.click(lambda: (gr.update(interactive=False)),outputs=[start]) \
        .then (load_models) \
        .then (process_firstpass, inputs=[proc_order,do_scratch,do_face_res,is_hr,use_cpu,image_input],outputs=image_output)
    




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

    def process_firstpass(self, pp, **args):

        if args["enable"] and args["proc_order"] == "Restoration First":

            do_scratch: bool = args["do_scratch"]
            do_face_res: bool = args["do_face_res"]
            is_hr: bool = args["is_hr"]
            use_cpu: bool = args["use_cpu"]

            img = pp.image
            pp.image = main(img, do_scratch, is_hr, do_face_res, use_cpu)
    """    


    def process(self, pp, **args):

        if args["enable"] and args["proc_order"] == "Upscale First":

            do_scratch: bool = args["do_scratch"]
            do_face_res: bool = args["do_face_res"]
            is_hr: bool = args["is_hr"]
            use_cpu: bool = args["use_cpu"]

            img = pp.image
            pp.image = main(img, do_scratch, is_hr, do_face_res, use_cpu)
