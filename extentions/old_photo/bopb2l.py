import gradio as gr

#!from modules import scripts_postprocessing
#!from modules.ui_components import InputAccordion
from extentions.old_photo.bopb2l_main import main
from modules.model_loader import load_file_from_url
import numpy as np
from pathlib import Path
from PIL import Image
import extentions.batch as batch
import os
import modules.config
temp_dir=modules.config.temp_path+os.path.sep

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

        #!if proc_order == "Restoration First":

            #!do_scratch: bool = args["do_scratch"]
            #!do_face_res: bool = args["do_face_res"]
            #!is_hr: bool = args["is_hr"]
            #!use_cpu: bool = args["use_cpu"]

            #!img = pp.image
            
        img =np.array(main(img, do_scratch, is_hr, do_face_res, use_cpu))
        
        return img
def process(proc_order,do_scratch,do_face_res,is_hr,use_cpu,do_color):
    batch_path=f"{temp_dir}batch_old_photo"
    batch_temp=f"{temp_dir}batch_temp"
    batch_files=sorted([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    batch_all=len(batch_files)
    passed=1
    for f_name in batch_files:
        print (f"\033[91m[OldPhotoRestoration QUEUE] {passed} / {batch_all}. Filename: {f_name} \033[0m")
        gr.Info(f"OldPhotoRestoration Batch: start element generation {passed}/{batch_all}. Filename: {f_name}") 
        img = Image.open(batch_path+os.path.sep+f_name)
        yield gr.update(value=img,visible=True),gr.update(visible=False)
        #!image=np.array(img)
        img_old=Image.fromarray(process_firstpass(proc_order,do_scratch,do_face_res,is_hr,use_cpu,img))
        yield gr.update(value=img,visible=True),gr.update(visible=False)
        if do_color:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            from modelscope.outputs import OutputKeys
            from modelscope.hub.snapshot_download import snapshot_download
            path_color=Path(__file__).parent / "color_model"
            os.makedirs(path_color, exist_ok=True)
            model_dir = snapshot_download('iic/cv_ddcolor_image-colorization',cache_dir=path_color)
            img_colorization = pipeline(task=Tasks.image_colorization,model=model_dir)
            rgb_image = img[..., ::-1] if img.shape[-1] == 3 else img
            output = img_colorization(rgb_image)
            result = output[OutputKeys.OUTPUT_IMG].astype(np.uint8)
            img = result[...,::-1]
            yield gr.update(value=img,visible=True),gr.update(visible=False)
            del img_colorization

            
        name, ext = os.path.splitext(f_name)
        filename =  batch_temp + os.path.sep + name +'_old'+ext
        img_old.save(filename)
        passed+=1
    return gr.update(value=None,visible=False),gr.update(visible=True)





    
def ui():
    file_in,files_single,image_single,enable_zip,file_out,preview, image_out = batch.ui_batch()
    with gr.Row():
        proc_order = gr.Radio(
            choices=("Restoration First", "Upscale First"),
            value="Restoration First",
            label="Processing Order",interactive=True,visible=False
            )

    with gr.Row():
        do_scratch = gr.Checkbox(False, label="Process Scratch")
        do_face_res = gr.Checkbox(False, label="Face Restore")
        do_color = gr.Checkbox(False, label="Colorization")
    with gr.Row():
        is_hr = gr.Checkbox(False, label="High Resolution")
        use_cpu = gr.Checkbox(True, label="Use CPU",interactive=True)
    with gr.Row():
        start=gr.Button(value='Start')
    #!args = {enable,proc_order,do_scratch,do_face_res,is_hr,use_cpu}
    with gr.Row(visible=False):
        ext_dir=gr.Textbox(value='batch_old_photo',visible=False) 
    start.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False),gr.update(visible=False)),outputs=[start,file_out,image_out]) \
              .then(fn=batch.clear_dirs,inputs=ext_dir) \
              .then (load_models) \
              .then(fn=batch.unzip_file,inputs=[file_in,files_single,enable_zip,ext_dir]) \
              .then(fn=process, inputs=[proc_order, do_scratch, do_face_res, is_hr, use_cpu, do_color],outputs=[preview,file_out],show_progress=False) \
              .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out,preview],show_progress=False) \
              .then(fn=batch.output_zip_image, outputs=[image_out,file_out]) \
              .then(lambda: (gr.update(visible=True, interactive=True)),outputs=start)

    




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
