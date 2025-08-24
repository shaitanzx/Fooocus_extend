import gradio as gr
import os
from modules.launch_util import delete_folder_content
import modules.config
import zipfile
import modules.util 

temp_dir=modules.config.temp_path+os.path.sep

def ui_batch():
    def zip_enable(enable,single_file):
        if enable:
            return gr.update(visible=True),gr.update(visible=False),gr.update(visible=False)
        else:
            if single_file and len(single_file)==1:
                return gr.update(visible=False),gr.update(visible=False),gr.update(visible=True)
            else:
                return gr.update(visible=False),gr.update(visible=True),gr.update(visible=False)
    def clear_single(image):
        return gr.update(value=None,visible=False),gr.update(value=None,visible=True)
    def single_image(single_upload):
        if len(single_upload) == 1:
            return gr.update (value=single_upload[0].name,visible=True),gr.update(visible=False)
        else:
            return gr.update (visible=False),gr.update(visible=True)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                file_in=gr.File(label="Upload a ZIP file",file_count='single',file_types=['.zip'],visible=False,height=260)
                files_single = gr.Files(label="Drag (Select) 1 or more reference images",
                                            file_types=["image"],visible=True,interactive=True,height=260)
                image_single=gr.Image(label="Reference image",visible=False,height=260,interactive=True,type="filepath")
            with gr.Row():
                enable_zip = gr.Checkbox(label="Upload ZIP-file", value=False)
        with gr.Column():
            with gr.Row():
                file_out=gr.File(label="Download a ZIP file", file_count='single',height=260,visible=True)
                preview=gr.Image(label="Process preview",visible=False,height=260,interactive=False)
    enable_zip.change(fn=zip_enable,inputs=[enable_zip,files_single],outputs=[file_in,files_single,image_single],show_progress=False)
    image_single.clear(fn=clear_single,inputs=image_single,outputs=[image_single,files_single],show_progress=False)
    files_single.upload(fn=single_image,inputs=files_single,outputs=[image_single,files_single],show_progress=False)
    return file_in,files_single,image_single,enable_zip,file_out,preview

def clear_dirs(ext_dir):
    result=delete_folder_content(f"{temp_dir}{ext_dir}", '')
    result=delete_folder_content(f"{temp_dir}batch_temp", '')
    return
def unzip_file(zip_file_obj,files_single,enable_zip,ext_dir):
    extract_folder = f"{temp_dir}{ext_dir}"
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    if enable_zip:
        zip_ref=zipfile.ZipFile(zip_file_obj.name, 'r')
        zip_ref.extractall(extract_folder)
        zip_ref.close()
    else:
        for file in files_single:
            original_name = os.path.basename(getattr(file, 'orig_name', file.name))
            save_path = os.path.join(extract_folder, original_name)
            try:
                with open(file.name, 'rb') as src:
                    with open(save_path, 'wb') as dst:
                        while True:
                            chunk = src.read(8192)  # Читаем по 8KB за раз
                            if not chunk:
                                break
                            dst.write(chunk)
            except Exception as e:
                print(f"copy error {original_name}: {str(e)}")
    return
def output_zip():
    directory=f"{temp_dir}batch_temp"
    _, _, filename = modules.util.generate_temp_filename(folder=temp_dir)
    name, ext = os.path.splitext(filename)
    zip_file = os.path.join(temp_dir, f"output_{name[:-5]}.zip")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, directory))
    zipf.close()
    return zip_file