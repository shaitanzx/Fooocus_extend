import gradio as gr
from PIL import Image
import sys
import os
import zipfile
import modules.util 
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY

def ui():
    with gr.Row():
        poDoVector = gr.Checkbox(label="Enabled", value=False)
    with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        poTransPNG = gr.Checkbox(label="Transparent PNG",value=False)
                        poTransPNGEps = gr.Slider(label="Noise Tolerance",minimum=0,maximum=128,value=16,interactive=True)
                        poTransPNGQuant = gr.Slider(label="Quantize",minimum=2,maximum=255,value=16,interactive=True)  
            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        with gr.Row():
                            poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
                            poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5,interactive=True)
    with gr.Row():
        gr.Markdown(value="Recommended to use vector styles")
    with gr.Row():
        gr.HTML('* \"Vector\" is powered by GeorgLegato. <a href="https://github.com/GeorgLegato/stable-diffusion-webui-vectorstudio" target="_blank">\U0001F4D4 Document</a>') 
    return poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poDoVector,poTransPNGQuant

def trans(image,poTransPNGQuant,poTransPNGEps):
                imgQ = image.quantize(colors=poTransPNGQuant, kmeans=0, palette=None)
                histo = image.histogram()

                # get first pixel and assume it is background, best with Sticker style
                if (imgQ):
                    bgI = imgQ.getpixel((0,0)) # return pal index
                    bg = list(imgQ.palette.colors.keys())[bgI]

                    E = poTransPNGEps # tolerance range if noisy

                imgT=imgQ.convert('RGBA')
                datas = imgT.getdata()
                newData = []
                for item in datas:
                    if (item[0] > bg[0]-E and item[0] < bg[0]+E) and (item[1] > bg[1]-E and item[1] < bg[1]+E) and (item[2] > bg[2]-E and item[1] < bg[2]+E):
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)

                imgT.putdata(newData)
                return imgT
def save_svg(image,poThreshold,filename):
            bm = Bitmap(image, blacklevel=0.5)
        # bm.invert()
            plist = bm.trace(
                turdsize=2,
                turnpolicy=POTRACE_TURNPOLICY_MINORITY,
                alphamax=1,
                opticurve=False,
                opttolerance=poThreshold,
            )



            with open(f"{filename}.svg", "w") as fp:
                fp.write(
                    f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">''')
                parts = []
                for curve in plist:
                    fs = curve.start_point
                    parts.append(f"M{fs.x},{fs.y}")
                    for segment in curve.segments:
                        if segment.is_corner:
                            a = segment.c
                            b = segment.end_point
                            parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                        else:
                            a = segment.c1
                            b = segment.c2
                            c = segment.end_point
                            parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
                    parts.append("z")
                fp.write(f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>')
                fp.write("</svg>")

def delete_out(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_out(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    return
def unzip_file(zip_file_obj,files_single,enable_zip):
    extract_folder = os.path.join(os.getcwd(), 'batch_vector')
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
def clear_make_dir():
    
    directory = os.path.join(os.getcwd(), 'batch_vector')
    delete_out(directory)
    directory = os.path.join(os.getcwd(), 'batch_temp')
    delete_out(directory)
    os.makedirs(directory, exist_ok=True)
    return
def process(poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poTransPNGQuant):
    batch_path=os.path.join(os.getcwd(), 'batch_vector')
    batch_temp=os.path.join(os.getcwd(), 'batch_temp')
    batch_files=sorted([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    batch_all=len(batch_files)
    passed=1
    for f_name in batch_files:
        print (f"\033[91m[Vector QUEUE] {passed} / {batch_all}. Filename: {f_name} \033[0m")
        gr.Info(f"Vector Batch: start element generation {passed}/{batch_all}. Filename: {f_name}") 
        img = Image.open(batch_path+os.path.sep+f_name)
        yield gr.update(value=img,visible=True),gr.update(visible=False)
        if poTransPNG:
            img = trans(img,poTransPNGQuant,poTransPNGEps)
            name, ext = os.path.splitext(f_name)
            filename =  batch_temp + os.path.sep + name +'_trans'+ext
            if not poKeepPnm:
                img.save(filename)
        name, ext = os.path.splitext(f_name)
        filename =  batch_temp + os.path.sep + name
        save_svg(img,poThreshold,filename)
        passed+=1
    return gr.update(value=None,visible=False),gr.update(visible=True)
def output_zip():
    directory=os.path.join(os.getcwd(), 'batch_temp')
    _, _, filename = modules.util.generate_temp_filename(folder=os.path.join(os.getcwd(), 'batch_temp'))
    name, ext = os.path.splitext(filename)
    new_filename = f"output_{name[:-5]}{ext}"
    zip_file = os.path.join(directory, new_filename)
    #zip_file='outputs.zip'
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, directory))
    zipf.close()
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "outputs.zip")
    return file_path




def ui_module():
    with gr.Row():
        with gr.Column():
            with gr.Row():
                file_in=gr.File(label="Upload a ZIP file",file_count='single',file_types=['.zip'],visible=False,height=260)
                files_single = gr.Files(label="Drag (Select) 1 or more reference images",
                                            file_types=["image"],visible=True,interactive=True,height=260) 
            with gr.Row():
                enable_zip = gr.Checkbox(label="Upload ZIP-file", value=False)
        with gr.Column():
            with gr.Row():
                file_out=gr.File(label="Download a ZIP file", file_count='single',height=260,visible=True)
                preview=gr.Image(label="Process preview",visible=False,height=260,interactive=False)

    with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        poTransPNG = gr.Checkbox(label="Transparent PNG",value=False)
                        poTransPNGEps = gr.Slider(label="Noise Tolerance",minimum=0,maximum=128,value=16,interactive=True)
                        poTransPNGQuant = gr.Slider(label="Quantize",minimum=2,maximum=255,value=16,interactive=True)  
            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        with gr.Row():
                            poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
                            poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5,interactive=True)
    with gr.Row():
        start = gr.Button(value='Start vectorization')

    with gr.Row():
        gr.HTML('* \"Vector\" is powered by GeorgLegato. <a href="https://github.com/GeorgLegato/stable-diffusion-webui-vectorstudio" target="_blank">\U0001F4D4 Document</a>') 
    enable_zip.change(lambda x: (gr.update(visible=x),gr.update(visible=not x)), inputs=enable_zip,
                                        outputs=[file_in,files_single], queue=False)
    
    start.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False)),outputs=[start,file_out]) \
              .then(fn=clear_make_dir) \
              .then(fn=unzip_file,inputs=[file_in,files_single,enable_zip]) \
              .then(fn=process, inputs=[poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poTransPNGQuant],
                        outputs=[preview,file_out],show_progress=False) \
              .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out,preview],show_progress=False) \
              .then(fn=output_zip, outputs=file_out) \
              .then(lambda: (gr.update(visible=True, interactive=True)),outputs=start)
