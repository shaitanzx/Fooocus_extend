import gradio as gr
from PIL import Image
import modules.gradio_hijack as grh
import sys
import os
import modules.config



import extentions.batch as batch
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY
temp_dir=modules.config.temp_path+os.path.sep

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


def process(poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poTransPNGQuant):
    batch_path=f"{temp_dir}batch_vector"
    batch_temp=f"{temp_dir}batch_temp"
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







def ui_module():
    ext_dir='batch vector'
    batch.ui_batch()
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


    start.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False)),outputs=[start,file_out]) \
              .then(fn=batch.clear_dirs,inputs=ext_dir) \
              .then(fn=batch.unzip_file,inputs=[file_in,files_single,enable_zip,ext_dir]) \
              .then(fn=process, inputs=[poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poTransPNGQuant],
                        outputs=[preview,file_out],show_progress=False) \
              .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out,preview],show_progress=False) \
              .then(fn=batch.output_zip, outputs=file_out) \
              .then(lambda: (gr.update(visible=True, interactive=True)),outputs=start)

