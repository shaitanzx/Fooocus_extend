import gradio as gr
from PIL import Image
import sys
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
