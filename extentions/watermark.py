import gradio as gr

def watermark():
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
                image_in=gr.File(label="Upload a ZIP file of Source Images",file_count='single',file_types=['.zip'],visible=False,height=260)
                image_in_multi = gr.Files(label="Drag (Select) 1 or more Source images",file_count="multiple",
                                            file_types=["image"],visible=True,interactive=True,height=260)
                image_in_single=gr.Image(label="Source Image",visible=False,height=260,interactive=True,type="filepath")
            with gr.Row():
                enable_zip_image = gr.Checkbox(label="Upload ZIP-file", value=False)
        with gr.Column():
            with gr.Row():
                logo_image=gr.Image(label="Source Image",visible=True,height=260,interactive=True,type="filepath")
        with gr.Column():
            with gr.Row():
                file_out=gr.File(label="Download a ZIP file", file_count='single',height=260,visible=True)
                preview_out=gr.Image(label="Process preview",visible=False,height=260,interactive=False)
                image_out=gr.Image(label="Output image",visible=False,height=260,interactive=False)
    with gr.Row():
            watermark_start=gr.Button(value='Start paste watermark')
    with gr.Row(visible=False):
        ext_dir_face=gr.Textbox(value='batch_insw_face',visible=False)
    enable_zip_image.change(fn=zip_enable,inputs=[enable_zip_image,image_in_multi],outputs=[image_in,image_in_multi,image_in_single],show_progress=False)
    image_in_single.clear(fn=clear_single,inputs=image_in_single,outputs=[image_in_single,image_in_multi],show_progress=False)
    image_in_multi.upload(fn=single_image,inputs=image_in_multi,outputs=[image_in_single,image_in_multi],show_progress=False)
    

#    inswap_start.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False),gr.update(visible=False)),
#                        outputs=[inswap_start,file_out,image_out]) \
#              .then(fn=batch.clear_dirs,inputs=ext_dir_face) \
#              .then(fn=batch.clear_dirs,inputs=ext_dir_image) \
#              .then(fn=batch.unzip_file,inputs=[file_in_face,files_single_face,enable_zip_face,ext_dir_face]) \
#              .then(fn=batch.unzip_file,inputs=[file_in_image,files_single_image,enable_zip_image,ext_dir_image]) \
#              .then(lambda: (gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)),
#                        outputs=[file_in_face,files_single_face,image_single_face,file_in_image,files_single_image,image_single_image]) \
#              .then(fn=process_insw, inputs=[inswap_source_image_indicies, inswap_target_image_indicies],
#                        outputs=[preview_face,preview_image,file_out],show_progress=False) \
#              .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False),gr.update(visible=False)),outputs=[file_out,preview_face,preview_image],show_progress=False) \
#              .then(fn=batch.output_zip_image, outputs=[image_out,file_out]) \
#              .then(fn=zip_enable,inputs=[enable_zip_face,files_single_face],outputs=[file_in_face,files_single_face,image_single_face],show_progress=False) \
#              .then(fn=zip_enable,inputs=[enable_zip_image,files_single_image],outputs=[file_in_image,files_single_image,image_single_image],show_progress=False) \
#              .then(fn=single_image,inputs=files_single_face,outputs=[image_single_face,files_single_face],show_progress=False) \
#              .then(fn=single_image,inputs=files_single_image,outputs=[image_single_image,files_single_image],show_progress=False) \
#              .then(lambda: (gr.update(visible=True, interactive=True)),outputs=inswap_start)