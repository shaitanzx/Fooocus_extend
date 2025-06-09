def swap_to_gallery(files):
    # Преобразуем временные файлы в пути
    file_paths = [file.name for file in files]  # Получаем пути из временных файлов
    return gr.update(value=file_paths, visible=True), gr.update(visible=False)

def gui():
    with gr.Row():
        photomaker_enabled = gr.Checkbox(label="Enabled", value=False)
    with gr.Row():
        photomaker_images = gr.Files(label="Drag (Select) 1 or more photos of your face", file_types=["image"])
    with gr.Row():
        photomaker_gallery_images = gr.Gallery(label="Source Face Images", columns=5, rows=1, height=200)
    photomaker_images.upload(fn=swap_to_gallery, inputs=photomaker_images, outputs=[photomaker_gallery_images, photomaker_images])