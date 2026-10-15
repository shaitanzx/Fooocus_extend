import gradio as gr
import os
import sys
import cv2
import numpy as np
from PIL import Image
import math
import extentions.batch as batch
import modules.config
from extentions.tutorial import tutorial
temp_dir=modules.config.temp_path+os.path.sep

def calculate_logo_size(image_size, logo_original_size, target_ratio=0.1):
    img_width, img_height = image_size
    logo_width, logo_height = logo_original_size
    
    base_size = min(img_width, img_height) * target_ratio
    logo_aspect_ratio = logo_width / logo_height
    
    if logo_aspect_ratio > 1:
        new_width = int(base_size)
        new_height = int(new_width / logo_aspect_ratio)
    else:
        new_height = int(base_size)
        new_width = int(new_height * logo_aspect_ratio)
    
    return new_width, new_height

def detect_faces(image_np):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            return []
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    except:
        return []

def get_face_regions(faces, img_width, img_height, padding_ratio=0.3):
    forbidden_zones = []
    
    for (x, y, w, h) in faces:
        padding = int(min(w, h) * padding_ratio)
        
        zone_x1 = max(0, x - padding)
        zone_y1 = max(0, y - padding)
        zone_x2 = min(img_width, x + w + padding)
        zone_y2 = min(img_height, y + h + padding)
        
        forbidden_zones.append((zone_x1, zone_y1, zone_x2, zone_y2))
    
    return forbidden_zones

def is_position_valid(x, y, logo_width, logo_height, forbidden_zones):
    if not forbidden_zones:
        return True
    
    logo_x2, logo_y2 = x + logo_width, y + logo_height
    
    for zone in forbidden_zones:
        zone_x1, zone_y1, zone_x2, zone_y2 = zone
        
        if not (logo_x2 <= zone_x1 or x >= zone_x2 or logo_y2 <= zone_y1 or y >= zone_y2):
            return False
    
    return True

def calculate_background_complexity(image_np, x, y, width, height):
    img_height, img_width = image_np.shape[:2]
    
    x1 = max(0, x - width//2)
    y1 = max(0, y - height//2)
    x2 = min(img_width, x + width + width//2)
    y2 = min(img_height, y + height + height//2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.5  
    
    region = image_np[y1:y2, x1:x2]
    
    if region.size == 0:
        return 0.5
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    variation = np.std(gray_region)
    return min(1.0, variation / 64.0)

def add_adaptive_background(logo_pil, bg_color, complexity):
    width, height = logo_pil.size
    bg_alpha = int(100 + 155 * complexity) 
    bg_alpha = min(255, max(100, bg_alpha))
    bg_layer = Image.new('RGBA', (width, height), 
                       (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), bg_alpha))
    result = Image.alpha_composite(bg_layer, logo_pil)
    return result

def get_corner_positions(img_width, img_height, logo_width, logo_height, margin_ratio, corner_priority):   
    margin_x = int(img_width * margin_ratio)
    margin_y = int(img_height * margin_ratio)
    'top-left', 'top-right', 'bottom-left', 'bottom-right'
    corners = {
        'top-left': (margin_x, margin_y),  
        'top-right': (img_width - logo_width - margin_x, margin_y),  
        'bottom-left': (margin_x, img_height - logo_height - margin_y),  
        'bottom-right': (img_width - logo_width - margin_x, img_height - logo_height - margin_y)
    }
    
    return [corners[x] for x in corner_priority]

def get_corner_background_color(image_np, x, y, logo_width, logo_height):
    img_height, img_width = image_np.shape[:2]

    x1 = max(0, x - logo_width//4)
    y1 = max(0, y - logo_height//4)
    x2 = min(img_width, x + logo_width + logo_width//4)
    y2 = min(img_height, y + logo_height + logo_height//4)
    
    if x2 <= x1 or y2 <= y1:
        return (128, 128, 128)  # Серый по умолчанию
    
    region = image_np[y1:y2, x1:x2]
    
    if region.size == 0:
        return (128, 128, 128)
    avg_color = np.mean(region, axis=(0, 1))
    return (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))

def place_logo_in_corner(image_np, logo_pil,size_ratio,margin_ratio,min_complexity_for_bg,corner_priority):        
    img_height, img_width = image_np.shape[:2]    
    logo_original_size = logo_pil.size    
    logo_width, logo_height = calculate_logo_size(
        (img_width, img_height), logo_original_size, size_ratio
    )
    logo_pil = logo_pil.resize((logo_width, logo_height), Image.LANCZOS)    
    faces = detect_faces(image_np)
    forbidden_zones = get_face_regions(faces, img_width, img_height)
    
    corner_positions = get_corner_positions(
        img_width, img_height, logo_width, logo_height, margin_ratio,corner_priority
    )
    
    final_position = None
    chosen_corner = None

    for i, (x, y) in enumerate(corner_positions):
        if is_position_valid(x, y, logo_width, logo_height, forbidden_zones):
            final_position = (x, y)
            chosen_corner = corner_priority[i]  
            break
    if final_position is None:
        final_position = corner_positions[0]
        chosen_corner = f'{corner_priority[0]} (forced)'
    x, y = final_position
    bg_color = get_corner_background_color(image_np, x, y, logo_width, logo_height)
    bg_complexity = calculate_background_complexity(image_np, x, y, logo_width, logo_height)
    
    final_logo = logo_pil
    if bg_complexity > min_complexity_for_bg:
        final_logo = add_adaptive_background(logo_pil, bg_color, bg_complexity)
    
    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pil_image.paste(final_logo, (x, y), final_logo)
    print(f"Image: {img_width}x{img_height}px")
    print(f"Logo: {logo_width}x{logo_height}px")
    print(f"Position: {chosen_corner} ({x}, {y})")
    print(f"Faces: {len(faces)}")
    print(f"Background complexity: {bg_complexity:.2f}")
    print(f"Substrate: {'Yes' if bg_complexity > min_complexity_for_bg else 'No'}")
    return pil_image

def get_rotated_size(width, height, angle_deg):
    """Calculate bounding box size after rotation (without actually rotating)"""
    angle_normalized = abs(angle_deg) % 180
    if angle_normalized > 90:
        angle_normalized = 180 - angle_normalized
    
    angle_rad = math.radians(angle_normalized)
    w = abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad))
    h = abs(width * math.sin(angle_rad)) + abs(height * math.cos(angle_rad))
    return int(math.ceil(w)), int(math.ceil(h))
def apply_opacity(watermark, opacity):

    wm = watermark.copy()
    alpha = wm.getchannel("A")
    alpha = alpha.point(lambda p: int(p * opacity))
    wm.putalpha(alpha)
    return wm
def watermark_process(img, watermark, opacity, scale_factor, rotation_angle, spacing_x, spacing_y):

        img_h, img_w = img.shape[:2]
        wm_w, wm_h = watermark.size
        rotated_w, rotated_h = get_rotated_size(wm_w, wm_h, rotation_angle)
        scale_to_fit_width = img_w / rotated_w
        scale_to_fit_height = img_h / rotated_h
        max_fit_scale = min(scale_to_fit_width, scale_to_fit_height)
        final_scale = max_fit_scale * scale_factor
        wm_rotated = watermark.rotate(-rotation_angle, expand=True, resample=Image.BICUBIC)
        new_w = int(wm_rotated.width * final_scale)
        new_h = int(wm_rotated.height * final_scale)
        wm_tile = wm_rotated.resize((new_w, new_h), Image.LANCZOS)
        wm_tile = apply_opacity(wm_tile, opacity)
        
        tile_w, tile_h = wm_tile.size
         
        overlay = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        tiles_count = 1
        
        if scale_factor >= 0.95:
            pos_x = (img_w - tile_w) // 2
            pos_y = (img_h - tile_h) // 2
            overlay.paste(wm_tile, (pos_x, pos_y), wm_tile)
        else: 
            step_x = int(round(tile_w * (1 + spacing_x)))
            step_y = int(round(tile_h * (1 + spacing_y)))
            
            start_y = -((img_h + tile_h) % step_y) // 2 - tile_h // 2
            row = 0
            y = start_y
            tiles_count = 0
            
            while y < img_h + tile_h:

                offset_x = step_x // 2 if (row % 2 == 1 and spacing_x > 0.0) else 0
                
                start_x = -((img_w + tile_w) % step_x) // 2 - tile_w // 2 + offset_x
                x = start_x
                
                while x < img_w + tile_w:
                    if x + tile_w > 0 and y + tile_h > 0 and x < img_w and y < img_h:
                        overlay.paste(wm_tile, (int(x), int(y)), wm_tile)
                        tiles_count += 1
                    x += step_x
                
                y += step_y
                row += 1
        
        print(f"Opacity: {opacity}") 
        print(f"Scale factor: {scale_factor}")
        print(f"Rotation angle: {rotation_angle}")
        print(f"Offset X: {spacing_x}")
        print(f"Offset Y: {spacing_y}")
        return overlay

def process(logo,size_ratio,margin_ratio,min_complexity_for_bg,priority1,priority2,priority3,priority4,logo_mode,opacity_water,scale_water,angle_water,spacing_x,spacing_y):
    opacity_water=opacity_water / 100.0
    scale_water=scale_water / 100.0
    spacing_x=spacing_x / 100.0
    spacing_y=spacing_y / 100.0
    
    
    corner_priority=[priority1,priority2,priority3,priority4]
    batch_path=f"{temp_dir}batch_logo"
    batch_temp=f"{temp_dir}batch_temp"
    batch_files=sorted([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    batch_all=len(batch_files)
    passed=1
    for f_name in batch_files:
        print (f"\033[91m[Logo/Watermark QUEUE] {passed} / {batch_all}. Filename: {f_name} \033[0m")
        gr.Info(f"Logo/Watermark Batch: start element generation {passed}/{batch_all}. Filename: {f_name}") 
        img = Image.open(batch_path+os.path.sep+f_name)
        yield gr.update(value=img,visible=True),gr.update(visible=False)
        image_in=cv2.imread(batch_path+os.path.sep+f_name)
        if logo_mode=="Logo":
            image_out=place_logo_in_corner(image_in, logo,size_ratio,margin_ratio,min_complexity_for_bg,corner_priority)
        else:
            image_out=watermark_process(image_in,logo,opacity_water,scale_water,angle_water,spacing_x,spacing_y)
                    
        name, _ = os.path.splitext(f_name)
        filename =  batch_temp + os.path.sep + name +'_logo.png'
        image_out = image_out.convert('RGB')
        image_out.save(filename)
        passed+=1
    return gr.update(value=None,visible=False),gr.update(visible=True)

def watermark():
    priority=['top-left', 'top-right', 'bottom-left', 'bottom-right']
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
    #with gr.Row():
    with gr.Row():
        logo_mode=gr.Radio(
            choices=["Logo", "Watermark"],
            value="Logo",
            label="Select Paste Mode"
            )
        tutorial("watermark.md")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_in=gr.File(label="Upload a ZIP file of Source Images",file_count='single',file_types=['.zip'],visible=False,height=260,interactive=True)
                image_in_multi = gr.Files(label="Drag (Select) 1 or more Source images",file_count="multiple",
                                            file_types=["image"],visible=True,interactive=True,height=260)
                image_in_single=gr.Image(label="Source Image",visible=False,height=260,interactive=True,type="filepath")
            with gr.Row():
                enable_zip_image = gr.Checkbox(label="Upload ZIP-file", value=False)
        with gr.Column():
            with gr.Row():
                logo_image=gr.Image(label="Source Logo/Watermark",visible=True,height=260,interactive=True,type="pil", image_mode="RGBA")
        with gr.Column():
            with gr.Row():
                file_out=gr.File(label="Download a ZIP file", file_count='single',height=260,visible=True)
                preview_out=gr.Image(label="Process preview",visible=False,height=260,interactive=False)
                image_out=gr.Image(label="",visible=False,height=260,interactive=False)
    with gr.Row() as logo_set:
        size_ratio = gr.Slider(label='Size Ratio', minimum=0.0, maximum=1.0, step=0.01, value=0.2,interactive=True)
        margin_ratio = gr.Slider(label='Margin Ratio', minimum=0.0, maximum=1.0, step=0.01, value=0.02,interactive=True)
        min_complexity_for_bg = gr.Slider(label='Minimal complexity for background', minimum=0.0, maximum=1.0, step=0.01, value=0.7,interactive=True)
    with gr.Row() as logo_grp:
        with gr.Group():
            gr.Markdown("### Сorner priority")
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    priority1 = gr.Dropdown(choices=priority,value=priority[0],label="Priority 1")
                with gr.Column(scale=1, min_width=100):
                    priority2 = gr.Dropdown(choices=priority,value=priority[1],label="Priority 2")
                with gr.Column(scale=1, min_width=100):
                    priority3 = gr.Dropdown(choices=priority,value=priority[2],label="Priority 3")
                with gr.Column(scale=1, min_width=100):
                    priority4 = gr.Dropdown(choices=priority,value=priority[3],label="Priority 4")
    with gr.Row(visible=False) as water_set:
        opacity_water = gr.Slider(label='Opacity (%)', minimum=0, maximum=100, step=0.5, value=85,interactive=True)
        scale_water = gr.Slider(label='Scale (%)', minimum=0, maximum=100, step=0.05, value=20,interactive=True)
        angle_water = gr.Slider(label='Rotate angle (degrees)', minimum=-180, maximum=180, step=1.0, value=0,interactive=True)
    with gr.Row(visible=False) as water_set2:
        spacing_x = gr.Slider(label='Offset X (%)', minimum=0, maximum=100, step=0.5, value=50.0,interactive=True)
        spacing_y = gr.Slider(label='Offset Y (%)', minimum=0, maximum=100, step=0.5, value=50.0,interactive=True)
    with gr.Row():
            watermark_start=gr.Button(value='Start paste logo/watermark')
    with gr.Row(visible=False):
        ext_dir=gr.Textbox(value='batch_logo',visible=False)
    def change_mode(mode):
        return (
            gr.update(visible=(mode == "Logo")),
            gr.update(visible=(mode == "Logo")),
            gr.update(visible=(mode == "Watermark")),
            gr.update(visible=(mode == "Watermark"))
            )
    logo_mode.change(fn=change_mode,inputs=[logo_mode],outputs=[logo_set,logo_grp,water_set,water_set2])
    enable_zip_image.change(fn=zip_enable,inputs=[enable_zip_image,image_in_multi],outputs=[image_in,image_in_multi,image_in_single],show_progress=False)
    image_in_single.clear(fn=clear_single,inputs=image_in_single,outputs=[image_in_single,image_in_multi],show_progress=False)
    image_in_multi.upload(fn=single_image,inputs=image_in_multi,outputs=[image_in_single,image_in_multi],show_progress=False)
    watermark_start.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False),gr.update(visible=False)),
                        outputs=[watermark_start,file_out,image_out]) \
                .then(fn=batch.clear_dirs,inputs=ext_dir) \
                .then(fn=batch.unzip_file,inputs=[image_in,image_in_multi,enable_zip_image,ext_dir]) \
                .then(fn=process,
                        inputs=[logo_image,size_ratio,margin_ratio,min_complexity_for_bg,
                            priority1,priority2,priority3,priority4,
                            logo_mode,opacity_water,scale_water,angle_water,spacing_x,spacing_y],
                        outputs=[preview_out,file_out],show_progress=False) \
                .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out,preview_out],show_progress=False) \
                .then(fn=batch.output_zip_image, outputs=[image_out,file_out]) \
                .then(lambda: (gr.update(visible=True, interactive=True)),outputs=watermark_start)  
