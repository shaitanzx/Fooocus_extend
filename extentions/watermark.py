import gradio as gr
import os
import sys
import cv2
import numpy as np
from PIL import Image
import math
import time
import extentions.batch as batch
import modules.config
temp_dir=modules.config.temp_path+os.path.sep

def calculate_logo_size(image_size, logo_original_size, target_ratio=0.1):
    """Рассчитывает размер логотипа для сохранения относительного размера"""
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
    """Обнаруживает лица на изображении (может не быть лиц)"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            return []
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    except:
        return []  # Если что-то пошло не wrong, возвращаем пустой список

def get_face_regions(faces, img_width, img_height, padding_ratio=0.3):
    """Возвращает запрещенные зоны вокруг лиц (если лица есть)"""
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
    """Проверяет, не пересекается ли позиция с запрещенными зонами"""
    if not forbidden_zones:  # Если нет запрещенных зон
        return True
    
    logo_x2, logo_y2 = x + logo_width, y + logo_height
    
    for zone in forbidden_zones:
        zone_x1, zone_y1, zone_x2, zone_y2 = zone
        
        # Проверка пересечения прямоугольников
        if not (logo_x2 <= zone_x1 or x >= zone_x2 or logo_y2 <= zone_y1 or y >= zone_y2):
            return False
    
    return True

def calculate_background_complexity(image_np, x, y, width, height):
    """Оценивает сложность фона в области (работает для любого изображения)"""
    img_height, img_width = image_np.shape[:2]
    
    # Берем область вокруг позиции логотипа
    x1 = max(0, x - width//2)
    y1 = max(0, y - height//2)
    x2 = min(img_width, x + width + width//2)
    y2 = min(img_height, y + height + height//2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.5  # Средняя сложность по умолчанию
    
    region = image_np[y1:y2, x1:x2]
    
    if region.size == 0:
        return 0.5
    
    # Конвертируем в grayscale и вычисляем вариацию
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    variation = np.std(gray_region)
    
    # Нормализуем к 0-1 (0 - простой фон, 1 - сложный)
    return min(1.0, variation / 64.0)

def add_adaptive_background(logo_pil, bg_color, complexity):
    """Добавляет адаптивную подложку для улучшения читаемости"""
    width, height = logo_pil.size
    
    # Определяем непрозрачность подложки based on complexity фона
    bg_alpha = int(100 + 155 * complexity)  # 100-255 alpha в зависимости от сложности
    bg_alpha = min(255, max(100, bg_alpha))
    
    # Создаем подложку
    bg_layer = Image.new('RGBA', (width, height), 
                       (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), bg_alpha))
    
    # Комбинируем с логотипом
    result = Image.alpha_composite(bg_layer, logo_pil)
    return result

def get_corner_positions(img_width, img_height, logo_width, logo_height, margin_ratio, corner_priority):
    """Возвращает позиции для 4 углов с отступом"""
    margin_x = int(img_width * margin_ratio)
    margin_y = int(img_height * margin_ratio)
    'top-left', 'top-right', 'bottom-left', 'bottom-right'
    corners = {
        'top-left': (margin_x, margin_y),  # Top-left
        'top-right': (img_width - logo_width - margin_x, margin_y),  # Top-right
        'bottom-left': (margin_x, img_height - logo_height - margin_y),  # Bottom-left
        'bottom-right': (img_width - logo_width - margin_x, img_height - logo_height - margin_y)  # Bottom-right
    }
    
    return [corners[x] for x in corner_priority]

def get_corner_background_color(image_np, x, y, logo_width, logo_height):
    """Получает средний цвет фона в области угла"""
    img_height, img_width = image_np.shape[:2]
    
    # Берем небольшую область в углу
    x1 = max(0, x - logo_width//4)
    y1 = max(0, y - logo_height//4)
    x2 = min(img_width, x + logo_width + logo_width//4)
    y2 = min(img_height, y + logo_height + logo_height//4)
    
    if x2 <= x1 or y2 <= y1:
        return (128, 128, 128)  # Серый по умолчанию
    
    region = image_np[y1:y2, x1:x2]
    
    if region.size == 0:
        return (128, 128, 128)
    
    # Средний цвет (BGR to RGB)
    avg_color = np.mean(region, axis=(0, 1))
    return (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))



def place_logo_in_corner(image_np, logo_pil,size_ratio,margin_ratio,min_complexity_for_bg,corner_priority):
    """
    Размещает логотип в углу изображения для любого типа контента
    """
    start_time = time.time()
    
    # Загрузка изображения
    load_start = time.time()
    #image_np = cv2.imread(image_path)
    #if image_np is None:
    #    raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    load_time = time.time() - load_start
    
    img_height, img_width = image_np.shape[:2]
    
    # Загрузка логотипа
    logo_load_start = time.time()
    #logo_pil = Image.open(logo_path).convert("RGBA")
    logo_original_size = logo_pil.size
    logo_load_time = time.time() - logo_load_start
    
    # Расчет размера логотипа
    size_calc_start = time.time()
    logo_width, logo_height = calculate_logo_size(
        (img_width, img_height), logo_original_size, size_ratio
    )
    logo_pil = logo_pil.resize((logo_width, logo_height), Image.LANCZOS)
    size_calc_time = time.time() - size_calc_start
    
    # Детекция лиц (может занять время)
    face_detect_start = time.time()
    faces = detect_faces(image_np)
    forbidden_zones = get_face_regions(faces, img_width, img_height)
    face_detect_time = time.time() - face_detect_start
    
    # Получаем позиции углов
    corners_start = time.time()
    corner_positions = get_corner_positions(
        img_width, img_height, logo_width, logo_height, margin_ratio,corner_priority
    )
    corners_time = time.time() - corners_start
    
    # Выбираем первый валидный угол
    position_start = time.time()
    final_position = None
    chosen_corner = None
    
    #corner_names = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    
    #for i, (x, y) in enumerate(corner_positions):
    #    if is_position_valid(x, y, logo_width, logo_height, forbidden_zones):
    #        final_position = (x, y)
    #        chosen_corner = corner_names[i]
    #        break

    for i, (x, y) in enumerate(corner_positions):
        if is_position_valid(x, y, logo_width, logo_height, forbidden_zones):
            final_position = (x, y)
            chosen_corner = corner_priority[i]  # Берем имя из переданного приоритета
            break
    
    # Если все углы заняты (маловероятно без лиц), используем первый угол
    if final_position is None:
        final_position = corner_positions[0]
        chosen_corner = f'{corner_names[0]} (forced)'

    position_time = time.time() - position_start
    
    # Анализ фона
    bg_analysis_start = time.time()
    x, y = final_position
    bg_color = get_corner_background_color(image_np, x, y, logo_width, logo_height)
    bg_complexity = calculate_background_complexity(image_np, x, y, logo_width, logo_height)
    bg_analysis_time = time.time() - bg_analysis_start
    
    # Добавляем подложку если нужно
    bg_processing_start = time.time()
    final_logo = logo_pil
    if bg_complexity > min_complexity_for_bg:
        final_logo = add_adaptive_background(logo_pil, bg_color, bg_complexity)
    bg_processing_time = time.time() - bg_processing_start
    
    # Наложение логотипа
    paste_start = time.time()
    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pil_image.paste(final_logo, (x, y), final_logo)
    paste_time = time.time() - paste_start
    
    # Сохранение
    save_start = time.time()

    save_time = time.time() - save_start
    
    total_time = time.time() - start_time
    
    # Вывод информации о времени выполнения
    print(f"=== ВРЕМЯ ВЫПОЛНЕНИЯ ===")
    print(f"Загрузка изображения: {load_time:.3f}с")
    print(f"Загрузка логотипа: {logo_load_time:.3f}с")
    print(f"Расчет размера: {size_calc_time:.3f}с")
    print(f"Детекция лиц: {face_detect_time:.3f}с")
    print(f"Определение углов: {corners_time:.3f}с")
    print(f"Выбор позиции: {position_time:.3f}с")
    print(f"Анализ фона: {bg_analysis_time:.3f}с")
    print(f"Обработка подложки: {bg_processing_time:.3f}с")
    print(f"Наложение: {paste_time:.3f}с")
    print(f"Сохранение: {save_time:.3f}с")
    print(f"ОБЩЕЕ ВРЕМЯ: {total_time:.3f}с")
    print(f"=== РЕЗУЛЬТАТ ===")
    print(f"Изображение: {img_width}x{img_height}px")
    print(f"Логотип: {logo_width}x{logo_height}px")
    print(f"Позиция: {chosen_corner} ({x}, {y})")
    print(f"Обнаружено лиц: {len(faces)}")
    print(f"Сложность фона: {bg_complexity:.2f}")
    print(f"Подложка: {'ДА' if bg_complexity > min_complexity_for_bg else 'НЕТ'}")
    
    #return output_path
    return pil_image


def process(logo,size_ratio,margin_ratio,min_complexity_for_bg,priority1,priority2,priority3,priority4):
    corner_priority=[priority1,priority2,priority3,priority4]
    batch_path=f"{temp_dir}batch_watermark"
    batch_temp=f"{temp_dir}batch_temp"
    batch_files=sorted([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    batch_all=len(batch_files)
    passed=1
    for f_name in batch_files:
        print (f"\033[91m[Watermarkr QUEUE] {passed} / {batch_all}. Filename: {f_name} \033[0m")
        gr.Info(f"Watermark Batch: start element generation {passed}/{batch_all}. Filename: {f_name}") 
        img = Image.open(batch_path+os.path.sep+f_name)
        yield gr.update(value=img,visible=True),gr.update(visible=False)
        image_in=cv2.imread(batch_path+os.path.sep+f_name)
        image_out=place_logo_in_corner(image_in, logo,size_ratio,margin_ratio,min_complexity_for_bg,corner_priority)
        name, _ = os.path.splitext(f_name)
        filename =  batch_temp + os.path.sep + name +'_watermark.png'
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
                logo_image=gr.Image(label="Source Logo",visible=True,height=260,interactive=True,type="pil", image_mode="RGBA")
        with gr.Column():
            with gr.Row():
                file_out=gr.File(label="Download a ZIP file", file_count='single',height=260,visible=True)
                preview_out=gr.Image(label="Process preview",visible=False,height=260,interactive=False)
                image_out=gr.Image(label="",visible=False,height=260,interactive=False)
    with gr.Row():
        size_ratio = gr.Slider(label='Size Ratio', minimum=0.0, maximum=1.0, step=0.01, value=0.2,interactive=True)
        margin_ratio = gr.Slider(label='Margin Ratio', minimum=0.0, maximum=1.0, step=0.01, value=0.02,interactive=True)
        min_complexity_for_bg = gr.Slider(label='Minimal complexity for background', minimum=0.0, maximum=1.0, step=0.01, value=0.3,interactive=True)
    with gr.Row():
        with gr.Group():
            gr.Markdown("### Сorner priority")
            with gr.Row():
                with gr.Column():
                    priority1 = gr.Dropdown(choices=priority,value=priority[0],label="Priority 1")
                with gr.Column():
                    priority2 = gr.Dropdown(choices=priority,value=priority[1],label="Priority 2")
                with gr.Column():
                    priority3 = gr.Dropdown(choices=priority,value=priority[2],label="Priority 2")
                with gr.Column():
                    priority4 = gr.Dropdown(choices=priority,value=priority[3],label="Priority 2")

    with gr.Row():
            watermark_start=gr.Button(value='Start paste watermark')
    with gr.Row(visible=False):
        ext_dir=gr.Textbox(value='batch_watermark',visible=False)
    enable_zip_image.change(fn=zip_enable,inputs=[enable_zip_image,image_in_multi],outputs=[image_in,image_in_multi,image_in_single],show_progress=False)
    image_in_single.clear(fn=clear_single,inputs=image_in_single,outputs=[image_in_single,image_in_multi],show_progress=False)
    image_in_multi.upload(fn=single_image,inputs=image_in_multi,outputs=[image_in_single,image_in_multi],show_progress=False)
    watermark_start.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False),gr.update(visible=False)),
                        outputs=[watermark_start,file_out,image_out]) \
                .then(fn=batch.clear_dirs,inputs=ext_dir) \
                .then(fn=batch.unzip_file,inputs=[image_in,image_in_multi,enable_zip_image,ext_dir]) \
                .then(fn=process, inputs=[logo_image,size_ratio,margin_ratio,min_complexity_for_bg,priority1,priority2,priority3,priority4],
                        outputs=[preview_out,file_out],show_progress=False) \
                .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out,preview_out],show_progress=False) \
                .then(fn=batch.output_zip_image, outputs=[image_out,file_out]) \
                .then(lambda: (gr.update(visible=True, interactive=True)),outputs=watermark_start)  
