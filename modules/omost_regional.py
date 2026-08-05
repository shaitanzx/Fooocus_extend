import torch
import numpy as np
import modules.default_pipeline as pipeline

def build_regional_conditioning(bag_of_conditions, global_strength=0.2, region_strength=0.8):
    """
    Создает conditioning в формате Fooocus [[tensor, dict]] с масками.
    Маски нативно применяются через ldm_patched.
    """
    final_conditioning = []
    
    for i, cond in enumerate(bag_of_conditions):
        is_global = (i == 0)
        
        if is_global:
            full_prompt = ", ".join(cond['prefixes'] + cond['suffixes'])
        else:
            # Как в ComfyUI_omost: пропускаем глобальный префикс для регионов,
            # т.к. он уже есть в глобальном условии
            full_prompt = ", ".join(cond['prefixes'][1:] + cond['suffixes'])
        
        encoded = pipeline.clip_encode([full_prompt], pool_top_k=1)
        
        if not encoded or len(encoded) == 0:
            print(f"[Omost] Failed to encode region {i}")
            continue
        
        cond_tensor = encoded[0][0]
        pooled_output = encoded[0][1]["pooled_output"]
        
        mask = torch.from_numpy(np.ascontiguousarray(cond['mask'])).float()
        mask_sum = float(mask.sum())
        
        if mask_sum <= 0.0 and not is_global:
            print(f"[Omost] WARNING: region {i} has EMPTY mask, skipping")
            continue
        
        strength = global_strength if is_global else region_strength
        
        entry = [
            cond_tensor,
            {
                "pooled_output": pooled_output,
                "mask": mask,
                "strength": strength,
            }
        ]
        final_conditioning.append(entry)
        print(f"[Omost] Region {i}: mask_sum={mask_sum:.1f}, strength={strength}")
    
    print(f"[Omost] Built regional conditioning with {len(final_conditioning)} regions")
    return final_conditioning


def get_initial_latent(canvas_data):
    """
    Возвращает initial_latent из Canvas (numpy array 90x90x3 RGB, 0-255).
    Это цветная карта композиции, которая используется как стартовая точка диффузии.
    """
    if canvas_data is None:
        return None
    
    # process() вернул словарь с initial_latent
    # Нам нужно повторно вызвать process() на самом Canvas для получения initial_latent,
    # но у нас уже есть bag_of_conditions. Поэтому вернем initial_latent, сохраненный ранее.
    return canvas_data.get('initial_latent', None)


import cv2
import os
import modules.config

def visualize_masks(bag_of_conditions, target_height, target_width, filename="omost_masks.png"):
    """
    Сохраняет визуализацию масок как PNG для отладки.
    """
    # Создаем пустое цветное изображение
    img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Цвета для разных регионов (RGB)
    colors = [
        (255, 0, 0),      # красный
        (0, 255, 0),      # зеленый
        (0, 0, 255),      # синий
        (255, 255, 0),    # желтый
        (255, 0, 255),    # фиолетовый
        (0, 255, 255),    # голубой
        (255, 128, 0),    # оранжевый
        (128, 0, 255),    # пурпурный
    ]
    
    latent_height = target_height // 8
    latent_width = target_width // 8
    
    for i, cond in enumerate(bag_of_conditions):
        mask_np = cond['mask']  # (90, 90)
        is_global = (i == 0)
        
        # Масштабируем маску до размера изображения
        mask_resized = cv2.resize(
            mask_np, 
            (target_width, target_height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        color = colors[i % len(colors)]
        
        if is_global:
            # Глобальный регион рисуем очень прозрачным
            mask_bool = mask_resized > 0.5
            img[mask_bool] = (img[mask_bool] * 0.9 + np.array(color) * 0.1).astype(np.uint8)
        else:
            # Региональные рисуем яркими
            mask_bool = mask_resized > 0.5
            img[mask_bool] = color
        
        # Добавляем текст с номером региона
        if not is_global:
            # Находим центр маски
            ys, xs = np.where(mask_resized > 0.5)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(img, f"R{i}", (cx-20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Сохраняем в папку outputs
    output_path = os.path.join(modules.config.path_outputs, filename)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[Omost] Masks visualization saved to: {output_path}")
    return output_path