import torch
import numpy as np
import modules.default_pipeline as pipeline


def convert_masks_to_tensors(bag_of_conditions, target_height, target_width):
    """
    Конвертирует numpy-маски из Omost в torch-тензоры и масштабирует их под размер latent space.
    Omost возвращает маски 90x90. Для SDXL latent space (1024x1024) нужно масштабировать до 128x128.
    """
    masks = []
    # Latent space в 8 раз меньше пиксельного (для SDXL 1024x1024 -> 128x128)
    latent_height = target_height // 8
    latent_width = target_width // 8
    
    for cond in bag_of_conditions:
        mask_np = cond['mask']  # numpy array (90, 90)
        
        # Конвертируем в тензор и добавляем batch и channel измерения
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 90, 90)
        
        # Масштабируем под размер latent space
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=(latent_height, latent_width), mode='nearest'
        )
        
        # Убираем batch измерение, оставляем (1, latent_height, latent_width)
        mask_tensor = mask_tensor.squeeze(0)
        masks.append(mask_tensor)
    
    return masks

def encode_regional_prompts(bag_of_conditions):
    """
    Кодирует текст для каждого региона через CLIP и добавляет маски.
    Возвращает список conditioning с масками.
    """
    regional_conditioning = []
    
    for i, cond in enumerate(bag_of_conditions):
        # Склеиваем prefixes и suffixes в один промпт
        # Prefixes содержат глобальный стиль, suffixes - детали региона
        full_prompt = ", ".join(cond['prefixes'] + cond['suffixes'])
        
        # Кодируем через стандартную функцию Fooocus
        # clip_encode возвращает [[cond_tensor, {"pooled_output": pooled}]]
        encoded = pipeline.clip_encode([full_prompt], pool_top_k=1)
        
        if encoded and len(encoded) > 0:
            cond_tensor = encoded[0][0]
            pooled_output = encoded[0][1]["pooled_output"]
            
            # Сохраняем conditioning (маску добавим позже, когда будем знать размеры)
            regional_conditioning.append({
                'cond': cond_tensor,
                'pooled': pooled_output,
                'prompt': full_prompt,
                'index': i
            })
        else:
            print(f"[Omost] Failed to encode region {i}")
    
    return regional_conditioning



def build_regional_conditioning(bag_of_conditions, global_strength=0.2, region_strength=0.8):
    """
    Создает финальный список conditioning с масками для регионального промптинга.
    Формат полностью совпадает с тем, что использует ComfyUI_omost + ConditioningSetMask.
    """
    final_conditioning = []
    
    for i, cond in enumerate(bag_of_conditions):
        is_global = (i == 0)  # Первый регион - глобальный
        
        # Склеиваем префиксы и суффиксы в один промпт
        if is_global:
            full_prompt = ", ".join(cond['prefixes'] + cond['suffixes'])
        else:
            # Для регионов пропускаем глобальный префикс (как в OmostComfyLayoutNode)
            full_prompt = ", ".join(cond['prefixes'][1:] + cond['suffixes'])
        
        # Кодируем через стандартную функцию Fooocus
        encoded = pipeline.clip_encode([full_prompt], pool_top_k=1)
        
        if not encoded or len(encoded) == 0:
            print(f"[Omost] Failed to encode region {i}")
            continue
        
        cond_tensor = encoded[0][0]
        pooled_output = encoded[0][1]["pooled_output"]
        
        # Маска из Omost (numpy 90x90) -> torch tensor
        mask = torch.from_numpy(np.ascontiguousarray(cond['mask'])).float()
        
        # Сила воздействия маски
        strength = global_strength if is_global else region_strength
        
        # Формат conditioning ldm_patched/ComfyUI:
        # [cond_tensor, {"pooled_output": ..., "mask": ..., "strength": ...}]
        entry = [
            cond_tensor,
            {
                "pooled_output": pooled_output,
                "mask": mask,
                "strength": strength,
            }
        ]
        
        final_conditioning.append(entry)
        print(f"[Omost] Region {i} encoded. Mask shape: {tuple(mask.shape)}, strength: {strength}")
    
    print(f"[Omost] Built regional conditioning with {len(final_conditioning)} regions")
    return final_conditioning


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