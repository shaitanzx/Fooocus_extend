import torch
import copy
import numpy as np


@torch.inference_mode()
def encode_bag_of_subprompts_greedy(clip, prefixes: list, suffixes: list):
    """
    Адаптация encode_bag_of_subprompts_greedy для CLIP из ldm_patched (Fooocus).
    Возвращает conds и pooler.
    """
    print(f"[Omost] encode_bag_of_subprompts_greedy START")
    print(f"[Omost] Prefixes count: {len(prefixes)}, Suffixes count: {len(suffixes)}")
    
    # 1. Токенизируем префиксы (глобальное описание)
    prefix_text = ", ".join(prefixes)
    print(f"[Omost] Prefix text length: {len(prefix_text)} chars")
    
    # 2. Жадная упаковка суффиксов
    suffix_texts = suffixes
    max_len = 70 # Оставляем запас (в символах, грубая оценка для корзины)
    
    bags = []
    current_bag = []
    current_len = 0
    
    # Оцениваем длину префикса (грубо, 1 токен ~ 4 символа)
    prefix_len = len(prefix_text) // 4 
    
    for s in suffix_texts:
        s_len = len(s) // 4
        if current_len + s_len > max_len - prefix_len:
            if current_bag:
                bags.append(current_bag)
            current_bag = [s]
            current_len = s_len
        else:
            current_bag.append(s)
            current_len += s_len
            
    if current_bag:
        bags.append(current_bag)
        
    print(f"[Omost] Greedy partition created {len(bags)} bags")
        
    # 3. Кодируем каждую корзину
    conds = []
    poolers = []
    
    for i, bag in enumerate(bags):
        full_text = prefix_text + ", " + ", ".join(bag)
        print(f"[Omost] Encoding bag {i+1}/{len(bags)}, text length: {len(full_text)} chars")
        
        tokens = clip.tokenize(full_text)
        cond, pooler = clip.encode_from_tokens(tokens, return_pooled=True)
        
        print(f"[Omost] Bag {i+1} encoded. Cond shape: {cond.shape}, Pooler shape: {pooler.shape}")
        conds.append(cond)
        poolers.append(pooler)
        
    # 4. Конкатенируем
    if not conds:
        print("[Omost] WARNING: No bags created, falling back to prefix only")
        tokens = clip.tokenize(prefix_text)
        cond, pooler = clip.encode_from_tokens(tokens, return_pooled=True)
        print(f"[Omost] encode_bag_of_subprompts_greedy END (fallback)")
        return cond, pooler
        
    conds_merged = torch.cat(conds, dim=1)
    pooler_merged = poolers[0]
    
    print(f"[Omost] Merged conds shape: {conds_merged.shape}")
    print(f"[Omost] encode_bag_of_subprompts_greedy END")
    
    return conds_merged, pooler_merged


@torch.inference_mode()
def all_conds_from_canvas(clip, canvas_outputs, negative_prompt):
    """
    Генерирует список кондишенов и масок для Omost.
    """
    print("[Omost] all_conds_from_canvas START")
    positive_results = []
    negative_results = []
    
    # Глобальный негатив
    print(f"[Omost] Encoding negative prompt: '{negative_prompt[:50]}...'")
    neg_tokens = clip.tokenize(negative_prompt)
    neg_cond, neg_pooler = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    mask_all = torch.ones((1, 90, 90), dtype=torch.float32)
    negative_results.append((neg_cond, mask_all))
    print(f"[Omost] Negative cond shape: {neg_cond.shape}")
    
    # Регионы из canvas
    positive_pooler = None
    bag_of_conditions = canvas_outputs.get('bag_of_conditions', [])
    print(f"[Omost] Processing {len(bag_of_conditions)} regions from canvas")
    
    for i, item in enumerate(bag_of_conditions):
        mask_np = item['mask']
        prefixes = item['prefixes']
        suffixes = item['suffixes']
        
        print(f"[Omost] Region {i+1}/{len(bag_of_conditions)}: prefixes={len(prefixes)}, suffixes={len(suffixes)}")
        
        cond, pooler = encode_bag_of_subprompts_greedy(clip, prefixes, suffixes)
        
        if positive_pooler is None:
            positive_pooler = pooler
            
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0) # [1, 90, 90]
        print(f"[Omost] Region {i+1} mask shape: {mask_tensor.shape}, sum: {mask_tensor.sum().item()}")
        positive_results.append((cond, mask_tensor))
        
    print(f"[Omost] all_conds_from_canvas END. Positive regions: {len(positive_results)}, Negative regions: {len(negative_results)}")
    return positive_results, negative_results, positive_pooler, neg_pooler