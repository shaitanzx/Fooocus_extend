import torch
import copy
import numpy as np


@torch.inference_mode()
def encode_bag_of_subprompts_greedy(clip, prefixes: list, suffixes: list):

    prefix_text = ", ".join(prefixes)

    

    suffix_texts = suffixes
    max_len = 70 
    
    bags = []
    current_bag = []
    current_len = 0
    
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

        
    # 3. Кодируем каждую корзину
    conds = []
    poolers = []
    
    for i, bag in enumerate(bags):
        full_text = prefix_text + ", " + ", ".join(bag)
        
        tokens = clip.tokenize(full_text)
        cond, pooler = clip.encode_from_tokens(tokens, return_pooled=True)

        conds.append(cond)
        poolers.append(pooler)
        
    if not conds:
        tokens = clip.tokenize(prefix_text)
        cond, pooler = clip.encode_from_tokens(tokens, return_pooled=True)
        return cond, pooler
        
    conds_merged = torch.cat(conds, dim=1)
    pooler_merged = poolers[0]
    return conds_merged, pooler_merged


@torch.inference_mode()
def all_conds_from_canvas(clip, canvas_outputs, negative_prompt):

    positive_results = []
    negative_results = []
    

    neg_tokens = clip.tokenize(negative_prompt)
    neg_cond, neg_pooler = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    mask_all = torch.ones((1, 90, 90), dtype=torch.float32)
    negative_results.append((neg_cond, mask_all))

    

    positive_pooler = None
    bag_of_conditions = canvas_outputs.get('bag_of_conditions', [])

    
    for i, item in enumerate(bag_of_conditions):
        mask_np = item['mask']
        prefixes = item['prefixes']
        suffixes = item['suffixes']
        

        
        cond, pooler = encode_bag_of_subprompts_greedy(clip, prefixes, suffixes)
        
        if positive_pooler is None:
            positive_pooler = pooler
            
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0) # [1, 90, 90]

        positive_results.append((cond, mask_tensor))
        
    return positive_results, negative_results, positive_pooler, neg_pooler