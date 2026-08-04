import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.lib_omost.canvas import Canvas as OmostCanvas
from modules.lib_omost.canvas import system_prompt

# Глобальные переменные для хранения модели, чтобы не перезагружать её каждый раз
_global_llm = None
_global_tokenizer = None

def load_local_llm(model_name="lllyasviel/omost-llama-3-8b-4bits"):
    global _global_llm, _global_tokenizer
    if _global_llm is not None:
        return _global_llm, _global_tokenizer
        
    print(f"[Omost] Loading LLM: {model_name}...")
    
    # Используем fp16, если поддерживается, иначе fp32
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    _global_llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto", # Автоматически распределит слои между GPU и CPU, если не влезут
        trust_remote_code=True,
    )
    _global_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[Omost] LLM loaded successfully.")
    return _global_llm, _global_tokenizer

def unload_llm():
    """Принудительная выгрузка LLM из VRAM перед запуском диффузии"""
    global _global_llm, _global_tokenizer
    if _global_llm is not None:
        print("[Omost] Unloading LLM from VRAM...")
        del _global_llm
        del _global_tokenizer
        _global_llm = None
        _global_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[Omost] VRAM cleared.")

def generate_canvas(user_prompt: str, model_name="lllyasviel/omost-llama-3-8b-4bits"):
    """
    Генерирует структуру Canvas (список регионов и промптов) на основе текста пользователя.
    Возвращает список словарей OmostCanvasCondition или None при ошибке.
    """
    llm, tokenizer = load_local_llm(model_name)
    
    # Формируем диалог с системным промптом Omost
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Применяем шаблон чата (ChatML / Llama-3 формат)
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    ).to(llm.device)
    
    input_length = input_ids.shape[1]
    
    # Генерация ответа
    output_ids = llm.generate(
        input_ids=input_ids,
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
    )
    
    generated_text = tokenizer.decode(
        output_ids[0][input_length:], 
        skip_special_tokens=True
    )
    
    print(f"[Omost] LLM generated response:\n{generated_text[:100]}...")
    
    # Парсим Python-код в структуру Canvas
    try:
        canvas = OmostCanvas.from_bot_response(generated_text).process()
        print(f"[Omost] Canvas parsed successfully. Found {len(canvas)} regions.")
        return canvas
    except Exception as e:
        print(f"[Omost] Error parsing canvas: {e}")
        return None

# Тестовый блок для проверки работы
if __name__ == "__main__":
    canvas = generate_canvas("A cozy living room, a cat sleeping on the left, a hot cup of tea on the right.")
    
    if canvas:
        print(f"Canvas type: {type(canvas)}")
        print(f"Canvas length: {len(canvas)}")
        print(f"Canvas content: {canvas}")
        
        for i, cond in enumerate(canvas):
            print(f"Region {i} type: {type(cond)}")
            if isinstance(cond, dict):
                print(f"Region {i}: {cond.get('rect', 'No rect key')}")
                print(f"Region {i} keys: {cond.keys()}")
            else:
                print(f"Region {i} is not a dict. Content: {cond}")
    else:
        print("Canvas is None")
        
    unload_llm()