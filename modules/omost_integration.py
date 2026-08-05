import torch
import gc
import threading
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from modules.lib_omost.canvas import Canvas as OmostCanvas
from modules.lib_omost.canvas import system_prompt

# Укажите путь к папке, где будет храниться модель
# Пример для Linux: "/content/Fooocus_extend/models/omost"
# Пример для Windows: "C:/Fooocus_extend/models/omost"
#MODEL_CACHE_DIR = os.path.join("models","omost")

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
        cache_dir=os.path.join("models","omost"),  # Указываем папку для кэша
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _global_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=os.path.join("models","omost"),  # Указываем папку для кэша
    )
    #print(f"[Omost] LLM loaded successfully. Cached in: {cache_dir}")
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

def generate_canvas(user_prompt: str, model_name="lllyasviel/omost-llama-3-8b-4bits", cache_dir=None):
    llm, tokenizer = load_local_llm(model_name, cache_dir=cache_dir)
    
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    ).to(llm.device)
    
    input_length = input_ids.shape[1]
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        streamer=streamer,
    )
    
    thread = threading.Thread(target=llm.generate, kwargs=generation_kwargs)
    thread.start()
    
    print("[Omost] LLM generating code in real-time:")
    print("-" * 50)
    
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text
    
    print("\n" + "-" * 50)
    thread.join()
    
    try:
        canvas_dict = OmostCanvas.from_bot_response(generated_text).process()
        
        if canvas_dict and 'bag_of_conditions' in canvas_dict:
            bag_of_conditions = canvas_dict['bag_of_conditions']
            initial_latent = canvas_dict.get('initial_latent', None)
            print(f"[Omost] Canvas parsed. Found {len(bag_of_conditions)} regions. Initial latent: {'yes' if initial_latent is not None else 'no'}")
            return {
                'bag_of_conditions': bag_of_conditions,
                'initial_latent': initial_latent
            }
        else:
            print("[Omost] Canvas missing 'bag_of_conditions'.")
            return None
            
    except Exception as e:
        print(f"[Omost] Error parsing canvas: {e}")
        return None

# Тестовый блок для проверки работы
if __name__ == "__main__":
    bag_of_conditions = generate_canvas(
        "A cozy living room, a cat sleeping on the left, a hot cup of tea on the right."
    )
    
    if bag_of_conditions:
        for i, cond in enumerate(bag_of_conditions):
            print(f"Region {i}:")
            print(f"  Mask shape: {cond['mask'].shape}")
            print(f"  Prefixes: {cond['prefixes']}")
            print(f"  Suffixes count: {len(cond['suffixes'])}")
    else:
        print("Failed to generate bag_of_conditions")
        
    unload_llm()