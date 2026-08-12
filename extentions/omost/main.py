import gradio as gr
from extentions.omost.chat_interface import ChatInterface
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import os
import numpy as np
import extentions.omost.lib_omost.canvas as omost_canvas
import ldm_patched.modules.model_management as mm
import modules.default_pipeline as pipeline
import modules.core as core

#import extentions.omost.lib_omost.memory_management as memory_management
#from transformers.generation.stopping_criteria import StoppingCriteriaList
from threading import Thread


# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True
import gc


llm_model = None
llm_tokenizer = None
llm_name = None




def get_vram_info():
    """Возвращает кортеж: (allocated_gb, reserved_gb, total_gb, free_gb)"""
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0, 0.0)
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    
    # ИСПРАВЛЕНИЕ: total_memory вместо total_mem
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Более точный способ получить свободную память
    free, total_from_cuda = torch.cuda.mem_get_info()
    free = free / (1024**3)
    
    return (allocated, reserved, total, free)


def format_vram_info(prefix=""):
    """Форматирует VRAM инфо в красивую строку для лога."""
    allocated, reserved, total, free = get_vram_info()
    return (
        f"{prefix}VRAM: "
        f"allocated={allocated:.2f}GB | "
        f"reserved={reserved:.2f}GB | "
        f"free={free:.2f}GB / {total:.2f}GB total"
    )



# @torch.inference_mode()
# def unload_model():
#     """Принудительная выгрузка LLM из VRAM перед запуском диффузии или сменой модели"""
#     global llm_model, llm_tokenizer, llm_name
    
#     # Запоминаем VRAM ДО
#     allocated_before, reserved_before, total, _ = get_vram_info()
#     print(f"\033[93m[Omost] BEFORE unload:\033[0m allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB / {total:.2f}GB")
    
#     if llm_model is not None:
#         print("\033[92m[Omost] Unloading LLM from VRAM...\033[0m")
        
#         # === КРИТИЧЕСКИ ВАЖНО: Удаляем хуки accelerate ===
#         try:
#             from accelerate.hooks import remove_hook_from_module
#             remove_hook_from_module(llm_model, recurse=True)
#             print("[Omost] Accelerate hooks removed")
#         except Exception as e:
#             print(f"[Omost] Warning: Could not remove hooks: {e}")
        
#         # Удаляем модели
#         del llm_model
#         del llm_tokenizer
        
#         # НЕ используем del llm_name — просто присваиваем None
#         llm_model = None
#         llm_tokenizer = None
#         llm_name = None
        
#         # Агрессивная очистка
#         import gc
#         gc.collect()
        
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#             # Дополнительная очистка через CUDA allocator
#             try:
#                 torch.cuda.synchronize()
#                 torch.cuda.reset_peak_memory_stats()
#             except:
#                 pass
            
#             # Считаем VRAM ПОСЛЕ
#             allocated_after, reserved_after, _, _ = get_vram_info()
#             freed_allocated = allocated_before - allocated_after
#             freed_reserved = reserved_before - reserved_after
            
#             print(f"\033[92m[Omost] AFTER  unload:\033[0m  allocated={allocated_after:.2f}GB, reserved={reserved_after:.2f}GB / {total:.2f}GB")
#             print(f"\033[96m[Omost] FREED:\033[0m allocated={freed_allocated:.2f}GB, reserved={freed_reserved:.2f}GB")
            
#             if freed_allocated < 1.0:
#                 print(f"\033[91m[Omost] WARNING: Freed less than 1GB! Possible memory leak.\033[0m")
            
#             print("\033[92m[Omost] ✓ VRAM cleared successfully.\033[0m")
#     else:
#         print("[Omost] LLM was not loaded, nothing to unload.")





def post_chat(history):
    canvas_outputs = None
    try:
        if history:
            history = [(user, assistant) for user, assistant in history if isinstance(user, str) and isinstance(assistant, str)]
            last_assistant = history[-1][1] if len(history) > 0 else None
            canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
            canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)

    unload_model()
    return canvas_outputs, gr.update(visible=canvas_outputs is not None), gr.update(interactive=len(history) > 0)

def defragment_vram():
    if not torch.cuda.is_available():
        return
    
    # Замер ДО
    before = torch.cuda.memory_reserved() / (1024**3)
    
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Замер ПОСЛЕ
        after = torch.cuda.memory_reserved() / (1024**3)
        freed = before - after
        
        if freed > 0.01:
            print(f"[Omost] ✓ VRAM defragmented. Freed reserved: {freed:.2f} GB")
        else:
            print(f"[Omost] ✓ VRAM defragmented (reserved was already minimal).")
    except Exception as e:
        print(f"[Omost] Warning during VRAM defragmentation: {e}")

def debug_loaded_models():
    mm.unload_all_models()
    mm.soft_empty_cache()
    defragment_vram()
    
    free_gb = mm.get_free_memory(mm.get_torch_device()) / (1024**3)
    print(f"[Omost] Final free VRAM: {free_gb:.2f}GB")
    print(f"[Omost DEBUG] ===================================\n")


def unload_fooocus_completely():
    """
    Полностью выгружает все модели Fooocus из GPU и RAM.
    
    После этого Fooocus не сможет генерировать картинки,
    пока не будет вызван refresh_everything() заново.
    """
    
    print(f"\n[Omost] === Unloading ALL Fooocus models ===")
    
    # Запоминаем память ДО
    allocated_before, reserved_before, total, _ = get_vram_info()
    print(f"[Omost] BEFORE: allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB")
    
    # === Шаг 1: Выгружаем всё из GPU через model_management ===
    print(f"[Omost] Step 1: Unloading from GPU via model_management...")
       
    # === Шаг 2: Обнуляем ссылки в pipeline ===
    print(f"[Omost] Step 2: Clearing pipeline references...")
    try:
        pipeline.final_unet = None
        pipeline.final_clip = None
        pipeline.final_vae = None
        pipeline.final_refiner_unet = None
        pipeline.final_refiner_vae = None
        pipeline.final_expansion = None
        pipeline.loaded_ControlNets = {}
        print(f"[Omost] ✓ Pipeline references cleared")
    except Exception as e:
        print(f"[Omost] Warning: {e}")
    
    # === Шаг 3: Пересоздаём model_base и model_refiner как пустые ===
    print(f"[Omost] Step 3: Resetting model_base and model_refiner...")
    try:
        pipeline.model_base = core.StableDiffusionModel()
        pipeline.model_refiner = core.StableDiffusionModel()
        print(f"[Omost] ✓ model_base and model_refiner reset")
    except Exception as e:
        print(f"[Omost] Warning: {e}")
    
    # === Шаг 4: Агрессивная очистка памяти ===
    print(f"[Omost] Step 4: Aggressive memory cleanup...")
    import gc
    
    # Первый проход сборщика мусора
    gc.collect()
    
    if torch.cuda.is_available():
        # Синхронизация GPU
        torch.cuda.synchronize()
        
        # Очистка кэша PyTorch
        torch.cuda.empty_cache()
        
        # Сборка IPC памяти
        torch.cuda.ipc_collect()
        
        # Сброс пиковых статистик
        torch.cuda.reset_peak_memory_stats()
        
        # Второй проход сборщика мусора
        gc.collect()
        
        # Повторная синхронизация
        torch.cuda.synchronize()
    
    # === Шаг 5: Дефрагментация ===
    defragment_vram()
    
    # Запоминаем память ПОСЛЕ
    allocated_after, reserved_after, _, _ = get_vram_info()
    freed_allocated = allocated_before - allocated_after
    freed_reserved = reserved_before - reserved_after
    
    print(f"[Omost] AFTER:  allocated={allocated_after:.2f}GB, reserved={reserved_after:.2f}GB")
    print(f"[Omost] FREED:  allocated={freed_allocated:.2f}GB, reserved={freed_reserved:.2f}GB")
    print(f"[Omost] ✓ ALL Fooocus models unloaded")
    print(f"[Omost] ===================================\n")


@torch.inference_mode()
def unload_model():
    """Принудительная выгрузка LLM из VRAM перед запуском диффузии или сменой модели"""
    global llm_model, llm_tokenizer, llm_name
    try:
        allocator_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')
        print(f"[Omost] PYTORCH_CUDA_ALLOC_CONF = {allocator_conf}")
    
        # Проверяем backend аллокатора
        backend = torch.cuda.get_allocator_backend()
        print(f"[Omost] CUDA allocator backend: {backend}")
    except Exception as e:
        print(f"[Omost] Could not check allocator: {e}")    
    # Запоминаем VRAM ДО
    allocated_before, reserved_before, total, _ = get_vram_info()
    print(f"\033[93m[Omost] BEFORE unload:\033[0m allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB / {total:.2f}GB")
    
    if llm_model is not None:
        print("\033[92m[Omost] Unloading LLM from VRAM...\033[0m")
        
        # === КРИТИЧЕСКИ ВАЖНО: Удаляем хуки accelerate ===
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(llm_model, recurse=True)
            print("[Omost] Accelerate hooks removed")
        except Exception as e:
            print(f"[Omost] Warning: Could not remove hooks: {e}")
        
        # Удаляем модели
        del llm_model
        del llm_tokenizer
        
        # НЕ используем del llm_name — просто присваиваем None
        llm_model = None
        llm_tokenizer = None
        llm_name = None
        
        # Агрессивная очистка
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize() 

       
            # Считаем VRAM ПОСЛЕ
            allocated_after, reserved_after, _, _ = get_vram_info()
            freed_allocated = allocated_before - allocated_after
            freed_reserved = reserved_before - reserved_after
            
            print(f"\033[92m[Omost] AFTER  unload:\033[0m  allocated={allocated_after:.2f}GB, reserved={reserved_after:.2f}GB / {total:.2f}GB")
            print(f"\033[96m[Omost] FREED:\033[0m allocated={freed_allocated:.2f}GB, reserved={freed_reserved:.2f}GB")
            
            if freed_allocated < 1.0:
                print(f"\033[91m[Omost] WARNING: Freed less than 1GB! Possible memory leak.\033[0m")
            
            print("\033[92m[Omost] ✓ VRAM cleared successfully.\033[0m")
    else:
        print("[Omost] LLM was not loaded, nothing to unload.")

@torch.inference_mode()
def chat_fn(message: str, history: list, seed:int, temperature: float, top_p: float, max_new_tokens: int, model_base: str) -> str:
    global llm_model, llm_tokenizer, llm_name
    print(f'[OMOST] model_base {model_base}')
    print(f'[OMOST] llm_name {llm_name}')
    if llm_name == None:
        unload_fooocus_completely()
    if llm_name is not None and llm_name != model_base:
        unload_model()
    if llm_name == None:
        print(f"[Omost] Loading LLM: {model_base}...")

        llm_model = AutoModelForCausalLM.from_pretrained(
            f"lllyasviel/{model_base}",
            cache_dir=os.path.join("models","omost"),  # Указываем папку для кэша
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=None,        
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(
            f"lllyasviel/{model_base}",
            cache_dir=os.path.join("models","omost"),  # Указываем папку для кэша
            token=None
        )
        llm_name = model_base



    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    for user, assistant in history:
        if isinstance(user, str) and isinstance(assistant, str):
            if len(user) > 0 and len(assistant) > 0:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role": "user", "content": message})


    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    streamer = TextIteratorStreamer(llm_tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)

    # def interactive_stopping_criteria(*args, **kwargs) -> bool:
    #     if getattr(streamer, 'user_interrupted', False):
    #         print('User stopped generation')
    #         return True
    #     else:
    #         return False

    # stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])

    # def interrupter():
    #     streamer.user_interrupted = True
    #     return

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        #stopping_criteria=stopping_criteria,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    if temperature == 0:
        generate_kwargs['do_sample'] = False

    Thread(target=llm_model.generate, kwargs=generate_kwargs).start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        # print(outputs)
        yield "".join(outputs) #, interrupter

    return
def model_loading(llm_name="lllyasviel/omost-llama-3-8b-4bits"):  
#def model_loading(llm_name="lllyasviel/omost-dolphin-2.9-llama3-8b-4bits"):  
#def model_loading(llm_name="lllyasviel/omost-phi-3-mini-128k-8bits"):  
#def model_loading(llm_name="lllyasviel/omost-llama-3-8b"):  
#def model_loading(llm_name="lllyasviel/omost-dolphin-2.9-llama3-8b"):  
#def model_loading(llm_name="lllyasviel/omost-phi-3-mini-128k"):  
 

    global llm_model, llm_tokenizer
        
    print(f"[Omost] Loading LLM: {llm_name}...")
    

    
  
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        cache_dir=os.path.join("models","omost"),  # Указываем папку для кэша
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=None,
        
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_name,
        cache_dir=os.path.join("models","omost"),  # Указываем папку для кэша
        token=None
    )
    #print(f"[Omost] LLM loaded successfully. Cached in: {cache_dir}")
    return gr.update(visible=False)


def gui():
    models_name = ["omost-llama-3-8b-4bits","omost-dolphin-2.9-llama3-8b-4bits","omost-phi-3-mini-128k-8bits","omost-llama-3-8b","omost-dolphin-2.9-llama3-8b","omost-phi-3-mini-128k"] 
    with gr.Row(elem_classes='outer_parent'):
        with gr.Column(scale=25):
            with gr.Row():
                model_base = gr.Dropdown(choices=models_name, value=models_name[0], label='LLM model')
            with gr.Row():
                load_model = gr.Button("Load Model", variant="secondary", size="sm", min_width=60)
            with gr.Row():
                clear_btn = gr.Button("New Chat", variant="secondary", size="sm", min_width=60)
                retry_btn = gr.Button("Retry", variant="secondary", size="sm", min_width=60, visible=False)
                undo_btn = gr.Button("Edit Last Input", variant="secondary", size="sm", min_width=60, interactive=False)
        
            seed = gr.Number(label="Random Seed", value=12345, precision=0)

            with gr.Accordion(open=True, label='LLM settings'):
                with gr.Group():
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.01,
                            value=0.6,
                            label="Temperature")
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.9,
                            label="Top P")
                    max_new_tokens = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        step=1,
                        value=4096,
                        label="Max New Tokens")
            # with gr.Accordion(open=True, label='Image Diffusion Model'):
            #     with gr.Group():
            #         with gr.Row():
            #             image_width = gr.Slider(label="Image Width", minimum=256, maximum=2048, value=896, step=64)
            #             image_height = gr.Slider(label="Image Height", minimum=256, maximum=2048, value=1152, step=64)

            #         with gr.Row():
            #             num_samples = gr.Slider(label="Image Number", minimum=1, maximum=12, value=1, step=1)
            #             steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)

            with gr.Accordion(open=False, label='Advanced'):
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
                highres_scale = gr.Slider(label="HR-fix Scale (\"1\" is disabled)", minimum=1.0, maximum=2.0, value=1.0, step=0.01)
                highres_steps = gr.Slider(label="Highres Fix Steps", minimum=1, maximum=100, value=20, step=1)
                highres_denoise = gr.Slider(label="Highres Fix Denoise", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')

            render_button = gr.Button("Render the Image!", size='lg', variant="primary", visible=False)

            clear_llm = gr.Button("CLEAR LLM", size='lg', variant="primary", visible=True)
            mem_llm = gr.Button("memory", size='lg', variant="primary", visible=True)

            examples = gr.Dataset(
                samples=[
                    ['generate an image of the fierce battle of warriors and a dragon'],
                    ['change the dragon to a dinosaur']
                ],
                components=[gr.Textbox(visible=False)],
                label='Quick Prompts'
            )
        with gr.Column(scale=75, elem_classes='inner_parent'):
            canvas_state = gr.State(None)
            chatbot = gr.Chatbot(label='Omost chat', scale=1, show_copy_button=True, render=False)
            chatInterface = ChatInterface(
                fn=chat_fn,
                post_fn=post_chat,
                post_fn_kwargs=dict(inputs=[chatbot], outputs=[canvas_state, render_button, undo_btn]),
                pre_fn=lambda: gr.update(visible=False),
                pre_fn_kwargs=dict(outputs=[render_button]),
                chatbot=chatbot,
                retry_btn=retry_btn,
                undo_btn=undo_btn,
                clear_btn=clear_btn,
                additional_inputs=[seed, temperature, top_p, max_new_tokens, model_base],
                examples=examples
            )
    load_model.click(
        fn=model_loading,
        outputs=[load_model]
    )
    #render_button.click(unload_model)

    clear_llm.click(unload_model)
    mem_llm.click(debug_loaded_models)
    # render_button.click(
    #      fn=diffusion_fn, inputs=[
    #          chatInterface.chatbot, canvas_state,
    #          num_samples, seed, image_width, image_height, highres_scale,
    #          steps, cfg, highres_steps, highres_denoise, n_prompt
    #      ], outputs=[chatInterface.chatbot]).then(
    #      fn=lambda x: x, inputs=[
    #          chatInterface.chatbot
    #      ], outputs=[chatInterface.chatbot_state])
    return render_button, canvas_state, n_prompt