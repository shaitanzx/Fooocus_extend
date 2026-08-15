import gradio as gr
from extentions.omost.chat_interface import ChatInterface
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import os
import numpy as np
import extentions.omost.lib_omost.canvas as omost_canvas
from transformers.generation.stopping_criteria import StoppingCriteriaList
import ldm_patched.modules.model_management
import modules.default_pipeline as pipeline
import modules.core as core

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
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[Omost] Warning during VRAM defragmentation: {e}")
def unload_model_by_name(target_name):
    """Удаляет модель по имени класса через существующую unload_model_clones."""
    for lm in ldm_patched.modules.model_management.current_loaded_models:
        try:
            name = lm.model.model.__class__.__name__
        except:
            name = "?"
        
        if name == target_name:
            print(f"[ModelMgmt] Unloading: {name}")
            ldm_patched.modules.model_management.unload_model_clones(lm.model)
            ldm_patched.modules.model_management.soft_empty_cache()
            return True
    
    print(f"[ModelMgmt] Model '{target_name}' not found")
    return False

def unload_fooocus_completely():
    """
    Полностью выгружает все модели Fooocus из GPU и RAM.
    
    После этого Fooocus не сможет генерировать картинки,
    пока не будет вызван refresh_everything() заново.
    """
    

    try:
        #ldm_patched.modules.model_management.unload_all_models()
        #unload_model_by_name('GPT2LMHeadModel')
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

    try:
        pipeline.model_base = core.StableDiffusionModel()
        pipeline.model_refiner = core.StableDiffusionModel()
    except Exception as e:
        print(f"[Omost] Warning: {e}")
    
    # === Шаг 4: Агрессивная очистка памяти ===

    
    # Первый проход сборщика мусора
    gc.collect()
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

@torch.inference_mode()
def unload_model():
    """Принудительная выгрузка LLM из VRAM перед запуском диффузии или сменой модели"""
    global llm_model, llm_tokenizer, llm_name
    
    if llm_model is not None:
        
        
        # === КРИТИЧЕСКИ ВАЖНО: Удаляем хуки accelerate ===
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(llm_model, recurse=True)

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
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize() 
    else:
        print("[Omost] LLM was not loaded, nothing to unload.")

@torch.inference_mode()
def chat_fn(message: str, history: list, seed:int, temperature: float, top_p: float, max_new_tokens: int, model_base: str, full_history: bool) -> str:
    global llm_model, llm_tokenizer, llm_name
    print(f'[OMOST] model_base {model_base}')
    print(f'[OMOST] llm_name {llm_name}')
    if llm_name is not None and llm_name != model_base:
        unload_model()
    if llm_name == None:
        unload_fooocus_completely()
        print(f"[Omost] Loading LLM: {model_base}...")

        llm_model = AutoModelForCausalLM.from_pretrained(
            f"lllyasviel/{model_base}",
            cache_dir=os.path.join("models","omost"),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=None,        
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(
            f"lllyasviel/{model_base}",
            cache_dir=os.path.join("models","omost"),
            token=None
        )
        llm_name = model_base

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]
    
    if full_history == False:
        select_history = history[-1:]
    else:
        select_history = history
    
    for user, assistant in select_history:
        if isinstance(user, str) and isinstance(assistant, str):
            if len(user) > 0 and len(assistant) > 0:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role": "user", "content": message})

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    # === НОВОЕ: streamer с флагом прерывания ===
    streamer = TextIteratorStreamer(llm_tokenizer, timeout=100.0, skip_prompt=True, skip_special_tokens=True)

    # === НОВОЕ: критерий остановки ===
    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        if getattr(streamer, 'user_interrupted', False):
            print('[Omost] User stopped generation')
            return True
        return False

    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])

    # === НОВОЕ: interrupter функция, возвращается в ChatInterface ===
    def interrupter():
        streamer.user_interrupted = True
        return

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        stopping_criteria=stopping_criteria,  # ← ВАЖНО: передаём критерий
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
        # === ВАЖНО: возвращаем TUPLE (текст, interrupter) ===
        yield "".join(outputs), interrupter

    return
def model_loading(llm_name):
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
    return gr.update(visible=False)


def gui():
    models_name = ["omost-llama-3-8b-4bits","omost-dolphin-2.9-llama3-8b-4bits","omost-phi-3-mini-128k-8bits","omost-llama-3-8b","omost-dolphin-2.9-llama3-8b","omost-phi-3-mini-128k"] 
    with gr.Row(elem_classes='outer_parent'):
        with gr.Column(scale=25):
            with gr.Row():
                model_base = gr.Dropdown(choices=models_name, value=models_name[0], label='LLM model')
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
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=4096,
                            step=1,
                            value=4096,
                            label="Max New Tokens")
                    with gr.Row():
                        full_history = gr.Checkbox(label='Use full history', value=False, elem_classes='min_check')

            # with gr.Accordion(open=False, label='Advanced'):
            #     cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
            #     highres_scale = gr.Slider(label="HR-fix Scale (\"1\" is disabled)", minimum=1.0, maximum=2.0, value=1.0, step=0.01)
            #     highres_steps = gr.Slider(label="Highres Fix Steps", minimum=1, maximum=100, value=20, step=1)
            #     highres_denoise = gr.Slider(label="Highres Fix Denoise", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
            #     n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')

            render_button = gr.Button("Render the Image!", size='lg', variant="primary", visible=False)

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
                additional_inputs=[seed, temperature, top_p, max_new_tokens, model_base, full_history],
                examples=examples
            )

    return render_button, canvas_state