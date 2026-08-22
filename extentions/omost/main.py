import gradio as gr
from extentions.omost.chat_interface import ChatInterface
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import os
import numpy as np
import extentions.omost.lib_omost.canvas as omost_canvas
from transformers.generation.stopping_criteria import StoppingCriteriaList
import random
from threading import Thread
import ldm_patched.modules.model_management as mm
import modules.default_pipeline as pipeline
import modules.core as core
import gc
import torch

# Импорт общего состояния LLM (создаётся в omost_state.py)
import extentions.omost.omost_state as omost_state
from extentions.omost.omost_state import reset_llm_state

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel
Phi3PreTrainedModel._supports_sdpa = True


# ============================================================
# ОБЁРТКА ДЛЯ LLM ИЗ TRANSFORMERS
# Совместима с model_management, поддерживает 4-bit/8-bit модели
# ============================================================

class LLMModelPatcher:
    """
    Обёртка для LLM из transformers, совместимая с model_management.
    Правильно обрабатывает 4-bit/8-bit модели bitsandbytes.
    """
    
    def __init__(self, model, tokenizer, load_device, offload_device, model_name="LLM"):
        self.model = model
        self.tokenizer = tokenizer
        self.load_device = load_device
        self.offload_device = offload_device
        self.model_name = model_name
        # Маркер для идентификации в async_worker
        self.is_omost_llm = True
    
    def model_size(self):
        try:
            return sum(p.nelement() * p.element_size() for p in self.model.parameters())
        except Exception:
            return 0
    
    @property
    def current_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return self.offload_device
    
    def patch_model(self, device_to=None):
        """Для LLM ничего не делаем, возвращаем модель как есть."""
        return self.model
    
    def unpatch_model(self, device_to=None):
        """
        Снятие патчей.
        
        Для 4-bit моделей: НЕ перемещаем на CPU (не поддерживается bitsandbytes).
        Полагаемся на удаление ссылок и gc для освобождения памяти.
        """
        print(f"[LLMModelPatcher] unpatch_model: начало очистки")
        
        if self.model is None:
            print(f"[LLMModelPatcher] Модель уже None, пропуск")
            return self.model
        
        # === Шаг 1: Удаление хуков accelerate ===
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(self.model, recurse=True)
            print(f"[LLMModelPatcher] ✓ Хуки accelerate удалены")
        except Exception as e:
            print(f"[LLMModelPatcher] ⚠ Не удалось удалить хуки: {e}")
        
        # === Шаг 2: Проверка квантования ===
        is_quantized = False
        try:
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and hasattr(module.weight, 'quant_state'):
                    is_quantized = True
                    break
                if 'BitsAndBytes' in type(module).__name__ or 'bnb' in type(module).__module__:
                    is_quantized = True
                    break
        except Exception:
            pass
        
        if is_quantized:
            print(f"[LLMModelPatcher] ℹ Модель квантованная (4/8-bit), пропуск .to('cpu')")
            print(f"[LLMModelPatcher] ℹ Освобождение памяти через удаление ссылок + gc")
        else:
            # Для обычных моделей — перемещаем на CPU
            try:
                self.model.to('cpu')
                print(f"[LLMModelPatcher] ✓ Модель перемещена на CPU")
            except Exception as e:
                print(f"[LLMModelPatcher] ⚠ Не удалось переместить на CPU: {e}")
        
        # === Шаг 3: Очистка CUDA кэша ===
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"[LLMModelPatcher] ✓ CUDA кэш очищен")
            except Exception as e:
                print(f"[LLMModelPatcher] ⚠ Ошибка очистки CUDA: {e}")
        
        print(f"[LLMModelPatcher] unpatch_model: очистка завершена")
        return self.model
    
    def model_patches_to(self, device):
        """Для LLM нет патчей, просто возвращаем себя."""
        return self
    
    def model_dtype(self):
        try:
            return next(self.model.parameters()).dtype
        except Exception:
            return torch.float32
    
    def is_clone(self, other):
        if hasattr(other, 'model'):
            return self.model is other.model
        return False


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_vram_info():
    """Возвращает кортеж: (allocated_gb, reserved_gb, total_gb, free_gb)"""
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0, 0.0)
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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


def defragment_vram():
    """Дефрагментация VRAM."""
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
    for lm in mm.current_loaded_models:
        try:
            name = lm.model.model.__class__.__name__
        except:
            name = "?"
        
        if name == target_name:
            print(f"[ModelMgmt] Unloading: {name}")
            mm.unload_model_clones(lm.model)
            mm.soft_empty_cache()
            return True
    
    print(f"[ModelMgmt] Model '{target_name}' not found")
    return False


# ============================================================
# ПОЛНАЯ ВЫГРУЗКА МОДЕЛЕЙ FOOOCUS
# ============================================================

def unload_fooocus_completely():
    """Правильная полная выгрузка Fooocus: GPU → ссылки → gc."""
    
    print(f"\n[Omost] === Unloading ALL Fooocus models ===")
    
    if torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / (1024**3)
        reserved_before = torch.cuda.memory_reserved() / (1024**3)
        print(f"[Omost] BEFORE: allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB")
    
    # === ШАГ 1: Выгрузка моделей из GPU ===
    print(f"[Omost] Step 1: Proper GPU unload via unload_all_models()...")
    try:
        mm.unload_all_models()
        print(f"[Omost] ✓ Models properly unloaded from GPU")
    except Exception as e:
        print(f"[Omost] Warning during unload_all_models: {e}")
    
    # === ШАГ 2: Очистка current_loaded_models ===
    print(f"[Omost] Step 2: Verifying current_loaded_models...")
    if len(mm.current_loaded_models) > 0:
        print(f"[Omost] ⚠ {len(mm.current_loaded_models)} models still in current_loaded_models, forcing removal...")
        for i in range(len(mm.current_loaded_models) - 1, -1, -1):
            try:
                m = mm.current_loaded_models.pop(i)
                m.model_unload()
                del m
            except Exception as e:
                print(f"[Omost] Warning: {e}")
    print(f"[Omost] ✓ current_loaded_models is now empty: {len(mm.current_loaded_models)}")
    
    # === ШАГ 3: Обнуление ссылок в pipeline ===
    print(f"[Omost] Step 3: Clearing pipeline references...")
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
    
    # === ШАГ 4: Пересоздание model_base и model_refiner ===
    print(f"[Omost] Step 4: Resetting model_base and model_refiner...")
    try:
        pipeline.model_base = core.StableDiffusionModel()
        pipeline.model_refiner = core.StableDiffusionModel()
        print(f"[Omost] ✓ model_base and model_refiner reset")
    except Exception as e:
        print(f"[Omost] Warning: {e}")
    
    # === ШАГ 5: Агрессивная очистка памяти ===
    print(f"[Omost] Step 5: Aggressive memory cleanup...")
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.synchronize()
    
    # === ШАГ 6: Дефрагментация ===
    defragment_vram()
    
    if torch.cuda.is_available():
        allocated_after = torch.cuda.memory_allocated() / (1024**3)
        reserved_after = torch.cuda.memory_reserved() / (1024**3)
        freed_allocated = allocated_before - allocated_after
        freed_reserved = reserved_before - reserved_after
        print(f"[Omost] AFTER:  allocated={allocated_after:.2f}GB, reserved={reserved_after:.2f}GB")
        print(f"[Omost] FREED:  allocated={freed_allocated:.2f}GB, reserved={freed_reserved:.2f}GB")
    print(f"[Omost] ✓ ALL Fooocus models unloaded")
    print(f"[Omost] ===================================\n")


# ============================================================
# ЗАГРУЗКА И ВЫГРУЗКА LLM ЧЕРЕЗ MODEL_MANAGEMENT
# ============================================================

def load_llm_model(model_base):
    """
    Загружает модель из transformers и регистрирует её в model_management.
    
    Args:
        model_base: имя модели из списка (например, 'omost-llama-3-8b-4bits')
    
    Returns:
        bool: True если загрузка успешна
    """
    print(f"\n[LLM Load] === Загрузка модели: {model_base} ===")
    
    # === Шаг 1: Загружаем модель через transformers ===
    print(f"[LLM Load] Шаг 1: Загрузка модели через transformers...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            f"lllyasviel/{model_base}",
            cache_dir=os.path.join("models", "omost"),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=None,
        )
        print(f"[LLM Load] ✓ Модель загружена")
    except Exception as e:
        print(f"[LLM Load] ✗ Ошибка загрузки модели: {e}")
        return False
    
    # === Шаг 2: Загружаем токенайзер ===
    print(f"[LLM Load] Шаг 2: Загрузка токенизатора...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            f"lllyasviel/{model_base}",
            cache_dir=os.path.join("models", "omost"),
            token=None,
        )
        print(f"[LLM Load] ✓ Токенайзер загружен")
    except Exception as e:
        print(f"[LLM Load] ✗ Ошибка загрузки токенизатора: {e}")
        del model
        return False
    
    # === Шаг 3: Создаём обёртку ===
    print(f"[LLM Load] Шаг 3: Создание обёртки LLMModelPatcher...")
    
    try:
        load_device = mm.get_torch_device()
        offload_device = torch.device('cpu')
        
        patcher = LLMModelPatcher(
            model=model,
            tokenizer=tokenizer,
            load_device=load_device,
            offload_device=offload_device,
            model_name=model_base,
        )
        
        model_size_mb = patcher.model_size() / (1024**2)
        print(f"[LLM Load] ✓ Обёртка создана, размер модели: {model_size_mb:.1f}MB")
    except Exception as e:
        print(f"[LLM Load] ✗ Ошибка создания обёртки: {e}")
        del model, tokenizer
        return False
    
    # === Шаг 4: Регистрируем в model_management ===
    print(f"[LLM Load] Шаг 4: Регистрация в model_management...")
    
    try:
        loaded_model = mm.LoadedModel(patcher)
        mm.current_loaded_models.insert(0, loaded_model)
        print(f"[LLM Load] ✓ Модель зарегистрирована в current_loaded_models")
        print(f"[LLM Load] ✓ Всего моделей в памяти: {len(mm.current_loaded_models)}")
    except Exception as e:
        print(f"[LLM Load] ⚠ Ошибка регистрации в model_management: {e}")
    
    # === Шаг 5: Обновляем глобальные переменные через omost_state ===
    omost_state.llm_patcher = patcher
    omost_state.llm_model = model
    omost_state.llm_tokenizer = tokenizer
    omost_state.llm_name = model_base
    
    print(f"[LLM Load] ✓ Модель готова к работе")
    print(f"[LLM Load] === Загрузка завершена ===\n")
    
    return True


def unload_llm_model():
    """
    Выгружает LLM из памяти через model_management.
    Обнуляет глобальные ссылки через omost_state.
    """
    print(f"\n[LLM Unload] === Выгрузка модели ===")
    
    if omost_state.llm_patcher is None and omost_state.llm_model is None:
        print(f"[LLM Unload] Модель не загружена, нечего выгружать")
        print(f"[LLM Unload] === Выгрузка завершена ===\n")
        return True
    
    # === Шаг 1: Удаляем из model_management ===
    if omost_state.llm_patcher is not None:
        try:
            mm.unload_model_clones(omost_state.llm_patcher)
            print(f"[LLM Unload] ✓ Модель удалена из current_loaded_models")
        except Exception as e:
            print(f"[LLM Unload] ⚠ Ошибка удаления из model_management: {e}")
            try:
                for i in range(len(mm.current_loaded_models) - 1, -1, -1):
                    if mm.current_loaded_models[i].model is omost_state.llm_patcher:
                        mm.current_loaded_models.pop(i).model_unload()
                        print(f"[LLM Unload] ✓ Модель удалена вручную")
                        break
            except Exception as e2:
                print(f"[LLM Unload] ⚠ Ошибка ручного удаления: {e2}")
    
    # === Шаг 2: Явный вызов unpatch_model ===
    if omost_state.llm_patcher is not None:
        try:
            omost_state.llm_patcher.unpatch_model()
            print(f"[LLM Unload] ✓ unpatch_model() вызван явно")
        except Exception as e:
            print(f"[LLM Unload] ⚠ Ошибка unpatch_model: {e}")
    
    # === Шаг 3: Обнуление глобальных ссылок через omost_state ===
    reset_llm_state()
    print(f"[LLM Unload] ✓ omost_state обнулён")
    
    # === Шаг 4: Агрессивная очистка памяти ===
    gc.collect()
    print(f"[LLM Unload] ✓ gc.collect() (первый проход)")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        print(f"[LLM Unload] ✓ CUDA кэш очищен")
    
    gc.collect()
    print(f"[LLM Unload] ✓ gc.collect() (второй проход)")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[LLM Unload] ✓ CUDA синхронизирован")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free, total = torch.cuda.mem_get_info()
        print(f"[LLM Unload] Состояние VRAM после выгрузки:")
        print(f"[LLM Unload]   Allocated: {allocated:.2f} GB")
        print(f"[LLM Unload]   Reserved:  {reserved:.2f} GB")
        print(f"[LLM Unload]   Free:      {free/(1024**3):.2f} GB")
    
    print(f"[LLM Unload] === Выгрузка завершена ===\n")
    
    return True


# ============================================================
# ОБРАБОТЧИКИ ЧАТА
# ============================================================

def post_chat(history):
    """Обработчик после завершения чата."""
    last_assistant = ""
    print(f"\n[Omost post_chat] CALLED")
    print(f"[Omost post_chat] Input history length: {len(history) if history else 0}")
    if history:
        print(f"[Omost post_chat] Input history: {history}")
    
    canvas_outputs = None
    try:
        if history:
            history = [(user, assistant) for user, assistant in history if isinstance(user, str) and isinstance(assistant, str)]
            print(f"[Omost post_chat] Filtered history length: {len(history) if history else 0}")
            if history:
                print(f"[Omost post_chat] Filtered history: {history}")
                last_assistant = history[-1][1] if len(history) > 0 else None
                canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
                canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)

    render_visible = canvas_outputs is not None 
    print(f"[Omost post_chat] Render visible: {render_visible}")
    
    return (
        gr.update(value=last_assistant),
        gr.update(value=omost_state.omost_seed),
        canvas_outputs,
        gr.update(visible=render_visible),
        gr.update(visible=render_visible),
        gr.update(interactive=len(history) > 0 if history else False)
    )


@torch.inference_mode()
def chat_fn(message: str, history: list, seed: int, temperature: float, top_p: float, 
            max_new_tokens: int, model_base: str, full_history: bool, seed_random: bool) -> str:
    """Основная функция генерации чата с LLM."""
    
    print(f'\n[Chat] === Начало генерации чата ===')
    print(f'[Chat] Запрошенная модель: {model_base}')
    print(f'[Chat] Текущая модель:     {omost_state.llm_name}')
    
    # === НОВАЯ ПРОВЕРКА: реальное состояние LLM ===
    def is_llm_actually_loaded():
        """Проверяет, реально ли LLM загружена в GPU."""
        if omost_state.llm_patcher is None or omost_state.llm_model is None:
            return False
        
        for lm in mm.current_loaded_models:
            if getattr(lm.model, 'is_omost_llm', False):
                return True
        
        return False
    
    llm_loaded = is_llm_actually_loaded()
    print(f'[Chat] LLM реально в GPU: {llm_loaded}')
    
    # === Принятие решения о загрузке ===
    need_reload = False
    
    if not llm_loaded:
        print(f'[Chat] LLM не загружена (или была выгружена)')
        reset_llm_state()
        need_reload = True
    elif omost_state.llm_name != model_base:
        print(f'[Chat] Модель изменилась: {omost_state.llm_name} → {model_base}')
        need_reload = True
    else:
        print(f'[Chat] Модель уже загружена, используем её')
    
    if need_reload:
        # Выгружаем старую модель
        if omost_state.llm_model is not None:
            print(f'[Chat] Выгружаем старую модель...')
            unload_llm_model()
        
        # Выгружаем модели диффузии
        print(f'[Chat] Выгружаем модели диффузии...')
        unload_fooocus_completely()
        
        # Загружаем новую модель
        print(f'[Chat] Загружаем новую модель...')
        if not load_llm_model(model_base):
            print(f'[Chat] ✗ Не удалось загрузить модель')
            yield "Ошибка загрузки модели", None
            return
    
    # === Настройка seed ===
    if seed_random:
        seed = random.randint(0, 2**32 - 1)
    omost_state.omost_seed = seed
    print(f'[Chat] Using seed: {seed} (random={seed_random})')
    
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    
    # === Формирование conversation ===
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
    
    input_ids = omost_state.llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(omost_state.llm_model.device)
    
    # === Streamer с флагом прерывания ===
    streamer = TextIteratorStreamer(
        omost_state.llm_tokenizer, 
        timeout=100.0, 
        skip_prompt=True, 
        skip_special_tokens=True
    )
    
    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        if getattr(streamer, 'user_interrupted', False):
            print('[Chat] User stopped generation')
            return True
        return False
    
    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])
    
    def interrupter():
        streamer.user_interrupted = True
        return
    
    # === Генерация ===
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    if temperature == 0:
        generate_kwargs['do_sample'] = False
    
    Thread(target=omost_state.llm_model.generate, kwargs=generate_kwargs).start()
    
    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs), interrupter
    
    print(f'[Chat] === Генерация завершена ===\n')
    return


# ============================================================
# GUI
# ============================================================

def gui():
    models_name = [
        "omost-llama-3-8b-4bits",
        "omost-dolphin-2.9-llama3-8b-4bits",
        "omost-phi-3-mini-128k-8bits",
        "omost-llama-3-8b",
        "omost-dolphin-2.9-llama3-8b",
        "omost-phi-3-mini-128k"
    ]
    
    with gr.Row(elem_classes='outer_parent'):
        with gr.Column(scale=25):
            with gr.Row():
                model_base = gr.Dropdown(choices=models_name, value=models_name[0], label='LLM model')
            with gr.Row():
                clear_btn = gr.Button("New Chat", variant="secondary", size="sm", min_width=60)
                retry_btn = gr.Button("Retry", variant="secondary", size="sm", min_width=60, visible=False)
                undo_btn = gr.Button("Edit Last Input", variant="secondary", size="sm", min_width=60, interactive=False)
            
            with gr.Accordion(open=True, label='LLM settings'):
                with gr.Group():
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.01, value=0.6, label="Temperature")
                        top_p = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Top P")
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=128, maximum=4096, step=1, value=4096, label="Max New Tokens")
                    with gr.Row():
                        seed_random = gr.Checkbox(label='Random Seed', value=True, elem_classes='min_check')
                    with gr.Row():
                        seed = gr.Number(label="Seed Value", value=12345, precision=0, visible=False)
                    with gr.Row():
                        full_history = gr.Checkbox(label='Use full history', value=False, elem_classes='min_check')

            with gr.Row():
                render_button = gr.Button("Render the Image!", size='lg', variant="primary", visible=False)
            with gr.Row(visible=False) as prompt_button:
                prompt_key = gr.Button("Convert to prompt!", size='lg', variant="primary")
                prompt_agress = gr.Radio(
                    choices=['normal', 'aggressive', 'short'], 
                    value='aggressive',
                    interactive=True
                )

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
            with gr.Row():
                prompt_code = gr.Textbox(label="Answer", value='', visible=True)
            chatbot = gr.Chatbot(label='Omost chat', scale=1, show_copy_button=True, render=False)
            chatInterface = ChatInterface(
                fn=chat_fn,
                post_fn=post_chat,
                post_fn_kwargs=dict(
                    inputs=[chatbot], 
                    outputs=[prompt_code, seed, canvas_state, render_button, prompt_button, undo_btn]
                ),
                pre_fn=lambda: gr.update(visible=False),
                pre_fn_kwargs=dict(outputs=[render_button]),
                chatbot=chatbot,
                retry_btn=retry_btn,
                undo_btn=undo_btn,
                clear_btn=clear_btn,
                additional_inputs=[seed, temperature, top_p, max_new_tokens, model_base, full_history, seed_random],
                examples=examples
            )
        
        seed_random.change(
            lambda x: gr.update(visible=not x),
            inputs=seed_random,
            outputs=seed,
            queue=False,
            show_progress=False
        )

    return render_button, canvas_state, prompt_key, prompt_agress, prompt_code