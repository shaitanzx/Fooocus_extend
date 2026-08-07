import gradio as gr
from extentions.omost.chat_interface import ChatInterface
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import os
import numpy as np
import extentions.omost.lib_omost.canvas as omost_canvas
#import extentions.omost.lib_omost.memory_management as memory_management
from transformers.generation.stopping_criteria import StoppingCriteriaList
from threading import Thread
import warnings
warnings.filterwarnings("ignore", message="MatMul8bitLt")

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True
import gc

llm_model = None
llm_tokenizer = None


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


@torch.inference_mode()
def unload_model():
    """Принудительная выгрузка LLM из VRAM перед запуском диффузии"""
    global llm_model, llm_tokenizer
    
    # Запоминаем VRAM ДО
    allocated_before, reserved_before, total, _ = get_vram_info()
    print(f"\033[93m[Omost] BEFORE unload:\033[0m allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB / {total:.2f}GB")
    
    if llm_model is not None:
        print("\033[92m[Omost] Unloading LLM from VRAM...\033[0m")
        del llm_model
        del llm_tokenizer
        llm_model = None
        llm_tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Считаем VRAM ПОСЛЕ
            allocated_after, reserved_after, _, _ = get_vram_info()
            freed_allocated = allocated_before - allocated_after
            freed_reserved = reserved_before - reserved_after
            
            print(f"\033[92m[Omost] AFTER  unload:\033[0m  allocated={allocated_after:.2f}GB, reserved={reserved_after:.2f}GB / {total:.2f}GB")
            print(f"\033[96m[Omost] FREED:\033[0m allocated={freed_allocated:.2f}GB, reserved={freed_reserved:.2f}GB")
            print("\033[92m[Omost] ✓ VRAM cleared successfully.\033[0m")
    else:
        print("[Omost] LLM was not loaded, nothing to unload.")

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

    return canvas_outputs, gr.update(visible=canvas_outputs is not None), gr.update(interactive=len(history) > 0)

@torch.inference_mode()
def chat_fn(message: str, history: list, seed:int, temperature: float, top_p: float, max_new_tokens: int) -> str:
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

    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        if getattr(streamer, 'user_interrupted', False):
            print('User stopped generation')
            return True
        else:
            return False

    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])

    def interrupter():
        streamer.user_interrupted = True
        return

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

    Thread(target=llm_model.generate, kwargs=generate_kwargs).start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        # print(outputs)
        yield "".join(outputs), interrupter

    return
#def model_loading(llm_name="lllyasviel/omost-llama-3-8b-4bits"):  
#def model_loading(llm_name="lllyasviel/omost-dolphin-2.9-llama3-8b-4bits"):  
#def model_loading(llm_name="lllyasviel/omost-phi-3-mini-128k-8bits"):  
#def model_loading(llm_name="lllyasviel/omost-llama-3-8b"):  
#def model_loading(llm_name="lllyasviel/omost-dolphin-2.9-llama3-8b"):  
def model_loading(llm_name="lllyasviel/omost-phi-3-mini-128k"):  
 

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
@torch.inference_mode()
def diffusion_fn(chatbot, canvas_outputs, num_samples, seed, image_width, image_height,
                 highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt):

    use_initial_latent = False
    eps = 0.05

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])

    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)

    if highres_scale > 1.0 + eps:
        pixels = [
            resize_without_crop(
                image=p,
                target_width=int(round(W * highres_scale / 64.0) * 64),
                target_height=int(round(H * highres_scale / 64.0) * 64)
            ) for p in pixels
        ]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

        memory_management.load_models_to_gpu([unet])
        latents = latents.to(device=unet.device, dtype=unet.dtype)

        latents = pipeline(
            initial_latent=latents,
            strength=highres_denoise,
            num_inference_steps=highres_steps,
            batch_size=num_samples,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    for i in range(len(pixels)):
        unique_hex = uuid.uuid4().hex
        image_path = os.path.join(gradio_temp_dir, f"{unique_hex}_{i}.png")
        image = Image.fromarray(pixels[i])
        image.save(image_path)
        chatbot = chatbot + [(None, (image_path, 'image'))]

    return chatbot

def gui():
    with gr.Row(elem_classes='outer_parent'):
        with gr.Column(scale=25):
            with gr.Row():
                load_model = gr.Button("Load Model", variant="secondary", size="sm", min_width=60)
            with gr.Row():
                clear_btn = gr.Button("➕ New Chat", variant="secondary", size="sm", min_width=60)
                retry_btn = gr.Button("Retry", variant="secondary", size="sm", min_width=60, visible=False)
                undo_btn = gr.Button("✏️️ Edit Last Input", variant="secondary", size="sm", min_width=60, interactive=False)
        
            seed = gr.Number(label="Random Seed", value=12345, precision=0)

            with gr.Accordion(open=True, label='Language Model'):
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
            with gr.Accordion(open=True, label='Image Diffusion Model'):
                with gr.Group():
                    with gr.Row():
                        image_width = gr.Slider(label="Image Width", minimum=256, maximum=2048, value=896, step=64)
                        image_height = gr.Slider(label="Image Height", minimum=256, maximum=2048, value=1152, step=64)

                    with gr.Row():
                        num_samples = gr.Slider(label="Image Number", minimum=1, maximum=12, value=1, step=1)
                        steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)

            with gr.Accordion(open=False, label='Advanced'):
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
                highres_scale = gr.Slider(label="HR-fix Scale (\"1\" is disabled)", minimum=1.0, maximum=2.0, value=1.0, step=0.01)
                highres_steps = gr.Slider(label="Highres Fix Steps", minimum=1, maximum=100, value=20, step=1)
                highres_denoise = gr.Slider(label="Highres Fix Denoise", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')

            render_button = gr.Button("Render the Image!", size='lg', variant="primary", visible=False)

            clear_llm = gr.Button("CLEAR LLM", size='lg', variant="primary", visible=True)

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
            chatbot = gr.Chatbot(label='Omost', scale=1, show_copy_button=True, layout="panel", render=False)
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
                additional_inputs=[seed, temperature, top_p, max_new_tokens],
                examples=examples
            )
    load_model.click(
        fn=model_loading,
        outputs=[load_model]
    )
    render_button.click(unload_model)

    clear_llm.click(unload_model)
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