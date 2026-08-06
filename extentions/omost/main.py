import gradio as gr
from extentions.omost.chat_interface import ChatInterface
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


llm_model = None
llm_tokenizer = None
@torch.inference_mode()
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

    memory_management.load_models_to_gpu(llm_model)

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

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
def loading_model(llm_name="lllyasviel/omost-llama-3-8b-4bits"):    
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
        token=HF_TOKEN
    )
    #print(f"[Omost] LLM loaded successfully. Cached in: {cache_dir}")
    return gr.update(visible=False)


llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_name,
    token=HF_TOKEN
)


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
    # render_button.click(
    #     fn=diffusion_fn, inputs=[
    #         chatInterface.chatbot, canvas_state,
    #         num_samples, seed, image_width, image_height, highres_scale,
    #         steps, cfg, highres_steps, highres_denoise, n_prompt
    #     ], outputs=[chatInterface.chatbot]).then(
    #     fn=lambda x: x, inputs=[
    #         chatInterface.chatbot
    #     ], outputs=[chatInterface.chatbot_state])