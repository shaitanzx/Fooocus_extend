import gradio as gr
import os


def ui_tab(mode, submode, eladd):
    """Создает компоненты для конкретной вкладки режима."""
    vret = None
    if mode == "Matrix":
        with gr.Row():
            mguide = gr.HTML(value = r"""<a href="https://github.com/hako-mikan/sd-webui-regional-prompter#2d-region-assignment">Matrix mode guide</a>""") 
        with gr.Row():
            mmode = gr.Radio(label="Main Splitting", choices=submode, value="Columns", type="value", interactive=True, elem_id="RP_main_splitting" + eladd)
            ratios = gr.Textbox(label="Divide Ratio", lines=1, value="1,1", interactive=True, elem_id="RP_divide_ratio" + eladd, visible=True)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    twid = gr.Slider(label="Width", minimum=64, maximum=2048, value=512, step=8, elem_id="RP_matrix_width" + eladd)
                    thei = gr.Slider(label="Height", minimum=64, maximum=2048, value=512, step=8, elem_id="RP_matrix_height" + eladd)
                maketemp = gr.Button(value="visualize and make template")
                template = gr.Textbox(label="template", interactive=True, visible=True, elem_id="RP_matrix_template" + eladd)
                flipper = gr.Checkbox(label='flip "," and ";"', value=False, elem_id="RP_matrix_flip" + eladd)
                overlay = gr.Slider(label="Overlay Ratio", minimum=0, maximum=1, step=0.1, value=0.5, elem_id="RP_matrix_overlay" + eladd)
            with gr.Column():
                areasimg = gr.Image(type="pil", show_label=False, height=256, width=256, source="upload", interactive=True)
        vret = [mmode, ratios, maketemp, template, areasimg, flipper, thei, twid, overlay]

    elif mode == "Mask":
        with gr.Row():
            xguide = gr.HTML(value = r"""<a href="https://github.com/hako-mikan/sd-webui-regional-prompter#mask-regions-aka-inpaint-experimental-function">Inpaint+ mode guide</a>""")
        with gr.Row():
            xmode = gr.Radio(label="Mask mode", choices=submode, value="Mask", type="value", interactive=True, elem_id="RP_mask_mode" + eladd)
        with gr.Row():
            # Упрощено для примера: создание холста для рисования маски
            polymask = gr.Image(label="Draw Mask Here", elem_id="polymask" + eladd, source="upload", mirror_webcam=False, type="numpy", tool="sketch")
            
            with gr.Column():
                num = gr.Slider(label="Region", minimum=-1, maximum=10, step=1, value=1, elem_id="RP_mask_region" + eladd)
                canvas_width = gr.Slider(label="Inpaint+ Width", minimum=64, maximum=2048, value=512, step=8, elem_id="RP_mask_width" + eladd)
                canvas_height = gr.Slider(label="Inpaint+ Height", minimum=64, maximum=2048, value=512, step=8, elem_id="RP_mask_height" + eladd)
                btn = gr.Button(value="Draw region + show mask")
                cbtn = gr.Button(value="Create mask area")
            with gr.Column():
                showmask = gr.Image(label="Mask", shape=(512, 512))
                uploadmask = gr.Image(label="Upload mask here", source="upload", type="numpy")
                
            # Привязка событий (кнопок к функциям)
            btn.click(fn=lambda *args: args, inputs=[polymask, num], outputs=[polymask, num, showmask]) # Заглушка для draw_region
            cbtn.click(fn=lambda w, h: gr.update(value=None), inputs=[canvas_height, canvas_width], outputs=[polymask]) # Заглушка для create_canvas
            uploadmask.upload(fn=lambda x: (x, None, x), inputs=[uploadmask], outputs=[polymask, uploadmask, showmask]) # Заглушка для draw_image

        vret = [xmode, polymask, num, canvas_width, canvas_height, showmask, uploadmask]
        
    elif mode == "Prompt":
        with gr.Row():
            pguide = gr.HTML(value = r"""<a href="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/prompt_en.md">Prompt mode guide</a>""")
        with gr.Row():
            pmode = gr.Radio(label="Prompt mode", choices=submode, value="Prompt", type="value", interactive=True, elem_id="RP_prompt_mode" + eladd)
            threshold = gr.Textbox(label="threshold", value=0.4, interactive=True, elem_id="RP_prompt_threshold" + eladd)
        vret = [pmode, threshold]

    return vret

def ui(self, is_img2img):

    with gr.Row():
        active_checkbox = gr.Checkbox(value=False, label="Active", interactive=True, elem_id="RP_active" + eladd)
        urlguide = gr.HTML(value=r"""<a href="https://github.com/hako-mikan/sd-webui-regional-prompter">Usage guide</a>""")
            
    with gr.Row():
        calcmode = gr.Radio(label="Generation Mode", choices=["Attention", "Latent"], value="Attention", type="value", interactive=True, elem_id="RP_generation_mode" + eladd)
            
    with gr.Row(visible=True):
        baseratios = gr.Textbox(label="Base Ratio", lines=1, value="0.2", interactive=True, elem_id="RP_base_ratio" + eladd, visible=True)
            
    with gr.Row():
        usebase = gr.Checkbox(value=False, label="Use base prompt", interactive=True, elem_id="RP_usebase" + eladd)
        usecom = gr.Checkbox(value=False, label="Use common prompt", interactive=True, elem_id="RP_usecommon" + eladd)
        usencom = gr.Checkbox(value=False, label="Use common negative prompt", interactive=True, elem_id="RP_usecommon_negative" + eladd)
            
    # 2. Вкладки режимов (Tabs)
    RPMODES = [
        ("Matrix", ("Columns", "Rows", "Random")),
        ("Mask", ("Mask",)),
        ("Prompt", ("Prompt", "Prompt-Ex")),
    ]
            
    with gr.Tabs(elem_id="RP_mode" + eladd) as tabs:
        rp_selected_tab = gr.State("Matrix") # Скрытое состояние для хранения текущей вкладки
        ltabp = []
        for (i, (md, smd)) in enumerate(RPMODES):
            with gr.TabItem(label=md, elem_id="RP_" + md) as tab:
                # Вызываем функцию из Части 2 для наполнения вкладки
                ltabp.append(ui_tab(md, smd, eladd))
            # При клике на вкладку обновляем скрытое состояние
            tab.select(fn=lambda tabnum=i: RPMODES[tabnum][0], inputs=[], outputs=[rp_selected_tab])

    # Распаковка возвращенных компонентов из вкладок для привязки событий
    (mmode, ratios, maketemp, template, areasimg, flipper, thei, twid, overlay) = ltabp[0]
    (xmode, polymask, num, canvas_width, canvas_height, showmask, uploadmask) = ltabp[1]
    (pmode, threshold) = ltabp[2]
            
    # 3. Секция пресетов
    with gr.Accordion("Presets", open=False):
        with gr.Row():
            availablepresets = gr.Dropdown(label="Presets", choices=["Preset 1", "Preset 2"], type="index")
            applypresets = gr.Button(value="Apply Presets", variant='primary', elem_id="RP_applysetting" + eladd)
        with gr.Row():
            presetname = gr.Textbox(label="Preset Name", lines=1, value="", interactive=True, elem_id="RP_preset_name" + eladd, visible=True)
            savesets = gr.Button(value="Save to Presets", variant='primary', elem_id="RP_savesetting" + eladd)
            
    # 4. Дополнительные настройки (LoRA, Options)
    with gr.Row():
        lstop = gr.Textbox(label="LoRA stop step", value="0", interactive=True, elem_id="RP_ne_tenc_ratio" + eladd, visible=True)
        lstop_hr = gr.Textbox(label="LoRA Hires stop step", value="0", interactive=True, elem_id="RP_ne_unet_ratio" + eladd, visible=True)
        lnter = gr.Textbox(label="LoRA in negative textencoder", value="0", interactive=True, elem_id="RP_ne_tenc_ratio_negative" + eladd, visible=True)
        lnur = gr.Textbox(label="LoRA in negative U-net", value="0", interactive=True, elem_id="RP_ne_unet_ratio_negative" + eladd, visible=True)
            
    with gr.Row():
        options = gr.CheckboxGroup(value=False, label="Options", choices=["Option A", "Option B"], interactive=True, elem_id="RP_options" + eladd)
        options_text = gr.Textbox(visible=False, value="")
            
    # Скрытые технические поля
    mode = gr.Textbox(value="Matrix", visible=False, elem_id="RP_divide_mode" + eladd)
    dummy_img = gr.Image(type="pil", show_label=False, height=256, width=256, source="upload", interactive=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)

    # 5. Привязка событий (Event Listeners)
    # Пример: при загрузке картинки обновляем dummy_img
    areasimg.upload(fn=lambda x: x, inputs=[areasimg], outputs=[dummy_img])
    areasimg.clear(fn=lambda: None, outputs=[dummy_img])

    # Пример: функция переключения вкладок программно
    def changetabs(mode_val):
        modes = ["Matrix", "Mask", "Prompt"]
        if mode_val not in modes: mode_val = "Matrix"
        return gr.Tabs.update(selected="t" + mode_val)
            
    mode.change(fn=changetabs, inputs=[mode], outputs=[tabs])
            
    # Пример: кнопки пресетов (заглушки функций для наглядности)
    maketemp.click(fn=lambda *args: args, inputs=[ratios, mmode, usecom, usebase, flipper, thei, twid, options, dummy_img, overlay], outputs=[areasimg, template])
    applypresets.click(fn=lambda *args: args, inputs=[availablepresets], outputs=[rp_selected_tab, mmode, xmode, pmode, ratios, baseratios, usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper])
    savesets.click(fn=lambda *args: ["Preset 1", "Preset 2"], inputs=[presetname], outputs=[availablepresets])
        
    # 6. Возврат списка всех компонентов, которые будут переданы в функцию process()
    return [active, dummy_false, rp_selected_tab, mmode, xmode, pmode, ratios, baseratios,
                usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper]