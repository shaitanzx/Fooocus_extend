import gradio as gr
import os


def ui(self, is_img2img):
    filepath = os.path.join(PTPRESET, FLJSON)

    presets = []

    eladd = "i2i" if is_img2img else "t2i"

    presets = loadpresets(filepath)
    presets = LPRESET.update(presets)


    with gr.Row():
        if use_old_active:
            active = gr.Checkbox(value=False, label="Active",interactive=True,elem_id="RP_active" + eladd)
        urlguide = gr.HTML(value = fhurl(GUIDEURL, "Usage guide"))
    with gr.Row():
        #smode = gr.Radio(label="Divide mode", choices=["Horizontal", "Vertical","Mask","Prompt","Prompt-Ex"], value="Horizontal",  type="value", interactive=True)
        calcmode = gr.Radio(label="Generation Mode", choices=["Attention", "Latent"], value="Attention",  type="value", interactive=True, elem_id="RP_generation_mode" + eladd,)
    with gr.Row(visible=True):
        # ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
        baseratios = gr.Textbox(label="Base Ratio", lines=1,value="0.2",interactive=True,  elem_id="RP_base_ratio" + eladd, visible=True)
    with gr.Row():
        usebase = gr.Checkbox(value=False, label="Use base prompt",interactive=True, elem_id="RP_usebase" + eladd)
        usecom = gr.Checkbox(value=False, label="Use common prompt",interactive=True,elem_id="RP_usecommon" + eladd)
        usencom = gr.Checkbox(value=False, label="Use common negative prompt",interactive=True,elem_id="RP_usecommon_negative" + eladd)
            
    # Tabbed modes.
    with gr.Tabs(elem_id="RP_mode" + eladd) as tabs:
        rp_selected_tab = gr.State("Matrix") # State component to document current tab for gen.
        # ltabs = []
        ltabp = []
        for (i, (md,smd)) in enumerate(RPMODES):
            with gr.TabItem(**fgrprop(md)) as tab: # Tabs with a formatted id.
                # ltabs.append(tab)
                ltabp.append(ui_tab(md, smd, eladd))
            # Tab switch tags state component.
            tab.select(fn = lambda tabnum = i: RPMODES[tabnum][0], inputs=[], outputs=[rp_selected_tab])

    # Hardcode expansion back to components for any specific events.
    (mmode, ratios, maketemp, template, areasimg, flipper, thei, twid, overlay) = ltabp[0]
    (xmode, polymask, num, canvas_width, canvas_height, showmask, uploadmask) = ltabp[1]
    (pmode, threshold) = ltabp[2]
            
    with gr.Accordion("Presets",open = False):
        with gr.Row():
            availablepresets = gr.Dropdown(label="Presets", choices=presets, type="index")
            applypresets = gr.Button(value="Apply Presets",variant='primary',elem_id="RP_applysetting" + eladd)
        with gr.Row():
            presetname = gr.Textbox(label="Preset Name",lines=1,value="",interactive=True,elem_id="RP_preset_name" + eladd,visible=True)
            savesets = gr.Button(value="Save to Presets",variant='primary',elem_id="RP_savesetting" + eladd)
    with gr.Row():
        lstop = gr.Textbox(label="LoRA stop step",value="0",interactive=True,elem_id="RP_ne_tenc_ratio" + eladd,visible=True)
        lstop_hr = gr.Textbox(label="LoRA Hires stop step",value="0",interactive=True,elem_id="RP_ne_unet_ratio" + eladd,visible=True)
        lnter = gr.Textbox(label="LoRA in negative textencoder",value="0",interactive=True,elem_id="RP_ne_tenc_ratio_negative" + eladd,visible=True)
        lnur = gr.Textbox(label="LoRA in negative U-net",value="0",interactive=True,elem_id="RP_ne_unet_ratio_negative" + eladd,visible=True)
    with gr.Row():
        options = gr.CheckboxGroup(value=False, label="Options",choices=OPTIONLIST, interactive=True, elem_id="RP_options" + eladd)
        options_text = gr.Textbox(visible=False, value = "")
    mode = gr.Textbox(value = "Matrix",visible = False, elem_id="RP_divide_mode" + eladd)

    dummy_img = gr.Image(type="pil", show_label  = False, height=256, width=256,source = "upload", interactive=True, visible = False)

    dummy_false = gr.Checkbox(value=False, visible=False)

    areasimg.upload(fn=lambda x: x,inputs=[areasimg],outputs = [dummy_img])
    areasimg.clear(fn=lambda : None,outputs = [dummy_img])

    def changetabs(mode):
        modes = ["Matrix", "Mask", "Prompt"]
        if mode not in modes: mode = "Matrix"
        return gr.Tabs.update(selected="t"+mode)
            
    def options_dealer(options_text):
        if options_text == "No Options":
            return []
        else:
            return [y for y in options_text.split(",")]

    mode.change(fn = changetabs,inputs=[mode],outputs=[tabs])
    options_text.change(fn=options_dealer, inputs=[options_text], outputs=[options])
    settings = [rp_selected_tab, mmode, xmode, pmode, ratios, baseratios, usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper]
        
    infotext_fields = [
                (active, "RP Active"),
                # (mode, "RP Divide mode"),
                (mode, "RP Divide mode"),
                (mmode, "RP Matrix submode"),
                (xmode, "RP Mask submode"),
                (pmode, "RP Prompt submode"),
                (calcmode, "RP Calc Mode"),
                (ratios, "RP Ratios"),
                (baseratios, "RP Base Ratios"),
                (usebase, "RP Use Base"),
                (usecom, "RP Use Common"),
                (usencom, "RP Use Ncommon"),
                (options_text,"RP Options"),
                (lnter,"RP LoRA Neg Te Ratios"),
                (lnur,"RP LoRA Neg U Ratios"),
                (threshold,"RP threshold"),
                (lstop,"RP LoRA Stop Step"),
                (lstop_hr,"RP LoRA Hires Stop Step"),
                (flipper, "RP Flip")
    ]

    for _,name in self.infotext_fields:
        self.paste_field_names.append(name)

    def setpreset(select, *settings):
        """Load preset from list.
            
        SBM: The only way I know how to get the old values in gradio,
        is to pass them all as input.
        Tab mode converts ui to single value.
        """
        # Need to swap all masked images to the source,
        # getting "valueerror: cannot process this value as image".
        # Gradio bug in components.postprocess, most likely.
        settings = [s["image"] if (isinstance(s,dict) and "image" in s) else s for s in settings]
        presets = loadpresets(filepath)
        preset = presets[select]
        preset = loadblob(preset)
        preset = [fmt(preset.get(k, vdef)) for (k,fmt,vdef) in PRESET_KEYS]
        preset = preset[1:] # Remove name.
        preset = expand_components(preset)
        # Change nulls to original value.
        preset = [settings[i] if p is None else p for (i,p) in enumerate(preset)]
        while  len(settings) >= len(preset):
                preset.append(0)
        # return [gr.update(value = pr) for pr in preset] # SBM Why update? Shouldn't regular return do the job? 
        if preset[0] == "Vertical":preset[0] = "Rows"
        if preset[0] == "Horizontal":preset[0] = "Columns"
        return preset

    maketemp.click(fn=makeimgtmp, inputs =[ratios,mmode,usecom,usebase,flipper,thei,twid,options,dummy_img,overlay],outputs = [areasimg,template])
    applypresets.click(fn=setpreset, inputs = [availablepresets, *settings], outputs=settings)
    savesets.click(fn=savepresets, inputs = [presetname,*settings],outputs=availablepresets)
        
    return [active, dummy_false, rp_selected_tab, mmode, xmode, pmode, ratios, baseratios,
                usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper]