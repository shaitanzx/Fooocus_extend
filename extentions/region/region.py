import gradio as gr
import os
import json




# OPT_RP_DISABLE_IMAGE_EDITOR = "regional_prompter_disable_iamgeeditor"
# disable_image_editor = getattr(shared.opts,"regprp_" + OPT_RP_DISABLE_IMAGE_EDITOR, False)

# USE_OLD_ACTIVE = "old_active_check"
# use_old_active = getattr(shared.opts,"regprp_" + USE_OLD_ACTIVE, False)

# forge = launch_utils.git_tag()[0:2] == "f2" or launch_utils.git_tag() == "neo" 
# reforge = launch_utils.git_tag()[0:2] == "f1" or launch_utils.git_tag() == "classic"
# print(f"Forge: {forge}, reForge: {reforge}")

KEYBRK_R = "RP_TEMP_REPLACE"
FLJSON = "regional_prompter_presets.json"
OPTAND = "disable convert 'AND' to 'BREAK'"
OPTUSEL = "Use LoHa or other"
OPTBREAK = "Use BREAK to change chunks"
OPTFLIP = "Flip prompts"
OPTCOUT = "Comment Out `#`"
OPTAHIRES = "Enabled only in Hires Fix"
OPTDHIRES = "Disabled in Hires Fix"

OPTIONLIST = [OPTAND,OPTUSEL,OPTBREAK,OPTFLIP,OPTCOUT,OPTAHIRES,OPTDHIRES,"debug", "debug2"]

# Modules.basedir points to extension's dir. script_path or scripts.basedir points to root.
PTPRESET = os.path.dirname(__file__)
PTPRESETALT = os.path.dirname(__file__)

def lange(l):
    return range(len(l))

# orig_batch_cond_uncond = shared.opts.batch_cond_uncond if hasattr(shared.opts,"batch_cond_uncond") else shared.batch_cond_uncond

PRESETSDEF =[
    ["Vertical-3", "Vertical",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-3", "Horizontal",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-7", "Horizontal",'1,1,1,1,1,1,1',"0.2",True,False,False,"Attention",False,"0","0"],
    ["Twod-2-1", "Horizontal",'1,2,3;1,1',"0.2",False,False,False,"Attention",False,"0","0"],
]

# ATTNSCALE = 8 # Initial image compression in attention layers.

fhurl = lambda url, label: r"""<a href="{}">{}</a>""".format(url, label)
GUIDEURL = r"https://github.com/hako-mikan/sd-webui-regional-prompter"
MATRIXURL = GUIDEURL + r"#2d-region-assignment"
MASKURL = GUIDEURL + r"#mask-regions-aka-inpaint-experimental-function"
PROMPTURL = GUIDEURL + r"/blob/main/prompt_en.md"
PROMPTURL2 = GUIDEURL + r"/blob/main/prompt_ja.md"
def presetfallback():
    """Swaps main json dir to alt if exists, attempts reload.
    
    """
    global PTPRESET
    global PTPRESETALT
    
    if PTPRESETALT is not None:
        print("Unknown preset error, fallback.")
        PTPRESET = PTPRESETALT
        PTPRESETALT = None
        return loadpresets(PTPRESET)
    else: # Already attempted swap.
        print("Presets could not be loaded.") 
        return None
def initpresets(filepath):
    lpr = PRESETSDEF
    # if not os.path.isfile(filepath):
    try:
        with open(filepath, mode='w', encoding="utf-8") as f:
            lprj = []
            for pr in lpr:
                plen = min(len(PRESET_KEYS), len(pr)) # Future setting additions ignored.
                prj = {PRESET_KEYS[i][0]:pr[i] for i in range(plen)}
                lprj.append(prj)
            #json.dump(json.dumps(lprj), f, indent = 2)
            json.dump(lprj, f, indent = 2)
            return lprj
    except Exception as e:
        return presetfallback()
def loadpresets(filepath):
    presets = []
    try:
        with open(filepath, encoding="utf-8") as f:
            # presets = json.loads(json.load(f))
            presets = json.load(f)
            # presets = loadblob(presets) # DO NOT load all blobs - that's the point.
    except OSError as e:
        print("Init / preset error.")
        presets = initpresets(filepath)
    except TypeError:
        print("Corrupted preset file, resetting.")
        presets = initpresets(filepath)
    except JSONDecodeError:
        print("Preset file could not be decoded.")
        presets = initpresets(filepath)
    return presets
def ui_tab(mode, submode, eladd):
    """Structures components for mode tab.
    
    Semi harcoded but it's clearer this way.
    """
    vret = None
    if mode == "Matrix":
        with gr.Row():
            mguide = gr.HTML(value = fhurl(MATRIXURL, "Matrix mode guide")) 
        with gr.Row():
            mmode = gr.Radio(label="Main Splitting", choices=submode, value="Columns", type="value", interactive=True,elem_id="RP_main_splitting" + eladd)
            ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio" + eladd,visible=True)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    twid = gr.Slider(label="Width", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_matrix_width" + eladd)
                    thei = gr.Slider(label="Height", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_matrix_height" + eladd)
                maketemp = gr.Button(value="visualize and make template")

                template = gr.Textbox(label="template",interactive=True,visible=True,elem_id="RP_matrix_template" + eladd)
                flipper = gr.Checkbox(label = 'flip "," and ";"', value = False,elem_id="RP_matrix_flip" + eladd)
                overlay = gr.Slider(label="Overlay Ratio", minimum=0, maximum=1, step=0.1, value=0.5,elem_id="RP_matrix_overlay" + eladd)

            with gr.Column():
                areasimg = gr.Image(type="pil", show_label  = False, height=256, width=256,source = "upload", interactive=True)
        # Need to add maketemp function based on base / common checks.
        vret = [mmode, ratios, maketemp, template, areasimg, flipper, thei, twid, overlay]
    elif mode == "Mask":
        with gr.Row():
            xguide = gr.HTML(value = fhurl(MASKURL, "Inpaint+ mode guide"))
        with gr.Row(): # Creep: Placeholder, should probably make this invisible.
            xmode = gr.Radio(label="Mask mode", choices=submode, value="Mask", type="value", interactive=True,elem_id="RP_mask_mode" + eladd)
        with gr.Row(): # CREEP: Css magic to make the canvas bigger? I think it's in style.css: #img2maskimg -> height.
            if IS_GRADIO_4:
                if disable_image_editor:
                    polymask = gr.Image(label = "Image",elem_id="polymask" + eladd,
                                source = "upload", mirror_webcam = False, type = "numpy")#.style(height=480)
                else:
                    polymask = gr.ImageEditor(elem_id="polymask" + eladd, image_mode = "RGB",canvas_size = (512,512),
                        source = "upload", mirror_webcam = False, type = "numpy", tool = "sketch",
                        brush = gr.Brush(colors=["#804040"], color_mode='fixed'))#.style(height=480)
            else:
                polymask = gr.Image(label = "Do not upload here until bugfix",elem_id="polymask" + eladd,
                                source = "upload", mirror_webcam = False, type = "numpy", tool = "sketch")#.style(height=480)
        with gr.Row():
           
            with gr.Column():
                num = gr.Slider(label="Region", minimum=-1, maximum=MAXCOLREG, step=1, value=1,elem_id="RP_mask_region" + eladd)
                canvas_width = gr.Slider(label="Inpaint+ Width", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_mask_width" + eladd)
                canvas_height = gr.Slider(label="Inpaint+ Height", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_mask_height" + eladd)
                btn = gr.Button(value = "Draw region + show mask")
                # btn2 = gr.Button(value = "Display mask") # Not needed.
                cbtn = gr.Button(value="Create mask area")
            with gr.Column():
                showmask = gr.Image(label = "Mask", shape=(IDIM, IDIM))
                # CONT: Awaiting fix for https://github.com/gradio-app/gradio/issues/4088.
                uploadmask = gr.Image(label="Upload mask here cus gradio",source = "upload", type = "numpy")
            # btn.click(detect_polygons, inputs = [polymask,num], outputs = [polymask,num])
            btn.click(draw_region, inputs = [polymask, num], outputs = [polymask, num, showmask])
            # btn2.click(detect_mask, inputs = [polymask,num], outputs = [showmask])
            cbtn.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[polymask])   
            uploadmask.upload(fn = draw_image, inputs = [uploadmask], outputs = [polymask, uploadmask, showmask])


            vret = [xmode, polymask, num, canvas_width, canvas_height, showmask, uploadmask]
        
    elif mode == "Prompt":
        with gr.Row():
            pguide = gr.HTML(value = fhurl(PROMPTURL, "Prompt mode guide"))
            pguide2 = gr.HTML(value = fhurl(PROMPTURL2, "Extended prompt guide (jp)"))
        with gr.Row():
            pmode = gr.Radio(label="Prompt mode", choices=submode, value="Prompt", type="value", interactive=True, elem_id="RP_prompt_mode" + eladd)
            threshold = gr.Textbox(label = "threshold", value = 0.4, interactive=True, elem_id="RP_prompt_threshold" + eladd)
        
        vret = [pmode, threshold]

    return vret
def gui():
    filepath = os.path.join(PTPRESET, FLJSON)

    presets = []

    eladd = "t2i"

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