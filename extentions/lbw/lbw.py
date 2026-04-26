import gradio as gr
import os

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"] # Len: 61
BLOCKNUMS = [12,17,20,26, len(BLOCKIDFLUX)]
BLOCKIDS=[BLOCKID12,BLOCKID17,BLOCKID20,BLOCKID26,BLOCKIDFLUX]

BLOCKS=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders",
"transformer_resblocks"]

DEF_WEIGHT_PRESET = f"\
NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n\
INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\n\
OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\n\
OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\n\
OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n\
ALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5\n\
FLUXALL:{','.join(['1']*61)}"

ELEMPRESETS="\
ATTNDEEPON:IN05-OUT05:attn:1\n\n\
ATTNDEEPOFF:IN05-OUT05:attn:0\n\n\
PROJDEEPOFF:IN05-OUT05:proj:0\n\n\
XYZ:::1"

ATYPES =["none","Block ID","values","seed","Original Weights","elements"]

def checkloadcond(line: str) -> bool:
    """Возвращает True, если строку нужно пропустить."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if ":" not in stripped:
        return True
    # Прямая проверка: количество запятых должно быть на 1 меньше числа блоков
    return not any(stripped.count(",") == x - 1 for x in BLOCKNUMS)

def ui():
    LWEIGHTSPRESETS = DEF_WEIGHT_PRESET
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lbw_path = os.path.join(script_dir, "lbwpresets.txt")
    elem_path = os.path.join(script_dir, "elempresets.txt")

    def load_or_init_preset(file_path, default_content):
        if not os.path.isfile(file_path):
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(default_content)
            except Exception as e:
                print(f"⚠️ Не удалось создать {file_path}: {e}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Ошибка чтения {file_path}: {e}")
            return default_content

    lbwpresets = load_or_init_preset(lbw_path, LWEIGHTSPRESETS)
    elempresets = load_or_init_preset(elem_path, ELEMPRESETS)

    lratios = {}
    for line in lbwpresets.splitlines():
        if checkloadcond(line):
            continue
        key, value = line.split(":", 1)
        lratios[key.strip()] = value.strip()

    ratiostags = ",".join(lratios.keys())


    with gr.Row():
        with gr.Column(min_width = 50, scale=1):
            
            lbw_useblocks = gr.Checkbox(value=False, label="Active",interactive=True,elem_id="lbw_active")
            debug =  gr.Checkbox(value = False,label="Debug",interactive =True,elem_id="lbw_debug")
        with gr.Column(scale=5):
            bw_ratiotags= gr.TextArea(label="",value=ratiostags,visible =True,interactive =True,elem_id="lbw_ratios") 
    with gr.Accordion("XYZ plot",open = False):
        gr.HTML(value='<p style= "word-wrap:break-word;">changeable blocks : BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11</p>')
        xyzsetting = gr.Radio(label = "Active",choices = ["Disable","XYZ plot","Effective Block Analyzer"], value ="Disable",type = "index") 
        with gr.Row(visible = False) as esets:
            diffcol = gr.Radio(label = "diff image color",choices = ["black","white"], value ="black",type = "value",interactive =True) 
            revxy = gr.Checkbox(value = False,label="change X-Y",interactive =True,elem_id="lbw_changexy")
            thresh = gr.Textbox(label="difference threshold",lines=1,value="20",interactive =True,elem_id="diff_thr")
        xtype = gr.Dropdown(label="X Types", choices=[x for x in ATYPES], value=ATYPES [2],interactive =True,elem_id="lbw_xtype")
        xmen = gr.Textbox(label="X Values",lines=1,value="0,0.25,0.5,0.75,1",interactive =True,elem_id="lbw_xmen")
        ytype = gr.Dropdown(label="Y Types", choices=[y for y in ATYPES], value=ATYPES [1],interactive =True,elem_id="lbw_ytype")    
        ymen = gr.Textbox(label="Y Values" ,lines=1,value="IN05-OUT05",interactive =True,elem_id="lbw_ymen")
        ztype = gr.Dropdown(label="Z type", choices=[z for z in ATYPES], value=ATYPES[0],interactive =True,elem_id="lbw_ztype")    
        zmen = gr.Textbox(label="Z values",lines=1,value="",interactive =True,elem_id="lbw_zmen")

        exmen = gr.Textbox(label="Range",lines=1,value="0.5,1",interactive =True,elem_id="lbw_exmen",visible = False) 
        eymen = gr.Textbox(label="Blocks (12ALL,17ALL,20ALL,26ALL also can be used)" ,lines=1,value="BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11",interactive =True,elem_id="lbw_eymen",visible = False)  
        ecount = gr.Number(value=1, label="number of seed", interactive=True, visible = True)           

    with gr.Accordion("Weights setting",open = True):
        with gr.Row():
            reloadtext = gr.Button(value="Reload Presets",variant='primary',elem_id="lbw_reload")
            reloadtags = gr.Button(value="Reload Tags",variant='primary',elem_id="lbw_reload")
            savetext = gr.Button(value="Save Presets",variant='primary',elem_id="lbw_savetext")
            openeditor = gr.Button(value="Open TextEditor",variant='primary',elem_id="lbw_openeditor")
        lbw_loraratios = gr.TextArea(label="",value=lbwpresets,visible =True,interactive  = True,elem_id="lbw_ratiospreset")      
            
    with gr.Accordion("Elemental",open = False):  
        with gr.Row():
            e_reloadtext = gr.Button(value="Reload Presets",variant='primary',elem_id="lbw_reload")
            e_savetext = gr.Button(value="Save Presets",variant='primary',elem_id="lbw_savetext")
            e_openeditor = gr.Button(value="Open TextEditor",variant='primary',elem_id="lbw_openeditor")
        elemsets = gr.Checkbox(value = False,label="print change",interactive =True,elem_id="lbw_print_change")
        elemental = gr.TextArea(label="Identifer:BlockID:Elements:Ratio,...,separated by empty line ",value = elempresets,interactive =True,elem_id="element") 

        d_true = gr.Checkbox(value = True,visible = False)
        d_false = gr.Checkbox(value = False,visible = False)
            
    with gr.Accordion("Make Weights",open = False):  
        with gr.Row():
            m_text = gr.Textbox(value="",label="Weights")
        with gr.Row():
            m_add = gr.Button(value="Add to presets",size="sm",variant='primary')
            m_add_save = gr.Button(value="Add to presets and Save",size="sm",variant='primary')
            m_name = gr.Textbox(value="",label="Identifier")
        with gr.Row():
            m_type = gr.Radio(label="Weights type",choices=["17(1.X/2.X)", "26(1.X/2.X full)", "12(XL)","20(XL full)"], value="17(1.X/2.X)")
        with gr.Row():
            m_set_0 = gr.Button(value="Set All 0",variant='primary')
            m_set_1 = gr.Button(value="Set All 1",variant='primary')
            m_custom = gr.Button(value="Set custom",variant='primary')
            m_custom_v = gr.Slider(show_label=False, minimum=-1.0, maximum=1, step=0.1, value=0, interactive=True)
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                    gr.Slider(visible=False)
            with gr.Column(scale=2, min_width=200):
                base = gr.Slider(label="BASE", minimum=-1, maximum=1, step=0.1, value=0.0)
            with gr.Column(scale=1, min_width=100):
                gr.Slider(visible=False)
        with gr.Row():
            with gr.Column(scale=2, min_width=200):
                ins = [gr.Slider(label=block, minimum=-1.0, maximum=1, step=0.1, value=0, interactive=True) for block in BLOCKID26[1:13]]
            with gr.Column(scale=2, min_width=200):
                outs = [gr.Slider(label=block, minimum=-1.0, maximum=1, step=0.1, value=0, interactive=True) for block in reversed(BLOCKID26[14:])]
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                gr.Slider(visible=False)
            with gr.Column(scale=2, min_width=200):
                m00 = gr.Slider(label="M00", minimum=-1, maximum=1, step=0.1, value=0.0)
            with gr.Column(scale=1, min_width=100):
                gr.Slider(visible=False)

            blocks = [base] + ins + [m00] + outs[::-1]
            for block in blocks:
                if block.label not in BLOCKID17:
                    block.visible = False

        m_set_0.click(fn=lambda x:[0]*26 + [",".join(["0"]*int(x[:2]))],inputs=[m_type],outputs=blocks + [m_text])
        m_set_1.click(fn=lambda x:[1]*26 + [",".join(["1"]*int(x[:2]))],inputs=[m_type],outputs=blocks + [m_text])
        m_custom.click(fn=lambda x,y:[x]*26 + [",".join([str(x)]*int(y[:2]))],inputs=[m_custom_v,m_type],outputs=blocks + [m_text])

        def addweights(weights, id, presets, save = False):
            if id == "":id = "NONAME"
            lines = presets.strip().split("\n")
            id_found = False
            for i, line in enumerate(lines):
                if line.startswith("#"):
                    continue
                if line.split(":")[0] == id:
                    lines[i] = f"{id}:{weights}"
                    id_found = True
                    break
            if not id_found:
                lines.append(f"{id}:{weights}")

            if save:
                with open(extpath,mode = 'w',encoding="utf-8") as f:
                    f.write("\n".join(lines))

            return "\n".join(lines)

        def changetheblocks(sdver,*blocks):
            sdver = int(sdver[:2])
            output = []
            targ_blocks = BLOCKIDS[BLOCKNUMS.index(sdver)]
            for i, block in enumerate(BLOCKID26):
                if block in targ_blocks:
                    output.append(str(blocks[i]))
            return [",".join(output)] + [gr.update(visible = True if block in targ_blocks else False) for block in BLOCKID26]
                
        m_add.click(fn=addweights, inputs=[m_text,m_name,lbw_loraratios],outputs=[lbw_loraratios])
        m_add_save.click(fn=addweights, inputs=[m_text,m_name,lbw_loraratios, d_true],outputs=[lbw_loraratios])
        m_type.change(fn=changetheblocks, inputs=[m_type] + blocks,outputs=[m_text] + blocks)

        d_true = gr.Checkbox(value = True,visible = False)
        d_false = gr.Checkbox(value = False,visible = False)

    def makeweights(sdver, *blocks):
        sdver = int(sdver[:2])
        output = []
        targ_blocks = BLOCKIDS[BLOCKNUMS.index(sdver)]
        for i, block in enumerate(BLOCKID26):
            if block in targ_blocks:
                output.append(str(blocks[i]))
        return ",".join(output)

    changes = [b.release(fn=makeweights,inputs=[m_type] + blocks,outputs=[m_text]) for b in blocks]

    import subprocess
    def openeditors(b):
        path = extpath if b else extpathe
        subprocess.Popen(['start', path], shell=True)
                  
    def reloadpresets(isweight):
        if isweight:
            try:
                with open(extpath,encoding="utf-8") as f:
                    return f.read()
            except OSError as e:
                pass
        else:
            try:
                with open(extpathe,encoding="utf-8") as f:
                    return f.read()
            except OSError as e:
                pass

    def tagdicter(presets):
        presets=presets.splitlines()
        wdict={}
        for l in presets:
            if checkloadcond(l) : continue
            w=[]
            if ":" in l :
                key = l.split(":",1)[0]
                w = l.split(":",1)[1]
            if any(len([w for w in w.split(",")]) == x for x in BLOCKNUMS):
                wdict[key.strip()]=w
        return ",".join(list(wdict.keys()))

    def savepresets(text,isweight):
        if isweight:
            with open(extpath,mode = 'w',encoding="utf-8") as f:
                f.write(text)
        else:
            with open(extpathe,mode = 'w',encoding="utf-8") as f:
                f.write(text)

    reloadtext.click(fn=reloadpresets,inputs=[d_true],outputs=[lbw_loraratios])
    reloadtags.click(fn=tagdicter,inputs=[lbw_loraratios],outputs=[bw_ratiotags])
    savetext.click(fn=savepresets,inputs=[lbw_loraratios,d_true],outputs=[])
    openeditor.click(fn=openeditors,inputs=[d_true],outputs=[])

    e_reloadtext.click(fn=reloadpresets,inputs=[d_false],outputs=[elemental])
    e_savetext.click(fn=savepresets,inputs=[elemental,d_false],outputs=[])
    e_openeditor.click(fn=openeditors,inputs=[d_false],outputs=[])

    def urawaza(active):
        if active > 0:
            register()
            scripts.scripts_txt2img.run = newrun
            scripts.scripts_img2img.run = newrun
            if active == 1:return [*[gr.update(visible = True) for x in range(6)],*[gr.update(visible = False) for x in range(4)]]
            else:return [*[gr.update(visible = False) for x in range(6)],*[gr.update(visible = True) for x in range(4)]]
        else:
            scripts.scripts_txt2img.run = runorigin
            scripts.scripts_img2img.run = runorigini
            return [*[gr.update(visible = True) for x in range(6)],*[gr.update(visible = False) for x in range(4)]]

    xyzsetting.change(fn=urawaza,inputs=[xyzsetting],outputs =[xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,esets])

    return lbw_loraratios,lbw_useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug