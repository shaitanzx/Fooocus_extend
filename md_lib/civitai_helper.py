import gradio as gr
from . import md_config

from . import util
from . import sections

js_msg_txtbox = gr.Textbox(
        label="Request Msg From Js",
        visible=False,
        lines=1,
        value="",
        elem_id="ch_js_msg_txtbox"
    )
def civitai_help():
# Used by some elements to pass messages to python
    # ====UI====

 def apply_settings(api_key):
        md_config.ch_civiai_api_key=api_key
        new_setting="Civitai API key="+str(md_config.ch_civiai_api_key)
        return setting_log.update(value=new_setting)   
### with shared.gradio_root:
 with gr.Box(elem_classes="ch_box"):
            with gr.Column(elem_classes="justify-bottom"):
                    gr.Markdown("### Settings")
                    dl_civitai_apikey = gr.Textbox(
                        label="Civitai API key",
                        lines=1,
                        max_lines=1,
                        value=md_config.ch_civiai_api_key,
                        placeholder="Civitai API key",
                        elem_id="ch_dl_url"
                    )
                    dl_apply = gr.Button(
                      value="Apply setting",
                      elem_classes="ch_vmargin",
                      variant="primary",
                      elem_id="ch_download_model_button"
                    )
                    setting_log = gr.Markdown(value="Civitai API key="+str(md_config.ch_civiai_api_key))
                    dl_apply.click(apply_settings,inputs=dl_civitai_apikey,outputs=setting_log)


 with gr.Box(elem_classes="ch_box"):
            sections.scan_models_section()

 with gr.Box(elem_classes="ch_box"):
            sections.get_model_info_by_url_section()

 with gr.Box(elem_classes="ch_box"):
            gr.Markdown("### Download Model")
            with gr.Tab("Single", elem_id="ch_dl_single_tab"):
                sections.download_section()
            with gr.Tab("Batch Download"):
                sections.download_multiple_section()

 with gr.Box(elem_classes="ch_box"):
            sections.scan_for_duplicates_section()

 with gr.Box(elem_classes="ch_box"):
            sections.check_new_versions_section(js_msg_txtbox)

        # ====Footer====
 gr.HTML('For apply emmbeding, in the prompt field use a record like (embedding:file_name:1.1)')
 gr.HTML('* \"Civitai Helper v1.8.10\" is powered by zixaphir. <a href="https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper" target="_blank">\U0001F4D4 Document</a>')
 gr.HTML('* Adaptation for Fooocus is powered by Shahmatist^RMDA')
        # ====hidden component for js, not in any tab====
 js_msg_txtbox.render()
