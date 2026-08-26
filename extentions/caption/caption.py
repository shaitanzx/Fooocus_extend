import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from tqdm import tqdm

from .utils.download import download_models
from .utils.image import get_image_paths
from .utils.inference import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_WITHOUT_WD, DEFAULT_USER_PROMPT_WITH_WD
from .utils.inference import get_caption_file_path, LLM, Tagger
from .utils.logger import Logger, print_title

DEFAULT_MODELS_SAVE_PATH = os.path.join("models", "caption")


@dataclass
class CaptionConfig:
    # Base
    data_path: str = ""
    recursive: bool = False
    
    # Logs
    log_level: str = "INFO"
    save_logs: bool = False
    
    # Download
    model_site: str = "huggingface"
    models_save_path: str = DEFAULT_MODELS_SAVE_PATH
    use_sdk_cache: bool = False
    download_method: str = "SDK"
    force_download: bool = False
    skip_download: bool = False
    
    # Caption
    caption_method: str = "wd+llm"
    run_method: str = "sync"
    caption_extension: str = ".txt"
    save_caption_together: bool = False
    save_caption_together_seperator: str = "|"
    image_size: int = 1024
    skip_exists: bool = False
    not_overwrite: bool = False
    custom_caption_save_path: Optional[str] = None
    
    # WD Caption
    wd_config: Optional[str] = None
    wd_model_name: Optional[str] = None
    wd_force_use_cpu: bool = False
    wd_caption_extension: str = ".wdcaption"
    wd_remove_underscore: bool = False
    wd_undesired_tags: str = ""
    wd_tags_frequency: bool = False
    wd_threshold: float = 0.35
    wd_general_threshold: Optional[float] = None
    wd_character_threshold: Optional[float] = None
    wd_add_rating_tags_to_first: bool = False
    wd_add_rating_tags_to_last: bool = False
    wd_character_tags_first: bool = False
    wd_always_first_tags: Optional[str] = None
    wd_caption_separator: str = ", "
    wd_tag_replacement: Optional[str] = None
    wd_character_tag_expand: bool = False
    
    # LLM Caption
    llm_choice: str = "llama"
    llm_config: Optional[str] = None
    llm_model_name: Optional[str] = None
    llm_patch: bool = False
    llm_use_cpu: bool = False
    llm_dtype: str = "fp16"
    llm_qnt: str = "none"
    llm_caption_extension: str = ".llmcaption"
    llm_read_wd_caption: bool = False
    llm_caption_without_wd: bool = False
    llm_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    llm_user_prompt: str = DEFAULT_USER_PROMPT_WITHOUT_WD
    llm_temperature: float = 0.0
    llm_max_tokens: int = 0


def get_default_config() -> CaptionConfig:
    return CaptionConfig()


class Caption:
    def __init__(self):
        self.use_wd = False
        self.use_joy = False
        self.use_llama = False
        self.use_qwen = False
        self.use_minicpm = False
        self.use_florence = False

        self.my_logger = None
        self.wd_model_path = None
        self.wd_tags_csv_path = None
        self.llm_models_paths = None
        self.my_tagger = None
        self.my_llm = None

    def check_path(self, config: CaptionConfig):
        if not config.data_path:
            print("`data_path` not defined, please set it in config!")
            raise ValueError
        if not os.path.exists(config.data_path):
            print(f"`{config.data_path}` not exists!!!")
            raise FileNotFoundError

    def set_logger(self, config: CaptionConfig):
        if config.save_logs:
            workspace_path = os.getcwd()
            data_dir_path = Path(config.data_path)
            log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

            if config.custom_caption_save_path:
                log_file_path = Path(config.custom_caption_save_path)

            log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = os.path.basename(data_dir_path) if os.path.exists(data_dir_path) else ""
            
            if not log_name:
                print(f'{data_dir_path} NOT FOUND!!!')
                raise FileNotFoundError

            log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
            log_file = os.path.join(log_file_path, log_file) if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
        else:
            log_file = None

        if str(config.log_level).lower() in 'debug, info, warning, error, critical':
            self.my_logger = Logger(config.log_level, log_file).logger
            self.my_logger.info(f'Set log level to "{config.log_level}"')
        else:
            self.my_logger = Logger('INFO', log_file).logger
            self.my_logger.warning('Invalid log level, set log level to "INFO"!')

        if config.save_logs:
            self.my_logger.info(f'Log file will be saved as "{log_file}".')

    def download_models(self, config: CaptionConfig):
        self.use_wd = config.caption_method in ["wd", "wd+llm"]
        self.use_joy = config.caption_method in ["llm", "wd+llm"] and config.llm_choice == "joy"
        self.use_llama = config.caption_method in ["llm", "wd+llm"] and config.llm_choice == "llama"
        self.use_qwen = config.caption_method in ["llm", "wd+llm"] and config.llm_choice == "qwen"
        self.use_minicpm = config.caption_method in ["llm", "wd+llm"] and config.llm_choice == "minicpm"
        self.use_florence = config.caption_method in ["llm", "wd+llm"] and config.llm_choice == "florence"
        
        models_save_path = Path(config.models_save_path) if os.path.exists(Path(config.models_save_path)) else Path(DEFAULT_MODELS_SAVE_PATH)
        if models_save_path == Path(DEFAULT_MODELS_SAVE_PATH):
            self.my_logger.warning(f"Models save path not defined or not exists, will download models into `{DEFAULT_MODELS_SAVE_PATH}`...")

        if self.use_wd:
            config_file = Path(config.wd_config) if config.wd_config else os.path.join(Path(__file__).parent, 'configs', 'default_wd.json')
            self.wd_model_path, self.wd_tags_csv_path = download_models(
                logger=self.my_logger, models_type="wd", args=config, config_file=config_file, models_save_path=models_save_path
            )

        llm_type_map = {
            "joy": ("default_joy.json", "joy"),
            "llama": ("default_llama_3.2V.json", "llama"),
            "qwen": ("default_qwen2_vl.json", "qwen"),
            "minicpm": ("default_minicpm.json", "minicpm"),
            "florence": ("default_florence.json", "florence"),
        }
        
        if config.llm_choice in llm_type_map and getattr(self, f"use_{config.llm_choice}"):
            default_cfg, model_type = llm_type_map[config.llm_choice]
            config_file = Path(config.llm_config) if config.llm_config else os.path.join(Path(__file__).parent, 'configs', default_cfg)
            self.llm_models_paths = download_models(
                logger=self.my_logger, models_type=model_type, args=config, config_file=config_file, models_save_path=models_save_path
            )

    def load_models(self, config: CaptionConfig):
        if self.use_wd:
            self.my_tagger = Tagger(logger=self.my_logger, args=config, model_path=self.wd_model_path, tags_csv_path=self.wd_tags_csv_path)
            self.my_tagger.load_model()

        if self.use_joy or self.use_llama or self.use_qwen or self.use_minicpm or self.use_florence:
            self.my_llm = LLM(logger=self.my_logger, models_type=config.llm_choice, models_paths=self.llm_models_paths, args=config)
            self.my_llm.load_model()

    def run_inference(self, config: CaptionConfig):
        start_inference_time = time.monotonic()
        
        if self.use_wd and config.caption_method == "wd+llm":
            if config.llm_user_prompt == DEFAULT_USER_PROMPT_WITHOUT_WD and not config.llm_caption_without_wd:
                self.my_logger.warning("LLM user prompt not defined, using default version with wd tags...")
                config.llm_user_prompt = DEFAULT_USER_PROMPT_WITH_WD
            
            if config.run_method == "sync":
                self.my_logger.info("Running in sync mode...")
                image_paths = get_image_paths(logger=self.my_logger, path=Path(config.data_path), recursive=config.recursive)
                pbar = tqdm(total=len(image_paths), smoothing=0.0)
                
                for image_path in image_paths:
                    try:
                        pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else image_path[:15]) + ' ... ' + image_path[-20:])
                        
                        wd_caption_file = get_caption_file_path(self.my_logger, data_path=config.data_path, image_path=Path(image_path), custom_caption_save_path=config.custom_caption_save_path, caption_extension=config.wd_caption_extension)
                        llm_caption_file = get_caption_file_path(self.my_logger, data_path=config.data_path, image_path=Path(image_path), custom_caption_save_path=config.custom_caption_save_path, caption_extension=config.llm_caption_extension if config.save_caption_together else config.caption_extension)
                        
                        image = Image.open(image_path)
                        tag_text, caption = "", ""

                        if not (config.skip_exists and os.path.isfile(wd_caption_file)):
                            tag_text, rating_tag_text, character_tag_text, general_tag_text = self.my_tagger.get_tags(image=image)
                            if not (config.not_overwrite and os.path.isfile(wd_caption_file)):
                                with open(wd_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(tag_text + "\n")
                            else:
                                self.my_logger.warning(f'`not_overwrite` ENABLED!!! WD Caption file {wd_caption_file} already exist, Skip save caption.')

                            self.my_logger.debug(f"Image path: {image_path}\nWD Caption path: {wd_caption_file}")
                            if config.wd_model_name and config.wd_model_name.lower().startswith("wd"):
                                self.my_logger.debug(f"WD Rating tags: {rating_tag_text}\nWD Character tags: {character_tag_text}")
                            self.my_logger.debug(f"WD General tags: {general_tag_text}")
                        else:
                            self.my_logger.warning(f'`skip_exists` ENABLED!!! WD Caption file {wd_caption_file} already exists, Skip save it!')

                        if not (config.skip_exists and os.path.isfile(llm_caption_file)):
                            caption = self.my_llm.get_caption(
                                image=image,
                                system_prompt=str(config.llm_system_prompt),
                                user_prompt=str(config.llm_user_prompt).format(wd_tags=tag_text),
                                temperature=config.llm_temperature,
                                max_new_tokens=config.llm_max_tokens
                            )
                            if not (config.not_overwrite and os.path.isfile(llm_caption_file)):
                                with open(llm_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(caption + "\n")
                                self.my_logger.debug(f"Image path: {image_path}\nLLM Caption path: {llm_caption_file}\nLLM Caption content: {caption}")
                            else:
                                self.my_logger.warning(f'`not_overwrite` ENABLED!!! LLM Caption file {llm_caption_file} already exist, skip save it!')
                        else:
                            self.my_logger.warning(f'`skip_exists` ENABLED!!! LLM Caption file {llm_caption_file} already exists, skip save it!')

                        if config.save_caption_together:
                            together_caption_file = get_caption_file_path(self.my_logger, data_path=config.data_path, image_path=Path(image_path), custom_caption_save_path=config.custom_caption_save_path, caption_extension=config.caption_extension)
                            self.my_logger.debug(f"`save_caption_together` Enabled, will save WD tags and LLM captions in a new file `{together_caption_file}`")
                            
                            if not (config.skip_exists and os.path.isfile(together_caption_file)):
                                if not tag_text or not caption:
                                    self.my_logger.warning("WD tags or LLM Caption is null, skip save them together in one file!")
                                    pbar.update(1)
                                    continue

                                if not (config.not_overwrite and os.path.isfile(together_caption_file)):
                                    with open(together_caption_file, "wt", encoding="utf-8") as f:
                                        together_caption = f"{tag_text} {config.save_caption_together_seperator} {caption}"
                                        f.write(together_caption + "\n")
                                    self.my_logger.debug(f"Together Caption save path: {together_caption_file}\nTogether Caption content: {together_caption}")
                                else:
                                    self.my_logger.warning(f'`not_overwrite` ENABLED!!! Together Caption file {together_caption_file} already exist, skip save it!')
                            else:
                                self.my_logger.warning(f'`skip_exists` ENABLED!!! LLM Caption file {llm_caption_file} already exists, skip save it!')

                    except Exception as e:
                        self.my_logger.error(f"Failed to caption image: {image_path}, skip it.\nerror info: {e}")
                    
                    pbar.update(1)
                pbar.close()

                if config.wd_tags_frequency:
                    sorted_tags = sorted(self.my_tagger.tag_freq.items(), key=lambda x: x[1], reverse=True)
                    self.my_logger.info('WD Tag frequencies:')
                    for tag, freq in sorted_tags:
                        self.my_logger.info(f'{tag}: {freq}')
            else:
                self.my_logger.info("Running in queue mode...")
                pbar = tqdm(total=2, smoothing=0.0)
                pbar.set_description('Processing with WD model...')
                self.my_tagger.inference()
                pbar.update(1)
                pbar.set_description(f'Processing with {config.llm_choice.capitalize()} model...')
                self.my_llm.inference()
                pbar.update(1)
                pbar.close()
        else:
            if self.use_wd:
                self.my_tagger.inference()
            elif self.use_joy or self.use_llama or self.use_qwen or self.use_minicpm or self.use_florence:
                self.my_llm.inference()

        total_inference_time = time.monotonic() - start_inference_time
        days = int(total_inference_time // (24 * 3600))
        hours = int((total_inference_time % (24 * 3600)) // 3600)
        minutes = int((total_inference_time % 3600) // 60)
        seconds = total_inference_time % 60
        
        time_str = f"{days} Day(s) " if days > 0 else ""
        time_str += f"{hours} Hour(s) " if hours > 0 or (days and hours == 0) else ""
        time_str += f"{minutes} Min(s) " if minutes > 0 or (hours and minutes == 0) else ""
        time_str += f"{seconds:.2f} Sec(s)"
        self.my_logger.info(f"All work done with in {time_str}.")

    def unload_models(self):
        if self.use_wd and self.my_tagger:
            self.my_tagger.unload_model()
        if (self.use_joy or self.use_llama or self.use_qwen or self.use_minicpm or self.use_florence) and self.my_llm:
            self.my_llm.unload_model()


def main():
    config = get_default_config()
    my_caption = Caption()
    # my_caption.check_path(config) # Раскомментируйте, если нужно проверять путь при запуске из CLI
    my_caption.set_logger(config)
    my_caption.download_models(config)
    my_caption.load_models(config)
    my_caption.run_inference(config)
    my_caption.unload_models()


if __name__ == "__main__":
    main()