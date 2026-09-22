import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .utils.download import download_models
from .utils.image import get_image_paths
from .utils.inference import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_WITHOUT_WD, DEFAULT_USER_PROMPT_WITH_WD
from .utils.inference import get_caption_file_path, LLM, Tagger
from .utils.logger import Logger, print_title

DEFAULT_MODELS_SAVE_PATH = str(os.path.join(os.getcwd(), "models"))


class Caption:
    def __init__(self):
        # Set flags
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

    def check_path(
            self,
            args: argparse.Namespace
    ):
        if not args.data_path:
            print(f"`data_path` not defined, use `--data_path` add your datasets path!!!")
            raise ValueError
        if not os.path.exists(args.data_path):
            print(f"`{args.data_path}` not exists!!!")
            raise FileNotFoundError

    def set_logger(
            self,
            args: argparse.Namespace
    ):
        # Set logger
        if args.save_logs:
            workspace_path = os.getcwd()
            data_dir_path = Path(args.data_path)

            log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

            if args.custom_caption_save_path:
                log_file_path = Path(args.custom_caption_save_path)

            log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            # caption_failed_list_file = f'Caption_failed_list_{log_time}.txt'

            if os.path.exists(data_dir_path):
                log_name = os.path.basename(data_dir_path)

            else:
                print(f'{data_dir_path} NOT FOUND!!!')
                raise FileNotFoundError

            log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
            log_file = os.path.join(log_file_path, log_file) \
                if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
        else:
            log_file = None

        if str(args.log_level).lower() in 'debug, info, warning, error, critical':
            self.my_logger = Logger(args.log_level, log_file).logger
            self.my_logger.info(f'Set log level to "{args.log_level}"')

        else:
            self.my_logger = Logger('INFO', log_file).logger
            self.my_logger.warning('Invalid log level, set log level to "INFO"!')

        if args.save_logs:
            self.my_logger.info(f'Log file will be saved as "{log_file}".')

    def download_models(
            self,
            args: argparse.Namespace
    ):
        # Set flags
        self.use_wd = True if args.caption_method in ["wd", "wd+llm"] else False
        self.use_joy = True if args.caption_method in ["llm", "wd+llm"] and args.llm_choice == "joy" else False
        self.use_llama = True if args.caption_method in ["llm", "wd+llm"] and args.llm_choice == "llama" else False
        self.use_qwen = True if args.caption_method in ["llm", "wd+llm"] and args.llm_choice == "qwen" else False
        self.use_minicpm = True if args.caption_method in ["llm", "wd+llm"] and args.llm_choice == "minicpm" else False
        self.use_florence = True if args.caption_method in ["llm", "wd+llm"] and \
                                    args.llm_choice == "florence" else False
        # Set models save path
        if os.path.exists(Path(args.models_save_path)):
            models_save_path = Path(args.models_save_path)
        else:
            self.my_logger.warning(
                f"Models save path not defined or not exists, will download models into `{DEFAULT_MODELS_SAVE_PATH}`...")
            models_save_path = Path(DEFAULT_MODELS_SAVE_PATH)

        if self.use_wd:
            # Check wd models path from json
            if not args.wd_config:
                wd_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_wd.json')
            else:
                wd_config_file = Path(args.wd_config)
            # Download wd models
            self.wd_model_path, self.wd_tags_csv_path = download_models(
                logger=self.my_logger,
                models_type="wd",
                args=args,
                config_file=wd_config_file,
                models_save_path=models_save_path,
            )

        if self.use_joy:
            # Check joy models path from json
            if not args.llm_config:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_joy.json')
            else:
                llm_config_file = Path(args.llm_config)
            # Download joy models
            self.llm_models_paths = download_models(
                logger=self.my_logger,
                models_type="joy",
                args=args,
                config_file=llm_config_file,
                models_save_path=models_save_path,
            )

        elif self.use_llama:
            # Check joy models path from json
            if not args.llm_config:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_llama_3.2V.json')
            else:
                llm_config_file = Path(args.llm_config)
            # Download Llama models
            self.llm_models_paths = download_models(
                logger=self.my_logger,
                models_type="llama",
                args=args,
                config_file=llm_config_file,
                models_save_path=models_save_path,
            )
        elif self.use_qwen:
            if not args.llm_config:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_qwen2_vl.json')
            else:
                llm_config_file = Path(args.llm_config)
            # Download Qwen models
            self.llm_models_paths = download_models(
                logger=self.my_logger,
                models_type="qwen",
                args=args,
                config_file=llm_config_file,
                models_save_path=models_save_path,
            )
        elif self.use_minicpm:
            if not args.llm_config:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_minicpm.json')
            else:
                llm_config_file = Path(args.llm_config)
            # Download Qwen models
            self.llm_models_paths = download_models(
                logger=self.my_logger,
                models_type="minicpm",
                args=args,
                config_file=llm_config_file,
                models_save_path=models_save_path,
            )
        elif self.use_florence:
            if not args.llm_config:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_florence.json')
            else:
                llm_config_file = Path(args.llm_config)
            # Download Qwen models
            self.llm_models_paths = download_models(
                logger=self.my_logger,
                models_type="florence",
                args=args,
                config_file=llm_config_file,
                models_save_path=models_save_path,
            )

    def load_models(
            self,
            args: argparse.Namespace
    ):
        if self.use_wd:
            # Load wd models
            self.my_tagger = Tagger(
                logger=self.my_logger,
                args=args,
                model_path=self.wd_model_path,
                tags_csv_path=self.wd_tags_csv_path
            )
            self.my_tagger.load_model()

        if self.use_joy:
            # Load Joy models
            self.my_llm = LLM(
                logger=self.my_logger,
                models_type="joy",
                models_paths=self.llm_models_paths,
                args=args,
            )
            self.my_llm.load_model()
        elif self.use_llama:
            # Load Llama models
            self.my_llm = LLM(
                logger=self.my_logger,
                models_type="llama",
                models_paths=self.llm_models_paths,
                args=args,
            )
            self.my_llm.load_model()
        elif self.use_qwen:
            # Load Qwen models
            self.my_llm = LLM(
                logger=self.my_logger,
                models_type="qwen",
                models_paths=self.llm_models_paths,
                args=args,
            )
            self.my_llm.load_model()
        elif self.use_minicpm:
            # Load Qwen models
            self.my_llm = LLM(
                logger=self.my_logger,
                models_type="minicpm",
                models_paths=self.llm_models_paths,
                args=args,
            )
            self.my_llm.load_model()
        elif self.use_florence:
            # Load Florence models
            self.my_llm = LLM(
                logger=self.my_logger,
                models_type="florence",
                models_paths=self.llm_models_paths,
                args=args,
            )
            self.my_llm.load_model()

    def run_inference(
            self,
            args: argparse.Namespace
    ):
        start_inference_time = time.monotonic()
        # Inference
        if self.use_wd and args.caption_method == "wd+llm":
            # Set joy user prompt
            if args.llm_user_prompt == DEFAULT_USER_PROMPT_WITHOUT_WD:
                if not args.llm_caption_without_wd:
                    self.my_logger.warning(f"LLM user prompt not defined, using default version with wd tags...")
                    args.llm_user_prompt = DEFAULT_USER_PROMPT_WITH_WD
            # run
            if args.run_method == "sync":
                self.my_logger.info(f"Running in sync mode...")
                image_paths = get_image_paths(logger=self.my_logger, path=Path(args.data_path),
                                              recursive=args.recursive)
                pbar = tqdm(total=len(image_paths), smoothing=0.0)
                for image_path in image_paths:
                    try:
                        pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                                     image_path[:15]) + ' ... ' + image_path[-20:])
                        # Caption file
                        wd_caption_file = get_caption_file_path(
                            self.my_logger,
                            data_path=args.data_path,
                            image_path=Path(image_path),
                            custom_caption_save_path=args.custom_caption_save_path,
                            caption_extension=args.wd_caption_extension
                        )
                        llm_caption_file = get_caption_file_path(
                            self.my_logger,
                            data_path=args.data_path,
                            image_path=Path(image_path),
                            custom_caption_save_path=args.custom_caption_save_path,
                            caption_extension=args.llm_caption_extension if args.save_caption_together else
                            args.caption_extension
                        )
                        # image to pillow
                        image = Image.open(image_path)
                        tag_text = ""
                        caption = ""

                        if not (args.skip_exists and os.path.isfile(wd_caption_file)):
                            # WD Caption
                            tag_text, rating_tag_text, character_tag_text, general_tag_text = self.my_tagger.get_tags(
                                image=image
                            )

                            if not (args.not_overwrite and os.path.isfile(wd_caption_file)):
                                # Write WD Caption file
                                with open(wd_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(tag_text + "\n")
                            else:
                                self.my_logger.warning(f'`not_overwrite` ENABLED!!! '
                                                       f'WD Caption file {wd_caption_file} already exist, '
                                                       f'Skip save caption.')

                            # Console output
                            self.my_logger.debug(f"Image path: {image_path}")
                            self.my_logger.debug(f"WD Caption path: {wd_caption_file}")
                            if args.wd_model_name.lower().startswith("wd"):
                                self.my_logger.debug(f"WD Rating tags: {rating_tag_text}")
                                self.my_logger.debug(f"WD Character tags: {character_tag_text}")
                            self.my_logger.debug(f"WD General tags: {general_tag_text}")
                        else:
                            self.my_logger.warning(f'`skip_exists` ENABLED!!! '
                                                   f'WD Caption file {wd_caption_file} already exists, '
                                                   f'Skip save it!')

                        if not (args.skip_exists and os.path.isfile(llm_caption_file)):
                            # LLM Caption
                            caption = self.my_llm.get_caption(
                                image=image,
                                system_prompt=str(args.llm_system_prompt),
                                user_prompt=str(args.llm_user_prompt).format(wd_tags=tag_text),
                                temperature=args.llm_temperature,
                                max_new_tokens=args.llm_max_tokens
                            )
                            if not (args.not_overwrite and os.path.isfile(llm_caption_file)):
                                # Write LLM Caption
                                with open(llm_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(caption + "\n")
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"LLM Caption path: {llm_caption_file}")
                                    self.my_logger.debug(f"LLM Caption content: {caption}")
                            else:
                                self.my_logger.warning(f'`not_overwrite` ENABLED!!! '
                                                       f'LLM Caption file {llm_caption_file} already exist, '
                                                       f'skip save it!')
                        else:
                            self.my_logger.warning(f'`skip_exists` ENABLED!!! '
                                                   f'LLM Caption file {llm_caption_file} already exists, '
                                                   f'skip save it!')

                        if args.save_caption_together:
                            together_caption_file = get_caption_file_path(
                                self.my_logger,
                                data_path=args.data_path,
                                image_path=Path(image_path),
                                custom_caption_save_path=args.custom_caption_save_path,
                                caption_extension=args.caption_extension
                            )
                            self.my_logger.debug(
                                f"`save_caption_together` Enabled, "
                                f"will save WD tags and LLM captions in a new file `{together_caption_file}`")
                            if not (args.skip_exists and os.path.isfile(together_caption_file)):
                                if not tag_text or not caption:
                                    self.my_logger.warning(
                                        "WD tags or LLM Caption is null, skip save them together in one file!")
                                    pbar.update(1)
                                    continue

                                if not (args.not_overwrite and os.path.isfile(together_caption_file)):
                                    with open(together_caption_file, "wt", encoding="utf-8") as f:
                                        together_caption = f"{tag_text} {args.save_caption_together_seperator} {caption}"
                                        f.write(together_caption + "\n")
                                    self.my_logger.debug(f"Together Caption save path: {together_caption_file}")
                                    self.my_logger.debug(f"Together Caption content: {together_caption}")
                                else:
                                    self.my_logger.warning(f'`not_overwrite` ENABLED!!! '
                                                           f'Together Caption file {together_caption_file} already exist, '
                                                           f'skip save it!')
                            else:
                                self.my_logger.warning(f'`skip_exists` ENABLED!!! '
                                                       f'LLM Caption file {llm_caption_file} already exists, '
                                                       f'skip save it!')

                    except Exception as e:
                        self.my_logger.error(f"Failed to caption image: {image_path}, skip it.\nerror info: {e}")
                        pbar.update(1)
                        continue

                    pbar.update(1)
                pbar.close()

                if args.wd_tags_frequency:
                    sorted_tags = sorted(self.my_tagger.tag_freq.items(), key=lambda x: x[1], reverse=True)
                    self.my_logger.info('WD Tag frequencies:')
                    for tag, freq in sorted_tags:
                        self.my_logger.info(f'{tag}: {freq}')
            else:
                self.my_logger.info(f"Running in queue mode...")
                pbar = tqdm(total=2, smoothing=0.0)
                pbar.set_description('Processing with WD model...')
                self.my_tagger.inference()
                pbar.update(1)
                if self.use_joy:
                    pbar.set_description('Processing with Joy model...')
                elif self.use_llama:
                    pbar.set_description('Processing with Llama model...')
                elif self.use_qwen:
                    pbar.set_description('Processing with Qwen model...')
                elif self.use_minicpm:
                    pbar.set_description('Processing with Mini-CPM model...')
                elif self.use_florence:
                    pbar.set_description('Processing with Florence model...')
                self.my_llm.inference()
                pbar.update(1)

                pbar.close()
        else:
            if self.use_wd:
                self.my_tagger.inference()
            elif self.use_joy or self.use_llama or self.use_qwen or self.use_minicpm or self.use_florence:
                self.my_llm.inference()

        total_inference_time = time.monotonic() - start_inference_time
        days = total_inference_time // (24 * 3600)
        total_inference_time %= (24 * 3600)
        hours = total_inference_time // 3600
        total_inference_time %= 3600
        minutes = total_inference_time // 60
        seconds = total_inference_time % 60
        days = f"{days:.0f} Day(s) " if days > 0 else ""
        hours = f"{hours:.0f} Hour(s) " if hours > 0 or (days and hours == 0) else ""
        minutes = f"{minutes:.0f} Min(s) " if minutes > 0 or (hours and minutes == 0) else ""
        seconds = f"{seconds:.2f} Sec(s)"
        self.my_logger.info(f"All work done with in {days}{hours}{minutes}{seconds}.")

    def unload_models(
            self
    ):
        # Unload models
        if self.use_wd:
            self.my_tagger.unload_model()
        if self.use_joy or self.use_llama or self.use_qwen or self.use_minicpm or self.use_florence:
            self.my_llm.unload_model()


def setup_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    base_args = args.add_argument_group("Base")
    base_args.add_argument(
        '--data_path',
        type=str,
        help='path for data.'
    )
    base_args.add_argument(
        '--recursive',
        action='store_true',
        help='Include recursive dirs'
    )

    log_args = args.add_argument_group("Logs")
    log_args.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='set log level, default is `INFO`'
    )
    log_args.add_argument(
        '--save_logs',
        action='store_true',
        help='save log file.'
    )

    download_args = args.add_argument_group("Download")
    download_args.add_argument(
        '--model_site',
        type=str,
        choices=['huggingface', 'modelscope'],
        default='huggingface',
        help='download models from model site huggingface or modelscope, default is `huggingface`.'
    )
    download_args.add_argument(
        '--models_save_path',
        type=str,
        default=DEFAULT_MODELS_SAVE_PATH,
        help='path to save models, default is `models`.'
    )
    download_args.add_argument(
        '--use_sdk_cache',
        action='store_true',
        help='use sdk\'s cache dir to store models. \
            if this option enabled, `--models_save_path` will be ignored.'
    )
    download_args.add_argument(
        '--download_method',
        type=str,
        choices=["SDK", "URL"],
        default='SDK',
        help='download models via SDK or URL, default is `SDK`.'
    )
    download_args.add_argument(
        '--force_download',
        action='store_true',
        help='force download even file exists.'
    )
    download_args.add_argument(
        '--skip_download',
        action='store_true',
        help='skip download if exists.'
    )

    caption_args = args.add_argument_group("Caption")
    caption_args.add_argument(
        '--caption_method',
        type=str,
        default='wd+llm',
        choices=['wd', 'llm', 'wd+llm'],
        help='method for caption [`wd`, `llm`, `wd+llm`], select wd or llm, or both of them to caption, '
             'default is `wd+llm`.',
    )
    caption_args.add_argument(
        '--run_method',
        type=str,
        default='sync',
        choices=['sync', 'queue'],
        help='''running method for wd+llm caption[`sync`, `queue`], need `caption_method` set to `wd+llm`.
             if sync, image will caption with wd models,
             then caption with joy models while wd captions in joy user prompt.
             if queue, all images will caption with wd models first,
             then caption all of them with joy models while wd captions in joy user prompt.
             default is `sync`.'''
    )
    caption_args.add_argument(
        '--caption_extension',
        type=str,
        default='.txt',
        help='extension of caption file, default is `.txt`. '
             'If `caption_method` not `wd+llm`, it will be wd or llm caption file extension.'
    )
    caption_args.add_argument(
        '--save_caption_together',
        action='store_true',
        help='Save WD tags and LLM captions in one file.'
    )
    caption_args.add_argument(
        '--save_caption_together_seperator',
        default='|',
        help='Seperator between WD and LLM captions, if they are saved in one file.'
    )
    caption_args.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='resize image to suitable, default is `1024`.'
    )
    caption_args.add_argument(
        '--skip_exists',
        action='store_true',
        help='not caption file if caption exists.'
    )
    caption_args.add_argument(
        '--not_overwrite',
        action='store_true',
        help='not overwrite caption file if exists.'
    )
    caption_args.add_argument(
        '--custom_caption_save_path',
        type=str,
        default=None,
        help='custom caption file save path.'
    )

    wd_args = args.add_argument_group("WD Caption")
    wd_args.add_argument(
        '--wd_config',
        type=str,
        help='configs json for wd tagger models, default is `default_wd.json`'
    )
    wd_args.add_argument(
        '--wd_model_name',
        type=str,
        help='wd tagger model name will be used for caption inference, default is `wd-eva02-large-tagger-v3`.'
    )
    wd_args.add_argument(
        '--wd_force_use_cpu',
        action='store_true',
        help='force use cpu for wd models inference.'
    )
    wd_args.add_argument(
        '--wd_caption_extension',
        type=str,
        default=".wdcaption",
        help='extension for wd captions files, default is `.wdcaption`.'
    )
    wd_args.add_argument(
        '--wd_remove_underscore',
        action='store_true',
        help='replace underscores with spaces in the output tags.',
    )
    wd_args.add_argument(
        "--wd_undesired_tags",
        type=str,
        default='',
        help='comma-separated list of undesired tags to remove from the output.'
    )
    wd_args.add_argument(
        '--wd_tags_frequency',
        action='store_true',
        help='Show frequency of tags for images.'
    )
    wd_args.add_argument(
        '--wd_threshold',
        type=float,
        default=0.35,
        help='threshold of confidence to add a tag, default value is `0.35`.'
    )
    wd_args.add_argument(
        '--wd_general_threshold',
        type=float,
        default=None,
        help='threshold of confidence to add a tag from general category, same as --threshold if omitted.'
    )
    wd_args.add_argument(
        '--wd_character_threshold',
        type=float,
        default=None,
        help='threshold of confidence to add a tag for character category, same as --threshold if omitted.'
    )
    # wd_args.add_argument(
    #     '--wd_maximum_cut_threshold',
    #     action = 'store_true',
    #     help = 'Enable Maximum Cut Thresholding, will overwrite every threshold value by its calculate value.'
    # )
    wd_args.add_argument(
        '--wd_add_rating_tags_to_first',
        action='store_true',
        help='Adds rating tags to the first.',
    )
    wd_args.add_argument(
        '--wd_add_rating_tags_to_last',
        action='store_true',
        help='Adds rating tags to the last.',
    )
    wd_args.add_argument(
        '--wd_character_tags_first',
        action='store_true',
        help='Always put character tags before the general tags.',
    )
    wd_args.add_argument(
        '--wd_always_first_tags',
        type=str,
        default=None,
        help='comma-separated list of tags to always put at the beginning, e.g. `1girl,solo`'
    )
    wd_args.add_argument(
        '--wd_caption_separator',
        type=str,
        default=', ',
        help='Separator for tags(include space if needed), default is `, `.'
    )
    wd_args.add_argument(
        '--wd_tag_replacement',
        type=str,
        default=None,
        help='tag replacement in the format of `source1,target1;source2,target2; ...`. '
             'Escape `,` and `;` with `\\`. e.g. `tag1,tag2;tag3,tag4`',
    )
    wd_args.add_argument(
        '--wd_character_tag_expand',
        action='store_true',
        help='expand tag tail parenthesis to another tag for character tags. e.g. '
             '`character_name_(series)` will be expanded to `character_name, series`.',
    )

    llm_args = args.add_argument_group("LLM Caption")
    llm_args.add_argument(
        '--llm_choice',
        type=str,
        default='llama',
        choices=['joy', 'llama', 'qwen', 'minicpm', 'florence'],
        help='select llm models[`joy`, `llama`, `qwen`, `minicpm`, `florence`], default is `llama`.',
    )
    llm_args.add_argument(
        '--llm_config',
        type=str,
        help='config json for LLM Caption models, default is `default_llama_3.2V.json`'
    )
    llm_args.add_argument(
        '--llm_model_name',
        type=str,
        help='model name for inference, default is `Llama-3.2-11B-Vision-Instruct`'
    )
    llm_args.add_argument(
        '--llm_patch',
        action='store_true',
        help='patch llm with lora for uncensored, only support `Llama-3.2-11B-Vision-Instruct` and `Joy-Caption-Pre-Alpha` now'
    )
    llm_args.add_argument(
        '--llm_use_cpu',
        action='store_true',
        help='load LLM models use cpu.'
    )
    llm_args.add_argument(
        '--llm_dtype',
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default='fp16',
        help='choice joy LLM load dtype, default is `fp16`.'
    )
    llm_args.add_argument(
        '--llm_qnt',
        type=str,
        choices=["none", "4bit", "8bit"],
        default='none',
        help='Enable quantization for LLM ["none","4bit", "8bit"]. default is `none`.'
    )
    llm_args.add_argument(
        '--llm_caption_extension',
        type=str,
        default='.llmcaption',
        help='extension of LLM caption file, default is `.llmcaption`'
    )
    llm_args.add_argument(
        '--llm_read_wd_caption',
        action='store_true',
        help='LLM will read wd tags for inference.\nOnly effect when `caption_method` is `llm`'
    )
    llm_args.add_argument(
        '--llm_caption_without_wd',
        action='store_true',
        help='LLM will not read WD tags for inference.\nOnly effect when `caption_method` is `wd+llm`.'
    )
    llm_args.add_argument(
        '--llm_system_prompt',
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help='system prompt for llm caption.'
    )
    llm_args.add_argument(
        '--llm_user_prompt',
        type=str,
        default=DEFAULT_USER_PROMPT_WITHOUT_WD,
        help='user prompt for llm caption.'
    )
    llm_args.add_argument(
        '--llm_temperature',
        type=float,
        default=0,
        help='temperature for LLM model, default is `0`，means use llm own default value.'
    )
    llm_args.add_argument(
        '--llm_max_tokens',
        type=int,
        default=0,
        help='max tokens for LLM model output, default is `0`, means use llm own default value.'
    )

    gradio_args = args.add_argument_group("Gradio dummy args, no effects")
    gradio_args.add_argument('--theme', type=str, default="default", choices=["default", "ocean", "origin"],
                             help="set themes")
    gradio_args.add_argument('--port', type=int, default="8282", help="port, default is `8282`")
    gradio_args.add_argument('--listen', action='store_true', help="allow remote connections")
    gradio_args.add_argument('--share', action='store_true', help="allow gradio share")
    gradio_args.add_argument('--inbrowser', action='store_true', help="auto open in browser")
    return args.parse_args()


def main():
    print_title()
    get_args = setup_args()
    my_caption = Caption()
    my_caption.check_path(get_args)
    my_caption.set_logger(get_args)
    my_caption.download_models(get_args)
    my_caption.load_models(get_args)
    my_caption.run_inference(get_args)
    my_caption.unload_models()


if __name__ == "__main__":
    main()
