import argparse
import json
import os
from pathlib import Path
from typing import Union, Optional

import requests
from tqdm import tqdm

from .logger import Logger


def url_download(
        logger: Logger,
        url: str,
        local_dir: Union[str, Path],
        skip_local_file_exist: bool = True,
        force_download: bool = False,
        force_filename: Optional[str] = None
) -> Path:
    # Download file via url by requests library
    filename = os.path.basename(url) if not force_filename else force_filename
    local_file = os.path.join(local_dir, filename)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info(f"Loading huggingface token from environment variable")
    response = requests.get(url, stream=True, headers={
        "Authorization": f"Bearer {hf_token}"} if "huggingface.co" in url and hf_token else None)
    total_size = int(response.headers.get('content-length', 0))

    def download_progress():
        desc = f'Downloading {filename}'

        if total_size > 0:
            pbar = tqdm(total=total_size, initial=0, unit='B', unit_divisor=1024, unit_scale=True,
                        dynamic_ncols=True,
                        desc=desc)
        else:
            pbar = tqdm(initial=0, unit='B', unit_divisor=1024, unit_scale=True, dynamic_ncols=True, desc=desc)

        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        with open(local_file, 'ab') as download_file:
            for data in response.iter_content(chunk_size=1024):
                if data:
                    download_file.write(data)
                    pbar.update(len(data))
        pbar.close()

    if not force_download and os.path.isfile(local_file):
        if skip_local_file_exist and os.path.exists(local_file):
            logger.info(f"`skip_local_file_exist` is Enable, Skipping download {filename}...")
        else:
            if total_size == 0:
                logger.info(
                    f'"{local_file}" already exist, but can\'t get its size from "{url}". Won\'t download it.')
            elif os.path.getsize(local_file) == total_size:
                logger.info(f'"{local_file}" already exist, and its size match with "{url}".')
            else:
                logger.info(
                    f'"{local_file}" already exist, but its size not match with "{url}"!\nWill download this file '
                    f'again...')
                download_progress()
    else:
        download_progress()

    return Path(os.path.join(local_dir, filename))


def download_models(
        logger: Logger,
        models_type: str,
        args: argparse.Namespace,
        config_file: Path,
        models_save_path: Path,
) -> tuple[Path] | tuple[Path, Path] | tuple[Path, Path, Path] | tuple[Path, Path, Path, Path]:
    if os.path.isfile(config_file):
        logger.info(f'Using config: {str(config_file)}')
    else:
        logger.error(f'{str(config_file)} NOT FOUND!')
        raise FileNotFoundError

    def read_json(config_file) -> tuple[str, dict[str]]:
        with open(config_file, 'r', encoding='utf-8') as config_json:
            datas = json.load(config_json)
            if models_type == "wd":
                model_name = list(datas.keys())[0] if not args.wd_model_name else args.wd_model_name
                args.wd_model_name = model_name
            elif models_type in ["joy", "llama", "qwen", "minicpm", "florence"]:
                model_name = list(datas.keys())[0] if not args.llm_model_name else args.llm_model_name
                args.llm_model_name = model_name
            else:
                logger.error("Invalid model type!")
                raise ValueError

            if model_name not in datas.keys():
                logger.error(f'"{str(model_name)}" NOT FOUND IN CONFIG!')
                raise FileNotFoundError
            return model_name, datas[model_name]

    model_name, model_info = read_json(config_file)
    models_save_path = Path(os.path.join(models_save_path, model_name))

    if args.use_sdk_cache:
        logger.warning('use_sdk_cache ENABLED! download_method force to use "SDK" and models_save_path will be ignored')
        args.download_method = 'sdk'
    else:
        logger.info(f'Models will be stored in {str(models_save_path)}.')

    if args.llm_model_name in ["Joy-Caption-Alpha-One", "Joy-Caption-Alpha-Two"]:
        logger.warning(f"{args.llm_model_name} will force using llm patch, auto changed `llm_patch` to `True`!")
        args.llm_patch = True

    def download_choice(
            args: argparse.Namespace,
            model_info: dict[str],
            model_site: str,
            models_save_path: Path,
            download_method: str = "sdk",
            use_sdk_cache: bool = False,
            skip_local_file_exist: bool = True,
            force_download: bool = False
    ):
        if model_site not in ["huggingface", "modelscope"]:
            logger.error('Invalid model site!')
            raise ValueError

        model_site_info = model_info[model_site]
        try:
            if download_method == "sdk":
                if model_site == "huggingface":
                    from huggingface_hub import hf_hub_download
                elif model_site == "modelscope":
                    from modelscope.hub.file_download import model_file_download

        except ModuleNotFoundError:
            if model_site == "huggingface":
                logger.warning('huggingface_hub not installed or download via it failed, '
                               'retrying with URL method to download...')
            elif model_site == "modelscope":
                logger.warning('modelscope not installed or download via it failed, '
                               'retrying with URL method to download...')

            models_path = download_choice(
                args,
                model_info,
                model_site,
                models_save_path,
                use_sdk_cache=False,
                download_method="url",
                skip_local_file_exist=skip_local_file_exist,
                force_download=force_download
            )
            return models_path

        models_path = []
        for sub_model_name in model_site_info:
            sub_model_info = model_site_info[sub_model_name]
            if sub_model_name == "patch" and not args.llm_patch:
                logger.warning(f"Found LLM patch, but llm_patch not enabled, won't download it.")
                continue
            if models_type == "joy" and args.llm_model_name == "Joy-Caption-Pre-Alpha" \
                    and sub_model_name == "llm" and args.llm_patch:
                logger.warning(f"LLM patch Enabled, will replace LLM to patched version.")
                continue
            sub_model_path = ""

            for filename in sub_model_info["file_list"]:
                if download_method.lower() == 'sdk':
                    if model_site == "huggingface":
                        logger.info(f'Will download "{filename}" from huggingface repo: "{sub_model_info["repo_id"]}".')
                        sub_model_path = hf_hub_download(
                            repo_id=sub_model_info["repo_id"],
                            filename=filename,
                            subfolder=sub_model_info["subfolder"] if sub_model_info["subfolder"] != "" else None,
                            repo_type=sub_model_info["repo_type"],
                            revision=sub_model_info["revision"],
                            local_dir=os.path.join(models_save_path, sub_model_name) if not use_sdk_cache else None,
                            local_files_only=skip_local_file_exist \
                                if os.path.exists(os.path.join(models_save_path, sub_model_name, filename)) else False,
                            # local_dir_use_symlinks=False if not use_sdk_cache else "auto",
                            # resume_download=True,
                            force_download=force_download
                        )
                    elif model_site == "modelscope":
                        local_file = os.path.join(models_save_path, sub_model_name, filename)
                        if skip_local_file_exist and os.path.exists(local_file):
                            logger.info(f"`skip_local_file_exist` is Enable, Skipping download {filename}...")
                            sub_model_path = local_file
                        else:
                            logger.info(
                                f'Will download "{filename}" from modelscope repo: "{sub_model_info["repo_id"]}".')
                            sub_model_path = model_file_download(
                                model_id=sub_model_info["repo_id"],
                                file_path=filename if sub_model_info["subfolder"] == ""
                                else os.path.join(sub_model_info["subfolder"], filename),
                                revision=sub_model_info["revision"],
                                local_files_only=False,
                                local_dir=os.path.join(models_save_path, sub_model_name) if not use_sdk_cache else None,
                            )
                else:
                    model_url = sub_model_info["file_list"][filename]
                    logger.info(f'Will download model from url: {model_url}')
                    sub_model_path = url_download(
                        logger=logger,
                        url=model_url,
                        local_dir=os.path.join(models_save_path, sub_model_name) if sub_model_info["subfolder"] == ""
                        else os.path.join(models_save_path, sub_model_name, sub_model_info["subfolder"]),
                        force_filename=filename,
                        skip_local_file_exist=skip_local_file_exist,
                        force_download=force_download
                    )
            models_path.append(sub_model_path)
        return models_path

    models_path = download_choice(
        args=args,
        model_info=model_info,
        model_site=str(args.model_site),
        models_save_path=Path(models_save_path),
        download_method=str(args.download_method).lower(),
        use_sdk_cache=args.use_sdk_cache,
        skip_local_file_exist=args.skip_download,
        force_download=args.force_download
    )

    if models_type == "wd":
        models_path = os.path.dirname(models_path[0])
        wd_model_path = Path(os.path.join(models_path, "model.onnx"))
        if os.path.isfile(os.path.join(models_path, "selected_tags.csv")):
            wd_tags_csv_path = Path(os.path.join(models_path, "selected_tags.csv"))
        else:
            wd_tags_csv_path = Path(os.path.join(models_path, "tags-selected.csv"))
        return wd_model_path, wd_tags_csv_path

    elif models_type == "joy":
        if args.llm_model_name == "Joy-Caption-Alpha-Two-Llava":
            return Path(os.path.dirname(models_path[0])),
        elif args.llm_patch:
            image_adapter_path = Path(os.path.dirname(models_path[0]))
            clip_path = Path(os.path.dirname(models_path[1]))
            llm_path = Path(os.path.dirname(models_path[2]))
            llm_patch_path = Path(os.path.dirname(models_path[3]))
            return image_adapter_path, clip_path, llm_path, llm_patch_path
        else:
            image_adapter_path = Path(os.path.dirname(models_path[0]))
            clip_path = Path(os.path.dirname(models_path[1]))
            llm_path = Path(os.path.dirname(models_path[2]))
            return image_adapter_path, clip_path, llm_path

    elif models_type == "llama":
        llm_path = Path(os.path.dirname(models_path[0]))
        if args.llm_patch:
            llm_patch_path = Path(os.path.dirname(models_path[1]))
            return llm_path, llm_patch_path
        else:
            return llm_path,

    elif models_type in ["qwen", "minicpm", "florence"]:
        return Path(os.path.dirname(models_path[0])),
