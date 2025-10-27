from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from rich import print  # noqa: A004  Shadowing built-in 'print'
from torchvision.transforms.functional import to_pil_image

REPO_ID = "Bingsu/adetailer"

T = TypeVar("T", int, float)


@dataclass
class PredictOutput(Generic[T]):
    bboxes: list[list[T]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def hf_download_to_dir(
    filename: str,
    repo_id: str,
    local_dir: Path,
    check_remote: bool = True,
) -> str:
    """
    Скачивает файл напрямую в local_dir без кеширования.
    Возвращает абсолютный путь к файлу или 'INVALID'.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename

    # Если файл уже есть — используем его
    if local_path.exists():
        return str(local_path.resolve())

    if not check_remote:
        return "INVALID"

    # Попытка 1: Официальный HF
    with suppress(Exception):
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # ← физический файл, не симлинк
            resume_download=True,
            etag_timeout=5,
        )
        return str(Path(path).resolve())

    # Попытка 2: Через зеркало
    with suppress(Exception):
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=5,
            endpoint="https://hf-mirror.com",
        )
        return str(Path(path).resolve())

    # Провал
    print(f"[-] ADetailer: Failed to download {filename!r} from {repo_id}")
    return "INVALID"


def download_models(local_dir: Path, *names: str, check_remote: bool = True) -> dict[str, str]:
    """
    Скачивает модели напрямую в local_dir.
    """
    models = OrderedDict()
    with ThreadPoolExecutor() as executor:
        futures = {}
        for name in names:
            repo_id = "Bingsu/yolo-world-mirror" if "-world" in name else REPO_ID
            future = executor.submit(
                hf_download_to_dir,
                filename=name,
                repo_id=repo_id,
                local_dir=local_dir,
                check_remote=check_remote,
            )
            futures[name] = future

        for name, future in futures.items():
            models[name] = future.result()

    return {k: v for k, v in models.items() if v != "INVALID"}

def scan_model_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix == ".pt"]
    
def get_models(
    *dirs: str | os.PathLike[str],
    huggingface: bool = True,
    download_dir: str | os.PathLike[str] | None = None,
) -> OrderedDict[str, str]:
    """
    Собирает список моделей.
    Если указан download_dir и huggingface=True — скачивает туда.
    """
    model_paths = []
    for dir_ in dirs:
        if not dir_:
            continue
        model_paths.extend(scan_model_dir(Path(dir_)))

    models = OrderedDict()

    # Скачиваем модели, если разрешено и указан download_dir
    if huggingface and download_dir is not None:
        to_download = [
            "face_yolov8n.pt",
            "face_yolov8s.pt",
            "hand_yolov8n.pt",
            "person_yolov8n-seg.pt",
            "person_yolov8s-seg.pt",
            "yolov8x-worldv2.pt",
        ]
        models.update(download_models(Path(download_dir), *to_download, check_remote=True))

    # MediaPipe заглушки
    models.update({
        "mediapipe_face_full": "mediapipe_face_full",
        "mediapipe_face_short": "mediapipe_face_short",
        "mediapipe_face_mesh": "mediapipe_face_mesh",
        "mediapipe_face_mesh_eyes_only": "mediapipe_face_mesh_eyes_only",
    })

    # Удаляем INVALID (на всякий случай)
    invalid_keys = [k for k, v in models.items() if v == "INVALID"]
    for key in invalid_keys:
        models.pop(key, None)

    # Добавляем локальные файлы (включая только что скачанные!)
    for path in model_paths:
        if path.name not in models:
            models[path.name] = str(path)

    return models

def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[list[int]]:
    """
    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        bboxes: list[list[float]]
        A list of bounding boxes

    """
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)  # noqa: PLW2901
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes


def ensure_pil_image(image: Any, mode: str = "RGB") -> Image.Image:
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != mode:
        image = image.convert(mode)
    return image
