from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from extentions.adetailer.adetailer.common import PredictOutput


POINT_LOW_Y = 0.62
MIN_BOX_FILL = 0.20
MAX_BOX_FILL = 0.90


def _to_rgb_np(image) -> np.ndarray:
    """Приводит изображение к contiguous RGB uint8 numpy-массиву."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Expected a 2D/3D image array, got shape {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 or 4 image channels, got {arr.shape[-1]}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def load_sam_predictor(model_type: str = "vit_b"):
    """
    Загружает веса через уже имеющийся Fooocus downloader,
    затем создаёт predictor с Fooocus model patcher.
    """
    import modules.config
    from segment_anything import sam_model_registry
    from extras.sam.predictor import SamPredictor

    if model_type not in modules.flags.inpaint_mask_sam_model:
        raise ValueError(f"Unsupported SAM model: {model_type}")

    checkpoint_path = modules.config.download_sam_model(model_type)
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)

    return SamPredictor(sam_model)


def free_sam_predictor(predictor) -> None:
    """
    Выгружает именно patcher этого SAM из model_management,
    не удаляя из списка другие модели.
    """
    try:
        from ldm_patched.modules import model_management

        loaded_models = model_management.current_loaded_models

        for index in range(len(loaded_models) - 1, -1, -1):
            loaded_model = loaded_models[index]

            if loaded_model.model is predictor.patcher:
                loaded_models.pop(index).model_unload()

        model_management.soft_empty_cache()

    except Exception as exc:
        # Ошибка освобождения памяти не должна скрывать первичную
        # ошибку сегментации или прерывать весь ADetailer workflow.
        print(f"[ADetailer] SAM unload warning: {exc}")


def _point_prompts(bbox):
    """
    Две положительные точки внутри bbox.
    Координаты задаются как (x, y).
    """
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + box_width * 0.5

    return [
        [center_x, y1 + box_height * 0.5],
        [center_x, y1 + box_height * POINT_LOW_Y],
    ]


def _choose_mask(candidates, scores, bbox, center):
    """
    Выбирает SAM-вариант, который содержит центр bbox.
    Среди подходящих масок предпочитает ту, которая покрывает
    от 20% до 90% площади bounding box, и имеет лучший SAM score.
    """
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    box_area = max((x2 - x1) * (y2 - y1), 1)

    center_x = int(center[0])
    center_y = int(center[1])

    preferred = []
    fallback = []

    for index in range(candidates.shape[0]):
        mask = candidates[index]

        if not (
            0 <= center_y < mask.shape[0]
            and 0 <= center_x < mask.shape[1]
            and mask[center_y, center_x]
        ):
            continue

        crop = mask[max(y1, 0):y2, max(x1, 0):x2]
        fill_ratio = crop.sum() / box_area

        if MIN_BOX_FILL <= fill_ratio <= MAX_BOX_FILL:
            preferred.append(index)
        else:
            fallback.append(index)

    pool = preferred or fallback

    if not pool:
        return None

    best_index = max(pool, key=lambda index: scores[index])
    return candidates[best_index]


def _fill_shape(mask: np.ndarray, convex_hull: bool = True) -> np.ndarray:
    """
    Заполняет внутренние пробелы в маске.
    При convex_hull=True использует выпуклую оболочку крупнейшего контура.
    """
    import cv2

    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return mask.astype(bool)

    result = np.zeros_like(mask_u8)

    if convex_hull:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        cv2.fillPoly(result, [hull], 1)
    else:
        cv2.drawContours(
            result,
            contours,
            contourIdx=-1,
            color=1,
            thickness=cv2.FILLED,
        )

    return result.astype(bool)


def sam_masks_from_bboxes(
    image,
    bboxes,
    model_type: str = "vit_b",
    min_pixels: int = 64,
    clip_to_bbox: bool = True,
    convex_hull: bool = True,
):
    """
    Возвращает список PIL L-масок той же длины, что и bboxes.
    None означает, что SAM не смог получить приемлемую маску;
    в этом случае вызывающий код оставит исходную маску детектора.
    """
    from segment_anything.utils.amg import remove_small_regions

    if not bboxes:
        return []

    image_rgb = _to_rgb_np(image)
    height, width = image_rgb.shape[:2]

    # Нормализуем боксы к обычным числам Python.
    boxes_np = np.asarray(bboxes, dtype=np.float32)
    if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError(f"Expected bboxes with shape [N, 4], got {boxes_np.shape}")

    points = [_point_prompts(box) for box in boxes_np]

    predictor = load_sam_predictor(model_type)

    try:
        predictor.set_image(image_rgb)

        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        boxes = predictor.transform.apply_boxes_torch(
            boxes,
            image_rgb.shape[:2],
        )

        point_coords = torch.as_tensor(points, dtype=torch.float32)
        point_coords = predictor.transform.apply_coords_torch(
            point_coords,
            image_rgb.shape[:2],
        )

        point_labels = torch.ones(
            point_coords.shape[:2],
            dtype=torch.int64,
        )

        masks, scores, _ = predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            multimask_output=True,
        )

        masks = masks.detach().cpu().numpy()
        scores = scores.detach().float().cpu().numpy()

    finally:
        free_sam_predictor(predictor)
        del predictor

    results = []

    for index, bbox in enumerate(boxes_np):
        x1b, y1b, x2b, y2b = bbox
        box_area = max((x2b - x1b) * (y2b - y1b), 1.0)
        center = points[index][0]

        mask = _choose_mask(
            masks[index],
            scores[index],
            bbox,
            center,
        )

        if mask is None:
            print(
                f"[ADetailer] SAM: no mask covers the box center "
                f"(detection {index + 1}); keeping detector mask."
            )
            results.append(None)
            continue

        mask = mask.astype(bool)

        # Убираем небольшие внутренние отверстия и отдельные островки.
        mask, _ = remove_small_regions(
            mask,
            400,
            mode="holes",
        )
        mask, _ = remove_small_regions(
            mask,
            max(400, int(box_area * 0.02)),
            mode="islands",
        )

        mask = _fill_shape(mask, convex_hull=convex_hull)

        if clip_to_bbox:
            x1 = max(int(np.floor(x1b)), 0)
            y1 = max(int(np.floor(y1b)), 0)
            x2 = min(int(np.ceil(x2b)), width)
            y2 = min(int(np.ceil(y2b)), height)

            clipped = np.zeros_like(mask)
            clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            mask = clipped

        center_x = int(center[0])
        center_y = int(center[1])

        # Защищаем индексы на случай bbox у границы изображения.
        center_x = min(max(center_x, 0), width - 1)
        center_y = min(max(center_y, 0), height - 1)

        fill_ratio = mask.sum() / box_area

        if (
            int(mask.sum()) < min_pixels
            or not mask[center_y, center_x]
            or fill_ratio < MIN_BOX_FILL
        ):
            print(
                f"[ADetailer] SAM: rejected mask for detection "
                f"{index + 1} ({fill_ratio:.0%} of box); "
                "keeping detector mask."
            )
            results.append(None)
        else:
            mask_image = Image.fromarray(
                mask.astype(np.uint8) * 255,
                mode="L",
            )
            results.append(mask_image)

    return results


def _overlay_preview(preview, masks):
    """Накладывает полупрозрачную magenta-заливку на preview."""
    if preview is None or not masks:
        return preview

    base = np.asarray(preview.convert("RGB")).astype(np.float32)
    color = np.array([255, 0, 255], dtype=np.float32)

    for mask in masks:
        resized = mask.convert("L").resize(preview.size)
        alpha = np.asarray(resized) > 127
        base[alpha] = base[alpha] * 0.55 + color * 0.45

    return Image.fromarray(
        base.clip(0, 255).astype(np.uint8),
        mode="RGB",
    )


def sam_refine(
    pred: PredictOutput,
    image,
    model_type: str = "vit_b",
) -> PredictOutput:
    """
    Заменяет маски детектора на SAM-маски.
    Если SAM не сработал для конкретного объекта, сохраняет исходную маску.
    """
    if not pred.bboxes:
        return pred

    try:
        sam_masks = sam_masks_from_bboxes(
            image=image,
            bboxes=pred.bboxes,
            model_type=model_type,
        )
    except Exception as exc:
        print(
            f"[ADetailer] SAM mask failed; "
            f"keeping detector masks: {exc}"
        )
        return pred

    refined_masks = []
    preview_masks = []
    refined_count = 0

    for old_mask, sam_mask in zip(pred.masks, sam_masks):
        if sam_mask is None:
            refined_masks.append(old_mask)
            continue

        if sam_mask.size != old_mask.size:
            sam_mask = sam_mask.resize(old_mask.size)

        refined_masks.append(sam_mask)
        preview_masks.append(sam_mask)
        refined_count += 1

    pred.masks = refined_masks
    pred.preview = _overlay_preview(pred.preview, preview_masks)

    print(
        f"[ADetailer] SAM ({model_type}): "
        f"{refined_count}/{len(sam_masks)} detections refined; "
        f"{len(sam_masks) - refined_count} kept as detector masks."
    )

    return pred
