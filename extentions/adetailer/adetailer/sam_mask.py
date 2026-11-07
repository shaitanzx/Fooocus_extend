from __future__ import annotations

# SAM refinement for ADetailer: turns bbox detections (e.g. face_yolov9c.pt)
# into shape masks by prompting Segment Anything with each bbox.

import numpy as np
from PIL import Image

from extentions.adetailer.adetailer.common import PredictOutput


def _to_rgb_np(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr.astype(np.uint8))


def load_sam_predictor(model_type: str = "vit_b"):
    # same loading path as extras/inpaint_mask.py (Enhance tab)
    import modules.config
    from segment_anything import sam_model_registry
    from extras.sam.predictor import SamPredictor

    sam_checkpoint = modules.config.download_sam_model(model_type)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    return SamPredictor(sam)


def free_sam_predictor(predictor) -> None:
    # unload SAM from VRAM so it does not stay next to SDXL during inpaint
    try:
        from ldm_patched.modules import model_management
        loaded = model_management.current_loaded_models
        for i in range(len(loaded) - 1, -1, -1):
            if loaded[i].model is predictor.patcher:
                loaded.pop(i).model_unload()
        model_management.soft_empty_cache()
    except Exception as e:
        print(f"[ADetailer] SAM unload warning: {e}")


def _fill_shape(m: np.ndarray, convex_hull: bool = True) -> np.ndarray:
    """Fill every hole inside the mask (eyes, brows, mouth), any size.
    With convex_hull=True the largest component is replaced by its convex
    hull, so eyes that touch the mask edge (not enclosed) are included too."""
    import cv2
    u8 = m.astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return m
    out = np.zeros_like(u8)
    if convex_hull:
        largest = max(contours, key=cv2.contourArea)
        cv2.fillPoly(out, [cv2.convexHull(largest)], 1)
    else:
        cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def _point_prompts(bbox):
    """Positive points for one face box: box center and a point lower on
    the face (nose/mouth area), so SAM picks the face skin, not a hair strand
    that is also inside the box."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx = x1 + bw * 0.5
    return [[cx, y1 + bh * 0.5], [cx, y1 + bh * POINT_LOW_Y]]


POINT_LOW_Y = 0.62     # second point, fraction of box height from the top
MIN_BOX_FILL = 0.20    # mask must cover at least this part of the box
MAX_BOX_FILL = 0.90    # preferred upper limit when choosing among SAM masks


def _choose_mask(cands, scores, bbox, center):
    """cands: [C, H, W] bool, scores: [C]. Prefer masks that contain the box
    center and cover 20-90% of the box, then the best score."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    box_area = max((x2 - x1) * (y2 - y1), 1)
    cx, cy = int(center[0]), int(center[1])
    good, ok = [], []
    for c in range(cands.shape[0]):
        m = cands[c]
        if not (0 <= cy < m.shape[0] and 0 <= cx < m.shape[1]) or not m[cy, cx]:
            continue
        fill = m[max(y1, 0):y2, max(x1, 0):x2].sum() / box_area
        (good if MIN_BOX_FILL <= fill <= MAX_BOX_FILL else ok).append(c)
    pool = good or ok
    if not pool:
        return None
    return cands[max(pool, key=lambda c: scores[c])]


def sam_masks_from_bboxes(image, bboxes, model_type: str = "vit_b",
                          min_pixels: int = 64, clip_to_bbox: bool = True,
                          convex_hull: bool = True):
    """Returns a list (same length as bboxes) of PIL 'L' masks, or None
    where SAM gave no usable mask (caller falls back to the rectangle)."""
    import torch
    from segment_anything.utils.amg import remove_small_regions

    if not bboxes:
        return []
    img = _to_rgb_np(image)
    h, w = img.shape[:2]
    points = [_point_prompts(b) for b in bboxes]

    predictor = load_sam_predictor(model_type)
    try:
        predictor.set_image(img)
        boxes = torch.tensor(bboxes, dtype=torch.float32)
        boxes = predictor.transform.apply_boxes_torch(boxes, img.shape[:2])
        coords = torch.tensor(points, dtype=torch.float32)            # [N, P, 2]
        coords = predictor.transform.apply_coords_torch(coords, img.shape[:2])
        labels = torch.ones(coords.shape[:2], dtype=torch.int64)       # [N, P]
        masks, scores, _ = predictor.predict_torch(
            point_coords=coords,
            point_labels=labels,
            boxes=boxes,
            multimask_output=True,
        )
        masks = masks.cpu().numpy()      # [N, 3, H, W] bool
        scores = scores.float().cpu().numpy()
    finally:
        free_sam_predictor(predictor)
        del predictor

    result = []
    for i, bbox in enumerate(bboxes):
        x1b, y1b, x2b, y2b = bbox
        box_area = max((x2b - x1b) * (y2b - y1b), 1)
        center = points[i][0]
        m = _choose_mask(masks[i], scores[i], bbox, center)
        if m is None:
            print(f"[ADetailer] SAM: no mask covers the box center (detection {i + 1}), using box")
            result.append(None)
            continue
        m, _ = remove_small_regions(m.astype(bool), 400, mode="holes")
        # drop small specks around the main shape (< 2% of the box)
        m, _ = remove_small_regions(m, max(400, int(box_area * 0.02)), mode="islands")
        m = _fill_shape(m, convex_hull)
        if clip_to_bbox:
            x1, y1, x2, y2 = bbox
            x1 = max(int(np.floor(x1)), 0)
            y1 = max(int(np.floor(y1)), 0)
            x2 = min(int(np.ceil(x2)), w)
            y2 = min(int(np.ceil(y2)), h)
            clipped = np.zeros_like(m)
            clipped[y1:y2, x1:x2] = m[y1:y2, x1:x2]
            m = clipped
        cx, cy = int(center[0]), int(center[1])
        fill = m.sum() / box_area
        if int(m.sum()) < min_pixels or not m[min(cy, h - 1), min(cx, w - 1)] or fill < MIN_BOX_FILL:
            print(f"[ADetailer] SAM: mask rejected (detection {i + 1}, {fill:.0%} of box), using box")
            result.append(None)
        else:
            result.append(Image.fromarray(m.astype(np.uint8) * 255, mode="L"))
    return result


def _overlay_preview(preview, masks):
    if preview is None or not masks:
        return preview
    base = np.array(preview.convert("RGB")).astype(np.float32)
    color = np.array([255, 0, 255], dtype=np.float32)
    for m in masks:
        a = np.array(m.convert("L").resize(preview.size)) > 127
        base[a] = base[a] * 0.55 + color * 0.45
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def sam_refine(pred: PredictOutput, image, model_type: str = "vit_b") -> PredictOutput:
    """Replace each detection's mask with a SAM mask (bbox as box prompt).
    Modifies pred in place (like the filters in mask.py) and returns it."""
    if not pred.bboxes:
        return pred
    try:
        sam_masks = sam_masks_from_bboxes(image, pred.bboxes, model_type)
    except Exception as e:
        print(f"[ADetailer] SAM mask failed, using detector masks: {e}")
        return pred

    new_masks = []
    used = 0
    for old, new in zip(pred.masks, sam_masks):
        if new is None:
            new_masks.append(old)
        else:
            new_masks.append(new.resize(old.size) if new.size != old.size else new)
            used += 1
    pred.masks = new_masks
    print(f"[ADetailer] SAM ({model_type}) mask: {used}/{len(sam_masks)} detections refined, "
          f"{len(sam_masks) - used} kept as box")
    pred.preview = _overlay_preview(pred.preview, [m for m, n in zip(new_masks, sam_masks) if n is not None])
    return pred
