"""YOLOv8 pose inference without the Ultralytics YOLO() high-level API.

Loads official ``*.pt`` checkpoints via ``torch.load``, runs the pickled ``nn.Module`` in
``eval()`` mode, decodes boxes/keypoints, and maps coordinates back to the full frame.

Requires ``ultralytics`` to be installed so the checkpoint can unpickle (same as training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:
    import torchvision
    from torchvision.ops import nms as tv_nms
except Exception:  # pragma: no cover
    torchvision = None  # type: ignore[assignment]
    tv_nms = None  # type: ignore[assignment]

from .torch_device import resolve_inference_device

# COCO-17 skeleton edges (per user spec)
COCO_POSE_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

NUM_KPTS = 17

_bbox_ema: Dict[int, List[float]] = {}


def smooth_bbox(track_id: int, bbox: List[float], alpha: float = 0.4) -> List[float]:
    if track_id not in _bbox_ema:
        _bbox_ema[track_id] = list(bbox)
        return list(bbox)
    s = _bbox_ema[track_id]
    s = [alpha * b + (1 - alpha) * p for b, p in zip(bbox, s)]
    _bbox_ema[track_id] = s
    return s


def _model_kpt_shape(model: Any) -> Tuple[int, int]:
    """Resolve ``(num_keypoints, dims)`` from a loaded Ultralytics pose ``nn.Module``."""
    ks = getattr(model, "kpt_shape", None)
    if ks is not None:
        t = tuple(int(x) for x in ks)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], 3)
    inner = getattr(model, "model", None)
    if inner is not None and len(inner) > 0:
        last = inner[-1]
        ks2 = getattr(last, "kpt_shape", None)
        if ks2 is not None:
            t2 = tuple(int(x) for x in ks2)
            return (t2[0], t2[1]) if len(t2) >= 2 else (t2[0], 3)
    return (NUM_KPTS, 3)


def _model_nc(model: Any) -> int:
    nc = getattr(model, "nc", None)
    if nc is not None:
        return int(nc)
    inner = getattr(model, "model", None)
    if inner is not None and len(inner) > 0:
        n2 = getattr(inner[-1], "nc", None)
        if n2 is not None:
            return int(n2)
    return 1


@dataclass
class LetterboxMeta:
    """Letterbox transform from original crop (H0, W0) to network input (H1, W1)."""

    ratio: Tuple[float, float]  # (gain_w, gain_h) — Ultralytics uses same gain both axes
    pad: Tuple[float, float]  # (pad_w, pad_h) left/top padding in input pixel space
    orig_hw: Tuple[int, int]  # (H0, W0)
    input_hw: Tuple[int, int]  # (H1, W1)


def _resolve_device(device: str) -> "torch.device":
    assert torch is not None
    return resolve_inference_device(device, context="YOLOv8 pose")


def load_pose_model(
    weights_path: str,
    device: str = "cuda",
    fuse: bool = True,
) -> "nn.Module":
    """Load ``yolov8*-pose.pt`` with ``torch.load`` and return the underlying ``nn.Module``."""
    if torch is None:
        raise RuntimeError("PyTorch is required for pose inference.")
    try:
        import ultralytics  # noqa: F401 — classes must be importable for checkpoint unpickling

        _ = ultralytics
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The ultralytics package is required so PyTorch can unpickle official YOLOv8 *.pt checkpoints. "
            "Install with: pip install ultralytics"
        ) from exc
    path = str(weights_path).strip()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
    elif isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format in {path!r}.")

    if not isinstance(model, nn.Module):
        raise TypeError("Checkpoint 'model' entry is not an nn.Module.")

    dev = _resolve_device(device)
    model = model.to(dev).float().eval()
    if fuse and hasattr(model, "fuse"):
        try:
            model.fuse(verbose=False)  # type: ignore[call-arg]
        except TypeError:
            model.fuse()  # older signature
    return model


def letterbox(
    bgr: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxMeta]:
    """Resize with aspect ratio and pad to ``new_shape`` (H, W). Returns BGR uint8 and meta for inverse mapping."""
    h0, w0 = bgr.shape[:2]
    h1, w1 = int(new_shape[0]), int(new_shape[1])
    r = min(h1 / h0, w1 / w0)
    new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (w1 - new_unpad_w) / 2.0, (h1 - new_unpad_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if (new_unpad_w, new_unpad_h) != (w0, h0):
        resized = cv2.resize(bgr, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = bgr
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    h_out, w_out = out.shape[:2]
    meta = LetterboxMeta(
        ratio=(r, r),
        pad=(float(left), float(top)),
        orig_hw=(h0, w0),
        input_hw=(h_out, w_out),
    )
    return out, meta


def preprocess_crop(
    crop_bgr: np.ndarray,
    imgsz: int = 640,
    device: Optional["torch.device"] = None,
) -> Tuple["torch.Tensor", LetterboxMeta]:
    """BGR crop → letterboxed RGB tensor (1,3,H,W) in [0,1], FP32 on ``device``."""
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty crop.")
    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        raise ValueError("Crop too small for pose.")
    lb, meta = letterbox(crop_bgr, new_shape=(imgsz, imgsz))
    x = lb[:, :, ::-1].transpose(2, 0, 1)  # BGR → CHW RGB
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
    if device is not None:
        t = t.to(device, non_blocking=True)
    return t, meta


def _scale_xy_to_orig(
    xy: "torch.Tensor",
    meta: LetterboxMeta,
) -> "torch.Tensor":
    """Map keypoint xy from letterboxed input space to original crop pixels."""
    gain = meta.ratio[0]
    pad_w, pad_h = meta.pad
    out = xy.clone()
    out[..., 0] = (out[..., 0] - pad_w) / gain
    out[..., 1] = (out[..., 1] - pad_h) / gain
    h0, w0 = meta.orig_hw
    out[..., 0].clamp_(0, w0)
    out[..., 1].clamp_(0, h0)
    return out


def _nms_select(
    pred: "torch.Tensor",
    nc: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> Optional["torch.Tensor"]:
    """``pred``: (A, 4+nc+nk) boxes xyxy + class scores + keypoints (flat)."""
    if tv_nms is None:
        raise RuntimeError("torchvision is required for NMS (torchvision.ops.nms).")
    if pred.numel() == 0:
        return None
    cls = pred[:, 4 : 4 + nc]
    if nc == 1:
        conf = cls[:, 0]
    else:
        conf, _ = cls.max(dim=1)
    mask = conf >= conf_thres
    if not mask.any():
        return None
    sel = pred[mask]
    boxes = sel[:, :4]
    conf = conf[mask]
    keep = tv_nms(boxes, conf, iou_thres)
    if keep.numel() == 0:
        return None
    order = conf[keep].argsort(descending=True)
    keep = keep[order[:max_det]]
    return sel[keep[0]]


def _normalize_forward_output(raw: Any) -> "torch.Tensor":
    """Reduce model(...) to shape (B, num_anchors, channels)."""
    assert torch is not None
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(raw)}")
    t = raw
    if t.dim() != 3:
        raise ValueError(f"Expected 3D prediction tensor, got shape {tuple(t.shape)}")
    _, d1, d2 = t.shape
    # (B, C, A) with few channels and many anchors → (B, A, C)
    if d1 <= 256 and d2 > d1 * 4:
        t = t.transpose(1, 2)
    return t


def run_pose_inference(
    model: nn.Module,
    batch: "torch.Tensor",
    _metas: Sequence[LetterboxMeta],
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 1,
) -> List[Optional[np.ndarray]]:
    """Run batched forward + per-image NMS. Returns list of arrays shape (17,3) xy conf in **letterboxed** space."""
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    nc = _model_nc(model)
    kpt_shape = _model_kpt_shape(model)
    nk = int(kpt_shape[0] * kpt_shape[1])

    with torch.inference_mode():
        raw = model(batch)
    pred_b = _normalize_forward_output(raw)

    out_list: List[Optional[np.ndarray]] = []
    for i in range(pred_b.shape[0]):
        p = pred_b[i]
        row = _nms_select(p, nc, conf_thres, iou_thres, max_det)
        if row is None:
            out_list.append(None)
            continue
        if row.numel() < 4 + nc + nk:
            out_list.append(None)
            continue
        kpts = row[4 + nc : 4 + nc + nk].reshape(kpt_shape[0], kpt_shape[1])
        out_list.append(kpts.detach().float().cpu().numpy())
    return out_list


def keypoints_crop_to_full_frame(
    kpts_crop: np.ndarray,
    meta: LetterboxMeta,
    bbox_xyxy: Tuple[float, float, float, float],
) -> np.ndarray:
    """Letterboxed crop coords → crop pixels → full-frame pixels. ``kpts_crop`` shape (17,3)."""
    assert torch is not None
    x1, y1, _, _ = bbox_xyxy
    t = torch.from_numpy(kpts_crop[..., :2].astype(np.float32))
    xy = _scale_xy_to_orig(t, meta).numpy()
    out = kpts_crop.astype(np.float32).copy()
    out[:, 0] = xy[:, 0] + float(x1)
    out[:, 1] = xy[:, 1] + float(y1)
    return out


def draw_pose_on_frame(
    bgr: np.ndarray,
    keypoints_xyv: np.ndarray,
    kpt_conf_thres: float,
    edges: Sequence[Tuple[int, int]] = COCO_POSE_EDGES,
    point_radius: int = 4,
    line_thickness: int = 2,
    point_color: Tuple[int, int, int] = (0, 255, 255),
    line_color: Tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Draw COCO keypoints (x,y,conf) and skeleton on ``bgr`` in place."""
    h, w = bgr.shape[:2]
    for i, j in edges:
        if i >= len(keypoints_xyv) or j >= len(keypoints_xyv):
            continue
        xi, yi, ci = keypoints_xyv[i]
        xj, yj, cj = keypoints_xyv[j]
        if ci < kpt_conf_thres or cj < kpt_conf_thres:
            continue
        p1 = (int(round(xi)), int(round(yi)))
        p2 = (int(round(xj)), int(round(yj)))
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(bgr, p1, p2, line_color, line_thickness, cv2.LINE_AA)
    for idx in range(len(keypoints_xyv)):
        x, y, c = keypoints_xyv[idx]
        if c < kpt_conf_thres:
            continue
        px, py = int(round(x)), int(round(y))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(bgr, (px, py), point_radius, point_color, -1, cv2.LINE_AA)


def infer_poses_for_person_tracks(
    model: Optional[nn.Module],
    frame_bgr: np.ndarray,
    tracks_tlbr: Sequence[Tuple[Any, Tuple[int, int, int, int]]],
    *,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    kpt_conf_thres: float = 0.25,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Batch-run pose on person crops; return structured dicts with keypoints in **full image** coords.

    ``tracks_tlbr`` is a list of ``(track_obj, (x1,y1,x2,y2 int))`` for persons only.
    """
    if model is None or torch is None:
        return []
    fh, fw = frame_bgr.shape[:2]
    print(f"[FRAME SIZE] height={fh} width={fw}")
    dev = _resolve_device(device)
    tensors: List[torch.Tensor] = []
    metas: List[LetterboxMeta] = []
    track_refs: List[Any] = []
    bboxes: List[Tuple[int, int, int, int]] = []

    for tr, (x1, y1, x2, y2) in tracks_tlbr:
        raw_bbox = [float(x1), float(y1), float(x2), float(y2)]
        smoothed = smooth_bbox(int(getattr(tr, "track_id", -1)), raw_bbox, alpha=0.4)
        x1, y1, x2, y2 = int(smoothed[0]), int(smoothed[1]), int(smoothed[2]), int(smoothed[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            t, meta = preprocess_crop(crop, imgsz=imgsz, device=dev)
        except ValueError:
            continue
        tensors.append(t.squeeze(0))
        metas.append(meta)
        track_refs.append(tr)
        bboxes.append((raw_bbox, smoothed, (x1, y1, x2, y2)))

    if not tensors:
        return []

    batch = torch.stack(tensors, dim=0)
    kpt_batches = run_pose_inference(
        model,
        batch,
        metas,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=1,
    )

    nkpt = int(_model_kpt_shape(model)[0])
    results: List[Dict[str, Any]] = []
    for tr, bbox_pack, meta, k_crop in zip(track_refs, bboxes, metas, kpt_batches):
        raw_bbox, smoothed, crop_xyxy = bbox_pack
        crop_tlbr = (float(crop_xyxy[0]), float(crop_xyxy[1]), float(crop_xyxy[2]), float(crop_xyxy[3]))
        if k_crop is None:
            k_full = np.zeros((nkpt, 3), dtype=np.float32)
        else:
            k_full = keypoints_crop_to_full_frame(k_crop, meta, crop_tlbr)
        low = k_full[:, 2] < kpt_conf_thres
        k_full = k_full.copy()
        k_full[low] = 0.0
        results.append(
            {
                "track_id": int(getattr(tr, "track_id", -1)),
                "bbox": raw_bbox,
                "smoothed_bbox": smoothed,
                "keypoints": k_full.tolist(),
                "frame_width": int(fw),
            }
        )
    return results


