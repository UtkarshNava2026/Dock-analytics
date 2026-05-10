from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config_loader import AppConfig
from .tracker_adapter import TrackedObject
from .yolov8_pose import draw_pose_on_frame

# --- Raw detector overlay ---


def _draw_raw_dets(
    out: np.ndarray,
    dets: np.ndarray,
    class_names: List[str],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    with_labels: bool = True,
) -> None:
    if dets.size == 0:
        return
    for i in range(dets.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in dets[i, :4]]
        cid = int(dets[i, 6])
        score = float(dets[i, 4] * dets[i, 5])
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness)
        if not with_labels:
            continue
        label = (
            f"{class_names[cid]} {score:.2f}"
            if 0 <= cid < len(class_names)
            else f"cls{cid} {score:.2f}"
        )
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_bgr,
            2,
            cv2.LINE_AA,
        )


# --- Person track + activity (production) ---

_PERSON_BOX_BGR = (0, 0, 255)
_DEFAULT_TRACK_BOX_BGR = (50, 220, 50)
_DEFAULT_TRACK_TEXT_BGR = (70, 255, 70)

_ACTIVITY_DISPLAY: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    "idle": ("Idle", (180, 180, 190)),
    "walking": ("Walking", (100, 200, 255)),
    "sitting": ("Sitting", (120, 160, 255)),
}

_LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX
_LABEL_SCALE = 1.2
_LABEL_THICK = 3
_LABEL_PAD = 12
_LABEL_MARGIN_ABOVE_BOX = 18
_LABEL_STACK_GAP = 10
_LABEL_PANEL_BG = (24, 26, 32)
_LABEL_PANEL_BORDER = (55, 58, 68)
_LABEL_TEXT = (248, 250, 252)
_STRIPE_W = 4


def _person_caption_line(person_id: int, activity: Optional[str]) -> Tuple[str, str]:
    """One-line label and internal style key (activity or unknown)."""
    pid = str(person_id) if person_id > 0 else "—"
    if not activity:
        return f"Person {pid}", "unknown"
    act = activity.lower().strip()
    title, _ = _ACTIVITY_DISPLAY.get(act, (act.capitalize(), (180, 180, 190)))
    return f"Person {pid} {title}", act if act in _ACTIVITY_DISPLAY else "unknown"


def _rect_intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _caption_size(text: str) -> Tuple[int, int, int]:
    (tw, th), bl = cv2.getTextSize(text, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICK)
    return tw, th, bl


def _layout_captions(
    items: List[Dict[str, Any]], frame_hw: Tuple[int, int]
) -> List[Tuple[Dict[str, Any], int]]:
    if not items:
        return []
    fh, fw = frame_hw[0], frame_hw[1]
    items = sorted(items, key=lambda it: int(it["y1"]))
    placed: List[Tuple[int, int, int, int]] = []
    out: List[Tuple[Dict[str, Any], int]] = []

    for it in items:
        text = str(it["caption"])
        tw, th, bl = _caption_size(text)
        pad = _LABEL_PAD
        x1, y1 = int(it["x1"]), int(it["y1"])
        # Match _draw_caption_panel: x1-x0 = stripe + tw + 3*pad
        total_w = _STRIPE_W + tw + 3 * pad

        baseline_y = y1 - _LABEL_MARGIN_ABOVE_BOX
        for _ in range(64):
            bg_top = baseline_y - th - pad
            bg_left = max(0, x1 - pad)
            bg_right = min(fw - 1, max(bg_left + 2, bg_left + int(total_w)))
            bg_bottom = baseline_y + bl + pad
            if bg_top < 0:
                baseline_y += 1 - bg_top
                continue
            rect = (bg_left, bg_top, bg_right, bg_bottom)
            if not any(_rect_intersects(rect, p) for p in placed):
                placed.append(rect)
                out.append((it, baseline_y))
                break
            baseline_y -= th + 2 * pad + _LABEL_STACK_GAP
        else:
            out.append((it, baseline_y))

    return out


def _draw_caption_panel(
    out: np.ndarray,
    caption: str,
    baseline_x: int,
    baseline_y: int,
    activity_key: str,
) -> None:
    tw, th, bl = _caption_size(caption)
    pad = _LABEL_PAD
    stripe_w = _STRIPE_W
    _, stripe_bgr = _ACTIVITY_DISPLAY.get(activity_key, ("", (180, 180, 190)))

    x0 = baseline_x - pad
    y0 = baseline_y - th - pad
    x1 = baseline_x + stripe_w + pad + tw + pad
    y1 = baseline_y + bl + pad

    cv2.rectangle(out, (x0, y0), (x1, y1), _LABEL_PANEL_BG, thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(out, (x0, y0), (x0 + stripe_w, y1), stripe_bgr, thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(out, (x0, y0), (x1, y1), _LABEL_PANEL_BORDER, thickness=1, lineType=cv2.LINE_AA)
    tx = x0 + stripe_w + pad
    cv2.putText(
        out,
        caption,
        (tx, baseline_y),
        _LABEL_FONT,
        _LABEL_SCALE,
        _LABEL_TEXT,
        _LABEL_THICK,
        cv2.LINE_AA,
    )


def draw_scene(
    bgr: np.ndarray,
    tracks: List[TrackedObject],
    class_names: List[str],
    cfg: AppConfig,
    raw_dets: Optional[np.ndarray] = None,
    person_reid_by_track: Optional[Dict[int, int]] = None,
    person_class_id: Optional[int] = None,
    person_poses: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    out = bgr.copy()
    fh, fw = out.shape[0], out.shape[1]

    if raw_dets is not None and raw_dets.size:
        under = (80, 120, 200) if tracks else (60, 180, 255)
        _draw_raw_dets(
            out,
            raw_dets,
            class_names,
            under,
            thickness=1 if tracks else 2,
            with_labels=not bool(tracks),
        )

    activity_by_track: Dict[int, str] = {}
    if person_poses:
        for pr in person_poses:
            tid, act = pr.get("track_id"), pr.get("activity")
            if tid is not None and act:
                activity_by_track[int(tid)] = str(act)

    caption_rows: List[Dict[str, Any]] = []

    if tracks:
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.tlbr]
            is_person = person_class_id is not None and t.class_id == person_class_id
            if is_person:
                pid = int(person_reid_by_track[t.track_id]) if person_reid_by_track and t.track_id in person_reid_by_track else -1
                cap, key = _person_caption_line(pid, activity_by_track.get(t.track_id))
                caption_rows.append({"x1": x1, "y1": y1, "caption": cap, "activity_key": key})
                cv2.rectangle(out, (x1, y1), (x2, y2), _PERSON_BOX_BGR, 2, lineType=cv2.LINE_AA)
            else:
                label = (
                    f"ID{t.track_id} {class_names[t.class_id]}"
                    if 0 <= t.class_id < len(class_names)
                    else f"ID{t.track_id} cls{t.class_id}"
                )
                cv2.rectangle(out, (x1, y1), (x2, y2), _DEFAULT_TRACK_BOX_BGR, 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    out,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _DEFAULT_TRACK_TEXT_BGR,
                    2,
                    cv2.LINE_AA,
                )

    for it, baseline_y in _layout_captions(caption_rows, (fh, fw)):
        _draw_caption_panel(
            out,
            str(it["caption"]),
            int(it["x1"]),
            baseline_y,
            str(it["activity_key"]),
        )

    if person_poses:
        kth = float(cfg.pose_keypoint_conf_threshold)
        for pr in person_poses:
            kpts = np.asarray(pr.get("keypoints", []), dtype=np.float32)
            if kpts.size == 0:
                continue
            draw_pose_on_frame(out, kpts, kpt_conf_thres=kth)

    return out
