from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config_loader import AppConfig, DockRegion
from .tracker_adapter import TrackedObject
from .yolov8_pose import draw_pose_on_frame


def _px_rect(region: DockRegion, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = region.region_xyxy
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def _draw_raw_dets(
    out: np.ndarray,
    dets: np.ndarray,
    class_names: List[str],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    with_labels: bool = True,
) -> None:
    """Draw YOLOX postprocess rows (N,7) xyxy, obj, cls, cls_id."""
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


# BGR — person tracks (distinct from other classes).
_PERSON_BOX_BGR = (0, 0, 255)
_PERSON_TEXT_BGR = (255, 255, 255)
_DEFAULT_TRACK_BOX_BGR = (50, 220, 50)
_DEFAULT_TRACK_TEXT_BGR = (70, 255, 70)


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
    h, w = out.shape[:2]
    for d in cfg.docks:
        x1, y1, x2, y2 = _px_rect(d, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (64, 158, 255), 2)
        cv2.putText(
            out,
            f"{d.name}",
            (x1 + 4, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 200, 255),
            2,
            cv2.LINE_AA,
        )
    # Always draw detector boxes when available; draw tracks on top with IDs when ByteTrack has outputs.
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
    if tracks:
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.tlbr]
            is_person = person_class_id is not None and t.class_id == person_class_id
            if is_person:
                if person_reid_by_track and t.track_id in person_reid_by_track:
                    pid = int(person_reid_by_track[t.track_id])
                    label = f"person id {pid}" if pid > 0 else "person id ?"
                else:
                    label = "person id ?"
                box_color = _PERSON_BOX_BGR
                text_color = _PERSON_TEXT_BGR
                font_scale = 0.65
                label_y = max(0, y1 - 12)
                text_thickness = 2
            else:
                label = (
                    f"ID{t.track_id} {class_names[t.class_id]}"
                    if 0 <= t.class_id < len(class_names)
                    else f"ID{t.track_id} cls{t.class_id}"
                )
                box_color = _DEFAULT_TRACK_BOX_BGR
                text_color = _DEFAULT_TRACK_TEXT_BGR
                font_scale = 0.5
                label_y = max(0, y1 - 8)
                text_thickness = 2
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                out,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )

    if person_poses:
        kth = float(cfg.pose_keypoint_conf_threshold)
        for pr in person_poses:
            kpts = np.asarray(pr.get("keypoints", []), dtype=np.float32)
            if kpts.size == 0:
                continue
            draw_pose_on_frame(out, kpts, kpt_conf_thres=kth)

    return out
