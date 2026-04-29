from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .byte_tracker import reset_global_track_id
from .byte_tracker.byte_tracker import BYTETracker


@dataclass
class TrackedObject:
    track_id: int
    tlbr: np.ndarray
    score: float
    class_id: int


def _iou_matrix(tlbr_a: np.ndarray, tlbr_b: np.ndarray) -> np.ndarray:
    if tlbr_a.size == 0 or tlbr_b.size == 0:
        return np.zeros((len(tlbr_a), len(tlbr_b)))
    from .byte_tracker.matching import bbox_overlaps_xyxy

    return bbox_overlaps_xyxy(np.ascontiguousarray(tlbr_a), np.ascontiguousarray(tlbr_b))


class TrackerWrapper:
    def __init__(
        self,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        mot20: bool,
        frame_rate: float,
    ):
        args = SimpleNamespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=mot20,
        )
        self._frame_rate = float(frame_rate)
        self.tracker = BYTETracker(args, frame_rate=self._frame_rate)
        self._last_cls: Dict[int, int] = {}

    def reset(self):
        reset_global_track_id()
        args = self.tracker.args
        self.tracker = BYTETracker(args, frame_rate=self._frame_rate)
        self._last_cls.clear()

    def update(
        self, dets_xyxy_score: np.ndarray, img_hw: Tuple[int, int], test_hw: Tuple[int, int]
    ) -> List[TrackedObject]:
        """
        dets: (N,7) x1,y1,x2,y2, obj_conf, cls_conf, cls_id (YOLOX postprocess)
        """
        h, w = img_hw
        if dets_xyxy_score.size == 0:
            dets6 = np.zeros((0, 6), dtype=np.float32)
            cls_ids = np.zeros((0,), dtype=np.int64)
        else:
            cls_ids = dets_xyxy_score[:, 6].astype(np.int64)
            scores = dets_xyxy_score[:, 4] * dets_xyxy_score[:, 5]
            dets6 = np.concatenate(
                [dets_xyxy_score[:, :4], scores[:, None], dets_xyxy_score[:, 5:6]], axis=1
            ).astype(np.float32)
            # BYTETracker expects tlbr in letterboxed model space, then divides by scale to image space.
            # YoloxDetector.infer() returns boxes already in original image pixels — re-scale for the tracker.
            scale = min(test_hw[0] / float(h), test_hw[1] / float(w))
            dets6 = dets6.copy()
            dets6[:, :4] *= scale

        stracks = self.tracker.update(dets6, (h, w), test_hw)
        if not stracks:
            return []

        track_tlbr = np.asarray([s.tlbr for s in stracks], dtype=np.float64)
        det_tlbr = dets_xyxy_score[:, :4] if dets_xyxy_score.size else np.zeros((0, 4))
        ious = _iou_matrix(track_tlbr, det_tlbr.astype(np.float64))
        out: List[TrackedObject] = []
        for i, st in enumerate(stracks):
            if ious.shape[1] == 0:
                cid = self._last_cls.get(st.track_id, -1)
            else:
                j = int(np.argmax(ious[i]))
                cid = int(cls_ids[j]) if ious[i, j] >= 0.1 else self._last_cls.get(st.track_id, -1)
            if cid >= 0:
                self._last_cls[st.track_id] = cid
            out.append(
                TrackedObject(
                    track_id=int(st.track_id),
                    tlbr=np.asarray(st.tlbr, dtype=np.float32),
                    score=float(st.score),
                    class_id=cid,
                )
            )
        return out
