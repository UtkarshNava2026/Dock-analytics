from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config_loader import AppConfig, DockRegion
from .tracker_adapter import TrackedObject


def _center_in_norm_rect(cx: float, cy: float, w: int, h: int, norm_xyxy: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = norm_xyxy
    px, py = cx / max(w, 1), cy / max(h, 1)
    return x1 <= px <= x2 and y1 <= py <= y2


def _bbox_center(tlbr: np.ndarray) -> Tuple[float, float]:
    return float((tlbr[0] + tlbr[2]) / 2), float((tlbr[1] + tlbr[3]) / 2)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / (union + 1e-6))


@dataclass
class DockFrameState:
    dock_id: int
    name: str
    dock_closed_in_region: bool
    dock_open_in_region: bool
    truck_track_ids_in_region: List[int]
    activity: bool
    idle: bool
    utilised: bool
    summary_line: str


@dataclass
class FrameAnalytics:
    global_dock_open: bool
    global_truck: bool
    global_person: bool
    global_box_in_hand: bool
    global_pallet: bool
    trio_labels: bool
    docks: List[DockFrameState]
    truck_presence_seconds: Dict[int, float]
    event_lines: List[str]


class AnalyticsEngine:
    # Merge duplicate truck detections on the same vehicle (IoU / center proximity).
    TRUCK_CLUSTER_IOU = 0.28
    TRUCK_CENTER_DIST_FRAC = 0.06
    # Consecutive frames (open + truck, no box_in_hand / pallet_load in ROI) before marking idle.
    IDLE_AFTER_FRAMES_NO_LOAD = 50

    def __init__(self, cfg: AppConfig, class_list: List[str]):
        self.cfg = cfg
        self.class_list = class_list
        self._name_to_idx = {n: i for i, n in enumerate(class_list)}
        self._resolve_aliases(cfg.class_names)
        self._truck_frame_counts: Dict[int, int] = defaultdict(int)
        self._frame_idx = 0
        # Last published dock snapshot per id — log only when this changes (clearer event log).
        self._dock_snapshots: Dict[int, Tuple] = {}
        self._dock_idle_streak: Dict[int, int] = defaultdict(int)

    def _resolve_aliases(self, aliases: Dict[str, str]):
        def idx(key: str):
            name = aliases.get(key, key)
            return self._name_to_idx.get(name)

        self.idx_dock_open = idx("dock_open")
        self.idx_dock_closed = idx("dock_closed")
        self.idx_truck = idx("truck")
        self.idx_person = idx("person")
        self.idx_box = idx("box_in_hand")
        self.idx_pallet = idx("pallet_load")

    def _is_dock_truck(self, class_id: int) -> bool:
        """Only the `truck` class counts for dock truck ROI / presence (not forklift / pallet jack)."""
        if class_id < 0 or self.idx_truck is None:
            return False
        return class_id == self.idx_truck

    def reset(self):
        self._truck_frame_counts.clear()
        self._frame_idx = 0
        self._dock_snapshots.clear()
        self._dock_idle_streak.clear()

    def _collect_global_flags(self, tracks: List[TrackedObject]):
        has_open = False
        has_truck = False
        has_person = False
        has_box = False
        has_pallet = False
        for t in tracks:
            if t.class_id < 0:
                continue
            if self.idx_dock_open is not None and t.class_id == self.idx_dock_open:
                has_open = True
            if self._is_dock_truck(t.class_id):
                has_truck = True
            if self.idx_person is not None and t.class_id == self.idx_person:
                has_person = True
            if self.idx_box is not None and t.class_id == self.idx_box:
                has_box = True
            if self.idx_pallet is not None and t.class_id == self.idx_pallet:
                has_pallet = True
        return has_open, has_truck, has_person, has_box, has_pallet

    def _truck_detections_in_region(
        self, tracks: List[TrackedObject], region: DockRegion, w: int, h: int
    ) -> List[TrackedObject]:
        if self.idx_truck is None:
            return []
        out: List[TrackedObject] = []
        for t in tracks:
            if not self._is_dock_truck(t.class_id):
                continue
            cx, cy = _bbox_center(t.tlbr)
            if _center_in_norm_rect(cx, cy, w, h, region.region_xyxy):
                out.append(t)
        return out

    def _merge_overlapping_truck_ids(
        self, in_region: List[TrackedObject], frame_w: int, frame_h: int
    ) -> List[int]:
        """One track id per overlapping truck cluster (same rig, multiple boxes)."""
        n = len(in_region)
        if n == 0:
            return []
        if n == 1:
            return [in_region[0].track_id]
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pj] = pi

        w, h = int(frame_w), int(frame_h)
        diag = float((w * w + h * h) ** 0.5) if (w and h) else 1.0
        max_center_dist = self.TRUCK_CENTER_DIST_FRAC * diag
        th_iou = self.TRUCK_CLUSTER_IOU
        for i in range(n):
            for j in range(i + 1, n):
                iou = _iou_xyxy(in_region[i].tlbr, in_region[j].tlbr)
                if iou >= th_iou:
                    union(i, j)
                    continue
                cxi, cyi = _bbox_center(in_region[i].tlbr)
                cxj, cyj = _bbox_center(in_region[j].tlbr)
                dist = float(((cxi - cxj) ** 2 + (cyi - cyj) ** 2) ** 0.5)
                if dist <= max_center_dist:
                    union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(in_region[i].track_id)

        return sorted(min(g) for g in clusters.values())

    def _trucks_in_region(self, tracks: List[TrackedObject], region: DockRegion, w: int, h: int) -> List[int]:
        in_roi = self._truck_detections_in_region(tracks, region, w, h)
        return self._merge_overlapping_truck_ids(in_roi, w, h)

    def _format_dock_status_line(
        self,
        region: DockRegion,
        open_here: bool,
        closed_here: bool,
        trucks_here: List[int],
        activity_here: bool,
        idle_here: bool,
        utilised: bool,
    ) -> str:
        parts: List[str] = []
        if closed_here:
            parts.append("closed")
        if open_here:
            parts.append("open")
        if trucks_here:
            parts.append("truck " + ", ".join(f"#{t}" for t in trucks_here))
        else:
            parts.append("no truck")
        if utilised:
            parts.append("utilised")
        if activity_here:
            parts.append("pallet activity")
        if idle_here and trucks_here:
            parts.append("idle")
        return f"{region.name}: " + " · ".join(parts)

    def _closed_in_region(self, tracks: List[TrackedObject], region: DockRegion, w: int, h: int) -> bool:
        if self.idx_dock_closed is None:
            return False
        for t in tracks:
            if t.class_id != self.idx_dock_closed:
                continue
            cx, cy = _bbox_center(t.tlbr)
            if _center_in_norm_rect(cx, cy, w, h, region.region_xyxy):
                return True
        return False

    def _open_in_region(self, tracks: List[TrackedObject], region: DockRegion, w: int, h: int) -> bool:
        if self.idx_dock_open is None:
            return False
        for t in tracks:
            if t.class_id != self.idx_dock_open:
                continue
            cx, cy = _bbox_center(t.tlbr)
            if _center_in_norm_rect(cx, cy, w, h, region.region_xyxy):
                return True
        return False

    def _class_centroid_in_roi(
        self,
        tracks: List[TrackedObject],
        class_idx: Optional[int],
        region: DockRegion,
        w: int,
        h: int,
    ) -> bool:
        if class_idx is None:
            return False
        for t in tracks:
            if t.class_id != class_idx:
                continue
            cx, cy = _bbox_center(t.tlbr)
            if _center_in_norm_rect(cx, cy, w, h, region.region_xyxy):
                return True
        return False

    def process(self, tracks: List[TrackedObject], frame_hw: Tuple[int, int]) -> FrameAnalytics:
        self._frame_idx += 1
        h, w = int(frame_hw[0]), int(frame_hw[1])
        g_open, g_truck, g_person, g_box, g_pallet = self._collect_global_flags(tracks)

        fps = max(self.cfg.frame_rate, 1e-3)
        event_lines: List[str] = []

        dock_states: List[DockFrameState] = []
        # Cumulative presence: one tick per merged logical vehicle per frame (not per raw track id).
        presence_counted: set[int] = set()

        for region in self.cfg.docks:
            closed_here = self._closed_in_region(tracks, region, w, h)
            open_here = self._open_in_region(tracks, region, w, h)
            trucks_here = self._trucks_in_region(tracks, region, w, h)
            for rid in trucks_here:
                if rid not in presence_counted:
                    self._truck_frame_counts[rid] += 1
                    presence_counted.add(rid)

            pallet_in_roi = self._class_centroid_in_roi(tracks, self.idx_pallet, region, w, h)
            box_in_roi = self._class_centroid_in_roi(tracks, self.idx_box, region, w, h)
            load_signal_in_roi = pallet_in_roi or box_in_roi

            activity_here = pallet_in_roi
            utilised_here = bool(open_here and trucks_here)

            if open_here and trucks_here:
                if load_signal_in_roi:
                    self._dock_idle_streak[region.id] = 0
                else:
                    self._dock_idle_streak[region.id] += 1
                idle_here = self._dock_idle_streak[region.id] >= self.IDLE_AFTER_FRAMES_NO_LOAD
            else:
                self._dock_idle_streak[region.id] = 0
                idle_here = False

            snap = (closed_here, open_here, tuple(trucks_here), activity_here, idle_here, utilised_here)
            if self._dock_snapshots.get(region.id) != snap:
                self._dock_snapshots[region.id] = snap
                event_lines.append(
                    self._format_dock_status_line(
                        region,
                        open_here,
                        closed_here,
                        trucks_here,
                        activity_here,
                        idle_here,
                        utilised_here,
                    )
                )

            dock_states.append(
                DockFrameState(
                    dock_id=region.id,
                    name=region.name,
                    dock_closed_in_region=closed_here,
                    dock_open_in_region=open_here,
                    truck_track_ids_in_region=trucks_here,
                    activity=activity_here,
                    idle=idle_here,
                    utilised=utilised_here,
                    summary_line=(
                        f"Dock {region.id} | open_roi={open_here} | closed_roi={closed_here} "
                        f"| trucks={trucks_here} | activity={activity_here} | utilised={utilised_here}"
                    ),
                )
            )

        truck_seconds = {tid: cnt / fps for tid, cnt in self._truck_frame_counts.items()}
        global_activity = any(ds.activity for ds in dock_states)

        return FrameAnalytics(
            global_dock_open=g_open,
            global_truck=g_truck,
            global_person=g_person,
            global_box_in_hand=g_box,
            global_pallet=g_pallet,
            trio_labels=global_activity,
            docks=dock_states,
            truck_presence_seconds=truck_seconds,
            event_lines=event_lines,
        )
