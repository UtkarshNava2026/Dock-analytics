"""Warehouse / dock worker activity from COCO-17 pose: walking | idle | sitting.

* ``walking`` — any channel’s smoothed motion exceeds active thresholds.
* ``idle`` — low motion everywhere for ``idle_consecutive_frames`` (standing / still).
* ``sitting`` — very low centroid/lower motion, moderate arm cap, optional left-zone gate;
  no pose geometry in the live path (see unused ``strict_warehouse_sitting_geometry``).

Motion is normalized by bbox height (prefer smoothed box when present). Per-track decisions
use rolling means and ``stable_activity_from_history`` to reduce flicker.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config_loader import AppConfig

# --- COCO-17 ---
L_SH, R_SH = 5, 6
L_ELB, R_ELB = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

LOWER_BODY_INDICES: Tuple[int, ...] = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK)
ARM_INDICES: Tuple[int, ...] = (L_SH, R_SH, L_ELB, R_ELB, L_WRIST, R_WRIST)
_ARM_SH_LO, _ARM_SH_HI = 0, 2
_ARM_ELB_LO, _ARM_ELB_HI = 2, 4
_ARM_WR_LO, _ARM_WR_HI = 4, 6


def deadzone_norm(d: float, eps: float) -> float:
    return 0.0 if d < eps else d


def bbox_height(bbox: List[float]) -> float:
    return max(float(bbox[3]) - float(bbox[1]), 1.0)


def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a[:2].astype(np.float64) - b[:2].astype(np.float64)
    bc = c[:2].astype(np.float64) - b[:2].astype(np.float64)
    n1 = float(np.linalg.norm(ba))
    n2 = float(np.linalg.norm(bc))
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def kpt_xy_valid(kpts: np.ndarray, i: int, conf_th: float) -> Tuple[np.ndarray, bool]:
    if kpts.shape[0] <= i:
        return np.zeros(3, dtype=np.float64), False
    row = kpts[i]
    return row, float(row[2]) >= conf_th


def joint_layer_xy(
    kpts: np.ndarray, indices: Tuple[int, ...], conf_th: float
) -> np.ndarray:
    layer = np.full((len(indices), 2), np.nan, dtype=np.float64)
    for j, idx in enumerate(indices):
        row, ok = kpt_xy_valid(kpts, idx, conf_th)
        if ok:
            layer[j, 0] = float(row[0])
            layer[j, 1] = float(row[1])
    return layer


def mean_displacement_norm(
    prev_layer: Optional[np.ndarray],
    cur_layer: np.ndarray,
    bbox_hv: float,
    motion_deadzone: float = 0.0,
) -> float:
    """Mean per-joint displacement / bbox_height (finite pairs only)."""
    if prev_layer is None:
        return 0.0
    vals: List[float] = []
    for j in range(cur_layer.shape[0]):
        a, b = prev_layer[j], cur_layer[j]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        d = float(np.hypot(a[0] - b[0], a[1] - b[1])) / bbox_hv
        vals.append(deadzone_norm(d, motion_deadzone))
    return float(np.mean(vals)) if vals else 0.0


def layer_centroid(layer: np.ndarray) -> Optional[Tuple[float, float]]:
    good = np.all(np.isfinite(layer), axis=1)
    if not np.any(good):
        return None
    m = np.mean(layer[good], axis=0)
    return float(m[0]), float(m[1])


def centroid_shift_norm(
    prev_c: Optional[Tuple[float, float]],
    cur_c: Optional[Tuple[float, float]],
    bbox_hv: float,
    motion_deadzone: float = 0.0,
) -> float:
    if prev_c is None or cur_c is None:
        return 0.0
    d = float(np.hypot(cur_c[0] - prev_c[0], cur_c[1] - prev_c[1])) / bbox_hv
    return deadzone_norm(d, motion_deadzone)


def weighted_mean_arm_motion_norm(
    prev_arm: Optional[np.ndarray],
    cur_arm: np.ndarray,
    bbox_hv: float,
    motion_deadzone: float,
    w_shoulder: float,
    w_elbow: float,
    w_wrist: float,
) -> float:
    """Weighted mean of per-joint arm displacements / bbox_h (missing joints omitted)."""
    if prev_arm is None or cur_arm.shape[0] < _ARM_WR_HI:
        return 0.0
    acc = 0.0
    w_eff = 0.0
    for j in range(_ARM_SH_LO, _ARM_SH_HI):
        a, b = prev_arm[j], cur_arm[j]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            d = deadzone_norm(
                float(np.hypot(a[0] - b[0], a[1] - b[1])) / bbox_hv, motion_deadzone
            )
            acc += w_shoulder * d
            w_eff += w_shoulder
    for j in range(_ARM_ELB_LO, _ARM_ELB_HI):
        a, b = prev_arm[j], cur_arm[j]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            d = deadzone_norm(
                float(np.hypot(a[0] - b[0], a[1] - b[1])) / bbox_hv, motion_deadzone
            )
            acc += w_elbow * d
            w_eff += w_elbow
    for j in range(_ARM_WR_LO, _ARM_WR_HI):
        a, b = prev_arm[j], cur_arm[j]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            d = deadzone_norm(
                float(np.hypot(a[0] - b[0], a[1] - b[1])) / bbox_hv, motion_deadzone
            )
            acc += w_wrist * d
            w_eff += w_wrist
    return float(acc / w_eff) if w_eff > 0 else 0.0


def lower_body_full_visibility(kpts: np.ndarray, conf_th: float) -> bool:
    """All hips, knees, ankles confident — avoids sitting from partial/noisy legs."""
    for idx in (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK):
        if not kpt_xy_valid(kpts, idx, conf_th)[1]:
            return False
    return True


def torso_compression_ratio(kpts: np.ndarray, bbox: List[float], conf_th: float) -> Optional[float]:
    """Shoulder–midhip vertical span / bbox_h. Lower = more compressed (chair height)."""
    ls, ok_ls = kpt_xy_valid(kpts, L_SH, conf_th)
    rs, ok_rs = kpt_xy_valid(kpts, R_SH, conf_th)
    lh, ok_lh = kpt_xy_valid(kpts, L_HIP, conf_th)
    rh, ok_rh = kpt_xy_valid(kpts, R_HIP, conf_th)
    if not (ok_ls and ok_rs and ok_lh and ok_rh):
        return None
    bh = bbox_height(bbox)
    mid_sy = 0.5 * (float(ls[1]) + float(rs[1]))
    mid_hy = 0.5 * (float(lh[1]) + float(rh[1]))
    return abs(mid_sy - mid_hy) / bh


def strict_warehouse_sitting_geometry(
    kpts: np.ndarray,
    bbox: List[float],
    conf_th: float,
    *,
    torso_span_max: float,
    leg_ratio_max: float,
    knee_straight_deg: float,
    hip_flex_deg: float,
    strong_compress_ratio: float,
) -> bool:
    """Conservative seated pose geometry (legacy / reference only — not used by ``PoseActivityTracker``)."""
    if not lower_body_full_visibility(kpts, conf_th):
        return False

    bh = bbox_height(bbox)
    leg_ratios: List[float] = []
    knee_angles: List[float] = []

    for hip_i, knee_i, ank_i in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        hip, ok_h = kpt_xy_valid(kpts, hip_i, conf_th)
        knee, ok_k = kpt_xy_valid(kpts, knee_i, conf_th)
        ank, ok_a = kpt_xy_valid(kpts, ank_i, conf_th)
        if not (ok_h and ok_k and ok_a):
            return False
        knee_angles.append(_angle_at_b(hip, knee, ank))
        leg_ratios.append((float(ank[1]) - float(hip[1])) / bh)

    span = torso_compression_ratio(kpts, bbox, conf_th)
    if span is None or span >= torso_span_max:
        return False

    if len(leg_ratios) != 2:
        return False
    avg_leg = float(np.mean(leg_ratios))
    if avg_leg > leg_ratio_max:
        return False
    if avg_leg > strong_compress_ratio:
        return False

    if not all(k < knee_straight_deg for k in knee_angles):
        return False

    ls, ok_ls = kpt_xy_valid(kpts, L_SH, conf_th)
    rs, ok_rs = kpt_xy_valid(kpts, R_SH, conf_th)
    lh, ok_lh = kpt_xy_valid(kpts, L_HIP, conf_th)
    rh, ok_rh = kpt_xy_valid(kpts, R_HIP, conf_th)
    lk, ok_lk = kpt_xy_valid(kpts, L_KNEE, conf_th)
    rk, ok_rk = kpt_xy_valid(kpts, R_KNEE, conf_th)
    if not (ok_ls and ok_rs and ok_lh and ok_rh and ok_lk and ok_rk):
        return False
    mid_s = (ls[:2] + rs[:2]) * 0.5
    mid_h = (lh[:2] + rh[:2]) * 0.5
    mid_k = (lk[:2] + rk[:2]) * 0.5
    hip_flex = _angle_at_b(mid_s, mid_h, mid_k)
    if hip_flex >= hip_flex_deg:
        return False

    return True


def stable_activity_from_history(
    history: Deque[str],
    previous_stable: Optional[str],
    min_votes: int,
) -> str:
    if not history:
        return previous_stable or "idle"
    cnt = Counter(history)
    ranked = cnt.most_common()
    best_n = ranked[0][1]
    tied = [s for s, n in ranked if n == best_n]
    if len(tied) > 1 and previous_stable in tied:
        leader = previous_stable
    else:
        leader = tied[0]
    n_lead = int(cnt[leader])
    if previous_stable is None or leader == previous_stable:
        return leader
    if n_lead >= min_votes:
        return leader
    return previous_stable


class PoseActivityTracker:
    def __init__(self, cfg: AppConfig):
        self._dz = float(cfg.pose_activity_motion_deadzone_norm)  # REASON: ignore sub-threshold jitter in px/bbox_h
        self._th_lower = float(cfg.pose_activity_active_lower_mean_threshold)
        self._th_cent = float(cfg.pose_activity_active_centroid_mean_threshold)
        self._th_arm = float(cfg.pose_activity_active_arm_mean_threshold)
        self._idle_max_lower = float(cfg.pose_activity_idle_lower_mean_max)
        self._idle_max_cent = float(cfg.pose_activity_idle_centroid_mean_max)
        self._idle_max_arm = float(cfg.pose_activity_idle_arm_mean_max)
        self._idle_consec = max(2, int(cfg.pose_activity_idle_consecutive_frames))
        self._sit_consec = max(2, int(cfg.pose_activity_sit_stable_min_frames))
        self._sit_cent_max = float(cfg.pose_activity_sit_centroid_mean_max)
        self._sit_lower_max = float(cfg.pose_activity_sit_lower_mean_max)

        pose_raw = cfg.raw.get("pose", {}) or {}  # REASON: AppConfig has no field for sit x2 gate yet
        self._sit_bbox_x2_max = float(pose_raw.get("activity_sit_bbox_x2_max", 0.35))

        self._w_wrist = float(cfg.pose_activity_arm_wrist_weight)
        self._w_elbow = float(cfg.pose_activity_arm_elbow_weight)
        self._w_shoulder = float(cfg.pose_activity_arm_shoulder_weight)

        self._win = max(self._idle_consec + 2, int(cfg.pose_activity_smooth_window))
        self._min_votes = max(4, min(self._win, int(cfg.pose_activity_stable_min_frames)))

        self._kpt_th = float(cfg.pose_keypoint_conf_threshold)

        self._prev_lower: Dict[int, np.ndarray] = {}
        self._prev_lower_centroid: Dict[int, Tuple[float, float]] = {}
        self._prev_arm: Dict[int, np.ndarray] = {}
        self._lower_norm_hist: Dict[int, Deque[float]] = {}
        self._centroid_norm_hist: Dict[int, Deque[float]] = {}
        self._arm_motion_hist: Dict[int, Deque[float]] = {}
        self._raw_history: Dict[int, Deque[str]] = {}
        self._stable_out: Dict[int, str] = {}
        self._idle_streak: Dict[int, int] = {}
        self._sit_streak: Dict[int, int] = {}
        self._last_raw: Dict[int, str] = {}

    def _prune(self, active: set[int]) -> None:
        for d in (
            self._prev_lower,
            self._prev_lower_centroid,
            self._prev_arm,
            self._lower_norm_hist,
            self._centroid_norm_hist,
            self._arm_motion_hist,
            self._raw_history,
            self._stable_out,
            self._idle_streak,
            self._sit_streak,
            self._last_raw,
        ):
            for k in list(d.keys()):
                if k not in active:
                    del d[k]

    def label_batch(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            self._prev_lower.clear()
            self._prev_lower_centroid.clear()
            self._prev_arm.clear()
            self._lower_norm_hist.clear()
            self._centroid_norm_hist.clear()
            self._arm_motion_hist.clear()
            self._raw_history.clear()
            self._stable_out.clear()
            self._idle_streak.clear()
            self._sit_streak.clear()
            self._last_raw.clear()
            return results

        active = {int(r["track_id"]) for r in results if "track_id" in r}
        #print(f"tid={tid} bbox_x2={bbox[2]:.0f} norm={bbox[2]/2688:.3f} sit_cand={sit_cand}")
        self._prune(active)

        out: List[Dict[str, Any]] = []
        for r in results:
            tid = int(r["track_id"])
            kpts = np.asarray(r.get("keypoints", []), dtype=np.float32)
            bbox = r.get("bbox", [0, 0, 1, 1])
            _sb = r.get("smoothed_bbox", bbox)  # REASON: EMA box stabilizes bbox_h used in motion norms
            bh = bbox_height(_sb)

            cur_lower = joint_layer_xy(kpts, LOWER_BODY_INDICES, self._kpt_th)
            prev_lower = self._prev_lower.get(tid)
            lower_inst = mean_displacement_norm(prev_lower, cur_lower, bh, self._dz)
            cur_lc = layer_centroid(cur_lower)
            prev_lc = self._prev_lower_centroid.get(tid)
            cent_inst = centroid_shift_norm(prev_lc, cur_lc, bh, self._dz)

            self._prev_lower[tid] = cur_lower.copy()
            if cur_lc is not None:
                self._prev_lower_centroid[tid] = cur_lc

            lnh = self._lower_norm_hist.setdefault(tid, deque(maxlen=self._win))
            lnh.append(lower_inst)
            cnh = self._centroid_norm_hist.setdefault(tid, deque(maxlen=self._win))
            cnh.append(cent_inst)

            cur_arm = joint_layer_xy(kpts, ARM_INDICES, self._kpt_th)
            prev_arm = self._prev_arm.get(tid)
            arm_inst = weighted_mean_arm_motion_norm(
                prev_arm,
                cur_arm,
                bh,
                self._dz,
                self._w_shoulder,
                self._w_elbow,
                self._w_wrist,
            )
            self._prev_arm[tid] = cur_arm.copy()

            amh = self._arm_motion_hist.setdefault(tid, deque(maxlen=self._win))
            amh.append(arm_inst)

            mean_lower = float(np.mean(lnh)) if lnh else 0.0
            mean_cent = float(np.mean(cnh)) if cnh else 0.0
            mean_arm = float(np.mean(amh)) if amh else 0.0

            walking_frame = (
                mean_lower >= self._th_lower
                or mean_cent >= self._th_cent
                or mean_arm >= self._th_arm
            )  # REASON: any locomotion / handling channel forces walking first

            if walking_frame:
                self._sit_streak[tid] = 0  # REASON: motion breaks seated latch
                self._idle_streak[tid] = 0  # REASON: motion breaks idle latch
                raw = "walking"
                sit_cand = False  # REASON: not evaluated in walking branch; avoids UnboundLocalError on debug print
            else:
                sit_arm_cap = self._idle_max_arm * 3.0  # REASON: spec — seated desk allows more arm motion than idle
                sit_motion_ok = (
                    mean_cent <= self._sit_cent_max
                    and mean_lower <= self._sit_lower_max
                    and mean_arm <= sit_arm_cap
                )
                fw = int(r.get("frame_width", 0) or 0)  # REASON: injected upstream for normalized x2 check
                spatial_disabled = self._sit_bbox_x2_max == 0.0  # REASON: 0.0 disables left-zone gate
                x2 = float(bbox[2])
                spatial_ok = spatial_disabled or (
                    fw > 0 and x2 / float(fw) <= self._sit_bbox_x2_max
                )  # REASON: guard desk fixed in left 35% of frame in this deployment
                sit_cand = sit_motion_ok and spatial_ok  # REASON: all sitting preconditions except streak length

                if sit_cand:
                    self._sit_streak[tid] = self._sit_streak.get(tid, 0) + 1
                else:
                    self._sit_streak[tid] = 0

                idle_cand = (
                    mean_lower <= self._idle_max_lower
                    and mean_cent <= self._idle_max_cent
                    and mean_arm <= self._idle_max_arm
                )  # REASON: stricter idle arm cap than sitting pre-arm cap
                if idle_cand:
                    self._idle_streak[tid] = self._idle_streak.get(tid, 0) + 1
                else:
                    self._idle_streak[tid] = 0

                sit_ok = self._sit_streak[tid] >= self._sit_consec  # REASON: temporal gate reduces false sitting
                idle_ok = self._idle_streak[tid] >= self._idle_consec

                if sit_ok:
                    raw = "sitting"  # REASON: priority 2 — wins over idle when both streaks qualify
                elif idle_ok:
                    raw = "idle"
                else:
                    raw = self._last_raw.get(tid, "walking")  # REASON: spec default for ambiguous in-between motion

            self._last_raw[tid] = raw
            print(f"tid={tid} bbox_x2={bbox[2]:.0f} norm={bbox[2]/2688:.3f} sit_cand={sit_cand}")
            rh = self._raw_history.setdefault(tid, deque(maxlen=self._win))
            rh.append(raw)
            stable = stable_activity_from_history(
                rh,
                self._stable_out.get(tid),
                self._min_votes,
            )
            self._stable_out[tid] = stable

            rr = dict(r)
            rr["activity"] = stable
            out.append(rr)
        return out
