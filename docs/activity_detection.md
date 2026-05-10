# Activity detection (pose-based)

This document describes how **Dock Analytics** classifies each tracked person as **Sitting**, **Walking**, **Working**, or **Idle** using COCO-17 pose keypoints from the YOLOv8 pose model.

## Pipeline overview

1. **Detection & tracking** — YOLOX + ByteTrack produce person boxes and stable `track_id`s.
2. **Re-ID (optional)** — Assigns a persistent `person_id` used in on-screen labels (`Person N …`).
3. **Pose** — Per-person crop is run through `yolov8*-pose.pt`; keypoints are mapped to full-frame coordinates.
4. **Activity** — `PoseActivityTracker` in `app/pose_activity.py` consumes pose results per `track_id`, applies geometry + motion rules, temporal smoothing, and writes `activity` on each pose dict (`sitting` | `walking` | `working` | `idle`).
5. **Overlay** — `app/overlay.py` draws `Person {id} {Activity}` above the box with a fixed panel style (no dock ROI on video; dock logic stays in config).

## Decision priority (exact order)

```
if sitting_geometry and low_leg_motion_gate:
    → sitting
elif walking_from_lower_body_only:
    → walking
elif arms_busy and lower_body_stationary:
    → working
else:
    → idle
```

Walking **does not** use wrists/elbows/shoulders. Working **does not** run if walking or sitting is already true.

## Motion normalization

All displacements are scaled by **person bbox height** (`Δpixels / bbox_h`) so thresholds stay meaningful across camera distances and resolutions.

## Sitting (hips, knees, shoulders)

- Bent knees, compressed leg span, hip flexion, optional relaxed keypoint confidence for partially visible legs.
- **Low leg motion**: mean lower-body motion norm over the smoothing window must stay below `activity_sit_leg_motion_norm_max`, unless **strong leg compression** is detected (very folded pose), or history is still short (warm-up frames).

## Walking (lower body only)

Uses **hips, knees, ankles** only:

- Peak **max joint displacement** norm over the lower-body layer across recent frames vs `activity_walk_norm_threshold`.
- Peak **centroid shift** of the lower-body keypoints vs `activity_walk_centroid_norm_threshold`.
- Optional **stride cue**: left vs right ankle dominance alternation over recent frames plus minimum ankle motion.

## Working (upper body: shoulders, elbows, wrists)

- **Arm motion norm**: max displacement across shoulder/elbow/wrist joints per frame, normalized by bbox height.
- **High activity** if peak ≥ `activity_work_arm_norm_threshold` **or** window mean ≥ threshold × `activity_work_arm_mean_factor`.
- Requires **stationary lower body**: instant lower-body motion and centroid shift below `activity_idle_lower_norm_threshold` (with a small slack on centroid), so walking arm swing does not become “working”.

## Idle

Default when none of the above gated conditions apply.

## Temporal smoothing

- Raw class is appended to a per-track deque of length `activity_smooth_window` (default **12**).
- Stable label changes only when the modal class has at least `activity_stable_min_frames` (**7**) votes; ties favor the previous stable label to reduce flicker (e.g. Idle ↔ Working, Walking ↔ Idle).

## Configuration (`pose:` in YAML)

| Key | Typical role |
|-----|----------------|
| `keypoint_conf_threshold` | Minimum keypoint confidence for geometry/motion |
| `activity_walk_norm_threshold` | Lower-body motion norm to trigger walking |
| `activity_walk_centroid_norm_threshold` | Lower-body centroid drift norm for walking |
| `activity_work_arm_norm_threshold` | Arm motion norm for working |
| `activity_work_arm_mean_factor` | Mean arm motion multiplier vs peak threshold |
| `activity_idle_lower_norm_threshold` | Upper bound for “stationary” lower body (working gate) |
| `activity_sit_leg_motion_norm_max` | Mean lower motion allowed while still accepting sitting |
| `activity_smooth_window` | History length (frames) |
| `activity_stable_min_frames` | Votes required to change stable label |
| `activity_sit_*` | Sitting geometry tuning (leg ratio, knee angle, hip flex, compress, relax scale) |

## On-screen label format

- `Person {reid_id} Sitting|Walking|Working|Idle`
- If Re-ID is off or not yet assigned, the numeric slot shows **—** until an ID exists.

## Limitations

- **2D pose** is view-dependent; extreme angles or occlusion will misclassify.
- **Working vs subtle motion** depends on thresholds; tune with your camera height and FPS.
- **Sitting + large leg fidget** can temporarily break the low leg-motion gate.

## Code map

| File | Role |
|------|------|
| `app/pose_activity.py` | `PoseActivityTracker`, sitting/walk/work math, smoothing |
| `app/yolov8_pose.py` | Pose inference, `infer_poses_for_person_tracks` |
| `app/ui_main.py` | Runs tracker after pose, passes `activity` into overlay |
| `app/overlay.py` | Person boxes + caption panels + optional skeleton |
| `app/config_loader.py` | Loads `pose.*` activity fields into `AppConfig` |
