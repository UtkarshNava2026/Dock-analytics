# Dock Analytics

PyQt5 desktop app for **YOLOX** detection + **ByteTrack** multi-object tracking with a dock / truck / loading activity dashboard.

## Environment (Conda)

```bash
cd dock_analytics
conda env create -f environment.yml
conda activate dock_analytics
```

If the YOLOX git install fails, install PyTorch from [pytorch.org](https://pytorch.org) for your CUDA version, then:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `config/default_config.yaml` and set:
   - `model.exp_file` — your YOLOX experiment Python file (`Exp`).
   - `model.checkpoint` — weights (`.pth`).
   - `model.class_file` — `class.txt` (one class name per line, same order as training).
   - `model.device` — `cuda` or `cpu`.
   - `docks` — each dock’s `region_xyxy` in **normalized** \[0,1\] coordinates: `[x1, y1, x2, y2]` (top-left to bottom-right).
   - `class_names` — optional mapping if your YAML keys differ from names in `class.txt`. Default YAML matches `class-dock.txt`: `person`, `dock_open`, `truck`, `dock_closed`, `forklift`, `box_in_hand`, `pallet_load`, `no_truck`.

2. Run:

```bash
python main.py
```

3. In the UI: choose **Video file** or **RTSP**, set the path or URL, click **Load model**, then **Start**.

## Behaviour summary

- **Dock closed (per ROI):** `dock_closed` detection whose box **centroid** lies inside that dock’s `region_xyxy`.
- **Dock open in ROI:** `dock_open` centroid inside the same region.
- **Truck in ROI:** Only the **`truck`** class (not `forklift`) counts for dock truck IDs and cumulative time. Overlapping truck boxes are merged to one ID. The event log updates **when dock status changes**, in one short line per dock.
- **Utilised (per dock):** `dock_open` in ROI **and** a **`truck`** in ROI (dock is in use).
- **Activity (per dock):** **`pallet_load`** centroid inside that dock ROI (pallet activity only).
- **Idle (per dock):** utilised (open + truck) but **no** `box_in_hand` and **no** `pallet_load` centroids in that ROI for **50 consecutive frames** (loading signals reset the counter).
- **Truck table:** cumulative time each **truck** track ID was in the ROI, shown as `M min Ss` (uses `tracking.frame_rate` from the config).

## Third-party code

ByteTrack tracker logic is adapted from [ByteTrack](https://github.com/ifzhang/ByteTrack) (MIT). YOLOX is used as a dependency from [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
