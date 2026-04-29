from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DockRegion:
    id: int
    name: str
    region_xyxy: tuple[float, float, float, float]


@dataclass
class AppConfig:
    raw: Dict[str, Any]
    exp_file: str
    checkpoint: str
    class_file: str
    device: str
    conf_threshold: float
    nms_threshold: float
    test_size: tuple[int, int]
    track_thresh: float
    track_buffer: int
    match_thresh: float
    mot20: bool
    frame_rate: float
    class_names: Dict[str, str]
    docks: List[DockRegion]
    default_rtsp: str
    reid_enabled: bool
    config_path: str
    pose_enabled: bool
    pose_weights: str
    pose_device: str
    pose_imgsz: int
    pose_conf_threshold: float
    pose_iou_threshold: float
    pose_keypoint_conf_threshold: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        m = d.get("model", {})
        t = d.get("tracking", {})
        cn = d.get("class_names", {})
        docks = []
        for item in d.get("docks", []):
            r = item["region_xyxy"]
            docks.append(
                DockRegion(
                    id=int(item["id"]),
                    name=str(item.get("name", f"Dock {item['id']}")),
                    region_xyxy=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                )
            )
        src = d.get("sources", {})
        reid = d.get("reid", {}) or {}
        reid_enabled = bool(reid.get("enabled", False))
        po = d.get("pose", {}) or {}
        pose_enabled = bool(po.get("enabled", False))
        pose_weights = os.path.expanduser(str(po.get("weights", "")))
        pose_device = str(po.get("device", m.get("device", "cuda")))
        pose_imgsz = int(po.get("imgsz", 640))
        pose_conf_threshold = float(po.get("conf_threshold", 0.25))
        pose_iou_threshold = float(po.get("iou_threshold", 0.7))
        pose_keypoint_conf_threshold = float(po.get("keypoint_conf_threshold", 0.25))
        return cls(
            raw=d,
            exp_file=os.path.expanduser(str(m.get("exp_file", ""))),
            checkpoint=os.path.expanduser(str(m.get("checkpoint", ""))),
            class_file=os.path.expanduser(str(m.get("class_file", ""))),
            device=str(m.get("device", "cuda")),
            conf_threshold=float(m.get("conf_threshold", 0.35)),
            nms_threshold=float(m.get("nms_threshold", 0.45)),
            test_size=tuple(int(x) for x in m.get("test_size", [640, 640])),
            track_thresh=float(t.get("track_thresh", 0.45)),
            track_buffer=int(t.get("track_buffer", 30)),
            match_thresh=float(t.get("match_thresh", 0.8)),
            mot20=bool(t.get("mot20", False)),
            frame_rate=float(t.get("frame_rate", 25)),
            class_names=dict(cn),
            docks=docks,
            default_rtsp=str(src.get("default_rtsp", "")),
            reid_enabled=reid_enabled,
            config_path="",
            pose_enabled=pose_enabled,
            pose_weights=pose_weights,
            pose_device=pose_device,
            pose_imgsz=pose_imgsz,
            pose_conf_threshold=pose_conf_threshold,
            pose_iou_threshold=pose_iou_threshold,
            pose_keypoint_conf_threshold=pose_keypoint_conf_threshold,
        )


def load_yaml(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = AppConfig.from_dict(data)
    cfg.config_path = str(Path(path).expanduser().resolve())
    return cfg


def load_classes(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names
