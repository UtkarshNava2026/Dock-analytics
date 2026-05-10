from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .analytics import AnalyticsEngine, FrameAnalytics
from .config_loader import AppConfig, load_classes, load_yaml
from .detector import YoloxDetector
from .overlay import draw_scene
from .person_reid import PersonReIDService, person_class_index
from .tracker_adapter import TrackerWrapper, TrackedObject
from .pose_activity import PoseActivityTracker
from .yolov8_pose import infer_poses_for_person_tracks, load_pose_model


DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #0f1218; color: #e4e9f2; font-size: 12px; }
QGroupBox {
  font-weight: 700;
  font-size: 13px;
  border: 1px solid #2d3548;
  border-radius: 10px;
  margin-top: 14px;
  padding: 14px 14px 12px 14px;
  background-color: #141922;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 14px;
  padding: 0 8px;
  color: #8ea3c4;
  letter-spacing: 0.4px;
}
QLabel#dashSubtitle { color: #7d8aa3; font-size: 12px; }
QLabel#videoPane {
  background-color: #080a0f;
  border: 1px solid #252b3a;
  border-radius: 10px;
  color: #5c6578;
}
QLineEdit, QComboBox, QTextEdit {
  background-color: #1a1f2c;
  border: 1px solid #323a50;
  border-radius: 8px;
  padding: 8px 10px;
  selection-background-color: #3d5a99;
  min-height: 20px;
}
QComboBox::drop-down { border: none; width: 24px; }
QPushButton {
  background-color: #2f6bff;
  color: #ffffff;
  border: none;
  border-radius: 8px;
  padding: 10px 16px;
  font-weight: 600;
}
QPushButton:hover { background-color: #4a7dff; }
QPushButton:pressed { background-color: #2554d9; }
QPushButton:disabled { background-color: #2a3142; color: #6b7280; }
QPushButton#secondary {
  background-color: #252d3d;
  color: #e4e9f2;
  border: 1px solid #323a50;
}
QPushButton#secondary:hover { background-color: #2f384c; }
QTableWidget {
  background-color: #121722;
  gridline-color: #252b3a;
  border: 1px solid #252b3a;
  border-radius: 8px;
  alternate-background-color: #161c28;
}
QHeaderView::section {
  background-color: #1a2130;
  padding: 8px;
  border: none;
  border-bottom: 1px solid #2d3548;
  font-weight: 700;
  color: #b4c0d9;
}
QScrollArea { border: none; background: transparent; }
QTextEdit#eventLog {
  font-family: "JetBrains Mono", "Consolas", "Monaco", monospace;
  font-size: 11px;
  background-color: #0c0f14;
  border: 1px solid #252b3a;
  border-radius: 8px;
  padding: 8px;
}
QLabel#metricChip { padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 11px; }
"""


class ProcessThread(QThread):
    frame_signal = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
        self.detector: Optional[YoloxDetector] = None
        self.tracker: Optional[TrackerWrapper] = None
        self.class_names: list[str] = []
        self.cfg: Optional[AppConfig] = None
        self.source_kind = "video"
        self.source_path = ""
        self._mutex_run = False
        self.person_reid: Optional[PersonReIDService] = None
        self._person_cls: Optional[int] = None
        self.pose_model: Optional[Any] = None
        self._pose_activity: Optional[PoseActivityTracker] = None

    def configure(
        self,
        cfg: AppConfig,
        detector: YoloxDetector,
        tracker: TrackerWrapper,
        class_names: list[str],
        source_kind: str,
        source_path: str,
        person_reid: Optional[PersonReIDService] = None,
        person_class_id: Optional[int] = None,
        pose_model: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.detector = detector
        self.tracker = tracker
        self.class_names = class_names
        self.source_kind = source_kind
        self.source_path = source_path
        self.person_reid = person_reid
        self._person_cls = person_class_id
        self.pose_model = pose_model
        self._pose_activity = (
            PoseActivityTracker(cfg)
            if cfg is not None and cfg.pose_enabled and pose_model is not None
            else None
        )

    def stop(self):
        self._stop = True

    def run(self):
        self._stop = False
        if self.detector is None or self.tracker is None or self.cfg is None:
            return
        if self.source_kind == "video":
            cap = cv2.VideoCapture(self.source_path)
        else:
            cap = cv2.VideoCapture(self.source_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.frame_signal.emit(None, "Could not open video / RTSP stream")
            return
        th, tw = self.detector.infer_hw[0], self.detector.infer_hw[1]
        while not self._stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.frame_signal.emit(None, "Stream ended or read failed")
                break
            preds, hw = self.detector.infer(frame)
            tracks = self.tracker.update(preds, hw, (th, tw))
            person_reid_map: dict[int, int] = {}
            if self.person_reid is not None and self._person_cls is not None:
                person_reid_map = self.person_reid.relabel_tracks(frame, tracks, self._person_cls)
            pose_results: List[Dict[str, Any]] = []
            if (
                self.pose_model is not None
                and self.cfg is not None
                and self.cfg.pose_enabled
                and self._person_cls is not None
            ):
                pairs: List[tuple[TrackedObject, tuple[int, int, int, int]]] = []
                for t in tracks:
                    if t.class_id != self._person_cls:
                        continue
                    x1, y1, x2, y2 = int(t.tlbr[0]), int(t.tlbr[1]), int(t.tlbr[2]), int(t.tlbr[3])
                    pairs.append((t, (x1, y1, x2, y2)))
                if pairs:
                    pose_results = infer_poses_for_person_tracks(
                        self.pose_model,
                        frame,
                        pairs,
                        imgsz=self.cfg.pose_imgsz,
                        conf_thres=self.cfg.pose_conf_threshold,
                        iou_thres=self.cfg.pose_iou_threshold,
                        kpt_conf_thres=self.cfg.pose_keypoint_conf_threshold,
                        device=self.cfg.pose_device,
                    )
                    if self._pose_activity is not None and pose_results:
                        pose_results = self._pose_activity.label_batch(pose_results)
            self.frame_signal.emit((frame, tracks, preds, person_reid_map, pose_results), None)
        cap.release()


class DockCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("dockCard")
        self.setStyleSheet(
            """
            QFrame#dockCard {
              background-color: #161c28;
              border: 1px solid #2d3548;
              border-radius: 12px;
            }
            QLabel#dockCardTitle { font-size: 15px; font-weight: 700; color: #f2f5fb; }
            QLabel#dockCardBody { color: #a8b5cc; font-size: 12px; line-height: 1.45; }
            """
        )
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 14, 16, 14)
        self.title = QLabel(title)
        self.title.setObjectName("dockCardTitle")
        self.body = QLabel("—")
        self.body.setObjectName("dockCardBody")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        lay.addWidget(self.title)
        lay.addWidget(self.body)

    def set_body_html(self, html: str):
        self.body.setText(html)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dock Analytics — YOLOX + ByteTrack")
        self.resize(1680, 960)
        self.setStyleSheet(DARK_STYLESHEET)

        self.cfg: Optional[AppConfig] = None
        self.class_names: list[str] = []
        self.detector: Optional[YoloxDetector] = None
        self.tracker: Optional[TrackerWrapper] = None
        self.analytics_engine = None
        self.person_reid: Optional[PersonReIDService] = None
        self.pose_model: Optional[Any] = None
        self._last_rgb_buf = None
        self.worker = ProcessThread(self)
        self.worker.frame_signal.connect(self._on_frame)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(10)
        self.video_label = QLabel()
        self.video_label.setObjectName("videoPane")
        self.video_label.setMinimumSize(960, 520)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(False)
        self.video_label.setText("Video preview")
        left_lay.addWidget(self.video_label, stretch=1)

        ctrl = QGroupBox("Controls")
        cform = QFormLayout(ctrl)
        self.config_edit = QLineEdit(str(Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"))
        browse_cfg = QPushButton("Browse…")
        browse_cfg.setObjectName("secondary")
        browse_cfg.clicked.connect(self._browse_config)
        row_cfg = QWidget()
        h = QHBoxLayout(row_cfg)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.config_edit)
        h.addWidget(browse_cfg)
        cform.addRow("Config YAML", row_cfg)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Video file", "RTSP"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        cform.addRow("Source", self.source_combo)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Pick a file or paste path / URL…")
        self.browse_video = QPushButton("Pick video…")
        self.browse_video.setObjectName("secondary")
        self.browse_video.clicked.connect(self._browse_video)
        row_path = QWidget()
        hp = QHBoxLayout(row_path)
        hp.setContentsMargins(0, 0, 0, 0)
        hp.addWidget(self.path_edit)
        hp.addWidget(self.browse_video)
        cform.addRow("Path / URL", row_path)
        self._on_source_changed(self.source_combo.currentIndex())

        self.btn_load = QPushButton("Load model")
        self.btn_load.setObjectName("secondary")
        self.btn_load.clicked.connect(self._load_model)
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("secondary")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        row_btn = QWidget()
        rb = QHBoxLayout(row_btn)
        rb.addWidget(self.btn_load)
        rb.addWidget(self.btn_start)
        rb.addWidget(self.btn_stop)
        cform.addRow(row_btn)

        left_lay.addWidget(ctrl, stretch=0)
        splitter.addWidget(left)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_inner = QWidget()
        rlay = QVBoxLayout(right_inner)

        dash = QGroupBox("Live dashboard")
        dlay = QVBoxLayout(dash)
        dlay.setSpacing(12)
        self.dash_subtitle = QLabel("Load configuration and model to begin.")
        self.dash_subtitle.setObjectName("dashSubtitle")
        self.dash_subtitle.setWordWrap(True)
        dlay.addWidget(self.dash_subtitle)

        self.metrics_strip = QWidget()
        self.metrics_layout = QHBoxLayout(self.metrics_strip)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setSpacing(8)
        dlay.addWidget(self.metrics_strip)

        self.dock_cards_wrap = QWidget()
        self.dock_cards_layout = QGridLayout(self.dock_cards_wrap)
        self.dock_cards: list[DockCard] = []
        dlay.addWidget(self.dock_cards_wrap)

        ev_title = QLabel("Event log")
        ev_title.setStyleSheet("font-weight: 700; color: #8ea3c4; font-size: 12px;")
        self.events_log = QTextEdit()
        self.events_log.setObjectName("eventLog")
        self.events_log.setReadOnly(True)
        self.events_log.setMaximumHeight(220)
        self.events_log.document().setMaximumBlockCount(100)
        dlay.addWidget(ev_title)
        dlay.addWidget(self.events_log)

        truck_box = QGroupBox("Truck presence (cumulative)")
        tlay = QVBoxLayout(truck_box)
        self.truck_table = QTableWidget(0, 2)
        self.truck_table.setAlternatingRowColors(True)
        self.truck_table.setShowGrid(False)
        self.truck_table.setHorizontalHeaderLabels(["Truck ID", "Time at dock"])
        self.truck_table.horizontalHeader().setStretchLastSection(True)
        tlay.addWidget(self.truck_table)

        rlay.addWidget(dash)
        rlay.addWidget(truck_box)
        right_scroll.setWidget(right_inner)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1100, 420])

    def _clear_metric_strip(self):
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _metric_chip(self, label: str, active: bool) -> QLabel:
        chip = QLabel(label)
        chip.setObjectName("metricChip")
        on_bg, on_fg = "#163d2a", "#7ee8a8"
        off_bg, off_fg = "#222833", "#8b95a8"
        bg, fg = (on_bg, on_fg) if active else (off_bg, off_fg)
        chip.setStyleSheet(
            f"QLabel#metricChip {{ background-color: {bg}; color: {fg}; "
            f"border-radius: 999px; padding: 6px 12px; font-weight: 600; font-size: 11px; }}"
        )
        return chip

    def _update_metric_strip(self, fa: FrameAnalytics):
        self._clear_metric_strip()
        for text, val in (
            ("Dock open", fa.global_dock_open),
            ("Truck", fa.global_truck),
            ("Person", fa.global_person),
            ("Box in hand", fa.global_box_in_hand),
            ("Pallet", fa.global_pallet),
            ("Pallet activity", fa.trio_labels),
        ):
            self.metrics_layout.addWidget(self._metric_chip(text, val))
        self.metrics_layout.addStretch(1)

    @staticmethod
    def _format_truck_duration(total_seconds: float) -> str:
        s = int(round(max(0.0, total_seconds)))
        m, sec = divmod(s, 60)
        if m <= 0:
            return f"{sec}s"
        if sec == 0:
            return f"{m} min"
        return f"{m} min {sec}s"

    @staticmethod
    def _bool_html(ok: bool) -> str:
        if ok:
            return "<span style='color:#6ee7a8;font-weight:700'>Yes</span>"
        return "<span style='color:#8b95a8'>No</span>"

    def _dock_card_html(self, ds) -> str:
        trucks = ", ".join(str(x) for x in ds.truck_track_ids_in_region) or "—"
        bh = self._bool_html
        return (
            f"<p style='margin:0 0 10px 0;'>"
            f"<b style='color:#c5d0e8'>Open in ROI</b> {bh(ds.dock_open_in_region)}"
            f"&nbsp;&nbsp;&nbsp;"
            f"<b style='color:#c5d0e8'>Closed in ROI</b> {bh(ds.dock_closed_in_region)}"
            f"</p>"
            f"<p style='margin:0 0 10px 0;'><b style='color:#c5d0e8'>Truck in ROI</b> "
            f"<span style='color:#dbe4f5'>{trucks}</span></p>"
            f"<p style='margin:0;'>"
            f"<b style='color:#c5d0e8'>Activity</b> {bh(ds.activity)} &nbsp;&nbsp; "
            f"<b style='color:#c5d0e8'>Idle</b> {bh(ds.idle)} &nbsp;&nbsp; "
            f"<b style='color:#c5d0e8'>Utilised</b> {bh(ds.utilised)}"
            f"</p>"
        )

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Config", "", "YAML (*.yaml *.yml)")
        if path:
            self.config_edit.setText(path)

    def _on_source_changed(self, index: int):
        is_video = index == 0
        self.browse_video.setVisible(is_video)
        if is_video:
            self.path_edit.setPlaceholderText("Pick a file or paste path…")
        else:
            self.path_edit.setPlaceholderText("Paste RTSP URL…")

    def _browse_video(self):
        filters = (
            "Video (*.mp4 *.avi *.mkv *.mov *.webm *.m4v *.wmv *.flv);;"
            "All files (*)"
        )
        start = str(Path(self.path_edit.text().strip()).expanduser().parent) if self.path_edit.text().strip() else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select video", start, filters)
        if path:
            self.path_edit.setText(path)

    def _load_model(self):
        try:
            cfg = load_yaml(self.config_edit.text().strip())
        except Exception as exc:
            QMessageBox.critical(self, "Config", str(exc))
            return
        if not cfg.exp_file or not Path(cfg.exp_file).is_file():
            QMessageBox.warning(self, "Config", "Set a valid model.exp_file in the YAML.")
            return
        if not cfg.checkpoint or not Path(cfg.checkpoint).is_file():
            QMessageBox.warning(self, "Config", "Set a valid model.checkpoint path.")
            return
        if not cfg.class_file or not Path(cfg.class_file).is_file():
            QMessageBox.warning(self, "Config", "Set a valid model.class_file path.")
            return
        try:
            classes = load_classes(cfg.class_file)
            det = YoloxDetector(
                cfg.exp_file,
                cfg.checkpoint,
                cfg.device,
                cfg.conf_threshold,
                cfg.nms_threshold,
                len(classes),
                cfg.test_size,
            )
            trk = TrackerWrapper(
                cfg.track_thresh,
                cfg.track_buffer,
                cfg.match_thresh,
                cfg.mot20,
                cfg.frame_rate,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Model", str(exc))
            return
        self.cfg = cfg
        self.class_names = classes
        self.detector = det
        self.tracker = trk
        self.analytics_engine = AnalyticsEngine(cfg, classes)
        self.person_reid = None
        if cfg.reid_enabled:
            if not cfg.config_path:
                QMessageBox.warning(self, "Re-ID", "Re-ID is enabled but config path is unknown; skipping Re-ID.")
            else:
                try:
                    self.person_reid = PersonReIDService(cfg.config_path)
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "Re-ID",
                        f"Re-ID could not be loaded and will be disabled:\n{exc}",
                    )
                    self.person_reid = None
        self.pose_model = None
        if cfg.pose_enabled:
            if not cfg.pose_weights or not Path(cfg.pose_weights).is_file():
                QMessageBox.warning(
                    self,
                    "Pose",
                    "Pose is enabled but pose.weights path is missing or not a file; pose disabled.",
                )
            else:
                try:
                    self.pose_model = load_pose_model(
                        cfg.pose_weights, cfg.pose_device, fuse=True
                    )
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "Pose",
                        f"Pose model could not be loaded and will be disabled:\n{exc}",
                    )
                    self.pose_model = None
        self._rebuild_dock_cards()
        self.dash_subtitle.setText(
            f"Model ready · {len(classes)} classes · {cfg.device} · {len(cfg.docks)} dock ROI"
        )
        self._clear_metric_strip()
        if cfg.default_rtsp and self.source_combo.currentIndex() == 1:
            self.path_edit.setText(cfg.default_rtsp)

    def _rebuild_dock_cards(self):
        for c in self.dock_cards:
            c.deleteLater()
        self.dock_cards.clear()
        if not self.cfg:
            return
        cols = 1
        for i, d in enumerate(self.cfg.docks):
            card = DockCard(f"{d.name}  ·  id {d.id}")
            self.dock_cards.append(card)
            self.dock_cards_layout.addWidget(card, i // cols, i % cols)

    def _start(self):
        if self.detector is None or self.tracker is None or self.cfg is None or self.analytics_engine is None:
            QMessageBox.warning(self, "Run", "Load the model first.")
            return
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Run", "Choose a video file or enter an RTSP URL.")
            return
        kind = "rtsp" if self.source_combo.currentIndex() == 1 else "video"
        if kind == "video" and not Path(path).is_file():
            QMessageBox.warning(self, "Run", "Video file does not exist.")
            return
        self._stop()
        self.tracker.reset()
        self.analytics_engine.reset()
        if self.person_reid is not None:
            self.person_reid.reset()
        self.events_log.clear()
        self.worker = ProcessThread(self)
        self.worker.frame_signal.connect(self._on_frame)
        pcls = person_class_index(self.class_names, self.cfg.class_names)
        self.worker.configure(
            self.cfg,
            self.detector,
            self.tracker,
            self.class_names,
            kind,
            path,
            person_reid=self.person_reid,
            person_class_id=pcls,
            pose_model=self.pose_model,
        )
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_frame(self, payload, err):
        if err:
            self.events_log.append(f"<span style='color:#ff8a8a'>{err}</span>")
            self._stop()
            return
        frame, tracks, preds, person_reid_map, pose_results = payload
        fa: FrameAnalytics = self.analytics_engine.process(tracks, frame.shape[:2])
        pcls = person_class_index(self.class_names, self.cfg.class_names)
        person_poses: List[Dict[str, Any]] = []
        for pr in pose_results or []:
            tid = int(pr.get("track_id", -1))
            pid = int(person_reid_map.get(tid, -1)) if person_reid_map else -1
            person_poses.append(
                {
                    "track_id": tid,
                    "person_id": pid,
                    "bbox": pr.get("bbox", []),
                    "keypoints": pr.get("keypoints", []),
                    "activity": pr.get("activity"),
                }
            )
        vis = draw_scene(
            frame,
            tracks,
            self.class_names,
            self.cfg,
            raw_dets=preds,
            person_reid_by_track=person_reid_map if person_reid_map else None,
            person_class_id=pcls,
            person_poses=person_poses if person_poses else None,
        )
        rgb = np.ascontiguousarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        self._last_rgb_buf = rgb
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pw = max(2, self.video_label.width() - 4)
        ph = max(2, self.video_label.height() - 4)
        if pw < 32 or ph < 32:
            pw, ph = max(w, 960), max(int(w * 9 / 16), 520)
        pm0 = QPixmap.fromImage(qimg)
        scaled = pm0.scaled(pw, ph, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        cx = max(0, (scaled.width() - pw) // 2)
        cy = max(0, (scaled.height() - ph) // 2)
        self.video_label.setPixmap(scaled.copy(cx, cy, pw, ph))

        self._update_metric_strip(fa)
        for card, ds in zip(self.dock_cards, fa.docks):
            card.set_body_html(self._dock_card_html(ds))
        for line in fa.event_lines:
            self.events_log.append(line)
        self.events_log.verticalScrollBar().setValue(self.events_log.verticalScrollBar().maximum())

        self.truck_table.setRowCount(0)
        for tid, sec in sorted(fa.truck_presence_seconds.items()):
            r = self.truck_table.rowCount()
            self.truck_table.insertRow(r)
            self.truck_table.setItem(r, 0, QTableWidgetItem(str(tid)))
            self.truck_table.setItem(r, 1, QTableWidgetItem(self._format_truck_duration(sec)))

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)


def run_app():
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
