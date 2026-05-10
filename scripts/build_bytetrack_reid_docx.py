#!/usr/bin/env python3
"""Build ByteTrack_ReID_Documentation.docx using only the Python standard library (OOXML zip)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def _t(text: str) -> str:
    return escape(text, entities={'"': "&quot;"})


def _p(text: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{_t(text)}</w:t></w:r></w:p>'


def _table(rows: list[tuple[str, str]]) -> str:
    cells = []
    for a, b in rows:
        cells.append(
            "<w:tr>"
            f'<w:tc><w:p><w:r><w:t xml:space="preserve">{_t(a)}</w:t></w:r></w:p></w:tc>'
            f'<w:tc><w:p><w:r><w:t xml:space="preserve">{_t(b)}</w:t></w:r></w:p></w:tc>'
            "</w:tr>"
        )
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>"
        + "".join(cells)
        + "</w:tbl>"
    )


def build_document_xml() -> str:
    parts: list[str] = []
    parts.append(_p("ByteTrack and Re-ID in Dock Analytics", "Title"))
    parts.append(
        _p(
            "This document describes how multi-object tracking (ByteTrack) and person "
            "re-identification (Re-ID) work in the dock_analytics codebase, how they connect, "
            "and what each configuration key does."
        )
    )

    parts.append(_p("1. End-to-end pipeline", "Heading1"))
    for s in [
        "YOLOX produces detections (x1, y1, x2, y2, obj_conf, cls_conf, cls_id) in original image pixels.",
        "TrackerWrapper (app/tracker_adapter.py) converts scores to obj_conf * cls_conf, rescales boxes into letterboxed model space as ByteTrack expects, then calls BYTETracker.update.",
        "ByteTrack returns tracklets with stable track_id and Kalman-smoothed boxes (image space after internal scaling).",
        "TrackerWrapper re-associates each track to a detection by IoU to attach class_id (and keeps last class per track_id when IoU is weak).",
        "If reid.enabled is true, PersonReIDService runs only on tracks whose class_id is the configured person class: crop → OSNet (or ResNet18 fallback) embedding → ReIDMemory → stable person_id (shown as person_N).",
    ]:
        parts.append(_p(s))

    parts.append(_p("2. ByteTrack — high-level algorithm", "Heading1"))
    parts.append(
        _p(
            "ByteTrack is association-based MOT: no appearance model inside the tracker; it uses "
            "geometry (IoU) and detection confidence in two passes (the “byte” idea: use low-score "
            "boxes to keep tracks alive through occlusion)."
        )
    )

    parts.append(_p("2.1 Per-frame steps", "Heading2"))
    for s in [
        "Parse detections: boxes scaled from letterbox space to image space (scale = min(test_H/img_h, test_W/img_w), then bboxes /= scale).",
        "Split detections by score relative to track_thresh: high-confidence (score > track_thresh) for primary association; low-confidence (0.1 < score < track_thresh) for the second association pass.",
        "Predict: merge tracked (confirmed) and lost tracks; each track gets a Kalman prediction (STrack.multi_predict).",
        "First association: cost = IoU distance (1 − IoU) between predicted boxes and high-conf detections. If mot20 is false, fuse_score blends IoU with detection score. Hungarian assignment (lap.lapjv) with threshold match_thresh.",
        "Second association: unmatched tracked tracks matched to low-conf detections with fixed cost threshold 0.5 (hardcoded in byte_tracker.py). Unmatched tracked tracks → Lost.",
        "Unconfirmed tracks: new tracks not yet activated matched to remaining high-conf dets (threshold 0.7 in code). Unmatched → removed.",
        "Birth: remaining high-conf detections with score ≥ det_thresh (track_thresh + 0.1) spawn new tracks.",
        "Prune lost: if a lost track’s age exceeds max_time_lost frames, remove it. max_time_lost = int(frame_rate / 30 * track_buffer).",
        "Deduplicate: remove near-duplicate overlaps between tracked and lost (IoU-related threshold 0.15).",
        "Output: return activated tracks (is_activated).",
    ]:
        parts.append(_p(s))

    parts.append(_p("2.2 TrackerWrapper responsibilities", "Heading2"))
    parts.append(
        _p(
            "ByteTrack does not know class IDs. TrackerWrapper rescales detections for the tracker, "
            "calls update, then matches each output track to the best IoU detection to set class_id, "
            "with IoU ≥ 0.1; otherwise it reuses the last class for that track_id."
        )
    )

    parts.append(_p("2.3 Constants in code (not in YAML)", "Heading2"))
    parts.append(
        _p(
            "Second-pass match threshold 0.5; unconfirmed match 0.7; low-score floor 0.1; duplicate IoU 0.15; "
            "det_thresh = track_thresh + 0.1; class reassignment IoU 0.1 in TrackerWrapper."
        )
    )

    parts.append(_p("3. YAML configuration", "Heading1"))

    parts.append(_p("3.1 tracking: (ByteTrack + wrapper)", "Heading2"))
    parts.append(
        _table(
            [
                ("Key", "Role"),
                (
                    "track_thresh",
                    "Detections above this join the primary association set; between 0.1 and this value go to the second low-score pass.",
                ),
                (
                    "track_buffer",
                    "With frame_rate, sets how long a lost track survives: max_time_lost = int(frame_rate/30 * track_buffer) frames.",
                ),
                (
                    "match_thresh",
                    "Max IoU distance (after optional score fusion) for the first Hungarian match. Lower = stricter; higher = more jitter tolerance.",
                ),
                (
                    "mot20",
                    "If true, skip fuse_score (pure IoU). If false, IoU cost is modulated by detection score (default).",
                ),
                ("frame_rate", "Scales track_buffer so timing matches real FPS."),
            ]
        )
    )

    parts.append(_p("3.2 reid: (embedding model)", "Heading2"))
    parts.append(
        _table(
            [
                ("Key", "Role"),
                ("enabled", "Turns PersonReIDService on/off (AppConfig.reid_enabled)."),
                ("device", "CUDA/CPU for the Re-ID network."),
                (
                    "model_name",
                    "Passed to torchreid.models.build_model when torchreid is installed (e.g. osnet_x1_0).",
                ),
                ("input_size", "[H, W] for resizing the person crop (default [256, 128])."),
            ]
        )
    )
    parts.append(
        _p(
            "If torchreid is missing, code falls back to torchvision ResNet18 (FC removed → embedding)."
        )
    )

    parts.append(_p("3.3 memory: (ReID gallery — ReIDMemory)", "Heading2"))
    parts.append(
        _table(
            [
                ("Key", "Role"),
                (
                    "cosine_threshold",
                    "Match existing person_id if cosine similarity ≥ this vs EMA centroid or any stored snapshot; else new person_id.",
                ),
                ("ttl_seconds", "> 0: remove gallery entries after wall-clock inactivity. 0 = no TTL."),
                ("ema_alpha", "EMA weight when updating centroid after a match; then re-normalized."),
                (
                    "max_feats_per_person",
                    "Ring buffer of recent embeddings; matching uses max cosine vs centroid or any feat.",
                ),
                ("max_gallery_size", "Read in ReIDMemory but not used for eviction in current code."),
                ("feature_dim", "Read in ReIDMemory but not used in matching (length comes from the network)."),
            ]
        )
    )

    parts.append(_p("4. ReID gallery logic (high level)", "Heading1"))
    for s in [
        "For each person track: extract L2-normalized embedding from the BGR crop.",
        "begin_frame: clear per-frame used person IDs; drop track_id → person_id for tracks that disappeared.",
        "match: If track_id maps to person_id not used this frame → update gallery and return. Else scan gallery (skip IDs already assigned), pick best cosine; if ≥ cosine_threshold assign and update; else new person.",
        "Frame-level mutex: at most one track per frame gets a given person_id (_frame_used_ids).",
    ]:
        parts.append(_p(s))

    parts.append(_p("5. ByteTrack track_id vs ReID person_id", "Heading1"))
    parts.append(
        _p(
            "ByteTrack track_id is short-term stable across frames for one tracklet; it can change after long occlusion. "
            "ReID person_id is longer-term: while a track is continuous, _track_to_person keeps the same person_id; "
            "when the track vanishes from active tracks, the bridge clears so a re-entering person can match the gallery by cosine similarity."
        )
    )

    parts.append(_p("6. One-frame flow (summary)", "Heading1"))
    parts.append(
        _p(
            "YOLOX detections → TrackerWrapper: scale boxes to letterbox space → BYTETracker: Kalman predict → "
            "Match high-conf detections (IoU + optional score fuse) → Match low-conf detections to unmatched tracked → "
            "Match unconfirmed / spawn / prune lost → TrackedObject list (track_id + class_id) → "
            "If reid.enabled and class is person: crop → OSNet embedding → ReIDMemory → map track_id → person_id for UI"
        )
    )

    parts.append(_p("7. Key source files", "Heading1"))
    parts.append(
        _table(
            [
                ("Component", "Path"),
                ("ByteTrack core", "app/byte_tracker/byte_tracker.py"),
                ("IoU, fuse_score, LAP", "app/byte_tracker/matching.py"),
                ("Detector ↔ tracker glue", "app/tracker_adapter.py"),
                ("Re-ID service", "app/person_reid.py"),
                ("Gallery", "app/reid_memory.py"),
                ("OSNet / fallback", "app/reid_osnet.py"),
                ("Default YAML", "config/default_config.yaml"),
                ("Flattened app config", "app/config_loader.py"),
            ]
        )
    )

    parts.append(_p("8. Upstream / license", "Heading1"))
    parts.append(
        _p(
            "ByteTrack-derived code is adapted from the MIT-licensed project: https://github.com/ifzhang/ByteTrack "
            "(see NOTICE and README.md in the repository)."
        )
    )

    body = "".join(parts)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {body}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
  </w:body>
</w:document>"""


def write_docx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document_xml = build_document_xml()

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""

    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>ByteTrack and Re-ID in Dock Analytics</dc:title>
  <dc:subject>Technical documentation</dc:subject>
  <dc:creator>dock_analytics</dc:creator>
</cp:coreProperties>"""

    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Python OOXML</Application>
</Properties>"""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "ByteTrack_ReID_Documentation.docx"
    write_docx(out)
    print(f"Wrote {out}")
