from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .reid_memory import ReIDMemory
from .reid_osnet import OSNetReID
from .tracker_adapter import TrackedObject


def person_class_index(class_names: List[str], class_aliases: Dict[str, str]) -> Optional[int]:
    name = class_aliases.get("person", "person") if class_aliases else "person"
    try:
        return class_names.index(name)
    except ValueError:
        return None


class PersonReIDService:
    """Runs OSNet embeddings + ReIDMemory for person tracks only."""

    def __init__(self, yaml_path: str):
        self.memory = ReIDMemory(yaml_path)
        self.model = OSNetReID(yaml_path)

    def reset(self) -> None:
        self.memory.reset()

    def relabel_tracks(
        self,
        frame_bgr: np.ndarray,
        tracks: List[TrackedObject],
        person_class_id: Optional[int],
    ) -> Dict[int, int]:
        """Map ByteTrack person track_id -> gallery person_id (person class only)."""
        out: Dict[int, int] = {}
        if person_class_id is None:
            return out
        h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        active_track_ids = {t.track_id for t in tracks if t.class_id == person_class_id}
        self.memory.begin_frame(active_track_ids)
        for t in tracks:
            if t.class_id != person_class_id:
                continue
            x1, y1, x2, y2 = int(t.tlbr[0]), int(t.tlbr[1]), int(t.tlbr[2]), int(t.tlbr[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                crop = None
            else:
                crop = frame_bgr[y1:y2, x1:x2].copy()
            emb = self.model.extract(crop)
            pid = self.memory.match(emb, track_id=t.track_id)
            out[t.track_id] = pid
        return out

    def label_for_person_id(self, person_id: int) -> str:
        return self.memory.get_label(person_id)
