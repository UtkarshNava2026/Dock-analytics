from __future__ import annotations

import time
from typing import Dict, Optional, Set

import numpy as np
import yaml


class ReIDMemory:
    """
    Gallery that stores embeddings per person and matches new embeddings
    using cosine similarity.

    Key guarantees:
    - ONE person_id per frame maximum (frame-level uniqueness mutex)
    - Stable track→person_id bridge: ByteTrack track_id caches its person_id
    - TTL: gallery entries expire after `ttl_seconds` of inactivity (`<= 0` = never expire)
    - EMA embedding update for smooth gallery evolution
    """

    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["memory"]

        self.cos_threshold = float(cfg["cosine_threshold"])
        self.max_gallery = int(cfg["max_gallery_size"])
        self.feature_dim = int(cfg["feature_dim"])
        self.ttl_seconds = float(cfg.get("ttl_seconds", 120))
        self.ema_alpha = float(cfg.get("ema_alpha", 0.3))
        self.max_feats = int(cfg.get("max_feats_per_person", 10))

        self.gallery: Dict[int, dict] = {}
        self.next_id = 1

        self._track_to_person: Dict[int, int] = {}
        self._last_active_tracks: Set[int] = set()
        self._frame_used_ids: Set[int] = set()

    def begin_frame(self, active_track_ids: Set[int]) -> None:
        self._frame_used_ids = set()
        gone = self._last_active_tracks - active_track_ids
        for tid in gone:
            self._track_to_person.pop(tid, None)
        self._last_active_tracks = set(active_track_ids)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _score_vs_gallery_entry(self, embedding: np.ndarray, entry: dict) -> float:
        """Best cosine vs EMA centroid and recent feature snapshots (helps re-id after pose/occlusion)."""
        best = self._cosine_similarity(embedding, entry["embedding"])
        for feat in entry.get("feats", ()):
            best = max(best, self._cosine_similarity(embedding, feat))
        return best

    def _expire_gallery(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        expired = [pid for pid, v in self.gallery.items() if now - v["last_seen"] > self.ttl_seconds]
        for pid in expired:
            del self.gallery[pid]
            print(f"[Memory] person_{pid} expired (TTL={self.ttl_seconds}s)")

    def _update_gallery(self, person_id: int, embedding: np.ndarray) -> None:
        entry = self.gallery[person_id]
        entry["embedding"] = (1 - self.ema_alpha) * entry["embedding"] + self.ema_alpha * embedding
        norm = np.linalg.norm(entry["embedding"])
        if norm > 1e-8:
            entry["embedding"] /= norm
        entry["feats"].append(embedding.copy())
        if len(entry["feats"]) > self.max_feats:
            entry["feats"].pop(0)
        entry["last_seen"] = time.time()

    def match(self, embedding: Optional[np.ndarray], track_id: Optional[int] = None) -> int:
        if embedding is None:
            return -1

        if track_id is not None and track_id in self._track_to_person:
            pid = self._track_to_person[track_id]
            if pid in self.gallery and pid not in self._frame_used_ids:
                self._update_gallery(pid, embedding)
                self._frame_used_ids.add(pid)
                return pid
            elif pid in self._frame_used_ids:
                del self._track_to_person[track_id]

        self._expire_gallery()

        best_id = -1
        best_score = -1.0

        for pid, entry in self.gallery.items():
            if pid in self._frame_used_ids:
                continue
            score = self._score_vs_gallery_entry(embedding, entry)
            if score > best_score:
                best_score = score
                best_id = pid

        if best_score >= self.cos_threshold:
            self._update_gallery(best_id, embedding)
            assigned_id = best_id
        else:
            assigned_id = self.next_id
            self.next_id += 1
            self.gallery[assigned_id] = {
                "embedding": embedding.copy(),
                "feats": [embedding.copy()],
                "last_seen": time.time(),
            }
            print(
                f"[Memory] New person: person_{assigned_id} "
                f"(best_score={best_score:.3f} < threshold={self.cos_threshold})"
            )

        self._frame_used_ids.add(assigned_id)
        if track_id is not None:
            self._track_to_person[track_id] = assigned_id

        return assigned_id

    def match_batch(self, embeddings: list, track_ids: Optional[list] = None) -> list:
        if track_ids is None:
            track_ids = [None] * len(embeddings)
        return [self.match(e, tid) for e, tid in zip(embeddings, track_ids)]

    def get_label(self, person_id: int) -> str:
        return f"person_{person_id}" if person_id > 0 else "person_?"

    def reset(self) -> None:
        self.gallery = {}
        self.next_id = 1
        self._track_to_person = {}
        self._last_active_tracks = set()
        self._frame_used_ids = set()
