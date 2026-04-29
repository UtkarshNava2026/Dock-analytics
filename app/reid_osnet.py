from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import torchreid  # type: ignore
except Exception:  # pragma: no cover
    torchreid = None  # type: ignore[assignment]

try:
    from torchvision import models as tv_models  # type: ignore
except Exception:  # pragma: no cover
    tv_models = None  # type: ignore[assignment]


class OSNetReID:
    """Person Re-ID embedding extractor (torchreid OSNet or torchvision ResNet18 fallback)."""

    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["reid"]

        if torch is None:
            raise RuntimeError("PyTorch is required for Re-ID but could not be imported.")

        dev = str(cfg.get("device", "cuda")).strip().lower()
        if dev == "cpu":
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device(str(cfg.get("device", "cuda")))
        else:
            print("[ReID] CUDA requested but not available; using CPU")
            self.device = torch.device("cpu")
        self.input_size = tuple(cfg["input_size"])  # (H, W) e.g. (256, 128)

        self.backend = "torchreid" if torchreid is not None else "torchvision_fallback"

        if torchreid is not None:
            self.model = torchreid.models.build_model(
                name=cfg["model_name"],
                num_classes=1000,
                pretrained=True,
            )
        else:
            if tv_models is None:
                raise RuntimeError("Neither torchreid nor torchvision is available for Re-ID.")
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            backbone.fc = torch.nn.Identity()
            self.model = backbone

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[ReID] ReID model loaded on {self.device} ({self.backend})")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, crop: np.ndarray) -> "torch.Tensor":
        img = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def extract(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        tensor = self.preprocess(crop)
        with torch.no_grad():
            feat = self.model(tensor)
        feat = feat.cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat

    def extract_batch(self, crops: list) -> list:
        return [self.extract(c) for c in crops if c is not None]
