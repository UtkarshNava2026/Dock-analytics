from __future__ import annotations

import importlib.util
from typing import Tuple

import numpy as np
import torch
from loguru import logger

from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, get_model_info, postprocess

from .torch_device import resolve_inference_device


def _load_exp_module(exp_path: str):
    spec = importlib.util.spec_from_file_location("yolox_user_exp", exp_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load exp file: {exp_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Exp"):
        raise RuntimeError("exp file must define class Exp")
    return mod.Exp()


class YoloxDetector:
    def __init__(
        self,
        exp_path: str,
        ckpt_path: str,
        device: str,
        conf_threshold: float,
        nms_threshold: float,
        num_classes: int,
        test_size: Tuple[int, int],
    ):
        self.device = resolve_inference_device(device, context="YOLOX detector")
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.test_size = test_size
        self.exp = _load_exp_module(exp_path)
        self.model = self.exp.get_model()
        exp_nc = getattr(self.exp, "num_classes", None)
        self.num_classes = int(exp_nc) if exp_nc is not None else int(num_classes)
        if exp_nc is not None and int(exp_nc) != int(num_classes):
            logger.warning(
                "exp.num_classes ({}) != class file count ({}); using exp value for inference",
                exp_nc,
                num_classes,
            )
        logger.info("Model summary: {}", get_model_info(self.model, (test_size[0], test_size[1])))
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            try:
                self.model = fuse_model(self.model)
            except Exception as exc:
                logger.warning("fuse_model skipped: {}", exc)
        self.preproc = ValTransform(legacy=False)
        # Match exp letterbox size (same as reference: ValTransform + exp.test_size).
        exp_ts = getattr(self.exp, "test_size", None)
        self._infer_hw = tuple(int(x) for x in (exp_ts if exp_ts is not None else self.test_size))

    @property
    def infer_hw(self) -> Tuple[int, int]:
        """Letterbox (H, W) used at inference — keep ByteTrack `img_size` in sync with this."""
        return self._infer_hw

    @torch.no_grad()
    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Returns (N,7) xyxy + obj + cls + cls_id in original image space, (H,W))."""
        h0, w0 = bgr.shape[:2]
        img, _ = self.preproc(bgr, None, self._infer_hw)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        out = self.model(img)
        # class_agnostic=False + batched_nms matches typical YOLOX demos / VOC reference;
        # class_agnostic=True uses single-class NMS and can wipe overlapping multi-class boxes (busy docks).
        preds = postprocess(
            out,
            self.num_classes,
            self.conf_threshold,
            self.nms_threshold,
            class_agnostic=False,
        )[0]
        if preds is None:
            return np.zeros((0, 7), dtype=np.float32), (h0, w0)
        preds = preds.cpu().numpy()
        ratio = min(self._infer_hw[0] / float(h0), self._infer_hw[1] / float(w0))
        preds[:, :4] /= ratio
        return preds.astype(np.float32), (h0, w0)
