from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import clip_xyxy, nms_xyxy
from .types import Detection


@dataclass(slots=True)
class DetectorConfig:
    weights: str = "yolo11n.pt"
    device: str | None = None
    imgsz: int = 1280
    conf: float = 0.18
    iou: float = 0.55
    keep_class_ids: tuple[int, ...] = (0,)
    agnostic_nms: bool = True
    augment: bool = False
    use_tiling: bool = False
    tile_size: int = 960
    tile_overlap: float = 0.20


class YOLODetector:
    """Ultralytics detector with optional SAHI-style tiled inference.

    The default class filter is ``0`` because both COCO person and the prepared
    VisDrone person-only dataset use class 0.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self.model = YOLO(config.weights)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if self.config.use_tiling:
            return self._predict_tiled(frame)
        return self._predict_one(frame)

    def _predict_one(self, frame: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        result = self.model.predict(
            frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            agnostic_nms=self.config.agnostic_nms,
            augment=self.config.augment,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = result.boxes.conf.cpu().numpy().astype(float)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        ox, oy = offset
        for xyxy, score, cls in zip(boxes, confs, classes, strict=False):
            if self.config.keep_class_ids and cls not in self.config.keep_class_ids:
                continue
            xyxy = xyxy.copy()
            xyxy[[0, 2]] += ox
            xyxy[[1, 3]] += oy
            detections.append(Detection(xyxy=xyxy, confidence=float(score), class_id=int(cls)))
        return detections

    def _predict_tiled(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        tile = min(self.config.tile_size, max(height, width))
        stride = max(1, int(tile * (1.0 - self.config.tile_overlap)))

        all_detections: list[Detection] = []
        # Include full-frame context so large nearby people are not chopped by tiles.
        all_detections.extend(self._predict_one(frame))
        for y in _starts(height, tile, stride):
            for x in _starts(width, tile, stride):
                crop = frame[y : min(y + tile, height), x : min(x + tile, width)]
                if crop.size == 0:
                    continue
                all_detections.extend(self._predict_one(crop, offset=(x, y)))

        if not all_detections:
            return []

        boxes = np.stack([d.xyxy for d in all_detections]).astype(np.float32)
        scores = np.array([d.confidence for d in all_detections], dtype=np.float32)
        boxes = clip_xyxy(boxes, width, height)
        keep = nms_xyxy(boxes, scores, self.config.iou)
        return [
            Detection(
                xyxy=boxes[i],
                confidence=all_detections[i].confidence,
                class_id=all_detections[i].class_id,
            )
            for i in keep
        ]


def _starts(length: int, tile: int, stride: int) -> Iterable[int]:
    if length <= tile:
        yield 0
        return
    starts = list(range(0, max(length - tile + 1, 1), stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    yield from starts


