from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque
from collections import deque

import numpy as np


@dataclass(slots=True)
class Detection:
    """Single detector output in xyxy pixel coordinates."""

    xyxy: np.ndarray
    confidence: float
    class_id: int = 0

    @property
    def width(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(slots=True)
class Track:
    """Active object track with a short center-point trail."""

    track_id: int
    xyxy: np.ndarray
    confidence: float
    class_id: int = 0
    age: int = 1
    hits: int = 1
    lost: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    history: Deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=32))

    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

    def append_center(self) -> None:
        self.history.append(self.center())


