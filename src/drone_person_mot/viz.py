from __future__ import annotations

import numpy as np

from .types import Track


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for drawing output video.") from exc

    canvas = frame.copy()
    for track in tracks:
        color = _color(track.track_id)
        x1, y1, x2, y2 = track.xyxy.astype(int).tolist()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"person #{track.track_id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_label = max(y1, th + baseline + 4)
        cv2.rectangle(canvas, (x1, y_label - th - baseline - 4), (x1 + tw + 6, y_label), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, y_label - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        pts = list(track.history)
        for p1, p2 in zip(pts, pts[1:], strict=False):
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
    return canvas


def draw_hud(frame: np.ndarray, fps: float, n_tracks: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for drawing output video.") from exc

    text = f"FPS {fps:.1f} | active IDs {n_tracks}"
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _color(track_id: int) -> tuple[int, int, int]:
    # BGR, intentionally varied so nearby IDs are easy to distinguish.
    palette = [
        (64, 128, 255),
        (80, 220, 120),
        (235, 120, 70),
        (220, 80, 220),
        (70, 200, 220),
        (210, 170, 60),
        (120, 120, 255),
    ]
    return palette[(track_id - 1) % len(palette)]


