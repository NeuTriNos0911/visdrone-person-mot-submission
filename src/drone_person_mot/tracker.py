from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import affine_warp_box, clip_xyxy, greedy_iou_assignment
from .types import Detection, Track


IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


@dataclass(slots=True)
class TrackerConfig:
    high_conf: float = 0.35
    low_conf: float = 0.08
    match_iou_high: float = 0.25
    match_iou_low: float = 0.12
    max_lost: int = 30
    min_hits: int = 1
    history: int = 32
    use_camera_motion: bool = True


class EgoMotionCompensator:
    """Estimate the dominant frame-to-frame camera motion with sparse optical flow."""

    def __init__(self, max_corners: int = 600):
        self.max_corners = max_corners
        self.prev_gray: np.ndarray | None = None

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for tracking.") from exc

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return IDENTITY_AFFINE.copy()

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if prev_pts is None or len(prev_pts) < 12:
            self.prev_gray = gray
            return IDENTITY_AFFINE.copy()

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
        if curr_pts is None or status is None:
            self.prev_gray = gray
            return IDENTITY_AFFINE.copy()

        good_prev = prev_pts[status.reshape(-1) == 1]
        good_curr = curr_pts[status.reshape(-1) == 1]
        if len(good_prev) < 12:
            self.prev_gray = gray
            return IDENTITY_AFFINE.copy()

        affine, _ = cv2.estimateAffinePartial2D(
            good_prev.reshape(-1, 2),
            good_curr.reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        self.prev_gray = gray
        if affine is None:
            return IDENTITY_AFFINE.copy()
        return affine.astype(np.float32)


class AerialTracker:
    """ByteTrack-inspired IOU tracker with optional ego-motion compensation."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracks: list[Track] = []
        self.next_id = 1
        self.cmc = EgoMotionCompensator() if config.use_camera_motion else None

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        affine = IDENTITY_AFFINE.copy()
        if self.cmc is not None and frame is not None:
            affine = self.cmc.estimate(frame)

        if frame is not None:
            height, width = frame.shape[:2]
        else:
            height = width = 10_000

        predicted = []
        for track in self.tracks:
            track.age += 1
            pred_box = track.xyxy + track.velocity
            pred_box = affine_warp_box(pred_box, affine)
            predicted.append(pred_box)
        if predicted:
            predicted_boxes = clip_xyxy(np.stack(predicted).astype(np.float32), width, height)
        else:
            predicted_boxes = np.empty((0, 4), dtype=np.float32)

        high_dets = [d for d in detections if d.confidence >= self.config.high_conf]
        low_dets = [
            d
            for d in detections
            if self.config.low_conf <= d.confidence < self.config.high_conf
        ]

        unmatched_track_ids = list(range(len(self.tracks)))
        self._apply_predictions(predicted_boxes)

        matches, unmatched_track_ids, unmatched_high_ids = self._match(
            unmatched_track_ids,
            high_dets,
            self.config.match_iou_high,
        )
        for track_idx, det_idx in matches:
            self._update_track(self.tracks[track_idx], high_dets[det_idx])

        if low_dets and unmatched_track_ids:
            matches_low, unmatched_track_ids, _ = self._match(
                unmatched_track_ids,
                low_dets,
                self.config.match_iou_low,
            )
            for track_idx, det_idx in matches_low:
                self._update_track(self.tracks[track_idx], low_dets[det_idx])

        for track_idx in unmatched_track_ids:
            self.tracks[track_idx].lost += 1

        for det_idx in unmatched_high_ids:
            self._start_track(high_dets[det_idx])

        self.tracks = [t for t in self.tracks if t.lost <= self.config.max_lost]
        visible = [
            t
            for t in self.tracks
            if t.lost == 0 and t.hits >= self.config.min_hits
        ]
        return visible

    def _apply_predictions(self, predicted_boxes: np.ndarray) -> None:
        for track, box in zip(self.tracks, predicted_boxes, strict=False):
            track.xyxy = box

    def _match(
        self,
        track_ids: list[int],
        detections: list[Detection],
        threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_ids or not detections:
            return [], track_ids.copy(), list(range(len(detections)))

        track_boxes = np.stack([self.tracks[i].xyxy for i in track_ids]).astype(np.float32)
        det_boxes = np.stack([d.xyxy for d in detections]).astype(np.float32)
        local_matches, unmatched_tracks_local, unmatched_dets = greedy_iou_assignment(
            track_boxes,
            det_boxes,
            threshold,
        )
        matches = [(track_ids[t], d) for t, d in local_matches]
        unmatched_tracks = [track_ids[i] for i in unmatched_tracks_local]
        return matches, unmatched_tracks, unmatched_dets

    def _start_track(self, detection: Detection) -> None:
        track = Track(
            track_id=self.next_id,
            xyxy=detection.xyxy.astype(np.float32),
            confidence=detection.confidence,
            class_id=detection.class_id,
        )
        track.history = track.history.__class__(maxlen=self.config.history)
        track.append_center()
        self.next_id += 1
        self.tracks.append(track)

    def _update_track(self, track: Track, detection: Detection) -> None:
        new_box = detection.xyxy.astype(np.float32)
        track.velocity = 0.75 * track.velocity + 0.25 * (new_box - track.xyxy)
        track.xyxy = new_box
        track.confidence = detection.confidence
        track.class_id = detection.class_id
        track.hits += 1
        track.lost = 0
        track.append_center()


