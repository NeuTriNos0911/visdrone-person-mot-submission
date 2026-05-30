from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from drone_person_mot.geometry import box_iou_matrix, greedy_iou_assignment
from drone_person_mot.tracker import AerialTracker, TrackerConfig
from drone_person_mot.types import Detection
from drone_person_mot.visdrone import read_mot_annotation


def test_iou_and_assignment() -> None:
    tracks = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=np.float32)
    detections = np.array([[1, 1, 11, 11], [300, 300, 310, 310]], dtype=np.float32)
    iou = box_iou_matrix(tracks, detections)
    assert iou.shape == (2, 2)
    matches, unmatched_tracks, unmatched_dets = greedy_iou_assignment(tracks, detections, 0.5)
    assert matches == [(0, 0)]
    assert unmatched_tracks == [1]
    assert unmatched_dets == [1]


def test_tracker_keeps_id_across_small_motion() -> None:
    tracker = AerialTracker(TrackerConfig(use_camera_motion=False, max_lost=2))
    first = [Detection(np.array([10, 10, 30, 40], dtype=np.float32), 0.9, 0)]
    tracks = tracker.update(first)
    assert len(tracks) == 1
    track_id = tracks[0].track_id

    second = [Detection(np.array([12, 11, 32, 41], dtype=np.float32), 0.88, 0)]
    tracks = tracker.update(second)
    assert len(tracks) == 1
    assert tracks[0].track_id == track_id


def test_visdrone_mot_parser() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "seq.txt"
        path.write_text("1,7,10,20,30,40,1,1,0,0\n", encoding="utf-8")
        rows = read_mot_annotation(path)
        assert len(rows) == 1
        assert rows[0].frame == 1
        assert rows[0].target_id == 7
        assert rows[0].class_id == 1


if __name__ == "__main__":
    test_iou_and_assignment()
    test_tracker_keeps_id_across_small_motion()
    test_visdrone_mot_parser()
    print("ok")


