from __future__ import annotations

import numpy as np


def clip_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def greedy_iou_assignment(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    ious = box_iou_matrix(track_boxes, det_boxes)
    pairs: list[tuple[float, int, int]] = []
    for t in range(ious.shape[0]):
        for d in range(ious.shape[1]):
            if ious[t, d] >= threshold:
                pairs.append((float(ious[t, d]), t, d))
    pairs.sort(reverse=True, key=lambda item: item[0])

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, t, d in pairs:
        if t in used_tracks or d in used_dets:
            continue
        used_tracks.add(t)
        used_dets.add(d)
        matches.append((t, d))

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in used_tracks]
    unmatched_dets = [i for i in range(len(det_boxes)) if i not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        denom = np.clip(areas[i] + areas[order[1:]] - inter, 1e-6, None)
        iou = inter / denom
        order = order[1:][iou <= iou_threshold]
    return keep


def affine_warp_box(xyxy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    if affine.shape != (2, 3):
        return xyxy.copy()
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float32,
    )
    warped = corners @ affine.T
    return np.array(
        [warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()],
        dtype=np.float32,
    )


