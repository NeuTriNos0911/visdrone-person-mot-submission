# Person Detection And Tracking On VisDrone MOT

## Goal

The task is to detect and track people from moving drone footage. The submitted output contains videos with bounding boxes, unique person IDs, and trajectory tails for every sequence in the VisDrone 2019 MOT validation archive.

## Implementation Summary

The pipeline has three main parts:

1. A compact YOLO11n detector for person candidates.
2. A two-pass tracker inspired by ByteTrack, using high-confidence detections for new tracks and lower-confidence detections only to recover existing tracks.
3. Global camera-motion compensation from sparse optical flow, used before IOU matching so drone ego-motion causes fewer ID breaks.

The package also includes a VisDrone conversion utility. It reads MOT annotations and converts the `pedestrian` and `people` categories into a single YOLO class named `person`.

## Why This Design

Drone footage is different from static CCTV: objects are small, the viewpoint changes quickly, and background motion can be larger than object motion. I kept the detector lightweight and put extra effort into the association step because many ID switches come from camera movement rather than from the detector alone.

For the submitted CPU run, I used:

```text
weights: yolo11n.pt
model size: 5.61 MB
input size: 960
confidence threshold: 0.08
camera-motion compensation: enabled
tiling: disabled
device: CPU
```

The code supports overlapping tiled inference. I left it disabled for the full CPU benchmark because it increases recall on very small people but makes CPU runtime much slower. On GPU or Jetson, tiled inference can be enabled selectively for high-altitude clips.

## Dataset Handling

Input archive:

```text
VisDrone2019-MOT-val.zip
```

Converted data summary:

```text
validation sequences: 7
frames processed: 2,846
person boxes after class merge: 50,312
merged source classes: pedestrian, people
```

## Tracking Details

Each frame is processed as follows:

1. Run detector and keep person detections.
2. Estimate global frame-to-frame motion with optical flow and an affine transform.
3. Predict active track locations using velocity and camera-motion compensation.
4. Match high-confidence detections to tracks by IOU.
5. Use lower-confidence detections to recover unmatched tracks.
6. Keep unmatched tracks alive for a short buffer to handle occlusion.
7. Draw bounding boxes, stable IDs, and recent center-point trails.

This is intentionally simple enough to run on edge hardware while still addressing the main drone-specific issue: moving-camera association noise.

## Full Validation Results

All videos and JSON files are in:

```text
results\full_validation
```

| Sequence | Frames | FPS | Output |
|---|---:|---:|---|
| uav0000086_00000_v | 464 | 11.990 | results\full_validation\uav0000086_00000_v.mp4 |
| uav0000117_02622_v | 349 | 7.553 | results\full_validation\uav0000117_02622_v.mp4 |
| uav0000137_00458_v | 233 | 7.680 | results\full_validation\uav0000137_00458_v.mp4 |
| uav0000182_00000_v | 363 | 14.382 | results\full_validation\uav0000182_00000_v.mp4 |
| uav0000268_05773_v | 978 | 5.170 | results\full_validation\uav0000268_05773_v.mp4 |
| uav0000305_00000_v | 184 | 9.560 | results\full_validation\uav0000305_00000_v.mp4 |
| uav0000339_00001_v | 275 | 11.050 | results\full_validation\uav0000339_00001_v.mp4 |

Overall:

```text
2,846 frames / 373.7842 seconds = 7.614 FPS
```

Benchmark hardware:

```text
OS: Windows 11
CPU: Intel64 Family 6 Model 186 Stepping 2
Python: 3.12.13
Runtime: PyTorch 2.12.0 CPU build
```

## Edge Deployment Plan

For a Jetson target, I would run the fine-tuned person detector through TensorRT:

1. Train or fine-tune with `tools\train_person_detector.py`.
2. Export TensorRT FP16 using `tools\export_edge_model.py`.
3. Profile detector, optical-flow camera motion, tracker, and video IO separately.
4. Start with `imgsz=960` for speed, then move to `1280` only if small-person recall needs it.
5. Use tiled inference only on the sequences where the altitude makes people too small.

The current model is far below the 300 MB requirement, leaving room for a slightly larger detector if the target Jetson can handle it.

## Limitations And Next Steps

The submitted videos use the compact base YOLO11n weights with the custom tracking pipeline. The included dataset converter and training script are ready for a fine-tuned one-class detector, which should improve recall for small and partially occluded people. A second improvement would be adding appearance embeddings for crowded crossings, but I avoided that in the submitted run to keep the pipeline lightweight.

