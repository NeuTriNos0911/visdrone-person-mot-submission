# Attribution

This submission uses the following external resources:

- VisDrone 2019 MOT validation dataset for drone video frames and annotations.
- Ultralytics YOLO11n weights as the compact detector backbone for the submitted run.
- OpenCV for video IO, optical flow, affine camera-motion estimation, and drawing.
- NumPy, Pillow, PyYAML, tqdm, requests, and Hugging Face Hub utilities for supporting data and runtime tasks.

Custom work in this package includes:

- Person-only VisDrone MOT to YOLO conversion.
- Drone-focused tracking pipeline with high/low confidence association.
- Camera-motion compensation before track matching.
- Video rendering with persistent IDs and trajectory tails.
- Full validation run scripts, FPS logging, and edge-export workflow.

