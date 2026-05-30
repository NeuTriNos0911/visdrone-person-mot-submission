from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .benchmark import FpsMeter
from .detector import DetectorConfig, YOLODetector
from .tracker import AerialTracker, TrackerConfig
from .viz import draw_hud, draw_tracks


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VisDrone Person MOT person tracking.")
    parser.add_argument("--source", required=True, help="Video path or VisDrone sequence directory.")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO weights path or model name.")
    parser.add_argument("--output", default="outputs/drone_person_mot_output.mp4")
    parser.add_argument("--stats", default=None, help="Optional JSON stats output path.")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cpu, cuda:0.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--keep-class-ids", default="0", help="Comma-separated class IDs to keep.")
    parser.add_argument("--tile", action="store_true", help="Enable overlapping tiled inference.")
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--no-camera-motion", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for image-sequence inputs.")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit. 0 means all frames.")
    parser.add_argument("--show-hud", action="store_true", help="Overlay measured FPS.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed. Run `pip install -r requirements.txt`.") from exc

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats) if args.stats else output.with_suffix(".json")

    keep_class_ids = tuple(int(part.strip()) for part in args.keep_class_ids.split(",") if part.strip())
    detector = YOLODetector(
        DetectorConfig(
            weights=args.weights,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            keep_class_ids=keep_class_ids,
            use_tiling=args.tile,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
    )
    tracker = AerialTracker(TrackerConfig(use_camera_motion=not args.no_camera_motion))

    reader = FrameReader(source, sequence_fps=args.fps)
    writer = None
    meter = FpsMeter()
    max_frames = args.max_frames if args.max_frames > 0 else None

    try:
        for index, frame in enumerate(reader.frames(), start=1):
            if max_frames is not None and index > max_frames:
                break
            start = meter.start()
            detections = detector.predict(frame)
            tracks = tracker.update(detections, frame)
            meter.stop(start)

            annotated = draw_tracks(frame, tracks)
            if args.show_hud:
                annotated = draw_hud(annotated, meter.fps, len(tracks))

            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output), fourcc, reader.fps, (w, h))
            writer.write(annotated)
    finally:
        reader.close()
        if writer is not None:
            writer.release()

    stats = {
        "source": str(source),
        "output": str(output),
        "frames": meter.frames,
        "elapsed_seconds": round(meter.elapsed, 4),
        "fps": round(meter.fps, 3),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "device_arg": args.device or "auto",
        },
        "settings": {
            "weights": args.weights,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "tile": args.tile,
            "camera_motion": not args.no_camera_motion,
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


class FrameReader:
    def __init__(self, source: Path, sequence_fps: float):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required.") from exc

        self.source = source
        self._cap = None
        self._images: list[Path] = []
        self.fps = float(sequence_fps)
        if source.is_dir():
            self._images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not self._images:
                raise FileNotFoundError(f"No images found in sequence directory: {source}")
        else:
            self._cap = cv2.VideoCapture(str(source))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {source}")
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if np.isfinite(fps) and fps > 0:
                self.fps = float(fps)

    def frames(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required.") from exc

        if self._images:
            for image_path in self._images:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                yield frame
            return

        assert self._cap is not None
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


if __name__ == "__main__":
    raise SystemExit(main())


