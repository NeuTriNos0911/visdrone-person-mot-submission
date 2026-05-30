from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Export trained YOLO weights for edge deployment.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript", "openvino"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run `pip install -r requirements.txt`.") from exc

    model = YOLO(args.weights)
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        simplify=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


