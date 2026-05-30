from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a compact YOLO detector on VisDrone persons.")
    parser.add_argument("--data", default="data/visdrone_person/visdrone_person.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="visdrone_person_yolo11n")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run `pip install -r requirements.txt`.") from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=5,
        close_mosaic=5,
        cos_lr=True,
        degrees=3.0,
        translate=0.08,
        scale=0.6,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.05,
        copy_paste=0.05,
        workers=4,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


