from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_person_mot.visdrone import convert_mot_to_yolo_person_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert VisDrone MOT val to one-class YOLO labels.")
    parser.add_argument("--mot-root", required=True, help="Path to VisDrone2019-MOT-val.")
    parser.add_argument("--out", default="data/visdrone_person")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--min-box-area", type=float, default=9.0)
    args = parser.parse_args()

    yaml_path = convert_mot_to_yolo_person_dataset(
        mot_root=Path(args.mot_root),
        output_root=Path(args.out),
        split_name=args.split_name,
        min_box_area=args.min_box_area,
    )
    print(f"YOLO data config: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

