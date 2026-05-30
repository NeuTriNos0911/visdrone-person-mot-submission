from __future__ import annotations

from pathlib import Path
import tempfile

from PIL import Image

from drone_person_mot.visdrone import convert_mot_to_yolo_person_dataset


def test_convert_mot_to_yolo_person_dataset() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "VisDrone2019-MOT-val"
        seq_dir = root / "sequences" / "uav000001"
        ann_dir = root / "annotations"
        seq_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        Image.new("RGB", (100, 50)).save(seq_dir / "0000001.jpg")
        (ann_dir / "uav000001.txt").write_text(
            "\n".join(
                [
                    "1,1,10,10,20,20,1,1,0,0",
                    "1,2,50,10,20,20,1,4,0,0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        out = Path(td) / "out"
        yaml_path = convert_mot_to_yolo_person_dataset(root, out)
        assert yaml_path.exists()
        label = (out / "labels" / "val" / "uav000001_0000001.txt").read_text(encoding="utf-8")
        assert label.startswith("0 ")
        assert len(label.strip().splitlines()) == 1


if __name__ == "__main__":
    test_convert_mot_to_yolo_person_dataset()
    print("ok")


