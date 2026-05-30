from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from PIL import Image


PERSON_CLASS_IDS = {1, 2}  # VisDrone: 1=pedestrian, 2=people


@dataclass(slots=True)
class MotObject:
    frame: int
    target_id: int
    x: float
    y: float
    w: float
    h: float
    score: float
    class_id: int
    truncation: float
    occlusion: int


@dataclass(slots=True)
class MotSequence:
    name: str
    image_dir: Path
    annotation_path: Path | None


def iter_mot_sequences(root: Path) -> list[MotSequence]:
    sequences_dir = root / "sequences"
    annotations_dir = root / "annotations"
    if not sequences_dir.exists():
        raise FileNotFoundError(f"Expected VisDrone MOT sequences directory: {sequences_dir}")

    sequences: list[MotSequence] = []
    for seq_dir in sorted(p for p in sequences_dir.iterdir() if p.is_dir()):
        ann = annotations_dir / f"{seq_dir.name}.txt"
        sequences.append(MotSequence(seq_dir.name, seq_dir, ann if ann.exists() else None))
    return sequences


def read_mot_annotation(path: Path) -> list[MotObject]:
    objects: list[MotObject] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 10:
            raise ValueError(f"{path}:{line_no} expected 10 MOT columns, got {len(parts)}")
        objects.append(
            MotObject(
                frame=int(float(parts[0])),
                target_id=int(float(parts[1])),
                x=float(parts[2]),
                y=float(parts[3]),
                w=float(parts[4]),
                h=float(parts[5]),
                score=float(parts[6]),
                class_id=int(float(parts[7])),
                truncation=float(parts[8]),
                occlusion=int(float(parts[9])),
            )
        )
    return objects


def convert_mot_to_yolo_person_dataset(
    mot_root: Path,
    output_root: Path,
    split_name: str = "val",
    min_box_area: float = 9.0,
    copy_images: bool = True,
) -> Path:
    """Convert VisDrone-MOT annotations to a one-class YOLO dataset.

    The challenge asks for persons, so VisDrone ``pedestrian`` and ``people`` are
    merged into one ``person`` class and all other object categories are skipped.
    """

    images_out = output_root / "images" / split_name
    labels_out = output_root / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    for seq in iter_mot_sequences(mot_root):
        if seq.annotation_path is None:
            continue
        by_frame: dict[int, list[MotObject]] = {}
        for obj in read_mot_annotation(seq.annotation_path):
            if obj.class_id not in PERSON_CLASS_IDS:
                continue
            if obj.w * obj.h < min_box_area:
                continue
            by_frame.setdefault(obj.frame, []).append(obj)

        for image_path in sorted(seq.image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            frame_id = _frame_number(image_path)
            out_stem = f"{seq.name}_{image_path.stem}"
            out_image = images_out / f"{out_stem}{image_path.suffix.lower()}"
            out_label = labels_out / f"{out_stem}.txt"
            width, height = Image.open(image_path).size

            rows: list[str] = []
            for obj in by_frame.get(frame_id, []):
                xc = (obj.x + obj.w * 0.5) / width
                yc = (obj.y + obj.h * 0.5) / height
                bw = obj.w / width
                bh = obj.h / height
                if bw <= 0 or bh <= 0:
                    continue
                rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            out_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            if rows:
                label_count += len(rows)
            if copy_images:
                shutil.copy2(image_path, out_image)
            image_count += 1

    yaml_path = output_root / "visdrone_person.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                f"train: images/{split_name}",
                f"val: images/{split_name}",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Converted {image_count} images and {label_count} person boxes to {output_root}")
    return yaml_path


def _frame_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer frame number from {path.name}")
    return int(digits)


