"""Stage 2 preprocessing for DeepPCB and PKU-Market-PCB."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    COMMON_CLASS_NAMES,
    COMMON_CLASS_TO_ID,
    IMAGE_SIZE,
    IMAGE_SUFFIXES,
    PATHS,
    PROJECT_ROOT,
    SPLITS,
    ProjectPaths,
)


DEEP_CLASS_TO_COMMON = {
    "open": "Open_circuit",
    "short": "Short",
    "mousebite": "Mouse_bite",
    "spur": "Spur",
    "copper": "Spurious_copper",
    "pin-hole": "Missing_hole",
}
PKU_FILENAME_CLASS_TOKENS = {
    "missing_hole": "Missing_hole",
    "mouse_bite": "Mouse_bite",
    "open_circuit": "Open_circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious_copper": "Spurious_copper",
}


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _validate_roots(paths: ProjectPaths) -> None:
    for name, root in (("DeepPCB", paths.deep_pcb), ("PKU", paths.pku)):
        missing = [
            root / split / folder
            for split in SPLITS
            for folder in ("images", "labels")
            if not (root / split / folder).is_dir()
        ]
        if missing:
            raise FileNotFoundError(f"{name} dataset is incomplete; missing {missing[0]}")


def _deep_class_names(root: Path) -> list[str]:
    return [
        name.strip(" '\"")
        for name in (root / "Class.txt").read_text(encoding="utf-8").strip().split(",")
    ]


def read_deep_boxes(label_path: Path, class_names: list[str]) -> list[dict[str, Any]]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain x1 y1 x2 y2 class_id")
        try:
            x1, y1, x2, y2, class_id = map(int, values)
        except ValueError as error:
            raise ValueError(f"Invalid DeepPCB annotation at {label_path}:{line_number}") from error
        if not 1 <= class_id <= len(class_names) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid DeepPCB box at {label_path}:{line_number}: {values}")
        source_class = class_names[class_id - 1]
        common_class = DEEP_CLASS_TO_COMMON[source_class]
        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "source_class": source_class,
            "class_name": common_class,
            "class_id": COMMON_CLASS_TO_ID[common_class],
        })
    return boxes


def pku_class_from_filename(image_path: Path) -> str:
    padded_stem = f"_{image_path.stem.lower()}_"
    matches = [
        common_class
        for token, common_class in PKU_FILENAME_CLASS_TOKENS.items()
        if f"_{token}_" in padded_stem
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot determine one PKU class from filename: {image_path.name}")
    return matches[0]


def _save_resized_rgb(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS).save(
        output_path, format="PNG"
    )


def _deep_pcb_samples(paths: ProjectPaths, class_names: list[str]) -> list[dict[str, Any]]:
    samples = []
    for split in SPLITS:
        images_dir = paths.deep_pcb / split / "images"
        labels_dir = paths.deep_pcb / split / "labels"
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing DeepPCB label: {label_path}")
            boxes = read_deep_boxes(label_path, class_names)
            with Image.open(image_path) as source:
                source_image = source.convert("RGB")
                original_width, original_height = source_image.size
                for box_index, box in enumerate(boxes):
                    x1 = max(0, min(box["x1"], original_width - 1))
                    y1 = max(0, min(box["y1"], original_height - 1))
                    x2 = max(x1 + 1, min(box["x2"], original_width))
                    y2 = max(y1 + 1, min(box["y2"], original_height))
                    output_path = (
                        paths.preprocessed / "DeepPCB" / split / box["class_name"]
                        / f"{image_path.stem}_box{box_index:03d}.png"
                    )
                    _save_resized_rgb(source_image.crop((x1, y1, x2, y2)), output_path)
                    samples.append({
                        "dataset": "DeepPCB",
                        "split": split,
                        "source_image": _relative_path(image_path),
                        "source_label": _relative_path(label_path),
                        "output_image": _relative_path(output_path),
                        "source_class": box["source_class"],
                        "class_id": box["class_id"],
                        "class_name": box["class_name"],
                        "crop_mode": "bounding_box",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "original_width": original_width,
                        "original_height": original_height,
                        "output_width": IMAGE_SIZE,
                        "output_height": IMAGE_SIZE,
                    })
    return samples


def _pku_samples(paths: ProjectPaths) -> list[dict[str, Any]]:
    samples = []
    for split in SPLITS:
        images_dir = paths.pku / split / "images"
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in image_paths:
            common_class = pku_class_from_filename(image_path)
            output_path = (
                paths.preprocessed / "PKU-Market-PCB" / split / common_class
                / f"{image_path.stem}.png"
            )
            with Image.open(image_path) as source:
                original_width, original_height = source.size
                _save_resized_rgb(source, output_path)
            samples.append({
                "dataset": "PKU-Market-PCB",
                "split": split,
                "source_image": _relative_path(image_path),
                "source_label": "",
                "output_image": _relative_path(output_path),
                "source_class": common_class,
                "class_id": COMMON_CLASS_TO_ID[common_class],
                "class_name": common_class,
                "crop_mode": "already_cropped",
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
                "original_width": original_width,
                "original_height": original_height,
                "output_width": IMAGE_SIZE,
                "output_height": IMAGE_SIZE,
            })
    return samples


def run_preprocessing(paths: ProjectPaths = PATHS) -> pd.DataFrame:
    """Create normalized images and return the relative-path metadata table."""
    _validate_roots(paths)
    class_names = _deep_class_names(paths.deep_pcb)
    deep_samples = _deep_pcb_samples(paths, class_names)
    pku_samples = _pku_samples(paths)
    metadata = pd.DataFrame(deep_samples + pku_samples)
    paths.metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(paths.metadata, index=False, encoding="utf-8")
    return metadata


def validate_metadata(metadata: pd.DataFrame, paths: ProjectPaths = PATHS) -> None:
    """Validate the generated metadata and representative output images."""
    required_columns = {
        "dataset", "split", "output_image", "class_id", "class_name", "crop_mode",
        "output_width", "output_height",
    }
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")
    if set(metadata["class_id"]) != set(range(len(COMMON_CLASS_NAMES))):
        raise ValueError("Metadata does not contain all six class IDs")
    if set(metadata["class_name"]) != set(COMMON_CLASS_NAMES):
        raise ValueError("Metadata does not contain all six class names")
    if set(metadata["split"]) != set(SPLITS):
        raise ValueError("Metadata does not preserve all splits")
    output_paths = [PROJECT_ROOT / path for path in metadata["output_image"]]
    if not all(path.is_file() for path in output_paths):
        raise FileNotFoundError("At least one generated output image is missing")
    if not (metadata["output_width"] == IMAGE_SIZE).all() or not (metadata["output_height"] == IMAGE_SIZE).all():
        raise ValueError("Not all outputs have the configured image size")
    if not paths.metadata.is_file():
        raise FileNotFoundError(paths.metadata)
