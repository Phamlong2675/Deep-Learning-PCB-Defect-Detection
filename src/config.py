"""Project paths and shared preprocessing configuration."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 224
COMMON_CLASS_NAMES = (
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
)
COMMON_CLASS_TO_ID = {name: index for index, name in enumerate(COMMON_CLASS_NAMES)}


def _project_path(environment_name: str, default: str) -> Path:
    value = Path(os.environ.get(environment_name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class ProjectPaths:
    deep_pcb: Path
    pku: Path
    preprocessed: Path

    @property
    def metadata(self) -> Path:
        return self.preprocessed / "preprocessing_metadata.csv"


PATHS = ProjectPaths(
    deep_pcb=_project_path("PCB_DATA_ROOT", "DeepPCB"),
    pku=_project_path("PKU_DATA_ROOT", "PKU-Market-PCB(Data enhanced version)"),
    preprocessed=_project_path("PCB_PREPROCESSED_ROOT", "preprocessed_data"),
)
