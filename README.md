# PCB Defect Detection

Object-detection project for locating and classifying PCB defects. The repository currently contains the data-audit and preprocessing notebooks; model training code will be added as the training stage is implemented.

## Repository layout

```text
Data/
  README.md                 Dataset placement and format
  raw_data/                 Local source dataset, not tracked by Git
  derived/                  Generated canonical targets and audit results
  processed/
    none_640/               Shared 640px reference dataset, tracked by Git
    resize_512/             Shared 512px candidate dataset, tracked by Git
Data_preprocessing.ipynb    Validation, transforms, export, and size audit
Data_understanding.ipynb    Dataset inventory and annotation analysis
requirements.txt
```

`Data/processed/none_640/` and `Data/processed/resize_512/` are intentionally tracked so a clone contains both variants for a controlled input-size comparison. Together they are about 129 MB, with the largest individual file around 1.4 MB, so Git LFS is not needed for these exports. `Data/raw_data/`, `Data/derived/`, and any other processed variants stay local/ignored. Share the licensed raw source through the team's approved storage. Do not commit model weights or run outputs.

## Setup

Use Python 3.10 or later, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Dataset

Place the raw dataset under `Data/raw_data/`, or set `PCB_DATA_ROOT` to another local path. Expected layout and annotation format are documented in [Data/README.md](Data/README.md). The notebooks can also be pointed at derived inputs/outputs with `PCB_TARGETS_CSV`, `PCB_CLASS_MAPPING_CSV`, and `PCB_PROCESSED_ROOT`.

## Workflow

1. Contributors who have the raw dataset can run `Data_understanding.ipynb` and `Data_preprocessing.ipynb` to audit/rebuild local artifacts.
2. For a 512px run, use `Data/processed/resize_512/data.yaml`; for the 640px reference, use `Data/processed/none_640/data.yaml`. Both point to included `train`, `valid`, and `test` image/label folders.
3. Train the same model/config on both variants, select size from validation AP/recall and speed, and keep test untouched until final evaluation.

The 512px variant is a direct resize; the 640px variant preserves source images byte-for-byte. Both include identical splits and class mapping. The processed datasets are ready in the clone, but a model training script and trainer-specific dependencies have not yet been added to this repository.

