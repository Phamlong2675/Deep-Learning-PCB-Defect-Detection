# PCB Defect Classification

This project prepares DeepPCB and PKU-Market-PCB for six-class PCB defect classification. The current work covers data understanding and Stage 2 preprocessing. The resulting metadata and images are ready for Stage 3 model training.

## Current Structure

```text
.
|-- Data_understanding.ipynb
|-- Data_Preprocessing.ipynb
|-- DeepPCB/                              # raw DeepPCB dataset
|-- PKU-Market-PCB(Data enhanced version)/ # raw PKU dataset
|-- preprocessed_data/
|   |-- DeepPCB/                          # generated 224x224 crops
|   |-- PKU-Market-PCB/                   # generated 224x224 images
|   `-- preprocessing_metadata.csv
|-- src/
|   |-- __init__.py
|   |-- config.py
|   `-- preprocessing.py
|-- requirements.txt
`-- README.md
```

## Dataset Classes

Both datasets are mapped to the same class IDs:

| ID | Class |
|---:|---|
| 0 | `Missing_hole` |
| 1 | `Mouse_bite` |
| 2 | `Open_circuit` |
| 3 | `Short` |
| 4 | `Spur` |
| 5 | `Spurious_copper` |

DeepPCB source labels are mapped as follows:

```text
open       -> Open_circuit
short      -> Short
mousebite  -> Mouse_bite
spur       -> Spur
copper     -> Spurious_copper
pin-hole   -> Missing_hole
```

PKU class labels are inferred from the image filename, for example `missing_hole`, `mouse_bite`, and `spurious_copper`.

## Data Understanding

Run `Data_understanding.ipynb` to inspect:

- image and annotation counts for both datasets;
- annotation format and invalid boxes;
- image dimensions and grayscale images;
- class, box, and split distributions;
- box-size statistics and class co-occurrence;
- representative annotated samples;
- duplicate and cross-split leakage checks for DeepPCB.

The notebook keeps the exploratory analysis separate from the reusable preprocessing code.

## Stage 2 Preprocessing

Run `Data_Preprocessing.ipynb` from the project root. The notebook calls `src.preprocessing.run_preprocessing()` and then validates the generated output.

The pipeline:

1. Reads `train`, `valid`, and `test` without reshuffling or re-splitting.
2. Reads DeepPCB pixel boxes in the format `x1 y1 x2 y2 class_id`.
3. Crops one sample per DeepPCB defect box.
4. Keeps PKU defect images as already-cropped samples.
5. Converts every sample to RGB.
6. Resizes every sample to `224x224` and saves PNG files.
7. Writes one metadata row per output sample.

The current run produces `13,517` samples:

```text
DeepPCB:          10,013 bounding-box crops
PKU-Market-PCB:    3,504 already-cropped images
```

Metadata is stored in `preprocessed_data/preprocessing_metadata.csv`. Paths in this file are relative to the project root, so they remain portable between machines using the same project structure.

Important metadata columns include:

- `dataset`, `split`
- `source_image`, `source_label`
- `output_image`
- `class_id`, `class_name`, `source_class`
- `crop_mode`
- original box coordinates and image dimensions
- output dimensions

## Installation

Use Python 3.10 or newer in an isolated environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pcb-classification
```

PyTorch installation can depend on the machine's CPU/CUDA setup. Install the appropriate PyTorch build before running Stage 3 if the default package is not suitable.

## Configurable Paths

The default locations are relative to the project root. They can be overridden with environment variables:

```powershell
$env:PCB_DATA_ROOT = "path\to\DeepPCB"
$env:PKU_DATA_ROOT = "path\to\PKU-Market-PCB(Data enhanced version)"
$env:PCB_PREPROCESSED_ROOT = "path\to\preprocessed_data"
```

## Stage 3 Handoff

Stage 3 can load `preprocessing_metadata.csv`, filter by `split`, read `output_image`, and use `class_id` as the single-label target. Because each output image contains one defect, the classification model should use six logits with `CrossEntropyLoss`.

The `dataset` column should be retained during evaluation so results can be reported for the combined test set and separately for DeepPCB and PKU-Market-PCB.

Raw datasets and generated image folders are local data artifacts and are ignored by Git. Team members need access to the raw datasets or a shared copy of `preprocessed_data/` before running Stage 2 or Stage 3.
