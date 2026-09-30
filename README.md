# PCB Multi-Label Classification

This project prepares DeepPCB and PKU-Market-PCB for PCB defect classification.

## Project layout

```text
.
├── Data_understanding.ipynb
├── Data_Preprocessing.ipynb
├── DeepPCB/                         # local raw data
├── PKU-Market-PCB(Data enhanced version)/
├── preprocessed_data/
│   └── preprocessing_metadata.csv
├── src/
│   ├── config.py
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer and create an isolated environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pcb-classification
```

PyTorch installation can vary by CPU/CUDA configuration. Install the appropriate PyTorch build from the official selector before running Stage 3.

## Run Stage 2

Open `Data_Preprocessing.ipynb` from the project root and run all cells. The pipeline:

- maps both datasets to six common classes;
- crops DeepPCB boxes and keeps PKU defect crops;
- converts images to RGB and resizes them to `224x224`;
- preserves `train`, `valid`, and `test` splits;
- writes relative paths to `preprocessed_data/preprocessing_metadata.csv`.

The default raw-data locations are the two folders in the project root. They can be overridden with `PCB_DATA_ROOT`, `PKU_DATA_ROOT`, and `PCB_PREPROCESSED_ROOT`.

## Sharing with teammates

Git tracks the source code, notebooks, metadata, `requirements.txt`, and this README. Raw datasets and generated PNG files are intentionally ignored because they are large.

Each teammate needs access to these local folders before running Stage 2:

```text
DeepPCB/
PKU-Market-PCB(Data enhanced version)/
```

After running Stage 2, the generated images are available under `preprocessed_data/`. For team sharing, store that folder in shared storage or use Git LFS/DVC rather than committing thousands of generated images to normal Git history.

The Stage 3 training notebook should read `preprocessed_data/preprocessing_metadata.csv` and resolve each relative `output_image` path from the project root.
