# Eastwood Otabil

ROI-guided relational YOLO-SegNet transformer for lightweight bone tumor segmentation and classification from X-ray images.

**Author:** Eastwood & Otabil

## Project Organization

```
.
├── .dvc
├── .dvcignore
├── .git
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── processed
│   └── raw
├── docs
├── dvc.lock
├── dvc.yaml
├── ml-env
├── models
├── notebooks
├── outputs
├── pyproject.toml
├── references
├── reports
├── requirements.in
├── requirements.txt
├── setup.cfg
└── src
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── build_fold_ultralytics.py
    │   ├── fold_calibration.py
    │   ├── make_folds.py
    │   ├── osgdf_preprocessing.py
    │   ├── prepare_dataset.py
    │   └── run_osgdf.py
    ├── features
    │   └── __init__.py
    ├── models
    │   ├── __init__.py
    │   ├── fheo.py
    │   ├── train.py
    │   └── train_handoff.py
    └── utils
        ├── __init__.py
        ├── allmlp_decoder.py
        ├── bilra_attention.py
        ├── efficient_attention.py
        ├── losses.py
        ├── osgdf_diagnostic.py
        ├── relational_head.py
        ├── roi_partition.py
        └── rtrb.py
```

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) — fast Python package manager (`pip install uv`)
- [DVC](https://dvc.org/) — data version control (`pip install dvc`)


### Setup

```bash
make setup        # create virtual environment and install all dependencies
```

`make setup` recompiles `requirements.txt` from `requirements.in` on your own
machine via `uv pip compile`, so `torch`/`ultralytics` resolve to whatever
build (CPU or a CUDA build matching your GPU/driver) works locally. Always
(re)run `make setup` on each machine rather than doing a plain
`pip install -r requirements.txt` from someone else's committed lock file —
it was generated for their hardware, not necessarily yours.

### Get the Dataset

There is **no shared DVC remote with the data pushed to it** — `data/`,
`outputs/`, and `runs/` are all git-ignored, and `data/raw/actual_dataset` is
a plain (non-DVC-tracked) pipeline dependency. Every clone must fetch the
raw dataset itself:

1. Download **"Bone_Tumor - vdataset bone_tumor-0krk1"** from Roboflow
   (YOLOv8 Oriented Object Detection export, 809 images — see
   `data/raw/actual_dataset/README.roboflow.txt` for the export this repo
   was built against).
2. Unzip it into `data/raw/actual_dataset` so it contains `train/`, `valid/`,
   and `data.yaml` directly (matching the layout Roboflow exports).
3. Verify it's the same data everyone else is using:
   ```bash
   dvc status
   ```
   `dvc.lock` records the expected hash for `data/raw/actual_dataset`
   (`b104f56072fa7b946c33130af46da51c.dir`, 1622 files). If `dvc status`
   reports that dependency as changed, your download differs from the one
   the pipeline outputs were built from — the pipeline will still run, but
   results may not match exactly.

### Run the Data Pipeline

```bash
make data            # dvc repro build_fold_ultralytics — CPU only, no GPU needed
```

Runs `prepare_dataset → make_folds → fold_calibration → build_fold_ultralytics`.
All of it is OpenCV/NumPy/SciPy, so it reproduces identically on a GPU-less machine.

### Train the Two-Stage Model

Training is split into two GPU-bound stages:

1. **Stage 1 — Train Detector (YOLOv8-seg):**
   ```bash
   make train_detector  # dvc repro train_detector — trains YOLOv8-seg per fold
   ```
   Produces the frozen detector weights (`best.pt` per fold) required by the relational head.

2. **Stage 2 — Train Relational Segmentation Head:**
   ```bash
   make train_head      # dvc repro train_head — trains BiLRA + RTrB + AllMLPDecoder head
   ```
   Hooks neck features from the frozen detector and trains the relational head end-to-end.

3. **Optional — Hyperparameter Tuning (FHEO):**
   ```bash
   make tune_fheo       # dvc repro tune_fheo — runs Fire Hawk + Election Optimizer
   ```
   Evaluates hyperparameter candidates on the `val` split for fold 0 and exports `outputs/fheo/best_params.json`.

### Run All Default Pipeline Steps

```bash
make all             # runs: data → train_detector → train_head
```

### Other Commands

```bash
make clean           # remove cache directories and virtual environment
```

## Data & Model Versioning

- **Git** tracks the directory structures of `data/` and `models/`, but their actual contents (large datasets and model weights) are excluded from Git using `.gitignore`. This ensures the folder hierarchy is preserved without committing heavy binary files.
- **DVC (Data Version Control)** is used to track and version the actual dataset files and model checkpoints.

> [!NOTE]
> `.dvc/config` points at a Google Drive remote, but it requires a
> `service_account.json` that is intentionally git-ignored and not shared.
> `dvc push`/`dvc pull` won't work for collaborators, since
> `dvc repro` never needs a remote; it just reruns pipeline stages locally
> based on file hashes. Data is reproduced.
