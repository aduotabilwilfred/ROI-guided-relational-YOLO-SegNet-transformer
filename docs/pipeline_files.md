# Pipeline files and what they do

Replication of Natarajan et al. (2026), "ROI-guided relational YOLO-SegNet
transformer for lightweight bone tumor segmentation and classification from
X-ray images." This covers the data and baseline-training pipeline. The RTrB
transformer block and FHEO tuning are not yet built.

## The flow, in order

Raw dataset (809 X-ray images, one class "cancer", axis-aligned box labels in
YOLOv8-OBB format) runs through: label-to-mask conversion, fold splitting,
per-fold OSGDF calibration, materialisation into Ultralytics format, then
five-fold training with pooled evaluation.

## Data preparation

**prepare_dataset.py** — Converts the raw box labels into segmentation masks
and YOLO bboxes, letterboxing every image to 512x512. Has an `--inspect` mode
that reports label contents, class balance, image sizes, and any parse
problems without writing anything. This is the tool that first revealed the
data is single-class, box-annotated, and roughly half background.

**make_folds.py** — Pools all 809 images and assigns them to five
cross-validation folds, stratified by tumour presence so each fold has the same
tumour/background ratio. Groups by source image so augmented copies (none in
the current data) could never straddle a fold boundary. Writes a manifest CSV;
copies no images. Runs on the raw dataset, so fold assignment is independent of
any preprocessing choice.

## Preprocessing (OSGDF)

**osgdf_preprocessing.py** — The core denoising library: a 2D Savitzky-Golay
filter, blind SNR and noise metrics, and the Tunicate Swarm Optimization that
tunes the filter's window size and polynomial order. Used by the calibration
and diagnostic steps.

**run_osgdf.py** — Standalone driver that calibrates and denoises a single
folder in one pass. Superseded by the per-fold calibration below and not part
of the cross-validation pipeline; kept as a manual tool.

**fold_calibration.py** — For each fold, tunes the OSGDF filter on that fold's
training images only (never validation or test), and writes the chosen window
and polynomial order per fold to a JSON. This is the step that keeps
preprocessing free of information leaked from held-out data. On the real data
every fold selected window 5, polynomial order 4.

**osgdf_diagnostic.py** — Observational tool. Applies a fold's chosen filter to
real images and writes before/after/residual panels plus SNR numbers, so the
filter's effect can be seen rather than trusted. Used to confirm window 5 gives
a real ~8.8 dB gain that removes noise-like texture without carving out
structure.

## Bridge to the model

**build_fold_ultralytics.py** — Materialises each fold into the folder layout
Ultralytics expects: denoised images written to disk (the paper denoises
offline before training), raw labels copied alongside, and a per-fold
dataset.yaml. Labels are copied rather than symlinked, because symlinks fail
silently on WSL/NTFS and make Ultralytics read every label as missing. This is
the step that connects our preprocessing to the training package.

## Training and evaluation

**train_folds.py** — Trains YOLOv8-seg across all five folds with the paper's
hyperparameters (50 epochs, AdamW, learning rate 1e-4, cosine schedule, batch
16) and only the paper's augmentations (rotation, horizontal flip, zoom, mild
brightness), with all other Ultralytics augmentations disabled. Then evaluates
by pooling predictions across the five test folds and computing a single Dice
over all 809 images, matching the paper rather than averaging five per-fold
scores. Portable: device auto-detects, paths are arguments, folds regenerate
from the manifest seed, so it runs unchanged on a collaborator's GPU machine.

## Orchestration and reference

**dvc.yaml** — Wires prepare_dataset, make_folds, and fold_calibration into a
reproducible DVC pipeline. The Ultralytics build and training are run manually
(GPU machine), not through DVC.

**paper_reference.md** — Summary of the paper, its reported results, and every
gap between what it claims and what the data supports.

## Two caveats that ride along with every result

The labels are axis-aligned boxes, not traced outlines, so any Dice measures
agreement with boxes rather than tumour-boundary delineation.

The dataset has one class, so "classification" is binary tumour-presence
derived from whether any box exists, not the paper's normal/benign/malignant.
