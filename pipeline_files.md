# Pipeline files and what they do

Replication of Natarajan et al. (2026), "ROI-guided relational YOLO-SegNet
transformer for lightweight bone tumor segmentation and classification from
X-ray images." This now covers the complete pipeline: data preparation, the
frozen YOLOv8 detector, the relational segmentation head and its components,
the two-stage training loop, and FHEO hyperparameter tuning.

## The pipeline in order

A raw dataset (one class, per-instance polygon or box labels) flows through:
label-to-mask conversion, stratified folds, per-fold OSGDF calibration,
materialisation into Ultralytics format, YOLOv8 detector training, then the
relational head trained on frozen detector features, evaluated by pooled Dice
and image-level classification accuracy. FHEO optionally tunes hyperparameters.

Two raw datasets are selectable via `params.yaml`'s `dataset` key:
**actual_dataset** (809 images, Roboflow export, axis-aligned/OBB box labels
only) and **BTXRD_yolo** (3725 images after exact-duplicate removal, traced
polygon labels — see `prepare_btxrd.py` below). Everything from
`prepare_dataset.py` onward is dataset-agnostic: both are raw YOLO-label
roots (`data/raw/<dataset>/<split>/{images,labels}`), so switching `dataset`
and running `dvc repro` is sufficient to retarget the whole pipeline.

## Data preparation (src/data/)

**prepare_dataset.py** — Converts raw box/polygon labels into segmentation masks
and YOLO bboxes, letterboxing every image to 512. The label parser already
accepts polygon lines (`class x1 y1 x2 y2 ...`), not just axis-aligned boxes,
which is what makes it dataset-agnostic. Has an `--inspect` mode reporting
label contents and class balance.

**prepare_btxrd.py** — One-off adapter, not part of the shared pipeline: reads
BTXRD's raw `Annotations/*.json` (LabelMe) + `dataset.csv` and writes the same
raw YOLO-label layout `prepare_dataset.py`/`make_folds.py` expect, to
`data/raw/BTXRD_yolo`. Keeps only the traced `polygon` shapes (each annotated
image's JSON also carries coarser `rectangle` pre-annotations of the same
instances, which are dropped — bboxes are derived from the polygon downstream
instead, same as for actual_dataset's own OBB labels). BTXRD has no
patient/case ID; hashing found 21 images filed under two different
`image_id`s each, which are collapsed to one copy here so no leakage-relevant
duplicate reaches fold assignment. Also writes `metadata.csv` (age, gender,
center, malignant/benign) alongside the labels, for reporting only — nothing
downstream reads it.

**validate_btxrd.py** — Re-derives masks/bboxes from the converted labels on a
sample, writes overlay PNGs (`reports/figures/btxrd_yolo_overlay_*.png`),
smoke-tests OSGDF on BTXRD's image sizes, and checks the fold manifest for
any source split across folds.

**make_folds.py** — Pools all images of the selected dataset into five folds,
stratified by tumour presence, grouped by source image so augmented/duplicate
copies never straddle a fold. Writes a manifest CSV; copies no images. Runs on
raw data, so fold assignment is independent of preprocessing. `cv_roles(fold)`
derives the 3/1/1 train/val/test rotation.

## Preprocessing, OSGDF (src/data/ and src/utils/)

**osgdf_preprocessing.py** — The denoising library: 2D Savitzky-Golay filter,
blind SNR/noise metrics, and Tunicate Swarm Optimization that tunes the filter's
window and polynomial order.

**run_osgdf.py** — Standalone single-folder driver. Superseded by per-fold
calibration; kept as a manual tool, not in the pipeline.

**fold_calibration.py** — Tunes OSGDF per fold on that fold's training images
only (never val or test), writing window/poly per fold to JSON. On the real data
every fold chose window 5, poly 4.

**osgdf_diagnostic.py** — Observational tool: applies a fold's filter to real
images and writes before/after/residual panels plus SNR numbers.

## Bridge to the detector (src/data/)

**build_fold_ultralytics.py** — Materialises each fold into Ultralytics layout:
denoised images written to disk (the paper denoises offline), raw labels copied
(not symlinked — symlinks fail silently on WSL/NTFS), per-fold dataset.yaml. This
is what the YOLOv8 detector trains on.

## Detector training (src/models/)

**train_folds.py** — Trains YOLOv8-seg across folds with the paper's
hyperparameters (50 epochs, AdamW, lr 1e-4, cosine, batch 16) and only the
paper's augmentations. This produces the frozen detector weights (best.pt per
fold) that the relational head consumes. Also has its own pooled-Dice
evaluation for the YOLOv8-only baseline.

## Relational segmentation head components (src/utils/)

Each is a self-contained, tested module.

**bilra_attention.py** — BiFormer bi-level routing attention (BiLRA), the
backbone's efficient long-range attention. Region top-k routing plus a
depth-wise-conv local context term (Eq. 6). Faithful to the BiFormer source.

**efficient_attention.py** — SegFormer spatial-reduction self-attention (Eqs.
7-9). The reduction ratio r the paper leaves unspecified is a constructor
argument (a value FHEO can tune).

**allmlp_decoder.py** — SegFormer All-MLP decoder (Eqs. 11-14): fuses multi-
scale features and upsamples to the mask.

**rtrb.py** — The Relational Transformer Block (Eqs. 15-20). Dual-head: self-
attention over tumour tokens f_m, cross-attention where f_m queries surrounding-
bone tokens f_w. ROI-restricted: attention runs only over the masked ROI tokens
(gather/scatter), so cost scales with tumour size, not image size. Output is
2*embed_dim channels (channel-concat, Eq. 20).

**roi_partition.py** — Builds the f_m and f_w soft masks from YOLO boxes via the
dilated-ring strategy: f_m = tokens inside the box, f_w = dilated box minus box
(surrounding ring). Handles zero-box (background) and multiple boxes.

**relational_head.py** — Composes the six components into the full segmentation
head: per-scale projection, a relational stage per scale (optional BiLRA then
RTrB), and the decoder. `HeadConfig` exposes embed_dim, heads, reduction,
dilation, and `use_bilra_in_head` (on by default, switchable). Trains and learns.

**losses.py** — dice_loss, bce_dice_loss (the paper's objective), and
dice_coefficient.

## Detector-to-head handoff (src/models/)

**train_handoff.py** (a.k.a. yolo_handoff) — Wraps a trained YOLOv8, frozen, to
yield the two things the head needs: the neck feature maps P3/P4/P5 (via forward
hooks on the layers feeding the Segment head) and the detected boxes (via
predict). Frozen: eval mode, no_grad, no gradient enters YOLO. This is the
two-stage connection.

## Head training (src/models/)

**train.py** — Trains the relational head across folds on frozen detector
features. Per fold: load that fold's YOLO weights frozen, loop the fold's
denoised images, extract features+boxes, forward the head, train with BCE+Dice.
Reports pooled Dice and image-level tumour-presence classification accuracy
(not pixel accuracy) across the five test folds, matching the paper. Logs to
MLflow (SQLite backend, degrades gracefully, `--no-mlflow` to disable). Saves
per-fold head weights.

## Hyperparameter tuning (src/models/)

**fheo.py** — Fire Hawk + Election Optimizer (Eqs. 21-26), an optional tuning
tool. `tune_relational_head` uses train.py as its fitness function: each
candidate trains the head briefly on one fold and returns validation Dice.
Population and iterations are configurable so it runs at a feasible scale. The
model trains fine on the paper's hyperparameters without it; FHEO is for
improving on them for this reconstruction.

## Orchestration and reference

**dvc.yaml** — Wires the data stages (prepare, folds, calibration, ultralytics
build) plus detector and head training into a reproducible pipeline. The head-
training stage depends on the detector weights and all head component modules.
`prepare_btxrd` and `validate_btxrd` are unchained (like `tune_fheo`): run
explicitly, not part of the default `dvc repro`.

**params.yaml** — Currently one key, `dataset` (`actual_dataset` or
`BTXRD_yolo`), interpolated into `prepare_dataset`/`make_folds`/
`build_fold_ultralytics`'s `--data-dir`.

**paper_reference.md** — Paper summary, reported results, and data gaps.
**model_decisions.md** — The reconstruction decisions (ROI soft mask, f_m/f_w
partition, BiLRA placement, FHEO approach) and how each was resolved.
**model_spec_and_ambiguities.md** — What the paper specifies vs leaves open.


## Two caveats on every result we get

On **actual_dataset**, masks are box-derived rectangles, so Dice measures box
agreement, not tumour-boundary delineation. On **BTXRD_yolo**, masks are traced
polygon outlines, so Dice there does measure boundary agreement — the two
numbers are not directly comparable across datasets. Both datasets are
single-class in the fed-in label, so classification stays binary tumour-
presence derived from the segmentation, not the paper's normal/benign/
malignant (BTXRD's malignant/benign flags are carried in `metadata.csv` for
reporting but are not used for stratification or as a model target).
