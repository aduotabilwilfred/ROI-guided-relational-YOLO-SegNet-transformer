# Reference: ROI-guided Relational YOLO-SegNet (Natarajan et al., 2026)

Full title: "ROI-guided relational YOLO-SegNet transformer for lightweight bone
tumor segmentation and classification from X-ray images." Scientific Reports
16:14603. DOI 10.1038/s41598-026-44297-8. Dataset:
https://universe.roboflow.com/vibhu-raj-ysy7d/bone_tumor

This file summarises the paper we are reimplementing and, more importantly,
records where the paper's description does not match the dataset we actually
have. Read the "Gaps" section before trusting any number here as a target.


## What the paper builds

A three-stage pipeline the authors describe as segmentation-first:

1. OSGDF preprocessing. Savitzky-Golay denoising whose window size and
   polynomial order are tuned per-image by Tunicate Swarm Optimization, to
   maximise SNR while preserving edges. Reported SNR gain 21.4 to 29.6 dB.
   Done offline before training.
2. Relational YOLO-SegNet. YOLOv8 detects a tumor region (ROI); a SegNet-style
   decoder segments inside that ROI; a Relational Transformer Block (RTrB)
   applies attention only within the ROI, not the whole image, to keep it
   lightweight. Self-attention head plus a cross-attention head that reads
   surrounding bone.
3. Classification. Normal vs tumor, derived from the segmented region via
   global average pooling and a small fully-connected head. Explicitly the
   secondary output; segmentation is called the primary task.

Hyperparameters (learning rate, batch size, dropout, embedding dim) are tuned
by a Fire Hawk / Election Optimizer, population 20, 50 iterations, fitness =
validation Dice.

Model size 12.3M params, 18.6 GFLOPs, 48 ms/image on an RTX 3090, memory below
2.1 GB. Training 50 epochs, batch 16, AdamW, cosine schedule, BCE loss for
classification plus Dice loss for segmentation.


## Reported results (the numbers being claimed)

Headline: 98.5% accuracy, 98.32% precision, 98.83% recall, 98.21% specificity,
98.57% F1, 97% Dice, 97.1% Jaccard, AUC 0.981, under five-fold CV (98.5 +/-
0.3%).

Five-fold breakdown (Table 3):

    Fold        Accuracy %   Dice %
    1           98.7         97.3
    2           98.1         96.8
    3           98.9         97.1
    4           98.3         96.6
    5           98.6         97.2
    mean +/- SD 98.5 +/- 0.3 97.0 +/- 0.3

Ablation (Table 2): without OSGDF 21.4 dB / 93.2% acc / 90.5% Dice; without
RTrB 95.6% / 93.1%; without FHEO 96.2% / 94.0%; full model 29.6 dB / 98.5% /
97%.

Confusion matrix (Fig. 8): normal 416 correct of 421 (5 wrong); tumor 381
correct of 388 (7 wrong). Total 809.


## Cross-validation protocol (what we are matching)

The dataset is randomly divided into five equal subsets. Each iteration uses
three subsets for training, one for validation, one for testing. Rotated five
times so every image is tested exactly once. Reported metrics are the mean
across folds. The paper also mentions a 70/15/15 split as an aspiration, but
the executed scheme is the 3/1/1 rotation, i.e. 60/20/20 per fold. There is no
permanent held-out test set; the test fold rotates.

Augmentation: rotation, flipping, zooming, contrast jitter, applied to
training data to increase diversity. Contrast enhancement deliberately NOT
applied to the base images, on the grounds it could introduce misleading
structure. All images resized to 512x512 and normalised to 0-1.


## Gaps between the paper and our actual data

These are the load-bearing discrepancies. Each one limits what we can honestly
claim to reproduce.

1. No pixel-level masks exist in the dataset.
   The paper states the dataset provides expert pixel-level segmentation masks,
   one per image. The public Roboflow export does not. Inspection of our copy:
   every annotation is a 4-point oriented bounding box (YOLOv8-OBB), one class
   ("cancer"), zero polygons. Consequence: any "segmentation mask" we build is
   a filled rotated rectangle. A Dice score computed against these measures
   agreement with rectangles, not tumor-boundary delineation. The paper's 97%
   Dice "against expert masks" is therefore not reproducible on this data,
   because those masks are not in the cited dataset. This must be stated
   plainly in our writeup: our segmentation target is box-derived, not expert
   pixel-traced.

2. Class counts differ slightly.
   Paper: 421 normal / 388 tumor. Our inspection: 416 background (empty label
   files) / 393 tumor-containing images, over the same 809. Same split to
   within a handful of images; likely a marginally different dataset version.
   The 393 tumor images carry 668 boxes total (some images have several).

3. Single class means no real classification task in the data.
   The dataset has one class, "cancer". The paper's normal-vs-tumor
   classification comes from image-level presence/absence of a tumor, which we
   can reconstruct (an image with any box = tumor, an empty label = normal),
   but there is no benign/malignant distinction. So "classification" here is
   binary presence detection derived from whether segmentation found anything.

4. Roughly half the images have no tumor.
   416 of 809 are tumor-free backgrounds. This makes folds needing to be
   stratified by tumor-present vs background, or per-fold class balance drifts
   and Dice becomes noisy. Not a flaw, but a design constraint the paper does
   not mention handling.

5. Image sizes are wildly heterogeneous.
   From 373x454 up to 3024x4032, portrait and landscape mixed. Letterboxing to
   512 is mandatory, and small images upscale while large ones downscale ~8x.
   Worth one sentence in methods; affects the denoiser's effective operating
   scale.


## Internal inconsistencies in the paper (do not propagate)

These are the paper's own errors. Noting them so we implement the sensible
version rather than copying a typo.

- FHEO is expanded three incompatible ways: "Fire Hawk Election Optimizer"
  (title/abstract), "Fire Hawk Optimizer + Election Optimizer" (methodology),
  and "Fused Harris Hawks and Equilibrium Optimizer" (FHEO section). The
  equations given are for Fire Hawk plus Election Optimizer, so that is the
  intended pair; the Harris Hawks / Equilibrium phrasing is an error.

- The SNR formula printed as Eq. 1 is a ratio of standard deviations,
  10*log10(std(signal) / std(signal - denoised)). That is not the dB SNR the
  text describes and does not match a 21.4-to-29.6 dB narrative cleanly. Our
  OSGDF module already uses a defensible blind-SNR definition; keep it and note
  the divergence rather than reproducing Eq. 1 literally.

- "Segmentation is the primary task" is repeated throughout, yet the dataset
  supports only box-level supervision. The paper's own pipeline is really
  detection-first with box-derived masks; our honest framing should say so.


## How this maps to our pipeline

Our build order (unchanged by the paper):

1. prepare_dataset.py converts OBB labels to masks + YOLO bboxes and
   letterboxes to 512. Masks are filled rotated boxes (see Gap 1).
2. make_folds.py (next to write): pool all 809, stratify 5 folds by
   tumor-present vs background, image-level, seeded, manifests only.
3. Per fold: calibrate OSGDF on that fold's training images only; denoise;
   augment training data only; train; evaluate on the held-out test fold.
4. Average metrics across 5 folds, report mean +/- SD.

What we can claim honestly: reproduction of the paper's *method* (OSGDF + ROI-
guided detection/segmentation + metaheuristic tuning + 5-fold CV) on the public
809-image dataset, reporting box-level Dice and binary presence classification.
What we cannot claim: 97% Dice against expert pixel masks, because those masks
are not in this dataset.
