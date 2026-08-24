# BTXRD Dataset Integration, Preprocessing, and 5-Fold Cross-Validation

## Objective

I have determined that the current dataset (`actual_dataset`) is not sufficiently suitable for the training process. I have another dataset, **BTXRD**, available in both raw and preprocessed forms:

* Raw BTXRD dataset: `raw/BTXRD`
* Preprocessed BTXRD dataset: `processed/Tumor_Bone_Kaggle`

The BTXRD dataset contains **actual tumor delineation coordinates/pixel-level annotations**, rather than only bounding-box coordinates around the tumor. This makes it potentially more suitable for the tumor segmentation/localization task.

There are visualization outputs showing the annotations overlaid on the original images in:

`reports/figures/`

The relevant visualization files contain **`overlay`** in their filenames.

There is also a file named approximately:

`mask_size analysis.jpg`

in the same directory, which contains analysis of the BTXRD masks.

Before making any implementation changes, inspect and verify these artifacts yourself.

The goal is to replace `actual_dataset` with BTXRD while preserving the existing project architecture and training logic as much as possible.

---

# 1. First: Audit the Existing Pipeline

**Do not immediately modify the code.**

First inspect the entire existing data and training pipeline, including:

* `raw/actual_dataset`
* `processed/`
* `src/data/`
* preprocessing scripts
* dataset loading code
* feature-generation code
* annotation parsing
* 5-fold cross-validation implementation
* model definition
* model input/output
* loss functions
* evaluation metrics
* training scripts
* configuration files
* `dvc.yaml`

Trace exactly how an example from `actual_dataset` flows through:

```text
raw dataset
    ↓
preprocessing
    ↓
processed dataset
    ↓
fold generation
    ↓
feature/target generation
    ↓
model input
    ↓
model prediction
    ↓
loss
    ↓
evaluation metrics
```

Document what representation is used at every stage.

---

# 2. Audit BTXRD

Inspect both:

```text
raw/BTXRD
processed/Tumor_Bone_Kaggle
```

Determine:

* image format
* annotation format
* mask format
* coordinate system
* image dimensions
* mask dimensions
* tumor/cancer labels
* patient identifiers
* study/case identifiers
* whether multiple images belong to the same patient/study
* whether there are duplicate or highly correlated images
* number of samples
* class distribution
* number of positive/negative samples if applicable
* whether annotations represent true segmentation masks
* whether masks can be converted to bounding boxes
* whether there are multiple tumors in one image
* whether there are empty masks
* whether there are malformed annotations
* whether the existing OSGDF preprocessing is compatible with the dataset

Also inspect the existing:

```text
reports/figures/overlay*.*
```

visualizations and:

```text
reports/figures/mask_size analysis.jpg
```

to verify that the annotations actually correspond to the tumor regions.

Do not assume that the dataset is suitable simply because it contains segmentation coordinates.

---

# 3. Compare `actual_dataset` and BTXRD

Produce a concise technical comparison before implementation.

At minimum compare:

| Component             | `actual_dataset` | BTXRD | Required change |
| --------------------- | ---------------- | ----- | --------------- |
| Image format          |                  |       |                 |
| Annotation format     |                  |       |                 |
| Bounding boxes        |                  |       |                 |
| Segmentation masks    |                  |       |                 |
| Patient/case ID       |                  |       |                 |
| Tumor/cancer label    |                  |       |                 |
| Image dimensions      |                  |       |                 |
| Preprocessing         |                  |       |                 |
| Target representation |                  |       |                 |
| Model input           |                  |       |                 |
| Model output          |                  |       |                 |
| Loss                  |                  |       |                 |
| Metrics               |                  |       |                 |
| Cross-validation      |                  |       |                 |

Clearly distinguish between:

1. changes that are **required**,
2. changes that are **optional**, and
3. things that should remain unchanged.

Do not modify the model merely because BTXRD contains segmentation masks.

---

# 4. Canonical Dataset Format

The preferred approach is to make BTXRD conform to the same canonical structure/schema expected by the existing `actual_dataset` pipeline.

The reason is to minimize unnecessary changes to the project.

Ideally, the existing pipeline should become:

```text
Dataset selection
      ↓
Dataset-specific adapter/input parsing
      ↓
Common canonical representation
      ↓
Existing preprocessing
      ↓
Existing folding logic
      ↓
Existing training pipeline
```

The goal is that changing the dataset in `dvc.yaml` is sufficient to select BTXRD.

For example, if the existing pipeline uses:

```text
actual_dataset
```

then after this work it should be possible to use:

```text
BTXRD
```

through configuration/DVC without duplicating the entire preprocessing pipeline.

### Important

Do **not** discard useful BTXRD information merely to force it into the old format.

If BTXRD provides a segmentation mask and the old dataset only provides a bounding box, preserve the segmentation mask if it can be useful.

If necessary, create a canonical representation capable of storing both:

```text
image
patient/case ID
tumor/cancer label
segmentation mask
bounding box
metadata
```

A bounding box may be derived from the segmentation mask where required by existing downstream code.

---

# 5. OSGDF Preprocessing

Use the **same OSGDF preprocessing logic** currently used by the project unless the BTXRD data representation makes a specific part incompatible.

Modify only the dataset-specific components necessary to correctly parse BTXRD.

Pay particular attention to geometric transformations.

If an image is:

* resized,
* cropped,
* rotated,
* flipped,
* padded,
* transformed,

the corresponding segmentation mask/coordinates must undergo the exact equivalent transformation.

The image and annotation must remain spatially aligned.

Do not silently apply image transformations without applying the corresponding transformation to the mask.

---

# 6. Prevent Data Leakage

This is a **hard requirement**.

The new BTXRD pipeline must use leakage-safe **5-fold cross-validation**.

Splitting must occur at the appropriate independent-unit level, preferably:

```text
patient / case / study
```

rather than simply splitting individual images.

First determine what identifier BTXRD provides that represents the independent patient/case/study.

If multiple images, slices, annotations, or derived samples belong to the same patient/case, they **must never be distributed across different folds**.

For example, this is invalid:

```text
Patient A
 ├── image 1 → Fold 1
 ├── image 2 → Fold 3
 └── image 3 → Fold 5
```

All correlated samples must remain in the same fold.

The correct structure is:

```text
Patient A
 ├── image 1
 ├── image 2
 └── image 3
        ↓
     Fold 1
```

---

# 7. Tumor/Cancer Stratification

Maintain **5-fold stratified cross-validation**.

Determine the appropriate tumor/cancer label from BTXRD and perform stratification at the patient/case level.

Do not blindly stratify individual images if this could cause patient leakage.

If the dataset contains multiple labels per patient/case, inspect the structure and determine the correct stratification strategy rather than making an arbitrary assumption.

Report the class distribution in each fold and verify that stratification is working as intended.

---

# 8. Fold-Specific Preprocessing

Ensure that preprocessing does not leak information across folds.

Any operation that learns parameters/statistics from the data must be fitted using the training portion of the fold and then applied to validation/test data.

Avoid doing dataset-wide operations before cross-validation if those operations can expose information from validation/test samples to the training process.

Also ensure that:

* augmentation is applied only to training data where appropriate,
* validation/test data are not augmented,
* derived features do not use information from another fold,
* cached/generated artifacts are correctly associated with their fold.

---

# 9. Investigate Model Compatibility

Do not assume that segmentation masks require a new model architecture.

First trace how the current model uses the annotations.

Determine:

1. What is currently used as the model input?
2. What is currently used as the target?
3. Is the current task classification, localization, detection, segmentation, or a combination?
4. Does the current model predict bounding boxes, masks, classes, or derived features?
5. How are features currently generated from the annotations?
6. What does the current loss function compare?
7. What metrics are currently calculated?

Then determine whether BTXRD can be converted into the existing target representation without losing information relevant to the research objective.

### Preferred approach

If the current model can operate correctly using a representation derived from the BTXRD masks, **keep the existing model architecture unchanged**.

Only modify the model if the research objective genuinely requires pixel-level segmentation and the current model cannot support it.

If a model modification is necessary, make the **minimum necessary change** and clearly explain why.

---

# 10. Investigate Feature Generation

Because BTXRD contains actual tumor delineations, determine whether the existing feature-generation process can take advantage of the more accurate tumor boundaries.

Investigate whether the current pipeline derives features such as:

* tumor area
* tumor dimensions
* centroid
* shape
* intensity statistics
* texture
* bounding-box dimensions
* mask-based measurements

Do not add new features simply because they are possible.

Only modify feature generation if:

1. the current pipeline is incompatible with BTXRD, or
2. the existing research objective requires the additional information.

Preserve the existing logic wherever possible.

---

# 11. Investigate Evaluation Metrics

Do not automatically replace the existing evaluation metrics.

First determine what the current metrics measure and whether they remain valid with BTXRD.

If the task is genuinely pixel-level segmentation, investigate appropriate segmentation metrics such as:

* Dice coefficient
* Intersection over Union (IoU)
* pixel-level precision
* pixel-level recall/sensitivity
* specificity, where appropriate

If the task remains bounding-box detection/localization, retain appropriate detection/localization metrics such as bounding-box IoU and the existing relevant metrics.

If classification is also part of the task, retain the appropriate classification metrics.

The final metric selection must correspond to the **actual model output and research objective**, not merely the annotation format.

Clearly document any metric changes and why they are necessary.

---

# 12. Validation of Masks and Preprocessing

Before training, create validation checks for BTXRD.

At minimum verify:

* every image has the expected annotation,
* image and mask dimensions correspond,
* masks contain valid values,
* masks are not unexpectedly empty,
* masks are not overwhelmingly the entire image,
* masks remain aligned after preprocessing,
* bounding boxes derived from masks are correct,
* transformed masks remain spatially correct,
* no samples are lost unexpectedly,
* no duplicate patient/case appears across folds.

Use the existing overlay visualization approach to visually verify a representative sample of processed images/masks.

---

# 13. DVC Integration

The final pipeline should integrate cleanly with the existing `dvc.yaml`.

The desired workflow should be approximately:

```text
Change dataset name/configuration
        ↓
dvc repro
        ↓
BTXRD preprocessing
        ↓
5-fold leakage-safe splitting
        ↓
feature/target generation
        ↓
training
        ↓
evaluation
        ↓
reports/results
```

Avoid creating a completely separate BTXRD pipeline if the existing pipeline can be reused.

Prefer modifying the existing dataset-dependent components so that the pipeline remains generic.

For example, rather than creating:

```text
preprocess_actual_dataset.py
preprocess_btxrd.py
```

with duplicated logic, prefer a common preprocessing pipeline with dataset-specific loading/parsing where appropriate.

---

# 14. Required Implementation Principles

Follow these principles throughout the implementation:

### Preserve existing behavior

Do not refactor unrelated parts of the project.

### Minimize changes

Only change code that is necessary to support BTXRD.

### Do not duplicate the pipeline

Reuse existing preprocessing, folding, training, and evaluation logic wherever possible.

### Do not silently change the research task

Changing from detection/localization to segmentation is a substantive methodological change. Do not make this decision implicitly.

### No data leakage

Patient/case-level separation is mandatory.

### Preserve information

Do not discard BTXRD's segmentation information unless there is a justified reason.

### Reproducibility

Maintain deterministic/random seeds where the current pipeline uses them.

### DVC compatibility

The final implementation must work through the existing DVC pipeline.

---

# 15. Expected Workflow

Follow this order:

```text
STEP 1
Audit current actual_dataset pipeline

↓

STEP 2
Audit BTXRD structure and annotations

↓

STEP 3
Inspect overlay visualizations and mask-size analysis

↓

STEP 4
Compare both dataset representations

↓

STEP 5
Determine the canonical dataset representation

↓

STEP 6
Determine whether the existing model/targets/features/metrics
are compatible with BTXRD

↓

STEP 7
Implement BTXRD dataset adapter/preprocessing

↓

STEP 8
Implement leakage-safe patient/case-level 5-fold stratification

↓

STEP 9
Integrate with existing OSGDF preprocessing

↓

STEP 10
Update DVC dataset configuration

↓

STEP 11
Run preprocessing and fold-generation tests

↓

STEP 12
Validate masks, overlays, class distributions and folds

↓

STEP 13
Run the training pipeline

↓

STEP 14
Compare results with the previous dataset where appropriate

↓

STEP 15
Document all changes
```

---

# 16. Before Coding: Report Your Findings

Before making significant code changes, provide a concise audit containing:

1. **BTXRD structure**
2. **Annotation representation**
3. **What the overlay visualizations confirm**
4. **What the mask-size analysis confirms**
5. **Current `actual_dataset` representation**
6. **Differences between the datasets**
7. **Current model input/output**
8. **Current loss**
9. **Current evaluation metrics**
10. **Current 5-fold splitting strategy**
11. **Potential leakage risks**
12. **Recommended canonical BTXRD representation**
13. **Which files need modification**
14. **Which files should remain unchanged**
15. **Whether the model needs modification**
16. **Whether the metrics need modification**
17. **Whether feature generation needs modification**

Do not proceed with a major architectural change until these findings have been established.

---

# 17. Final Acceptance Criteria

The implementation is considered successful only if:

* BTXRD can be selected through the DVC configuration.
* BTXRD is converted into the canonical dataset format expected by the pipeline.
* Existing OSGDF preprocessing is reused wherever applicable.
* Image/mask alignment is preserved.
* 5-fold cross-validation is maintained.
* Stratification is performed using the appropriate tumor/cancer label.
* Patient/case leakage is prevented.
* No patient/case appears in multiple folds.
* Fold distributions are validated.
* The existing model remains unchanged if it is compatible.
* Any model modification is justified by an actual incompatibility.
* Feature-generation changes are justified.
* Metric changes are justified.
* The pipeline runs successfully through DVC.
* Preprocessing and fold generation are reproducible.
* Representative processed masks/overlays are visually validated.
* No unrelated parts of the project are unnecessarily refactored.

## Important Final Rule

**Do not optimize for making BTXRD fit the existing code at the expense of correctness.**

The goal is:

> **Reuse the existing architecture where it is methodologically correct, while making the minimum necessary changes required to correctly represent and train on BTXRD's more accurate tumor delineations.**

Correctness, leakage prevention, and preservation of the segmentation information take priority over minimizing the number of changed files.
