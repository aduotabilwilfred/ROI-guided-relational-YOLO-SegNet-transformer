# Bone Tumor Detection with YOLO, SegNet, and Transformers
## Step-by-Step Implementation Roadmap

---

## Update 27-28/6/2026
# Complete End-to-End Pipeline
# ROI-Guided Relational YOLO-SegNet Transformer for Bone Tumor Detection
# Based on: Natarajan et al. (2026)

```

PROJECT PIPELINE OVERVIEW


CURRENT STATE: 809 annotated images in ./data/raw/actual_dataset/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/


PHASE 2: STAGE 1 - OSGDF CALIBRATION & FOLDING


INPUT:  ./data/raw/actual_dataset/ (809 raw images)
OUTPUT: ./outputs/folds/cv_manifest.csv (5-fold stratified manifest)
        ./outputs/folds/fold_params.json (Savitzky-Golay parameters per fold)

STEPS:
  2.1 Pool dataset and create stratified 5-fold splits
      - Use `make_folds.py` to group by source image ID (to prevent data leakage from augmented duplicates) and stratify by tumor presence.

  2.2 Calibrate Savitzky-Golay Digital Filter (SGDF) per fold on train images only
      - Use `fold_calibration.py` to optimize SG window size and polynomial order using TSO (Tunicate Swarm Optimization).
      - Prevents data leakage by ensuring validation/test sets are never seen during calibration.

  2.3 Diagnostic Visualisation
      - Use `osgdf_diagnostic.py` to compare before/after/residual panels and report blind SNR gains.

FILES TO CREATE:
  - osgdf_preprocessing.py (Savitzky-Golay + TSO)
  - run_osgdf.py (apply to all images)

EXPECTED OUTPUT:
  - SNR improvement: 21.4 → 29.6 dB
  - Noise-reduced training images
  - Parameter report (best SG window size, polynomial order)



PHASE 3: STAGE 2 - DETECTION + SEGMENTATION


INPUT:  ./data/raw/actual_dataset/ (809 images in train/valid splits)
        ./outputs/folds/cv_manifest.csv (cross-validation roles)
        ./outputs/folds/fold_params.json (OSGDF parameters per fold)

OUTPUT: Trained model (detection + segmentation)

3.1 DATA PREPARATION
    - Create PyTorch DataLoader
    - Split: 70% train, 15% val, 15% test
    - Load images, masks, bboxes
    - Apply data augmentation:
      * Random zoom (0.8-1.2x) [important for small objects]
      * Random rotation (±15°)
      * Random shift (±5% image size)
      * Random brightness/contrast
      * Mosaic augmentation (YOLO style)
      * Mixup augmentation

3.2 BUILD RELATIONAL YOLO-SEGNET ARCHITECTURE

    YOLOv8 Detection Head:
    ├─ Backbone: CSPDarknet (feature extraction)
    ├─ Neck: PANet (feature fusion)
    └─ Detection Head: YOLO head
        └─ Output: Bounding boxes + confidence scores

    Relational YoLo-SegNet Segmentation:
    ├─ BiFormer Attention Module
    │  └─ Bidirectional Transformer attention
    │     (captures relationships between tumor and surrounding tissue)
    │
    ├─ Relational Transformer Block (RTrB)
    │  ├─ Self-attention (within tumor)
    │  └─ Cross-attention (tumor vs background)
    │
    └─ SegFormer Decoder
       ├─ Multi-scale feature extraction
       ├─ Upsampling to original resolution
       └─ Pixel-wise segmentation output

3.3 TRAINING SETUP

    Loss Functions:
    ├─ Detection: YOLOv8 loss + Focal Loss (for small objects)
    └─ Segmentation: Dice Loss (0.5) + Focal Loss (0.5)

    Optimizer: AdamW
    ├─ Learning Rate: 1e-4 (with cosine annealing)
    ├─ Weight Decay: 1e-4
    └─ Gradient Clipping: 1.0

    Scheduler: Cosine Annealing
    ├─ T_max: 100 epochs
    └─ Eta_min: 1e-6

    Hyperparameters:
    ├─ Batch Size: 16 (8 for small objects)
    ├─ Epochs: 100+ (Natarajan used 50, but more is better)
    ├─ Patience (early stopping): 20
    └─ Warmup Epochs: 5

3.4 TRAINING PROCESS

    for epoch in range(num_epochs):
        # Training phase
        for batch in train_loader:
            images, masks, bboxes = batch

            # Forward pass
            detection_outputs = yolo_head(images)
            segmentation_outputs = segnet_decoder(features)

            # Loss computation
            detection_loss = yolo_loss(detection_outputs, bboxes)
            segmentation_loss = dice_loss(segmentation_outputs, masks) + \
                               focal_loss(segmentation_outputs, masks)
            total_loss = detection_loss + segmentation_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation phase
        val_metrics = validate(model, val_loader)

        # Early stopping check
        if val_metrics not improving:
            patience_counter += 1

        if patience_counter >= 20:
            break

3.5 EVALUATION METRICS

    Detection Metrics:
    ├─ mAP (mean Average Precision)
    ├─ Precision
    ├─ Recall
    └─ F1-Score

    Segmentation Metrics:
    ├─ Dice Coefficient (expected: 97%)
    ├─ Jaccard Index (expected: 97.1%)
    ├─ IoU (Intersection over Union)
    └─ Pixel Accuracy

FILES TO CREATE:
  - dataset_loader.py (PyTorch DataLoader)
  - relational_yolo_segnet.py (Model architecture)
  - train_stage2.py (Training loop)
  - evaluate_stage2.py (Evaluation metrics)

EXPECTED OUTPUT:
  - Trained model checkpoint
  - Detection + Segmentation predictions
  - Evaluation metrics (Dice ~97%, mAP ~98%)

TIME: 4-6 hours implementation + 8-12 hours training


PHASE 4: STAGE 3 - CLASSIFICATION + HYPERPARAMETER TUNING


INPUT:  Trained model from Stage 2
        Training data with class labels:
        ├─ Normal / Background (416 images)
        └─ Tumour / Cancer (393 images)

OUTPUT: Optimized model with classification head

4.1 ADD CLASSIFICATION HEAD

    After feature extraction:
    features (from YoLo-SegNet) → [Dense layers] → Classification head
                                      ↓
                                  [256 → 128 → 2]
                                      ↓
                                  Output: [Normal, Tumour/Cancer]

    Loss: Cross-Entropy + weighted loss (for class imbalance)
    ├─ Weight for Tumour/Cancer: balanced (393 samples)
    └─ Weight for Normal/Background: balanced (416 samples)

4.2 IMPLEMENT FIRE HAWK ELECTION OPTIMIZER (FHEO)

    Hyperparameters to optimize:
    ├─ Learning Rate: [1e-5, 1e-2] (log scale)
    ├─ Batch Size: [8, 16, 32]
    ├─ Dropout Rate: [0.1, 0.2, 0.3, 0.4, 0.5]
    ├─ Embedding Dimension: [128, 256, 384, 512]
    ├─ Weight Decay: [1e-5, 1e-4, 1e-3]
    └─ Optimizer (Adam vs AdamW vs SGD)

    FHEO Algorithm:
    ├─ Candidate Solutions: [lr, batch, dropout, emb_dim, weight_decay, optimizer]
    ├─ Fitness Function: Validation accuracy (maximize)
    ├─ Iterations: 50-100
    ├─ Population: 20-30
    └─ Election Process: Fire Hawk competition mechanism

    For each candidate hyperparameter set:
    ├─ Train model with these hyperparameters
    ├─ Evaluate on validation set
    ├─ Calculate fitness (accuracy)
    └─ Update best candidate based on fitness

4.3 TRAINING WITH OPTIMAL HYPERPARAMETERS

    After FHEO finds best hyperparameters:
    ├─ Retrain with optimal config
    ├─ Monitor validation accuracy
    ├─ Apply early stopping
    └─ Save best checkpoint

4.4 CLASSIFICATION EVALUATION

    Metrics:
    ├─ Overall Accuracy: 98.5% (target)
    ├─ Per-class Precision
    ├─ Per-class Recall
    ├─ Per-class F1-Score
    ├─ Confusion Matrix
    └─ ROC-AUC (expected: 0.981)

    Class-wise breakdown:
    ├─ Normal: 98%+ accuracy
    └─ Tumour/Cancer: 98%+ accuracy

FILES TO CREATE:
  - classification_head.py (Classification layer)
  - fire_hawk_optimizer.py (FHEO algorithm)
  - train_stage3.py (Training with FHEO)
  - evaluate_classification.py (Classification metrics)

EXPECTED OUTPUT:
  - Final optimized model
  - Classification accuracy: 98.5%
  - FHEO optimization report
  - Best hyperparameters found

TIME: 3-4 hours implementation + 20-30 hours optimization + training


PHASE 5: EVALUATION & ABLATION STUDIES


INPUT:  Trained & optimized model from Stage 3
        Test set (20% of data per fold = ~162 images)

OUTPUT: Comprehensive evaluation report

5.1 CROSS-VALIDATION (5-FOLD)

    Split data into 5 folds:
    ├─ Fold 1: Train on folds 2-5, test on fold 1
    ├─ Fold 2: Train on folds 1,3-5, test on fold 2
    ├─ Fold 3: Train on folds 1-2,4-5, test on fold 3
    ├─ Fold 4: Train on folds 1-3,5, test on fold 4
    └─ Fold 5: Train on folds 1-4, test on fold 5

    Report average metrics across folds:
    ├─ Mean accuracy ± Std Dev
    ├─ Mean Dice ± Std Dev
    ├─ Mean AUC ± Std Dev
    └─ Confidence intervals

5.2 ANATOMICAL EVALUATION

    Performance by bone location (Optional / Dependent on Metadata):
    ├─ Tibia
    ├─ Femur
    ├─ Fibula
    ├─ Humerus
    ├─ Hand
    ├─ Foot
    └─ Other sites...
    Note: Current dataset does not contain site/location labels. This evaluation requires manual categorization or additional metadata.

    Report: Accuracy, Dice, Precision, Recall per location (if available)

5.3 AGE-GROUP EVALUATION (Optional / Dependent on Metadata)

    Performance by age (if metadata is available):
    ├─ Child (1-18 years)
    ├─ Young Adult (19-35 years)
    ├─ Adult (36-60 years)
    └─ Senior (60+ years)
    Note: Current dataset does not contain patient age metadata.

    Report: How model performs across age groups (if available)

5.4 TUMOR-TYPE EVALUATION (Binary Classification)

    Performance by tumor type:
    ├─ Normal / Background (416 images)
    └─ Cancer / Tumour (393 images)
    Note: The current Roboflow dataset is annotated with a single `cancer` class (binary classification), so evaluation is Tumour vs Background.

    Report: Accuracy per class

5.5 ABLATION STUDIES

    Test impact of each component:
    ├─ Without OSGDF: Accuracy drops by X%
    ├─ Without BiFormer Attention: Dice drops by Y%
    ├─ Without RTrB: Segmentation accuracy drops by Z%
    ├─ Without Focal Loss: Small object detection drops by W%
    └─ Without Data Augmentation: Overall accuracy drops by V%

    Report: Contribution of each module

5.6 FAILURE ANALYSIS

    Identify and analyze misclassified cases:
    ├─ False negatives (missed tumors)
    ├─ False positives (normal detected as tumor)
    ├─ Misclassified types (Benign detected as Malignant)
    └─ Incorrect segmentation boundaries

    Report: Common failure patterns, edge cases

5.7 FINAL METRICS REPORT

    Summary Table (expected values from paper):
    ├─ Accuracy: 98.5%
    ├─ Precision: 98.32%
    ├─ Recall: 98.83%
    ├─ Dice: 97.0%
    ├─ Jaccard: 97.1%
    ├─ AUC: 0.981
    ├─ Model Parameters: 12.3M
    └─ Inference Time: 48ms per image (RTX 3090)

FILES TO CREATE:
  - cross_validation.py (5-fold CV)
  - anatomy_evaluation.py (Per-location analysis)
  - age_evaluation.py (Per-age-group analysis)
  - tumor_type_evaluation.py (Per-type analysis)
  - ablation_studies.py (Component analysis)
  - failure_analysis.py (Misclassification analysis)
  - final_report_generator.py (Comprehensive report)

EXPECTED OUTPUT:
  - 5-fold cross-validation scores
  - Per-anatomy performance breakdown
  - Per-age-group performance breakdown
  - Per-tumor-type performance breakdown
  - Ablation study results
  - Failure analysis with visualizations
  - Final comprehensive report (10-20 pages)



QUALITY CHECKPOINTS

After PHASE 2 (OSGDF):
  ✓ SNR improved by 8.2 dB
  ✓ Images visibly less noisy
  ✓ Ready for Stage 2

After PHASE 3 (Detection + Segmentation):
  ✓ mAP > 95%
  ✓ Dice > 95%
  ✓ Ready for Stage 3

After PHASE 4 (Classification + Tuning):
  ✓ Overall accuracy > 97%
  ✓ All hyperparameters optimized
  ✓ Ready for evaluation

After PHASE 5 (Evaluation):
  ✓ 5-fold CV confirms generalization
  ✓ All anatomy types evaluated
  ✓ Ablation studies completed
  ✓ Final accuracy: 98.5% ±0.5%


SUCCESS CRITERIA

Expected Results (from Natarajan et al. 2026):
├─ Accuracy: 98.5%
├─ Precision: 98.32%
├─ Recall: 98.83%
├─ Dice: 97%
├─ Jaccard: 97.1%
├─ AUC: 0.981
└─ Your Dataset (809 vs 1,867 in paper):
   └─ Target: Comparable performance (~98% accuracy) despite smaller dataset size

Acceptable Performance Thresholds:
├─ If accuracy > 98%:  Excellent (matches paper)
├─ If accuracy 97-98%:  Good (acceptable)
├─ If accuracy 95-97%:  Fair (investigate why)
└─ If accuracy < 95%:  Poor (debug issues)


PROJECT STRUCTURE


project_root/
├── data/
│   ├── raw/                           (original data)
│   │   └── actual_dataset/            (809 images in train/valid splits)
│   │       ├── train/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       └── valid/
│   │           ├── images/
│   │           └── labels/
│   │
│   └── processed/                     (previously used for flat dataset outputs)
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py           (phase 1)
│   │   └── osgdf_preprocessing.py     (phase 2)
│   │
│   ├── models/
│   │   ├── yolo_segnet.py            (phase 3 architecture)
│   │   ├── classification_head.py     (phase 4)
│   │   └── fire_hawk_optimizer.py     (phase 4 optimization)
│   │
│   ├── training/
│   │   ├── dataset_loader.py          (phase 3 data)
│   │   ├── train_stage2.py            (phase 3 training)
│   │   └── train_stage3.py            (phase 4 training)
│   │
│   └── evaluation/
│       ├── cross_validation.py        (phase 5)
│       ├── anatomy_evaluation.py      (phase 5)
│       ├── ablation_studies.py        (phase 5)
│       └── final_report.py            (phase 5)
│
├── outputs/
│   ├── models/
│   │   ├── stage2_best.pth            (best segmentation model)
│   │   └── stage3_best.pth            (best optimized model)
│   │
│   ├── results/
│   │   ├── cv_results.json
│   │   ├── anatomy_breakdown.csv
│   │   ├── ablation_results.json
│   │   └── final_report.pdf
│   │
│   └── visualizations/
│       ├── osgdf_comparison.png       (phase 2)
│       ├── training_curves.png        (phase 3/4)
│       ├── confusion_matrices.png     (phase 4)
│       └── ablation_results.png       (phase 5)
│
└── README.md                          (this pipeline)


1. START HERE: Implement PHASE 2 (OSGDF Calibration & Folding)
   - Create osgdf_preprocessing.py
   - Implement Savitzky-Golay Filter
   - Implement Tunicate Swarm Optimization
   - Generate stratified 5-folds using make_folds.py
   - Calibrate OSGDF per fold on train images only using fold_calibration.py

2. Then: Implement PHASE 3 (Detection + Segmentation)
   - Create PyTorch DataLoader
   - Build Relational YoLo-SegNet architecture
   - Train on noise-reduced images

3. Then: Implement PHASE 4 (Classification + Tuning)
   - Add classification head
   - Implement Fire Hawk Optimizer
   - Find optimal hyperparameters

4. Finally: Complete PHASE 5 (Evaluation)
   - Run 5-fold cross-validation
   - Evaluate by anatomy, age, tumor type
   - Generate comprehensive report












































































## Begining
## Phase 1 — Understanding & Setup (Week 1–2)

- Read the original paper thoroughly
- Understand how YOLO, SegNet, and Transformers each work individually before combining them
- Set up Python environment and install all libraries
- Download and explore your chosen bone tumor X-ray dataset
- Understand the annotation format (bounding boxes for YOLO, masks for segmentation)

---

## Phase 2 — Data Preparation (Week 2–3)

- Collect and organize X-ray images with:
  - Bounding box annotations for YOLO (tumor location)
  - Pixel-level masks for segmentation
  - Class labels for classification (e.g. benign vs. malignant, or tumor type)
- Apply preprocessing:
  - Resize images to consistent dimensions (e.g. 512×512)
  - Normalize pixel values
  - Apply data augmentation (flipping, rotation, brightness adjustment) using Albumentations
- Split into train/validation/test sets (e.g. 70/15/15)

---

## Phase 3 — Build the YOLO Detection Module (Week 3–4)

- Use YOLOv8 (from Ultralytics — very straightforward to fine-tune)
- Train YOLO on your X-ray dataset to detect and draw bounding boxes around bone tumors
- This produces the Region of Interest (ROI) — the cropped tumor area
- Evaluate: mAP (mean Average Precision), IoU

---

## Phase 4 — Build the SegNet + Transformer Segmentation Module (Week 4–6)

- Take the ROI crop from YOLO as input
- Implement SegNet encoder-decoder for pixel-level segmentation
- Add Transformer blocks (self-attention layers) to capture long-range spatial relationships within the ROI
- The "relational" part means the Transformer models relationships between different parts of the tumor region
- Train the segmentation head on masked X-ray ROIs
- Evaluate: Dice coefficient, IoU, Hausdorff distance

---

## Phase 5 — Build the Classification Module (Week 6–7)

- Add a classification head on top of the encoder features
- Classify the tumor (e.g. benign/malignant, or specific tumor type)
- You can share the encoder weights between segmentation and classification (multi-task learning)
- Evaluate: Accuracy, Precision, Recall, F1-score, AUC-ROC

---

## Phase 6 — Connect Everything into One Pipeline (Week 7–8)

- YOLO detects ROI → ROI fed into SegNet-Transformer → segmentation + classification output
- Make sure the whole thing runs end-to-end on a new X-ray image
- Test inference speed — the paper emphasizes lightweight, so track model size and inference time

---

## Phase 7 — Evaluation & Comparison (Week 8–9)

Compare your system against baselines:

- YOLO only (no segmentation)
- SegNet without Transformer
- SegNet without ROI guidance (uses full image)
- Standard U-Net

---

## Phase 8 — Ablation Study (Week 9–10)

- Remove the ROI guidance — does segmentation accuracy drop?
- Remove the Transformer — does it perform worse?
- Remove the relational module — what changes?

---

## Phase 9 — Write Up (Week 10–12)

- Introduction, related work, methodology, experiments, results, conclusion, future work

---

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|-----------------|
| 1-2 | Understanding & Setup | Environment ready, dataset explored |
| 2-3 | Data Preparation | Annotated and preprocessed data |
| 3-4 | YOLO Module | Trained detection model |
| 4-6 | SegNet + Transformer | Trained segmentation model |
| 6-7 | Classification Module | Trained classifier |
| 7-8 | Pipeline Integration | End-to-end system working |
| 8-9 | Evaluation & Comparison | Baseline comparisons complete |
| 9-10 | Ablation Study | Component importance analysis |
| 10-12 | Write Up | Final paper/report |

---

## Key Metrics to Track

### Detection (Phase 3)
- Mean Average Precision (mAP)
- Intersection over Union (IoU)

### Segmentation (Phase 4)
- Dice Coefficient
- IoU (Jaccard Index)
- Hausdorff Distance

### Classification (Phase 5)
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### System Performance (Phase 6)
- End-to-end inference time
- Model size
- Memory usage

---

## Important Notes

1. **Data Annotation Format**: Ensure consistency between bounding box format (YOLO uses normalized XYWH) and segmentation masks
2. **ROI Preprocessing**: After YOLO detection, standardize ROI size before feeding to segmentation module
3. **Multi-task Learning**: Consider weight sharing between segmentation and classification encoders for efficiency
4. **Lightweight Design**: Monitor model complexity — the paper emphasizes efficiency
5. **Baseline Comparisons**: Always include ablation studies to validate each component's contribution
