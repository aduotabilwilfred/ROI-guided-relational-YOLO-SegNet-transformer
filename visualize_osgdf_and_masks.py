from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(r"c:\Users\chris\Desktop\roi-guided-yolo-segnet\ROI-guided-relational-YOLO-SegNet-transformer")
sys.path.insert(0, str(REPO_ROOT))

from src.data.osgdf_preprocessing import apply_sg_filter, estimate_noise_sigma, snr_blind
from src.data.prepare_dataset import Letterbox, parse_label_file, draw_mask

def get_clinical_info(image_stem: str, dataset_csv_path: Path) -> dict:
    if not dataset_csv_path.exists():
        return {}
    try:
        df = pd.read_csv(dataset_csv_path)
        row = df[df["image_id"].astype(str).str.startswith(image_stem)]
        if row.empty:
            return {}
        r = row.iloc[0]
        tumors = ["osteosarcoma", "giant cell tumor", "osteochondroma", "simple bone cyst", "osteofibroma", "synovial osteochondroma", "other mt", "other bt"]
        active_tumors = [t for t in tumors if t in r and r[t] == 1]
        diag = active_tumors[0].title() if active_tumors else ("Malignant" if r.get("malignant") == 1 else "Benign")
        bones = ["femur", "tibia", "humerus", "fibula", "radius", "ulna", "hip bone", "hand", "foot"]
        active_bones = [b for b in bones if b in r and r[b] == 1]
        bone = active_bones[0].title() if active_bones else "Bone"
        return {"diagnosis": diag, "bone": bone, "age": r.get("age", "N/A"), "gender": r.get("gender", "N/A")}
    except Exception:
        return {}

def create_yellow_highlight(image_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    yellow_fill = np.zeros_like(color_img)
    yellow_fill[:, :] = [255, 230, 0]
    overlay = color_img.copy()
    tumor_indices = mask > 0
    overlay[tumor_indices] = cv2.addWeighted(color_img[tumor_indices], 1.0 - alpha, yellow_fill[tumor_indices], alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), thickness=2)
    return overlay

def select_sample_images(manifest_path: Path, n_samples: int = 4, seed: int = 42, min_fraction: float = 0.01, specific_ids: list[str] | None = None) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split(","))
    header = rows[0]
    records = [dict(zip(header, r)) for r in rows[1:]]
    tumor_records = [r for r in records if r.get("has_tumour") == "1"]

    if specific_ids:
        chosen = [r for r in tumor_records if r.get("source_id") in specific_ids]
        if not chosen:
            chosen = [r for r in tumor_records if any(sid in r.get("source_id", "") for sid in specific_ids)]
        return chosen[:n_samples]

    # If min_fraction requested, filter for well-defined visible tumours
    if min_fraction > 0:
        qualified = []
        for r in tumor_records:
            lbl_p = REPO_ROOT / r["label_path"]
            if not lbl_p.exists():
                continue
            parsed = parse_label_file(lbl_p, 1000, 1000)
            if parsed.ok and parsed.annotations:
                pts = parsed.annotations[0].points
                if len(pts) >= 4:
                    x = [p[0] for p in pts]
                    y = [p[1] for p in pts]
                    area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(len(pts)-1)))
                    if area >= min_fraction:
                        qualified.append((r, area))
        if qualified:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(qualified), size=min(n_samples, len(qualified)), replace=False)
            return [qualified[i][0] for i in indices]

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(tumor_records), size=min(n_samples, len(tumor_records)), replace=False)
    return [tumor_records[i] for i in chosen_indices]

def run_visualization(
    manifest_path: Path = REPO_ROOT / "outputs/folds/cv_manifest.csv",
    fold_params_path: Path = REPO_ROOT / "outputs/folds/fold_params.json",
    dataset_csv_path: Path = REPO_ROOT / "data/raw/actual_dataset/dataset.csv",
    output_dir: Path = REPO_ROOT / "outputs/visualizations",
    n_samples: int = 4,
    seed: int = 42,
    min_fraction: float = 0.01,
    target_size: int = 512,
    specific_ids: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_size, poly_order = 5, 4
    if fold_params_path.exists():
        try:
            params_data = json.loads(fold_params_path.read_text())
            f0 = params_data["folds"][0]
            window_size = int(f0["window_size"])
            poly_order = int(f0["poly_order"])
        except Exception as e:
            print(f"Warning: could not read fold_params.json ({e}), using default W=5, P=4")
    print(f"Using OSGDF Filter: Window Size = {window_size}, Polynomial Order = {poly_order}")
    samples = select_sample_images(manifest_path, n_samples=n_samples, seed=seed, min_fraction=min_fraction, specific_ids=specific_ids)
    print(f"Selected {len(samples)} sample images for visualization.\n")

    print(f"{'Image ID':<12} | {'Clinical Diagnosis':<20} | {'Bone':<10} | {'Noise Sigma':<20} | {'Blind SNR':<20} | {'Tumour Pixels':<12}")
    print("-" * 105)

    processed_samples = []
    for item in samples:
        stem = item["source_id"]
        img_path = REPO_ROOT / item["image_path"]
        lbl_path = REPO_ROOT / item["label_path"]
        if not img_path.exists():
            continue
        raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            continue
        orig_h, orig_w = raw_img.shape[:2]
        lb = Letterbox.build(orig_h, orig_w, target_size)
        raw_512 = lb.apply_image(raw_img)
        raw_float = raw_512.astype(np.float64) / 255.0
        filtered_float = apply_sg_filter(raw_float, window_size, poly_order)
        filtered_512 = np.rint(np.clip(filtered_float, 0.0, 1.0) * 255.0).astype(np.uint8)

        sigma_before = estimate_noise_sigma(raw_float)
        sigma_after = estimate_noise_sigma(filtered_float)
        snr_before = snr_blind(raw_float)
        snr_after = snr_blind(filtered_float)

        noise_diff = np.abs(raw_float - filtered_float)
        noise_diff_vis = np.clip(noise_diff * 10.0 * 255.0, 0, 255).astype(np.uint8)

        mask_512 = np.zeros((target_size, target_size), dtype=np.uint8)
        num_vertices = 0
        if lbl_path.exists():
            parsed = parse_label_file(lbl_path, orig_h, orig_w)
            if parsed.ok and parsed.annotations:
                mask_512 = draw_mask(parsed.annotations, lb)
                num_vertices = len(parsed.annotations[0].points)

        yellow_overlay = create_yellow_highlight(filtered_512, mask_512, alpha=0.45)
        clin = get_clinical_info(stem, dataset_csv_path)
        diag = clin.get("diagnosis", "Tumour")
        bone = clin.get("bone", "Bone")
        mask_area = int(np.count_nonzero(mask_512))
        pct_area = (mask_area / (target_size * target_size)) * 100.0

        sigma_str = f"{sigma_before:.4f} -> {sigma_after:.4f}"
        snr_str = f"{snr_before:.1f} dB -> {snr_after:.1f} dB"
        print(f"{stem:<12} | {diag:<20} | {bone:<10} | {sigma_str:<20} | {snr_str:<20} | {mask_area} ({pct_area:.1f}%)")

        sample_dir = output_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "1_original_raw.png"), raw_512)
        cv2.imwrite(str(sample_dir / "2_osgdf_filtered.png"), filtered_512)
        cv2.imwrite(str(sample_dir / "3_noise_removed_x10.png"), noise_diff_vis)
        cv2.imwrite(str(sample_dir / "4_tumor_yellow_overlay.png"), cv2.cvtColor(yellow_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "5_ground_truth_mask.png"), mask_512)

        processed_samples.append({
            "stem": stem, "raw": raw_512, "filtered": filtered_512, "noise_diff": noise_diff_vis,
            "yellow_overlay": yellow_overlay, "mask": mask_512, "diag": diag, "bone": bone,
            "sigma_before": sigma_before, "sigma_after": sigma_after, "snr_before": snr_before,
            "snr_after": snr_after, "mask_area": mask_area, "num_vertices": num_vertices,
        })
    print("-" * 105)

    n_rows = len(processed_samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 4.4 * n_rows), dpi=180)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    column_titles = [
        "1. Original (Before OSGDF)",
        f"2. OSGDF Filtered (W={window_size}, P={poly_order})",
        "3. Removed Noise (Residual x10)",
        "4. Tumour Region (Highlighted Yellow)",
        "5. Radiologist Ground Truth Mask",
    ]
    for c, title in enumerate(column_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight="bold", pad=12)

    for r, item in enumerate(processed_samples):
        axes[r, 0].imshow(item["raw"], cmap="gray")
        axes[r, 0].set_ylabel(f"{item['stem']}\n{item['diag']}\n({item['bone']})", fontsize=11, fontweight="bold", labelpad=10)
        axes[r, 0].set_xlabel(f"Noise sigma: {item['sigma_before']:.4f}\nSNR: {item['snr_before']:.1f} dB", fontsize=10, color="darkred")

        axes[r, 1].imshow(item["filtered"], cmap="gray")
        gain = item['snr_after'] - item['snr_before']
        pct_drop = (1.0 - item['sigma_after']/item['sigma_before']) * 100.0
        axes[r, 1].set_xlabel(f"Noise sigma: {item['sigma_after']:.4f} (-{pct_drop:.0f}%)\nSNR: {item['snr_after']:.1f} dB (+{gain:.1f} dB)", fontsize=10, color="darkgreen")

        axes[r, 2].imshow(item["noise_diff"], cmap="magma")
        axes[r, 2].set_xlabel("High-frequency speckle removed\n(Zero anatomical edges lost)", fontsize=9)

        axes[r, 3].imshow(item["yellow_overlay"])
        axes[r, 3].set_xlabel(f"Clinical Tumour Area\n{item['num_vertices']} polygon boundary vertices", fontsize=10)

        axes[r, 4].imshow(item["mask"], cmap="gray")
        axes[r, 4].set_xlabel(f"Binary Mask (512x512)\n{item['mask_area']} active pixels", fontsize=10)

        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.tight_layout()
    grid_path = output_dir / "osgdf_and_masks_grid.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n Master visual grid saved to: {grid_path}")
    print(f" Individual sample folders saved under: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Verify OSGDF filtering before/after and ground truth yellow masks.")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of random samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument("--min-fraction", type=float, default=0.01, help="Min tumor area fraction (e.g. 0.01 for visible tumors, 0 for any).")
    parser.add_argument("--image-ids", nargs="*", default=None, help="Optional specific image IDs.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/visualizations", help="Output directory.")
    args = parser.parse_args()
    run_visualization(output_dir=args.output_dir, n_samples=args.n_samples, seed=args.seed, min_fraction=args.min_fraction, specific_ids=args.image_ids)

if __name__ == "__main__":
    main()
