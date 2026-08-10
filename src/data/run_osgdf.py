from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow running either from the project root or from inside src/data/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from osgdf_preprocessing import (
    FITNESS_BLIND,
    FITNESS_PSEUDO_GT,
    FITNESS_SYNTHETIC,
    apply_sg_filter,
    equivalent_poly_order,
    estimate_noise_sigma,
    format_poly_order,
    image_metrics,
    load_grayscale,
    optimize_sg_parameters,
    save_grayscale,
    sg2d_kernel,
    validate_sg_params,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# helpers


def find_images(input_dir: Path) -> list[Path]:
    """Return every readable image file in `input_dir`, sorted by name."""
    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: input directory does not exist: {input_dir}")
    paths = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"ERROR: no images found in {input_dir}")
    return paths


def progress(index: int, total: int, start_time: float, every: int = 50) -> None:
    """Lightweight progress line so tqdm is not a hard dependency."""
    if index % every and index != total:
        return
    elapsed = time.time() - start_time
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    sys.stdout.write(
        f"\r  {index}/{total} images  ({rate:.1f} img/s, ~{remaining:.0f}s left)   "
    )
    sys.stdout.flush()
    if index == total:
        sys.stdout.write("\n")


def save_convergence_plot(history: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print(f"  (skipping convergence plot: {exc})")
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        history["iteration"], history["best_fitness"], label="best fitness", linewidth=2
    )
    ax.plot(
        history["iteration"],
        history["mean_fitness"],
        label="population mean",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_xlabel("TSO iteration")
    ax.set_ylabel("Fitness (SNR, dB)")
    ax.set_title("Tunicate Swarm Optimization convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  convergence plot -> {path}")


def save_comparison_figure(samples, window: int, poly: int, path: Path) -> None:
    """Original / filtered / residual triptych for a few example images."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print(f"  (skipping comparison figure: {exc})")
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.6 * n), squeeze=False)

    for row, (name, original, filtered) in enumerate(samples):
        residual = original - filtered

        axes[row][0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(f"{name}\noriginal", fontsize=9)

        axes[row][1].imshow(filtered, cmap="gray", vmin=0, vmax=1)
        axes[row][1].set_title(f"OSGDF (w={window}, p={poly})", fontsize=9)

        limit = float(np.abs(residual).max()) or 1e-6
        axes[row][2].imshow(residual, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[row][2].set_title("removed component (residual)", fontsize=9)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("OSGDF preprocessing - before / after / residual", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  comparison figure -> {path}")


# main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OSGDF preprocessing (Savitzky-Golay + Tunicate Swarm Optimization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/processed/BTXRD/images")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/BTXRD/osgdf_images")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("outputs/visualizations")
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=40,
        help="images used to calibrate the filter parameters",
    )
    parser.add_argument(
        "--agents", type=int, default=25, help="TSO population size (roadmap: 20-30)"
    )
    parser.add_argument(
        "--iters", type=int, default=60, help="TSO iterations (roadmap: 50-100)"
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--fitness",
        choices=[FITNESS_PSEUDO_GT, FITNESS_SYNTHETIC, FITNESS_BLIND],
        default=FITNESS_PSEUDO_GT,
        help="objective function; see module docstring. "
        "'synthetic' is biased toward under-smoothing on "
        "already-noisy input",
    )
    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=20.0,
        help="guard-term weight, blind fitness only",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["2d", "separable"],
        default="2d",
        help="true bivariate SG fit, or row-then-column",
    )
    parser.add_argument(
        "--no-grid-verify",
        action="store_true",
        help="skip the exhaustive check that TSO found the optimum",
    )

    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="use --window/--poly directly, no search",
    )
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--poly", type=int, default=3)

    parser.add_argument(
        "--output-format",
        choices=["png", "jpg"],
        default="png",
        help="png is lossless and strongly recommended",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="calibrate and report, but write no images",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="process only the first N images (0 = all)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("PHASE 2 - OSGDF PREPROCESSING")
    print("=" * 72)

    # 1. index
    paths = find_images(args.input_dir)
    if args.limit:
        paths = paths[: args.limit]
    print(f"\n[1/6] Found {len(paths)} images in {args.input_dir}")

    # 2. calibration subset
    sample_size = min(args.sample_size, len(paths))
    sample_idx = rng.choice(len(paths), size=sample_size, replace=False)
    calibration_paths = [paths[i] for i in sorted(sample_idx)]
    calibration = [load_grayscale(p) for p in calibration_paths]
    print(
        f"[2/6] Calibration subset: {sample_size} images "
        f"(searching on all {len(paths)} would be wasteful - the filter "
        f"parameters are global)"
    )

    # 3. noise level
    sigmas = [estimate_noise_sigma(im) for im in calibration]
    print(
        f"[3/6] Estimated noise sigma: median {np.median(sigmas):.5f}, "
        f"range [{np.min(sigmas):.5f}, {np.max(sigmas):.5f}] "
        f"(intensity units, 0-1 scale)"
    )

    # 4. optimisation
    if args.skip_optimization:
        window, poly = validate_sg_params(args.window, args.poly)
        print(f"[4/6] Optimisation skipped; using window={window}, poly={poly}")
        opt_result = {
            "window_size": window,
            "poly_order": poly,
            "fitness": None,
            "skipped": True,
        }
    else:
        print(
            f"[4/6] Running TSO ({args.agents} agents x {args.iters} iterations, "
            f"fitness='{args.fitness}')"
        )
        t0 = time.time()
        opt_result = optimize_sg_parameters(
            images=calibration,
            n_agents=args.agents,
            n_iterations=args.iters,
            fitness=args.fitness,
            ssim_weight=args.ssim_weight,
            filter_mode=args.filter_mode,
            seed=args.seed,
            verbose=True,
            verify_with_grid=not args.no_grid_verify,
        )
        window = opt_result["window_size"]
        poly = opt_result["poly_order"]
        print(
            f"      done in {time.time() - t0:.1f}s -> "
            f"window={window}, poly={poly}, fitness={opt_result['fitness']:.4f}"
        )
        print(
            f"      TSO made {opt_result['tso_evaluations']} calls but only "
            f"{opt_result['unique_configs_evaluated']} distinct configurations "
            f"exist in the cache"
        )
        if "tso_found_global_optimum" in opt_result:
            verdict = (
                "matched the exhaustive optimum"
                if opt_result["tso_found_global_optimum"]
                else "did NOT match the exhaustive optimum (grid result used)"
            )
            print(f"      grid verification: TSO {verdict}")

    kernel = sg2d_kernel(window, poly)
    partner = equivalent_poly_order(poly)
    if partner is not None:
        print(
            f"      NOTE: polynomial order {poly} and order {partner} give "
            f"bit-identical kernels, so the search cannot distinguish them. "
            f"Report them as a pair."
        )
    print(
        f"      kernel: {kernel.shape[0]}x{kernel.shape[1]}, "
        f"centre weight {kernel[window // 2, window // 2]:.4f}, "
        f"sum {kernel.sum():.6f}"
    )

    # 5. apply to everything
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[5/6] Applying OSGDF to {len(paths)} images "
        f"{'(dry run - nothing written)' if args.dry_run else f'-> {args.output_dir}'}"
    )

    rows = []
    figure_samples = []
    start = time.time()

    for i, path in enumerate(paths, start=1):
        original = load_grayscale(path)
        filtered = apply_sg_filter(original, window, poly, mode=args.filter_mode)

        metrics = image_metrics(original, filtered)
        metrics["filename"] = path.name
        rows.append(metrics)

        if not args.dry_run:
            # Keep the stem identical so masks/ and bboxes/ still line up.
            out_path = args.output_dir / f"{path.stem}.{args.output_format}"
            save_grayscale(out_path, filtered)

        if len(figure_samples) < 3 and i % max(1, len(paths) // 4) == 0:
            figure_samples.append((path.name, original, filtered))

        progress(i, len(paths), start)

    # 6. reports
    print("[6/6] Writing reports")

    snr_before = np.array([r["snr_before_db"] for r in rows])
    snr_after = np.array([r["snr_after_db"] for r in rows])
    edge = np.array([r["edge_preservation"] for r in rows])

    summary = {
        "n_images": len(rows),
        "window_size": int(window),
        "poly_order": int(poly),
        "filter_mode": args.filter_mode,
        "fitness_mode": None if args.skip_optimization else args.fitness,
        "snr_before_db": {
            "mean": float(snr_before.mean()),
            "std": float(snr_before.std()),
            "min": float(snr_before.min()),
            "max": float(snr_before.max()),
        },
        "snr_after_db": {
            "mean": float(snr_after.mean()),
            "std": float(snr_after.std()),
            "min": float(snr_after.min()),
            "max": float(snr_after.max()),
        },
        "mean_snr_gain_db": float((snr_after - snr_before).mean()),
        "mean_edge_preservation": float(edge.mean()),
        "snr_metric": (
            "blind SNR = 10*log10(mean_signal^2 / sigma_noise^2), "
            "sigma via Immerkaer (1996)"
        ),
        "output_format": args.output_format,
        "seed": args.seed,
    }

    params_path = args.report_dir / "osgdf_params.json"
    with open(params_path, "w") as fh:
        json.dump({"summary": summary, "optimization": opt_result}, fh, indent=2)
    print(f"  parameters + optimisation log -> {params_path}")

    csv_path = args.report_dir / "osgdf_snr_report.csv"
    fieldnames = [
        "filename",
        "snr_before_db",
        "snr_after_db",
        "snr_gain_db",
        "noise_sigma_before",
        "noise_sigma_after",
        "psnr_vs_original_db",
        "edge_preservation",
        "ssim_vs_original",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  per-image SNR report -> {csv_path}")

    if not args.skip_optimization and "history" in opt_result:
        save_convergence_plot(
            opt_result["history"], args.figure_dir / "tso_convergence.png"
        )
    if figure_samples:
        save_comparison_figure(
            figure_samples, window, poly, args.figure_dir / "osgdf_comparison.png"
        )

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(
        f"  Parameters       : window={window}, "
        f"polynomial order={format_poly_order(poly)}"
    )
    print(
        f"  SNR before       : {snr_before.mean():6.2f} +/- {snr_before.std():.2f} dB"
    )
    print(f"  SNR after        : {snr_after.mean():6.2f} +/- {snr_after.std():.2f} dB")
    print(f"  Mean gain        : {summary['mean_snr_gain_db']:+6.2f} dB")
    print(f"  Edge preservation: {edge.mean():6.3f}  (1.0 = structure fully retained)")
    print(f"  Elapsed          : {time.time() - start:.1f}s")
    print("=" * 72)
    print("\nNext: Phase 3 - point dataset_loader.py at the osgdf_images/ directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
