import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_FILES = [
    "01_radial_rotation_comparison.png",
    "02_longitudinal_shear_comparison.png",
    "03_timeoscillation_comparison.png",
    "04_peninsula_trajectories.png",
    "05_stommel_trajectories.png",
    "06_dampedoscillation_comparison.png",
    "07_brownian_histograms.png",
]


def resolve_input_dir(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    return Path(__file__).resolve().parent / "output"


def build_overview_figure(input_dir: Path, output_path: Path) -> None:
    missing = [name for name in PLOT_FILES if not (input_dir / name).exists()]
    if missing:
        missing_list = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Missing expected plot files in input directory:\n" + missing_list
        )

    fig, axes = plt.subplots(3, 3, figsize=(11.7, 8.3))
    axes = axes.flatten()
    labels = [f"({chr(ord('a') + idx)})" for idx in range(len(PLOT_FILES))]

    axis_order = [0, 1, 2, 3, 4, 5, 7]
    for idx, (name, label) in enumerate(zip(PLOT_FILES, labels, strict=True)):
        ax = axes[axis_order[idx]]
        image = plt.imread(input_dir / name)
        image = trim_white_borders(image)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
            fontstyle="italic",
            family="Arial",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
        )

    for idx in range(len(axes)):
        if idx not in axis_order:
            axes[idx].axis("off")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.00, hspace=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def trim_white_borders(image: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image[..., :3]

    mask = np.any(rgb < threshold, axis=-1)
    if not np.any(mask):
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return image[r0:r1, c0:c1]


def build_metrics_summary(input_dir: Path, output_path: Path) -> None:
    metric_files = sorted(input_dir.glob("*_metrics.txt"))
    if not metric_files:
        raise FileNotFoundError("No *_metrics.txt files found in input directory.")

    lines: list[str] = []
    for path in metric_files:
        lines.append(path.name)
        lines.append("-" * len(path.name))
        content = path.read_text(encoding="utf-8").strip()
        if content:
            lines.extend(content.splitlines())
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 7-panel benchmark overview figure and metrics summary."
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing benchmark plots and metrics.",
    )
    parser.add_argument(
        "--output-figure",
        help="Output path for the combined overview figure.",
    )
    parser.add_argument(
        "--output-metrics",
        help="Output path for the metrics summary text file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_figure = Path(args.output_figure) if args.output_figure else input_dir / "benchmarks_overview.png"
    output_metrics = Path(args.output_metrics) if args.output_metrics else input_dir / "benchmarks_metrics_summary.txt"

    build_overview_figure(input_dir, output_figure)
    build_metrics_summary(input_dir, output_metrics)


if __name__ == "__main__":
    main()
