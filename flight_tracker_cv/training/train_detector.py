"""
YOLOv8 aircraft detector training script.

Usage:
    python -m flight_tracker_cv.training.train_detector --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(config_path: str = "config.yaml") -> str:
    """
    Train YOLOv8 on the aircraft detection dataset.

    Returns path to the best checkpoint.
    """
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    det_cfg = cfg.get("training", {}).get("detector", {})
    model_cfg = cfg.get("models", {}).get("detector", {})
    data_cfg = cfg.get("data", {})

    dataset_yaml = str(
        Path(data_cfg.get("dataset_root", "data/aircraft_detection")) / "dataset.yaml"
    )

    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {dataset_yaml}. "
            "Run training/prepare_dataset.py first."
        )

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError("ultralytics must be installed: pip install ultralytics") from exc

    pretrained = model_cfg.get("pretrained", "yolov8m.pt")
    console.rule("[bold green]FlightTracker CV — Detector Training")
    console.print(f"  Pretrained base : {pretrained}")
    console.print(f"  Dataset         : {dataset_yaml}")
    console.print(f"  Epochs          : {det_cfg.get('epochs', 100)}")
    console.print(f"  Batch size      : {det_cfg.get('batch', 16)}")
    console.print(f"  Image size      : {det_cfg.get('imgsz', 640)}")

    model = YOLO(pretrained)

    # Register custom callbacks
    model.add_callback("on_train_epoch_end", _cb_log_epoch)
    model.add_callback("on_train_end", _cb_train_end)

    results = model.train(
        data=dataset_yaml,
        epochs=det_cfg.get("epochs", 100),
        batch=det_cfg.get("batch", 16),
        imgsz=det_cfg.get("imgsz", 640),
        lr0=det_cfg.get("lr0", 0.01),
        lrf=det_cfg.get("lrf", 0.01),
        momentum=det_cfg.get("momentum", 0.937),
        weight_decay=det_cfg.get("weight_decay", 0.0005),
        warmup_epochs=det_cfg.get("warmup_epochs", 3),
        patience=det_cfg.get("patience", 50),
        workers=det_cfg.get("workers", 8),
        project=det_cfg.get("project", "runs/train"),
        name=det_cfg.get("name", "aircraft_detector"),
        pretrained=det_cfg.get("pretrained", True),
        augment=det_cfg.get("augment", True),
        rect=det_cfg.get("rect", False),
        cos_lr=det_cfg.get("cos_lr", False),
        close_mosaic=det_cfg.get("close_mosaic", 10),
        amp=det_cfg.get("amp", True),
        verbose=True,
        device="",  # auto-select
    )

    # Find best weights path
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    chosen = best_weights if best_weights.exists() else last_weights
    console.print(f"\n[bold green]Training complete. Best weights: {chosen}")
    return str(chosen)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _cb_log_epoch(trainer) -> None:
    """Log loss metrics after each epoch."""
    metrics = trainer.metrics if hasattr(trainer, "metrics") else {}
    epoch = trainer.epoch if hasattr(trainer, "epoch") else "?"
    loss = metrics.get("train/box_loss", metrics.get("box_loss", "N/A"))
    logger.info("Epoch %s — box_loss=%s", epoch, loss)


def _cb_train_end(trainer) -> None:
    """Log final message when training finishes."""
    console.print("[bold cyan]Training loop finished.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model_path: str, dataset_yaml: str) -> dict:
    """
    Run validation on the test split and print per-class metrics.
    Returns a dict with mAP50, mAP50-95, precision, recall.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics required") from exc

    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml, split="test", verbose=True)

    results_dict = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    table = Table(title="Validation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in results_dict.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    # Per-class metrics
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        console.print("\n[bold]Per-class mAP50:")
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = model.names.get(int(cls_idx), f"class_{cls_idx}")
            ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
            console.print(f"  {cls_name:30s}: {ap50:.4f}")

    return results_dict


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_model(model_path: str, export_format: str = "onnx") -> str:
    """Export trained YOLO model to ONNX (or other Ultralytics-supported format)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics required") from exc

    model = YOLO(model_path)
    exported = model.export(format=export_format)
    console.print(f"[green]Model exported to: {exported}")
    return str(exported)


# ---------------------------------------------------------------------------
# Training curve plotting
# ---------------------------------------------------------------------------

def plot_training_curves(results_csv: str, output_dir: Optional[str] = None) -> None:
    """Parse Ultralytics results.csv and plot loss + mAP curves."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib / pandas not available. Skipping plot.")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("YOLOv8 Aircraft Detector Training Curves", fontsize=14)

    # Loss plots
    for ax, col, title in [
        (axes[0, 0], "train/box_loss", "Box Loss (train)"),
        (axes[0, 1], "val/box_loss",   "Box Loss (val)"),
        (axes[1, 0], "metrics/mAP50",  "mAP@0.50"),
        (axes[1, 1], "metrics/mAP50-95", "mAP@0.50-0.95"),
    ]:
        if col in df.columns:
            ax.plot(df["epoch"] if "epoch" in df.columns else range(len(df)),
                    df[col], linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title} (N/A)")

    plt.tight_layout()
    out_path = Path(output_dir or Path(results_csv).parent) / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Training curves saved to {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resume_training(last_pt: Optional[str] = None) -> str:
    """Resume training from the last saved checkpoint."""
    from ultralytics import YOLO

    # Auto-find last.pt if not specified
    if not last_pt:
        candidates = sorted(
            Path("runs/train").glob("**/weights/last.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "No last.pt found under runs/train/. "
                "Training may not have started yet — run without --resume."
            )
        last_pt = str(candidates[0])

    console.print(f"[cyan]Resuming from checkpoint:[/cyan] {last_pt}")
    model = YOLO(last_pt)
    results = model.train(resume=True)
    best = str(Path(results.save_dir) / "weights" / "best.pt")
    console.print(f"[green]Training complete. Best weights:[/green] {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 aircraft detector")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", nargs="?", const="auto", metavar="LAST_PT",
                        help="Resume from last checkpoint. Optionally pass path to last.pt")
    parser.add_argument("--validate-only", metavar="WEIGHTS",
                        help="Run validation only using given weights path")
    parser.add_argument("--export", metavar="WEIGHTS",
                        help="Export weights to ONNX")
    parser.add_argument("--plot-curves", metavar="RESULTS_CSV",
                        help="Plot training curves from a results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.resume:
        last_pt = None if args.resume == "auto" else args.resume
        best_path = resume_training(last_pt)
        console.print(f"[bold]Best checkpoint: {best_path}")

    elif args.validate_only:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        data_root = cfg.get("data", {}).get("dataset_root", "data/aircraft_detection")
        dataset_yaml = str(Path(data_root) / "dataset.yaml")
        validate(args.validate_only, dataset_yaml)

    elif args.export:
        export_model(args.export)

    elif args.plot_curves:
        plot_training_curves(args.plot_curves)

    else:
        best_path = train(args.config)
        console.print(f"[bold]Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
