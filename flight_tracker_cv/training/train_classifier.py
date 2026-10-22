"""
EfficientNetV2-M dual-head aircraft type + airline classifier trainer.

Usage:
    python -m flight_tracker_cv.training.train_classifier --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AircraftDataset:
    """
    Loads image crops + aircraft_type label + airline label from a directory layout:

        root/
            <aircraft_type>/<airline>/<image.jpg>
            OR
            <aircraft_type>/<image.jpg>   (airline label = "unknown")
    """

    def __init__(
        self,
        root: Path,
        aircraft_classes: List[str],
        airline_classes: List[str],
        transform=None,
    ) -> None:
        import torch
        from torch.utils.data import Dataset  # noqa: F401

        self.root = root
        self.aircraft_classes = aircraft_classes
        self.airline_classes = airline_classes
        self.transform = transform
        self.samples: List[Tuple[Path, int, int]] = []

        self._aircraft_to_id = {c: i for i, c in enumerate(aircraft_classes)}
        self._airline_to_id = {c: i for i, c in enumerate(airline_classes)}

        self._scan()

    def _scan(self) -> None:
        for type_dir in sorted(self.root.iterdir()):
            if not type_dir.is_dir():
                continue
            type_label = type_dir.name.replace("_", " ")
            type_id = self._fuzzy_type_id(type_label)

            for item in type_dir.rglob("*.jpg"):
                # Check if parent dir (one level up from image) is an airline
                airline_id = 0
                parent = item.parent
                if parent != type_dir:
                    airline_name = parent.name.replace("_", " ")
                    airline_id = self._fuzzy_airline_id(airline_name)
                self.samples.append((item, type_id, airline_id))

        logger.info("AircraftDataset: %d samples in %s", len(self.samples), self.root)

    def _fuzzy_type_id(self, label: str) -> int:
        label_norm = label.upper().replace(" ", "").replace("-", "")
        for name, idx in self._aircraft_to_id.items():
            n = name.upper().replace(" ", "").replace("-", "")
            if label_norm == n or label_norm in n or n in label_norm:
                return idx
        return 0

    def _fuzzy_airline_id(self, label: str) -> int:
        label_norm = label.lower()
        for name, idx in self._airline_to_id.items():
            if label_norm in name.lower() or name.lower() in label_norm:
                return idx
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import cv2
        import torch

        img_path, type_id, airline_id = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            import torchvision.transforms.functional as F
            from PIL import Image
            img = F.to_tensor(Image.fromarray(img))

        return img, torch.tensor(type_id, dtype=torch.long), torch.tensor(airline_id, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_model(num_aircraft: int, num_airline: int):
    """Create dual-head EfficientNetV2-M classifier."""
    try:
        import timm
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("timm and torch required") from exc

    class DualHeadClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnetv2_m",
                pretrained=True,
                num_classes=0,
            )
            feat_dim = self.backbone.num_features
            self.aircraft_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feat_dim, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_aircraft),
            )
            self.airline_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feat_dim, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_airline),
            )

        def forward(self, x):
            feats = self.backbone(x)
            return self.aircraft_head(feats), self.airline_head(feats)

    return DualHeadClassifier()


# ---------------------------------------------------------------------------
# Label-smoothing loss
# ---------------------------------------------------------------------------

def _label_smoothing_loss(logits, targets, num_classes: int, smoothing: float = 0.1):
    import torch
    import torch.nn.functional as F

    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    smooth_loss = -log_probs.mean(dim=-1)
    ce_loss = F.nll_loss(log_probs, targets, reduction="none")
    loss = confidence * ce_loss + smoothing * smooth_loss
    return loss.mean()


# ---------------------------------------------------------------------------
# Train / val epochs
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, device, num_ac: int, num_air: int, smoothing: float):
    import torch

    model.train()
    total_loss = 0.0
    correct_type = 0
    correct_airline = 0
    n = 0

    for imgs, type_labels, airline_labels in loader:
        imgs = imgs.to(device)
        type_labels = type_labels.to(device)
        airline_labels = airline_labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            type_logits, airline_logits = model(imgs)
            loss_type = _label_smoothing_loss(type_logits, type_labels, num_ac, smoothing)
            loss_airline = _label_smoothing_loss(airline_logits, airline_labels, num_air, smoothing)
            loss = loss_type + 0.5 * loss_airline

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct_type += (type_logits.argmax(1) == type_labels).sum().item()
        correct_airline += (airline_logits.argmax(1) == airline_labels).sum().item()
        n += imgs.size(0)

    return total_loss / n, correct_type / n, correct_airline / n


def val_epoch(model, loader, device, num_ac: int, num_air: int):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    correct_type = 0
    correct_airline = 0
    n = 0
    all_type_true: List[int] = []
    all_type_pred: List[int] = []

    with torch.no_grad():
        for imgs, type_labels, airline_labels in loader:
            imgs = imgs.to(device)
            type_labels = type_labels.to(device)
            airline_labels = airline_labels.to(device)

            type_logits, airline_logits = model(imgs)
            loss = (
                _label_smoothing_loss(type_logits, type_labels, num_ac)
                + 0.5 * _label_smoothing_loss(airline_logits, airline_labels, num_air)
            )
            total_loss += loss.item() * imgs.size(0)

            pred_type = type_logits.argmax(1)
            pred_airline = airline_logits.argmax(1)
            correct_type += (pred_type == type_labels).sum().item()
            correct_airline += (pred_airline == airline_labels).sum().item()
            n += imgs.size(0)

            all_type_true.extend(type_labels.cpu().numpy().tolist())
            all_type_pred.extend(pred_type.cpu().numpy().tolist())

    type_f1 = f1_score(all_type_true, all_type_pred, average="macro", zero_division=0)
    return total_loss / n, correct_type / n, correct_airline / n, type_f1


# ---------------------------------------------------------------------------
# Full training orchestration
# ---------------------------------------------------------------------------

def train(config_path: str = "config.yaml") -> str:
    """Orchestrate classifier training. Returns path to best checkpoint."""
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    cls_cfg = cfg.get("training", {}).get("classifier", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("models", {}).get("classifier", {})

    num_aircraft = model_cfg.get("num_aircraft_classes", 100)
    num_airline = model_cfg.get("num_airline_classes", 60)

    from flight_tracker_cv.models.classifier import AIRCRAFT_TYPE_LABELS, AIRLINE_LABELS
    from flight_tracker_cv.training.augmentation import (
        get_classifier_train_transforms,
        get_classifier_val_transforms,
    )

    classifier_root = Path(data_cfg.get("dataset_root", "data")) / ".." / "classifier_crops"
    if not classifier_root.exists():
        raise FileNotFoundError(
            f"Classifier dataset not found at {classifier_root}. "
            "Run prepare_dataset.py first."
        )

    train_transform = get_classifier_train_transforms(cls_cfg.get("image_size", 224))
    val_transform = get_classifier_val_transforms(cls_cfg.get("image_size", 224))

    full_dataset = AircraftDataset(
        classifier_root, AIRCRAFT_TYPE_LABELS, AIRLINE_LABELS, transform=None
    )
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms
    train_set.dataset.transform = train_transform
    # Note: val_set shares the same dataset object; transforms are applied per-item

    batch = cls_cfg.get("batch", 32)
    workers = cls_cfg.get("workers", 4)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    console.print(f"[cyan]Training classifier on device: {device}")

    model = _build_model(num_aircraft, num_airline).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cls_cfg.get("lr", 0.001),
        weight_decay=cls_cfg.get("weight_decay", 0.0001),
    )
    epochs = cls_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    smoothing = cls_cfg.get("label_smoothing", 0.1)
    patience = cls_cfg.get("patience", 10)
    ckpt_dir = Path(cls_cfg.get("checkpoint_dir", "weights/classifier_checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_path = str(ckpt_dir / "best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_type_acc, train_air_acc = train_epoch(
            model, train_loader, optimizer, scaler, device, num_aircraft, num_airline, smoothing
        )
        val_loss, val_type_acc, val_air_acc, val_f1 = val_epoch(
            model, val_loader, device, num_aircraft, num_airline
        )
        scheduler.step()
        elapsed = time.time() - t0

        console.print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"type_acc {train_type_acc:.3f}/{val_type_acc:.3f} | "
            f"air_acc {train_air_acc:.3f}/{val_air_acc:.3f} | "
            f"f1 {val_f1:.3f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_type_acc": val_type_acc,
                    "val_airline_acc": val_air_acc,
                    "val_f1": val_f1,
                    "num_aircraft_classes": num_aircraft,
                    "num_airline_classes": num_airline,
                },
                best_path,
            )
            console.print(f"  [green]Saved best checkpoint (val_loss={val_loss:.4f})[/green]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print(f"[yellow]Early stopping after {patience} epochs without improvement.")
                break

    console.print(f"\n[bold green]Training complete. Best checkpoint: {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# Confusion matrix + metrics
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        logger.warning("matplotlib/seaborn/sklearn not available.")
        return

    cm = confusion_matrix(y_true, y_pred)
    # Limit display to top-N classes for readability
    n_show = min(30, len(class_names))
    cm_show = cm[:n_show, :n_show]
    names_show = class_names[:n_show]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_show, annot=True, fmt="d", xticklabels=names_show,
        yticklabels=names_show, cmap="Blues", ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Aircraft Type Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", save_path)


def compute_metrics(model, loader, device: str) -> Dict[str, float]:
    """Compute per-class accuracy dict."""
    import torch
    from sklearn.metrics import classification_report

    model.eval()
    all_type_true: List[int] = []
    all_type_pred: List[int] = []

    with torch.no_grad():
        for imgs, type_labels, _ in loader:
            imgs = imgs.to(device)
            type_logits, _ = model(imgs)
            preds = type_logits.argmax(1).cpu().numpy()
            all_type_pred.extend(preds.tolist())
            all_type_true.extend(type_labels.numpy().tolist())

    report = classification_report(all_type_true, all_type_pred, output_dict=True, zero_division=0)
    return {str(k): v for k, v in report.items() if isinstance(v, (int, float, dict))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train aircraft classifier")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args.config)


if __name__ == "__main__":
    main()
