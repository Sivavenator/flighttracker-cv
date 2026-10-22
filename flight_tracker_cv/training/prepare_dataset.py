"""
Dataset preparation pipeline for FlightTracker CV.

Steps:
  1. Download FGVC-Aircraft 2013b dataset
  2. Convert to YOLO label format (normalised x_centre y_centre w h)
  3. Build train / val / test splits
  4. Augment training images
  5. Organise classifier crops
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FGVC_URL = "http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
FGVC_ARCHIVE = "fgvc-aircraft-2013b.tar.gz"

# Full 100-class list matching training/dataset.yaml
AIRCRAFT_CLASSES: List[str] = [
    "707-320", "727-200", "737-200", "737-300", "737-400", "737-500",
    "737-600", "737-700", "737-800", "737-900", "747-100", "747-200",
    "747-300", "747-400", "757-200", "757-300", "767-200", "767-300",
    "767-400", "777-200", "777-300", "787-8", "787-9", "787-10",
    "A300B4", "A310", "A318", "A319", "A320", "A321", "A330-200",
    "A330-300", "A340-200", "A340-300", "A340-500", "A340-600",
    "A350-900", "A350-1000", "A380", "A220-100", "A220-300",
    "ATR-42", "ATR-72", "An-12",
    "BAE 146-200", "BAE 146-300", "BAE-125",
    "Beechcraft 1900",
    "Boeing 717",
    "C-130", "C-47",
    "CRJ-200", "CRJ-700", "CRJ-900",
    "Cessna 172", "Cessna 208", "Cessna Citation",
    "Challenger 600",
    "DC-10", "DC-3", "DC-6", "DC-8", "DC-9-30",
    "DHC-1", "DHC-6",
    "Dash 8 Q400",
    "Dornier 328",
    "E-170", "E-175", "E-190", "E-195",
    "EMB-120",
    "ERJ-135", "ERJ-145",
    "Embraer Legacy 600",
    "Eurofighter Typhoon",
    "F/A-18",
    "Fokker 100", "Fokker 50", "Fokker 70",
    "Global Express",
    "Gulfstream IV", "Gulfstream V",
    "Hawk T1",
    "IL-76",
    "King Air 200",
    "L-1011",
    "MD-11", "MD-80", "MD-87", "MD-90",
    "Metroliner",
    "Model B200",
    "PA-28",
    "SR-20",
    "Saab 2000", "Saab 340",
    "Spitfire",
    "Tornado",
    "Tristar",
    "Tu-134", "Tu-154",
    "Yak-42",
    "737-MAX7", "737-MAX8", "737-MAX9", "737-MAX10",
    "A321neo", "A321XLR", "A330neo",
]
while len(AIRCRAFT_CLASSES) < 100:
    AIRCRAFT_CLASSES.append(f"unknown_{len(AIRCRAFT_CLASSES)}")
AIRCRAFT_CLASSES = AIRCRAFT_CLASSES[:100]

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(AIRCRAFT_CLASSES)}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_fgvc_aircraft(dest_dir: Path) -> Path:
    """Download and extract the FGVC-Aircraft 2013b dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / FGVC_ARCHIVE
    extracted_path = dest_dir / "fgvc-aircraft-2013b"

    if extracted_path.exists():
        logger.info("FGVC-Aircraft already extracted at %s", extracted_path)
        return extracted_path

    if not archive_path.exists():
        logger.info("Downloading FGVC-Aircraft dataset (~2.75 GB)...")
        response = requests.get(FGVC_URL, stream=True, timeout=300)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(archive_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc="FGVC-Aircraft"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
                pbar.update(len(chunk))
        logger.info("Download complete: %s", archive_path)

    logger.info("Extracting %s ...", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    logger.info("Extracted to %s", extracted_path)
    return extracted_path


# ---------------------------------------------------------------------------
# FGVC bounding-box annotation parser
# ---------------------------------------------------------------------------

def _parse_fgvc_boxes(fgvc_root: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Parse data/images_box.txt which has lines like:
        0034309  16  10 480 311
    Returns dict mapping image name (no extension) -> (x1, y1, x2, y2).
    """
    box_file = fgvc_root / "data" / "images_box.txt"
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    if not box_file.exists():
        logger.warning("images_box.txt not found at %s", box_file)
        return boxes
    with open(box_file, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            name = parts[0]
            x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            boxes[name] = (x1, y1, x2, y2)
    return boxes


def _parse_fgvc_labels(fgvc_root: Path) -> Dict[str, str]:
    """
    Parse FGVC variant label files. The dataset ships split-specific files:
        data/images_variant_train.txt
        data/images_variant_val.txt
        data/images_variant_test.txt
        data/images_variant_trainval.txt   (train+val combined)
    Each line: <image_name> <variant label>
    We merge all of them so every image gets a label regardless of split.
    Falls back to images_manufacturer.txt if none of the above exist.
    """
    data_dir = fgvc_root / "data"
    labels: Dict[str, str] = {}

    # Candidate files in priority order — collect from all that exist
    candidates = [
        data_dir / "images_variant_trainval.txt",
        data_dir / "images_variant_train.txt",
        data_dir / "images_variant_val.txt",
        data_dir / "images_variant_test.txt",
        data_dir / "images_variant.txt",          # older releases
        data_dir / "images_manufacturer.txt",     # coarser fallback
    ]

    found_any = False
    for label_file in candidates:
        if not label_file.exists():
            continue
        found_any = True
        logger.info("Loading labels from %s", label_file.name)
        with open(label_file, "r") as fh:
            for line in fh:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    img_name, variant = parts[0], parts[1].strip()
                    # Don't overwrite a variant-level label with a coarser one
                    if img_name not in labels:
                        labels[img_name] = variant

    if not found_any:
        # Last resort: list every *.txt in data/ to help diagnose
        txt_files = list(data_dir.glob("*.txt")) if data_dir.exists() else []
        logger.warning(
            "No label file found in %s/data/  "
            "Files present: %s",
            fgvc_root,
            [f.name for f in txt_files] or "none",
        )
    else:
        logger.info("Loaded %d image labels in total", len(labels))
    return labels


# ---------------------------------------------------------------------------
# YOLO conversion
# ---------------------------------------------------------------------------

def convert_to_yolo_format(fgvc_root: Path, output_dir: Path) -> None:
    """
    Convert FGVC bounding boxes + labels to YOLO format and build splits.
    """
    images_dir = fgvc_root / "data" / "images"
    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        return

    boxes = _parse_fgvc_boxes(fgvc_root)
    labels = _parse_fgvc_labels(fgvc_root)

    if not boxes or not labels:
        logger.error("Could not load bounding boxes or labels from FGVC dataset.")
        return

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    random.seed(42)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(0.70 * n)
    n_val = int(0.20 * n)
    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:],
    }
    logger.info("Total images: %d  train=%d  val=%d  test=%d",
                n, len(splits["train"]), len(splits["val"]), len(splits["test"]))

    skipped = 0
    for split, files in splits.items():
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(files, desc=f"Converting {split}"):
            stem = img_path.stem
            if stem not in boxes or stem not in labels:
                skipped += 1
                continue

            # Copy image
            dst_img = img_out / img_path.name
            shutil.copy2(img_path, dst_img)

            # Read image dimensions
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue
                h, w = img.shape[:2]
            except Exception:
                skipped += 1
                continue

            x1, y1, x2, y2 = boxes[stem]
            variant_label = labels[stem]

            # Map label to class ID
            class_id = _fuzzy_class_id(variant_label)

            # YOLO format: class x_centre y_centre width height (normalised)
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lbl_file = lbl_out / (stem + ".txt")
            with open(lbl_file, "w") as fh:
                fh.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    logger.info("Conversion done. Skipped %d images.", skipped)
    _write_dataset_yaml(output_dir)


def _fuzzy_class_id(label: str) -> int:
    """Map a variant label string to nearest class ID."""
    label_norm = label.strip().upper().replace(" ", "").replace("-", "")
    for class_name, cid in CLASS_TO_ID.items():
        cn_norm = class_name.upper().replace(" ", "").replace("-", "")
        if label_norm == cn_norm or label_norm in cn_norm or cn_norm in label_norm:
            return cid
    # Default to class 0 if no match
    return 0


def _write_dataset_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "dataset.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 100,
        "names": AIRCRAFT_CLASSES,
    }
    with open(yaml_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
    logger.info("Dataset YAML written to %s", yaml_path)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_dataset(images_dir: Path, labels_dir: Path, output_images: Path, output_labels: Path) -> None:
    """Apply albumentations augmentations to training images."""
    try:
        import albumentations as A
        import cv2
        from albumentations.pytorch import ToTensorV2  # noqa: F401
    except ImportError:
        logger.warning("albumentations not installed, skipping augmentation.")
        return

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CLAHE(p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.Rotate(limit=10, p=0.3),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    for img_path in tqdm(image_files, desc="Augmenting"):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        with open(lbl_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    bboxes.append([xc, yc, bw, bh])
                    class_labels.append(cid)

        if not bboxes:
            continue

        try:
            augmented = transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
        except Exception:
            continue

        aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        out_img_path = output_images / f"aug_{img_path.name}"
        out_lbl_path = output_labels / f"aug_{img_path.stem}.txt"
        cv2.imwrite(str(out_img_path), aug_img)
        with open(out_lbl_path, "w") as fh:
            for cid, (xc, yc, bw, bh) in zip(aug_labels, aug_bboxes):
                fh.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# ---------------------------------------------------------------------------
# Classifier dataset organiser
# ---------------------------------------------------------------------------

def create_classifier_dataset(fgvc_root: Path, output_dir: Path) -> None:
    """
    Organise FGVC crops into:
        output_dir/
            <ClassName>/
                image1.jpg ...
    """
    images_dir = fgvc_root / "data" / "images"
    boxes = _parse_fgvc_boxes(fgvc_root)
    labels = _parse_fgvc_labels(fgvc_root)

    if not boxes or not labels:
        logger.error("Missing boxes or labels for classifier dataset.")
        return

    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for img_path in tqdm(list(images_dir.glob("*.jpg")), desc="Cropping classifier"):
        stem = img_path.stem
        if stem not in boxes or stem not in labels:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        x1, y1, x2, y2 = boxes[stem]
        # Add small padding
        pad = 10
        h, w = img.shape[:2]
        x1m = max(0, x1 - pad)
        y1m = max(0, y1 - pad)
        x2m = min(w, x2 + pad)
        y2m = min(h, y2 + pad)

        crop = img[y1m:y2m, x1m:x2m]
        if crop.size == 0:
            continue

        class_name = labels[stem].strip().replace("/", "_").replace(" ", "_")
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(class_dir / img_path.name), crop)
        processed += 1

    logger.info("Classifier dataset: %d crops written to %s", processed, output_dir)


# ---------------------------------------------------------------------------
# Airline livery dataset note
# ---------------------------------------------------------------------------

def download_airline_livery_dataset() -> None:
    """
    Print instructions for obtaining airline livery training data.
    No automated scraping is performed to respect terms of service.
    """
    print("""
=== Airline Livery Dataset ===

No single public dataset covers airline liveries comprehensively.
Recommended sources:

1. Airliners.net — large photo database. Use their API or respectfully
   scrape (check ToS). Organise by airline name.

2. JetPhotos.com — similar to Airliners.net.

3. Plane Spotters Dataset on Kaggle — search for 'aircraft livery' datasets.

4. Open Images V7 (Google) — filter by 'aircraft' and manually label airlines.

5. Wikipedia aircraft type pages often link to freely-licensed images.

For each airline, collect at least 200 images of representative aircraft.
Organise into:
  data/classifier_airlines/<airline_iata>/img001.jpg ...

Then re-run create_classifier_dataset() pointing at this directory.
""")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FlightTracker CV datasets")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    fgvc_dest = data_root / "fgvc"
    yolo_out = data_root / "aircraft_detection"
    classifier_out = data_root / "classifier_crops"

    if not args.skip_download:
        fgvc_root = download_fgvc_aircraft(fgvc_dest)
    else:
        fgvc_root = fgvc_dest / "fgvc-aircraft-2013b"
        if not fgvc_root.exists():
            logger.error("FGVC root not found at %s. Run without --skip-download.", fgvc_root)
            return

    logger.info("Converting FGVC to YOLO format...")
    convert_to_yolo_format(fgvc_root, yolo_out)

    if not args.skip_augment:
        logger.info("Augmenting training images...")
        augment_dataset(
            images_dir=yolo_out / "images" / "train",
            labels_dir=yolo_out / "labels" / "train",
            output_images=yolo_out / "images" / "train",
            output_labels=yolo_out / "labels" / "train",
        )

    logger.info("Creating classifier crop dataset...")
    create_classifier_dataset(fgvc_root, classifier_out)

    download_airline_livery_dataset()

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
