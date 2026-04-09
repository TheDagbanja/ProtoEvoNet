"""
Dataset loaders for Phase 1 (HRSID) and Phase 2 (FUSAR-Ship).

HRSID
-----
High-Resolution SAR Images Dataset for ship detection and instance segmentation.
Directory layout expected::

    data/hrsid/
        images/       # .png SAR chips
        annotations/  # COCO-format JSON files

We load the image chips and (for contrastive learning) pair chips that
contain the same class of vessel.  For Phase 1 detection pre-training we
treat it as a binary classification task (ship / no-ship) with contrastive
learning to pull ship embeddings together and push background apart.

FUSAR-Ship
----------
16-class SAR ship classification dataset.
Directory layout expected::

    data/fusar_ship/
        <ClassName>/   # one folder per class
            *.png / *.jpg / *.tiff

Both datasets return single-channel (grayscale) images normalised to [0, 1].
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _load_sar_image(path: str, size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    """
    Load a SAR image chip, convert to float32 grayscale in [0, 1].

    Parameters
    ----------
    path:
        File path to the image.
    size:
        ``(H, W)`` to resize to.

    Returns
    -------
    torch.Tensor
        Shape ``(1, H, W)``, dtype float32.
    """
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# HRSID dataset
# ---------------------------------------------------------------------------

class HRSIDDataset(Dataset):
    """
    HRSID SAR ship-detection dataset.

    Loads image chips from ``images/`` and their COCO annotations.  Each
    sample is a (image, label, image_id) triple where label = 1 for ship,
    0 for background.

    Parameters
    ----------
    root:
        Path to ``data/hrsid/``.
    split:
        ``"train"`` or ``"test"``.
    image_size:
        ``(H, W)`` to resize images to.
    transform:
        Optional callable applied to the image tensor.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.image_dir = self.root / "images"
        anno_path = self.root / "annotations" / f"{split}.json"

        self.samples: List[Dict] = []

        if anno_path.exists():
            with open(anno_path) as fh:
                coco = json.load(fh)
            # Build image id → filename map
            id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
            # Build image id → has_ship map
            ship_ids = set(ann["image_id"] for ann in coco.get("annotations", []))
            for img in coco["images"]:
                fpath = self.image_dir / img["file_name"]
                if fpath.exists():
                    self.samples.append({
                        "path": str(fpath),
                        "label": 1 if img["id"] in ship_ids else 0,
                        "image_id": img["id"],
                    })
        else:
            # Fallback: load all images in the images/ directory as ship=1
            for fname in sorted(os.listdir(self.image_dir)):
                if fname.lower().endswith((".png", ".jpg", ".tif", ".tiff")):
                    self.samples.append({
                        "path": str(self.image_dir / fname),
                        "label": 1,
                        "image_id": len(self.samples),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        img = _load_sar_image(sample["path"], self.image_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, sample["label"], sample["image_id"]

    def get_positive_pairs(self, idx: int) -> List[int]:
        """Return indices of samples with the same label as *idx*."""
        label = self.samples[idx]["label"]
        return [i for i, s in enumerate(self.samples) if s["label"] == label and i != idx]


# ---------------------------------------------------------------------------
# FUSAR-Ship dataset
# ---------------------------------------------------------------------------

class FUSARShipDataset(Dataset):
    """
    FUSAR-Ship 16-class SAR ship classification dataset.

    Assumes the class-folder layout::

        root/
            Cargo/      image files …
            Tanker/     image files …
            …

    Parameters
    ----------
    root:
        Path to ``data/fusar_ship/``.
    image_size:
        ``(H, W)`` to resize images to.
    transform:
        Optional callable applied to the image tensor.
    classes:
        Explicit class list to use.  If ``None``, auto-detected from
        directory names (sorted alphabetically).
    """

    # Canonical 16-class list (used to ensure reproducible class indices)
    CANONICAL_CLASSES = [
        "Cargo", "DiveVessel", "Dredger", "Fishing",
        "HighSpeedCraft", "LawEnforce", "Other", "Passenger",
        "PortTender", "Reserved", "SAR", "Tanker",
        "Tug", "Unspecified", "WingInGrnd", "Other",
    ]

    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform

        # Discover classes
        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted([
                d.name for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
                    and not d.name.lower().endswith((".csv", ".xls", ".txt"))
            ])

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        # Build sample list — images may be nested under sub-class folders,
        # e.g. Cargo/BulkCarrier/*.tiff, so rglob is used to find all images
        # at any depth under each top-level class directory.
        self.samples: List[Dict] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            for fname in sorted(cls_dir.rglob("*")):
                if fname.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    self.samples.append({
                        "path": str(fname),
                        "class_name": cls,
                        "label": self.class_to_idx[cls],
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        img = _load_sar_image(sample["path"], self.image_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, sample["label"], sample["class_name"]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_class_samples(self, class_name: str) -> List[int]:
        """Return all sample indices belonging to *class_name*."""
        return [i for i, s in enumerate(self.samples) if s["class_name"] == class_name]

    def per_class_counts(self) -> Dict[str, int]:
        """Return a dict mapping class name → number of images."""
        counts: Dict[str, int] = {c: 0 for c in self.classes}
        for s in self.samples:
            counts[s["class_name"]] += 1
        return counts


# ---------------------------------------------------------------------------
# MSTAR dataset
# ---------------------------------------------------------------------------

class MSTARDataset(Dataset):
    """
    MSTAR SAR ground-vehicle classification dataset.

    Handles two common directory layouts automatically:

    Flat (class folders directly under root)::

        root/
            BMP2/   *.jpeg
            T72/    *.jpeg
            BTR70/  *.jpeg

    Split-wrapped (train / test wrapper above class folders)::

        root/
            test/
                BMP2/  *.jpeg
                T72/   *.jpeg
            train/
                BMP2/  *.jpeg

    The class label is inferred from the first subdirectory that is *not* a
    recognised split-wrapper name (train / test / val / data).

    Parameters
    ----------
    root:
        Path to the MSTAR data directory.
    image_size:
        ``(H, W)`` to resize images to.
    transform:
        Optional callable applied to the loaded image tensor.
    """

    _SPLIT_DIRS: set = {"train", "test", "val", "validation", "data"}
    _IMAGE_EXTS: set = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform

        self.samples: List[Dict] = []
        class_names: set = set()

        for img_path in sorted(self.root.rglob("*")):
            if img_path.suffix.lower() not in self._IMAGE_EXTS:
                continue

            relative = img_path.relative_to(self.root)
            parts = relative.parts   # e.g. ('test', 'BMP2', 'HB03871.jpeg')

            if len(parts) < 2:
                continue  # image at root level — no class directory

            # Walk path parts (excluding filename) to find first non-split dir
            cls_name: Optional[str] = None
            for part in parts[:-1]:
                if part.lower() not in self._SPLIT_DIRS:
                    cls_name = part
                    break

            if cls_name is None:
                continue

            class_names.add(cls_name)
            self.samples.append({
                "path": str(img_path),
                "class_name": cls_name,
            })

        self.classes: List[str] = sorted(class_names)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        for s in self.samples:
            s["label"] = self.class_to_idx[s["class_name"]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        img = _load_sar_image(sample["path"], self.image_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, sample["label"], sample["class_name"]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_class_samples(self, class_name: str) -> List[int]:
        """Return all sample indices belonging to *class_name*."""
        return [i for i, s in enumerate(self.samples) if s["class_name"] == class_name]

    def per_class_counts(self) -> Dict[str, int]:
        """Return a dict mapping class name → number of images."""
        counts: Dict[str, int] = {c: 0 for c in self.classes}
        for s in self.samples:
            counts[s["class_name"]] += 1
        return counts
