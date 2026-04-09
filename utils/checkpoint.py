"""
Checkpointing utilities for ProtoEvoNet.

Saves and restores the full system state: backbone, physics layer,
hippocampal system, memory store, inference engine, and optimisers.
"""

from __future__ import annotations

import os
import glob
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    epoch: int,
    state_dicts: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    path:
        Full file path for the checkpoint (.pt file).
    epoch:
        Current epoch number (stored for human readability).
    state_dicts:
        Mapping of name → ``state_dict`` for every component that should be
        persisted (e.g. ``{"backbone": model.backbone.state_dict(), ...}``).
    metrics:
        Optional dict of scalar evaluation metrics to embed in the file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "state_dicts": state_dicts,
        "metrics": metrics or {},
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint → %s  (epoch %d)", path, epoch)


def load_checkpoint(
    path: str,
    state_dict_targets: Dict[str, Any],
    strict: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore state dicts in-place.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file.
    state_dict_targets:
        Mapping of name → module whose ``load_state_dict`` should be called.
        Only keys present in **both** the file and this dict are restored.
    strict:
        Passed to ``load_state_dict``; set ``False`` to ignore missing/extra keys.
    device:
        Device to map tensors to (e.g. ``"cpu"`` or ``"cuda:0"``).

    Returns
    -------
    dict
        The full checkpoint payload (``epoch``, ``metrics``, …).
    """
    payload = torch.load(path, map_location=device)
    saved = payload.get("state_dicts", {})

    for name, module in state_dict_targets.items():
        if name in saved:
            missing, unexpected = module.load_state_dict(saved[name], strict=strict)
            if missing:
                logger.warning("[%s] missing keys: %s", name, missing)
            if unexpected:
                logger.warning("[%s] unexpected keys: %s", name, unexpected)
            logger.info("Restored '%s' from %s", name, path)
        else:
            logger.warning("Key '%s' not found in checkpoint %s", name, path)

    return payload


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Return the most recently modified ``.pt`` file in *checkpoint_dir*, or
    ``None`` if the directory is empty / does not exist.
    """
    pattern = os.path.join(checkpoint_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_best_checkpoint(checkpoint_dir: str, metric: str = "accuracy") -> Optional[str]:
    """
    Scan all checkpoints in *checkpoint_dir* and return the path of the one
    whose ``metrics[metric]`` is highest.
    """
    pattern = os.path.join(checkpoint_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        return None

    best_path: Optional[str] = None
    best_val: float = float("-inf")

    for fp in files:
        try:
            payload = torch.load(fp, map_location="cpu")
            val = payload.get("metrics", {}).get(metric, float("-inf"))
            if val > best_val:
                best_val = val
                best_path = fp
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read checkpoint %s: %s", fp, exc)

    return best_path


class CheckpointManager:
    """
    Manages a rolling window of saved checkpoints plus an always-up-to-date
    ``best.pt`` symlink (or copy on Windows).

    Parameters
    ----------
    directory:
        Folder where checkpoints are written.
    keep_last_n:
        How many recent checkpoints to retain (older ones are deleted).
    metric:
        Metric name used to decide the "best" checkpoint.
    higher_is_better:
        Whether a higher metric value is better.
    """

    def __init__(
        self,
        directory: str,
        keep_last_n: int = 5,
        metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.metric = metric
        self.higher_is_better = higher_is_better
        self._best_value: float = float("-inf") if higher_is_better else float("inf")
        self._saved: list[Path] = []

    def save(
        self,
        epoch: int,
        state_dicts: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save checkpoint and manage rolling window + best checkpoint."""
        metrics = metrics or {}
        fname = self.directory / f"epoch_{epoch:04d}.pt"
        save_checkpoint(str(fname), epoch, state_dicts, metrics)
        self._saved.append(fname)

        # Overwrite best if improved
        current = metrics.get(self.metric, None)
        if current is not None:
            improved = (
                current > self._best_value
                if self.higher_is_better
                else current < self._best_value
            )
            if improved:
                self._best_value = current
                best_path = self.directory / "best.pt"
                import shutil
                shutil.copy2(str(fname), str(best_path))
                logger.info(
                    "New best checkpoint (%.4f) → %s", current, best_path
                )

        # Prune old checkpoints
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()
                logger.debug("Removed old checkpoint %s", old)

        return fname
