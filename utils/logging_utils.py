"""
Logging utilities for ProtoEvoNet.

Sets up structured console + file logging and a lightweight TensorBoard /
CSV writer so experiments can be tracked without heavyweight dependencies.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Root logger configuration
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str,
    run_name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the root logger to write to both *stdout* and a log file.

    Parameters
    ----------
    log_dir:
        Directory in which to place the log file.
    run_name:
        Human-readable experiment label (used as the log file name stem).
        Defaults to a timestamp.
    level:
        Logging level for the root logger.

    Returns
    -------
    logging.Logger
        The root logger, now fully configured.
    """
    os.makedirs(log_dir, exist_ok=True)
    run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}.log")

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    root.info("Logging initialised → %s", log_path)
    return root


# ---------------------------------------------------------------------------
# MetricLogger — thin wrapper that writes CSV + optional TensorBoard
# ---------------------------------------------------------------------------

class MetricLogger:
    """
    Record scalar metrics during training and flush them to a CSV file.
    If TensorBoard is available, also write SummaryWriter events.

    Parameters
    ----------
    log_dir:
        Directory for the CSV file and TensorBoard events.
    run_name:
        Experiment label.
    use_tensorboard:
        Attempt to import and use TensorBoard SummaryWriter.
    """

    def __init__(
        self,
        log_dir: str,
        run_name: Optional[str] = None,
        use_tensorboard: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")

        # CSV writer
        csv_path = self.log_dir / f"{self.run_name}_metrics.csv"
        self._csv_file = open(csv_path, "w", newline="")  # noqa: WPS515
        self._csv_writer: Optional[csv.DictWriter] = None

        # TensorBoard (optional)
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                tb_dir = str(self.log_dir / "tensorboard" / self.run_name)
                self._tb_writer = SummaryWriter(log_dir=tb_dir)
            except ImportError:
                logging.getLogger(__name__).info(
                    "TensorBoard not available – using CSV only."
                )

        self._step = 0

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Record *metrics* at the given *step* (defaults to an internal counter).

        Parameters
        ----------
        metrics:
            Mapping of metric-name → scalar value.
        step:
            Global step / iteration number.
        """
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step + 1

        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}

        # CSV
        if self._csv_writer is None:
            fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # TensorBoard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                try:
                    self._tb_writer.add_scalar(k, float(v), global_step=step)
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        """Flush and close all writers."""
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# AverageMeter — running mean for per-batch metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    """
    Keep a running mean and count of a scalar quantity.

    Typical use::

        loss_meter = AverageMeter("loss")
        for batch in loader:
            loss = compute_loss(batch)
            loss_meter.update(loss.item(), n=len(batch))
        print(loss_meter)  # "loss: 0.3412 (avg)"
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.val:.4f} (avg {self.avg:.4f})"
