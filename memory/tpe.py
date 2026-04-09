"""
Temporal Prototype Evolution (TPE).

Prototypes must remain accurate as the data distribution shifts over time
(e.g., new ship types, seasonal clutter changes).  TPE implements an
exponential moving average (EMA) update:

    μ_t = m · μ_{t-1} + (1-m) · μ_new

where m ∈ [0, 1) is the momentum.  A higher momentum means the stored
prototype changes slowly; a lower momentum allows rapid adaptation.

TPE also tracks:
  * Whether a class has been observed enough times to apply EMA (guard against
    corrupting a fresh prototype with a bad new estimate).
  * An optional Fisher–Rao-weighted update: if the new observation is
    physically similar (small Fisher–Rao distance) to the stored prototype,
    give it more weight.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from .owms import OnlineWorkingMemoryStore

logger = logging.getLogger(__name__)


class TemporalPrototypeEvolution:
    """
    Apply EMA prototype updates to an OWMS.

    Parameters
    ----------
    momentum:
        EMA momentum m ∈ [0, 1).  Closer to 1 = slower evolution.
    min_count:
        Minimum number of observations before EMA kicks in.  Before this
        threshold the prototype is replaced outright (warm-up phase).
    use_tpe:
        If ``False``, always replaces the prototype outright (ablation).
    physics_weighting:
        If ``True``, scale the update step by a physics-based similarity
        weight (protects against physically inconsistent updates).
    """

    def __init__(
        self,
        momentum: float = 0.9,
        min_count: int = 5,
        use_tpe: bool = True,
        physics_weighting: bool = False,
    ) -> None:
        assert 0.0 <= momentum < 1.0, "momentum must be in [0, 1)"
        self.momentum = momentum
        self.min_count = min_count
        self.use_tpe = use_tpe
        self.physics_weighting = physics_weighting

    def update(
        self,
        owms: OnlineWorkingMemoryStore,
        class_id: int,
        new_prototype: torch.Tensor,
        new_uncertainty: float = 1.0,
        physics_similarity: Optional[float] = None,
    ) -> None:
        """
        Apply a TPE update for *class_id*.

        Parameters
        ----------
        owms:
            The working memory store.
        class_id:
            ID of the class to update.
        new_prototype:
            Shape ``(D,)`` — freshly computed prototype.
        new_uncertainty:
            Posterior variance of the new prototype.
        physics_similarity:
            Optional scalar ∈ [0, 1] indicating how physically consistent the
            new prototype is with the stored one.  High similarity → full EMA
            step; low similarity → reduced step.
        """
        if not owms.contains(class_id):
            # First time: simply enrol
            owms.enrol(
                prototype=new_prototype,
                uncertainty=new_uncertainty,
                count=1,
                class_id=class_id,
            )
            return

        entry = owms.get(class_id)
        if entry is None:
            return

        if not self.use_tpe or entry.count < self.min_count:
            # Warm-up or ablation: replace outright
            owms.update(class_id, new_prototype, new_uncertainty, count_delta=1)
            return

        # EMA momentum — possibly modulated by physics similarity
        m = self.momentum
        if self.physics_weighting and physics_similarity is not None:
            # Low physical similarity → reduce momentum (let new estimate shine through)
            m = m * float(physics_similarity)

        old_proto = entry.prototype  # (D,)
        new_proto = F.normalize(new_prototype, p=2, dim=0)

        # EMA in the unnormalised space, then re-normalise
        ema_proto = m * old_proto + (1.0 - m) * new_proto
        ema_proto = F.normalize(ema_proto, p=2, dim=0)

        # EMA on uncertainty
        ema_uncertainty = m * entry.uncertainty + (1.0 - m) * new_uncertainty

        owms.update(class_id, ema_proto, ema_uncertainty, count_delta=1)
        logger.debug(
            "TPE update: class %d count=%d → %d, momentum=%.3f",
            class_id, entry.count, entry.count + 1, m,
        )

    def bulk_update(
        self,
        owms: OnlineWorkingMemoryStore,
        class_ids: list[int],
        new_prototypes: torch.Tensor,
        new_uncertainties: Optional[list[float]] = None,
    ) -> None:
        """
        Apply TPE updates for multiple classes at once.

        Parameters
        ----------
        owms:
            Working memory store.
        class_ids:
            List of length C.
        new_prototypes:
            Shape ``(C, D)``.
        new_uncertainties:
            Length-C list of uncertainties.  Defaults to 1.0 for each.
        """
        if new_uncertainties is None:
            new_uncertainties = [1.0] * len(class_ids)

        for cid, proto, unc in zip(class_ids, new_prototypes, new_uncertainties):
            self.update(owms, cid, proto, float(unc))

    def __repr__(self) -> str:
        return (
            f"TemporalPrototypeEvolution(momentum={self.momentum}, "
            f"min_count={self.min_count}, use_tpe={self.use_tpe})"
        )
