"""
Prototype Interference Monitor (PIM).

As new classes are enrolled, two prototypes may become very similar in
embedding space (e.g., small cargo vs. medium cargo vessels).  When two
stored prototypes have a high cosine similarity, they "interfere" — a query
near either one is ambiguous.

PIM continuously monitors all pairwise prototype similarities.  When a pair
exceeds the interference threshold τ_pim, it:
  1. Flags the pair with an interference score.
  2. Optionally repels the two prototypes along the line joining them
     (soft-push update) to increase separation.
  3. Emits a warning to the logger for human inspection.

The repulsion step is a gradient-free in-place modification of OWMS entries;
it does NOT backpropagate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .owms import OnlineWorkingMemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InterferencePair:
    """A detected interference between two prototype slots."""
    class_id_a: int
    class_id_b: int
    cosine_similarity: float
    label_a: str = ""
    label_b: str = ""


# ---------------------------------------------------------------------------
# PIM module
# ---------------------------------------------------------------------------

class PrototypeInterferenceMonitor:
    """
    Monitor prototype interference and optionally apply repulsion.

    Parameters
    ----------
    threshold:
        Cosine-similarity threshold above which interference is flagged.
    repulsion_strength:
        Magnitude of the gradient-free prototype push (0 = no repulsion).
    use_pim:
        If ``False``, PIM is a no-op (ablation toggle).
    """

    def __init__(
        self,
        threshold: float = 0.85,
        repulsion_strength: float = 0.01,
        use_pim: bool = True,
    ) -> None:
        self.threshold = threshold
        self.repulsion_strength = repulsion_strength
        self.use_pim = use_pim

        # History of flagged pairs: (id_a, id_b) → list of similarities over time
        self._history: Dict[Tuple[int, int], List[float]] = {}

    def scan(
        self,
        owms: OnlineWorkingMemoryStore,
        apply_repulsion: bool = True,
    ) -> List[InterferencePair]:
        """
        Scan all prototype pairs for interference.

        Parameters
        ----------
        owms:
            The working memory store to scan.
        apply_repulsion:
            If ``True`` and ``repulsion_strength > 0``, push interfering
            prototypes apart in-place.

        Returns
        -------
        list of InterferencePair
            All pairs currently exceeding the threshold.
        """
        if not self.use_pim or len(owms) < 2:
            return []

        protos, ids = owms.all_prototypes()   # (C, D)
        C = protos.size(0)

        # Pairwise cosine similarity: (C, C)
        sim = protos @ protos.T   # L2-normalised, so dot = cos sim

        flagged: List[InterferencePair] = []

        for i in range(C):
            for j in range(i + 1, C):
                s = sim[i, j].item()
                if s > self.threshold:
                    id_a, id_b = ids[i], ids[j]
                    entry_a = owms.get(id_a)
                    entry_b = owms.get(id_b)
                    label_a = entry_a.label if entry_a else ""
                    label_b = entry_b.label if entry_b else ""

                    pair = InterferencePair(
                        class_id_a=id_a,
                        class_id_b=id_b,
                        cosine_similarity=s,
                        label_a=label_a,
                        label_b=label_b,
                    )
                    flagged.append(pair)
                    key = (id_a, id_b)
                    self._history.setdefault(key, []).append(s)

                    logger.warning(
                        "PIM interference: class %d (%s) ↔ class %d (%s) "
                        "cosine=%.4f (threshold=%.2f)",
                        id_a, label_a, id_b, label_b, s, self.threshold,
                    )

                    if apply_repulsion and self.repulsion_strength > 0:
                        self._repel(owms, id_a, id_b, protos[i], protos[j])

        return flagged

    def _repel(
        self,
        owms: OnlineWorkingMemoryStore,
        id_a: int,
        id_b: int,
        proto_a: torch.Tensor,
        proto_b: torch.Tensor,
    ) -> None:
        """
        Apply a small repulsion update to push two prototypes apart.

        The repulsion direction is the unit vector from proto_b to proto_a
        (for proto_a) and from proto_a to proto_b (for proto_b).
        """
        diff = proto_a - proto_b   # (D,)
        diff_norm = diff.norm().clamp(min=1e-8)
        direction = diff / diff_norm  # unit vector

        entry_a = owms.get(id_a)
        entry_b = owms.get(id_b)
        if entry_a is None or entry_b is None:
            return

        new_proto_a = F.normalize(
            entry_a.prototype + self.repulsion_strength * direction, p=2, dim=0
        )
        new_proto_b = F.normalize(
            entry_b.prototype - self.repulsion_strength * direction, p=2, dim=0
        )

        owms.update(id_a, new_proto_a, uncertainty=entry_a.uncertainty, count_delta=0)
        owms.update(id_b, new_proto_b, uncertainty=entry_b.uncertainty, count_delta=0)

        logger.debug(
            "PIM repulsion applied: class %d and %d pushed apart.", id_a, id_b
        )

    def interference_score(self, id_a: int, id_b: int) -> Optional[float]:
        """Return the most recent interference score for a pair, or ``None``."""
        key = (min(id_a, id_b), max(id_a, id_b))
        history = self._history.get(key)
        return history[-1] if history else None

    def persistent_pairs(self, min_occurrences: int = 3) -> List[Tuple[int, int]]:
        """Return pairs that have been flagged at least *min_occurrences* times."""
        return [
            pair
            for pair, hist in self._history.items()
            if len(hist) >= min_occurrences
        ]

    def reset_history(self) -> None:
        """Clear interference history."""
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"PrototypeInterferenceMonitor(threshold={self.threshold}, "
            f"repulsion={self.repulsion_strength}, use_pim={self.use_pim})"
        )
