"""
Online Working Memory Store (OWMS).

OWMS maintains the current set of class prototypes in a dictionary keyed by
class ID.  Each entry stores:
  * The prototype vector (D,)
  * The posterior uncertainty / variance (scalar)
  * Observation count N
  * Timestamp of last update

The store supports:
  * Enrolment (add a new class)
  * Refinement (update an existing class prototype)
  * Retrieval (get prototype for class ID)
  * Bulk retrieval (get all prototypes as a matrix for nearest-neighbour search)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Prototype entry
# ---------------------------------------------------------------------------

@dataclass
class PrototypeEntry:
    """One slot in the OWMS."""
    class_id: int
    label: str                          # human-readable class name
    prototype: torch.Tensor             # shape (D,), L2-normalised
    uncertainty: float = 1.0           # posterior variance from BPC-GP
    count: int = 0                      # number of support samples seen
    timestamp: float = field(default_factory=time.time)

    def to_device(self, device: torch.device) -> "PrototypeEntry":
        return PrototypeEntry(
            class_id=self.class_id,
            label=self.label,
            prototype=self.prototype.to(device),
            uncertainty=self.uncertainty,
            count=self.count,
            timestamp=self.timestamp,
        )


# ---------------------------------------------------------------------------
# OWMS
# ---------------------------------------------------------------------------

class OnlineWorkingMemoryStore:
    """
    Online Working Memory Store for class prototypes.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of prototype vectors.
    max_classes:
        Maximum number of prototype slots.
    device:
        Torch device for all stored tensors.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        max_classes: int = 200,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.max_classes = max_classes
        self.device = torch.device(device)

        self._store: Dict[int, PrototypeEntry] = {}
        self._next_id: int = 0

    # -----------------------------------------------------------------------
    # Basic CRUD
    # -----------------------------------------------------------------------

    def enrol(
        self,
        prototype: torch.Tensor,
        label: str = "",
        uncertainty: float = 1.0,
        count: int = 1,
        class_id: Optional[int] = None,
    ) -> int:
        """
        Enrol a new class prototype.

        Parameters
        ----------
        prototype:
            Shape ``(D,)``.  Will be L2-normalised before storage.
        label:
            Human-readable class name.
        uncertainty:
            Posterior variance from BPC-GP.
        count:
            Number of support samples used to create this prototype.
        class_id:
            Explicit class ID.  If ``None``, auto-incremented.

        Returns
        -------
        int
            Assigned class ID.
        """
        if len(self._store) >= self.max_classes:
            raise RuntimeError(
                f"OWMS capacity exceeded ({self.max_classes} classes).  "
                "Increase MemoryConfig.max_classes or evict old prototypes."
            )

        if class_id is None:
            cid = self._next_id
            self._next_id += 1
        else:
            cid = class_id
            self._next_id = max(self._next_id, cid + 1)

        proto = F.normalize(prototype.to(self.device), p=2, dim=0)
        self._store[cid] = PrototypeEntry(
            class_id=cid,
            label=label,
            prototype=proto,
            uncertainty=uncertainty,
            count=count,
        )
        return cid

    def update(
        self,
        class_id: int,
        new_prototype: torch.Tensor,
        uncertainty: float = 1.0,
        count_delta: int = 1,
    ) -> None:
        """
        Overwrite the stored prototype for *class_id*.

        Parameters
        ----------
        class_id:
            Must already exist in the store.
        new_prototype:
            Shape ``(D,)``.
        uncertainty:
            New posterior variance.
        count_delta:
            Number of new observations (added to existing count).
        """
        if class_id not in self._store:
            raise KeyError(f"Class {class_id} not found in OWMS.")
        entry = self._store[class_id]
        proto = F.normalize(new_prototype.to(self.device), p=2, dim=0)
        self._store[class_id] = PrototypeEntry(
            class_id=class_id,
            label=entry.label,
            prototype=proto,
            uncertainty=uncertainty,
            count=entry.count + count_delta,
            timestamp=time.time(),
        )

    def get(self, class_id: int) -> Optional[PrototypeEntry]:
        """Retrieve the prototype entry for *class_id*, or ``None``."""
        return self._store.get(class_id)

    def delete(self, class_id: int) -> None:
        """Remove a class from the store."""
        self._store.pop(class_id, None)

    def contains(self, class_id: int) -> bool:
        return class_id in self._store

    # -----------------------------------------------------------------------
    # Bulk retrieval
    # -----------------------------------------------------------------------

    def all_prototypes(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Return all stored prototypes as a matrix.

        Returns
        -------
        prototypes:
            Shape ``(C, D)`` where C = number of enrolled classes.
        class_ids:
            Ordered list of class IDs matching the rows of *prototypes*.
        """
        if not self._store:
            return (
                torch.zeros(0, self.embedding_dim, device=self.device),
                [],
            )
        ids = sorted(self._store.keys())
        protos = torch.stack([self._store[i].prototype for i in ids], dim=0)
        return protos, ids

    def all_uncertainties(self) -> Tuple[torch.Tensor, List[int]]:
        """Return all uncertainties as a vector."""
        if not self._store:
            return torch.zeros(0, device=self.device), []
        ids = sorted(self._store.keys())
        uncertainties = torch.tensor(
            [self._store[i].uncertainty for i in ids],
            device=self.device,
        )
        return uncertainties, ids

    def nearest(
        self,
        query: torch.Tensor,
        k: int = 1,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        Find the k nearest prototypes to a query embedding.

        Parameters
        ----------
        query:
            Shape ``(D,)`` or ``(N, D)`` — L2-normalised query embedding(s).
        k:
            Number of nearest neighbours.

        Returns
        -------
        distances:
            Shape ``(N, k)`` Euclidean distances.
        class_ids:
            List of length k (or (N, k) for batched input) with class IDs.
        prototypes:
            Shape ``(N, k, D)`` — nearest prototype vectors.
        """
        protos, ids = self.all_prototypes()
        if protos.size(0) == 0:
            empty = torch.zeros(0, device=self.device)
            return empty, [], protos

        if query.dim() == 1:
            query = query.unsqueeze(0)

        dists = torch.cdist(query.to(self.device), protos)  # (N, C)
        k = min(k, protos.size(0))
        top_dists, top_idx = dists.topk(k, dim=1, largest=False)  # (N, k)
        top_ids = [[ids[i.item()] for i in row] for row in top_idx]
        top_protos = protos[top_idx]  # (N, k, D)
        return top_dists, top_ids, top_protos

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise the entire store to a plain Python dict."""
        return {
            "embedding_dim": self.embedding_dim,
            "max_classes": self.max_classes,
            "next_id": self._next_id,
            "entries": {
                cid: {
                    "label": e.label,
                    "prototype": e.prototype.cpu(),
                    "uncertainty": e.uncertainty,
                    "count": e.count,
                    "timestamp": e.timestamp,
                }
                for cid, e in self._store.items()
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a state dict produced by ``state_dict()``."""
        self.embedding_dim = sd["embedding_dim"]
        self.max_classes = sd["max_classes"]
        self._next_id = sd["next_id"]
        self._store = {
            int(cid): PrototypeEntry(
                class_id=int(cid),
                label=v["label"],
                prototype=v["prototype"].to(self.device),
                uncertainty=v["uncertainty"],
                count=v["count"],
                timestamp=v["timestamp"],
            )
            for cid, v in sd["entries"].items()
        }

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"OnlineWorkingMemoryStore(classes={len(self)}, "
            f"dim={self.embedding_dim}, device={self.device})"
        )
