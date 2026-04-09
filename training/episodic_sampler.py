"""
Episodic (N-way K-shot) task sampler for Phase 2 meta-learning.

An *episode* consists of:
  * A support set: N classes × K shots = N·K images with labels in [0, N).
  * A query set:   N classes × Q queries = N·Q images with labels in [0, N).

The sampler randomly draws N classes from the dataset and then samples K
support + Q query images per class (without replacement within the episode).
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Sampler

from .datasets import FUSARShipDataset


# ---------------------------------------------------------------------------
# Episode batch structure
# ---------------------------------------------------------------------------

class EpisodeBatch:
    """
    Container for one N-way K-shot episode.

    Attributes
    ----------
    support_images:
        Shape ``(N, K, 1, H, W)``.
    support_labels:
        Shape ``(N, K)`` — episode-local class indices in [0, N).
    query_images:
        Shape ``(N*Q, 1, H, W)``.
    query_labels:
        Shape ``(N*Q,)`` — episode-local class indices.
    class_names:
        List of N class names (episode-local index → global class name).
    global_class_ids:
        List of N global class labels.
    """

    def __init__(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        class_names: List[str],
        global_class_ids: List[int],
    ) -> None:
        self.support_images = support_images
        self.support_labels = support_labels
        self.query_images = query_images
        self.query_labels = query_labels
        self.class_names = class_names
        self.global_class_ids = global_class_ids

    def to(self, device: torch.device) -> "EpisodeBatch":
        return EpisodeBatch(
            support_images=self.support_images.to(device),
            support_labels=self.support_labels.to(device),
            query_images=self.query_images.to(device),
            query_labels=self.query_labels.to(device),
            class_names=self.class_names,
            global_class_ids=self.global_class_ids,
        )

    @property
    def n_way(self) -> int:
        return self.support_images.size(0)

    @property
    def k_shot(self) -> int:
        return self.support_images.size(1)

    @property
    def n_query(self) -> int:
        return self.query_images.size(0) // self.n_way


# ---------------------------------------------------------------------------
# Episodic sampler
# ---------------------------------------------------------------------------

class EpisodicSampler:
    """
    Sample episodes from a ``FUSARShipDataset``.

    Parameters
    ----------
    dataset:
        The FUSAR-Ship dataset.
    n_way:
        Number of classes per episode.
    k_shot:
        Number of support images per class.
    n_query:
        Number of query images per class.
    num_episodes:
        Total number of episodes to yield per epoch.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: FUSARShipDataset,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        num_episodes: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_episodes = num_episodes

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Build class → indices map
        self._class_to_indices: Dict[str, List[int]] = {
            c: dataset.get_class_samples(c)
            for c in dataset.classes
        }

        # Only keep classes with enough samples
        min_samples = k_shot + n_query
        self._eligible_classes = [
            c for c, idxs in self._class_to_indices.items()
            if len(idxs) >= min_samples
        ]

        if len(self._eligible_classes) < n_way:
            raise ValueError(
                f"Need at least {n_way} classes with ≥ {min_samples} samples each. "
                f"Found {len(self._eligible_classes)} eligible classes."
            )

    def sample_episode(self) -> EpisodeBatch:
        """
        Sample a single episode.

        Returns
        -------
        EpisodeBatch
        """
        # Draw N classes
        chosen_classes = random.sample(self._eligible_classes, self.n_way)

        support_list: List[torch.Tensor] = []
        query_list: List[torch.Tensor] = []
        global_ids: List[int] = []

        for cls in chosen_classes:
            indices = self._class_to_indices[cls]
            selected = random.sample(indices, self.k_shot + self.n_query)
            support_idx = selected[:self.k_shot]
            query_idx = selected[self.k_shot:]

            support_imgs = torch.stack([self.dataset[i][0] for i in support_idx])  # (K, 1, H, W)
            query_imgs = torch.stack([self.dataset[i][0] for i in query_idx])      # (Q, 1, H, W)

            support_list.append(support_imgs)
            query_list.append(query_imgs)
            global_ids.append(self.dataset.class_to_idx[cls])

        # Stack and create episode-local labels
        support_images = torch.stack(support_list, dim=0)     # (N, K, 1, H, W)
        support_labels = torch.arange(self.n_way).unsqueeze(1).expand(self.n_way, self.k_shot)

        query_images = torch.cat(query_list, dim=0)            # (N*Q, 1, H, W)
        query_labels = torch.arange(self.n_way).repeat_interleave(self.n_query)  # (N*Q,)

        return EpisodeBatch(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
            query_labels=query_labels,
            class_names=chosen_classes,
            global_class_ids=global_ids,
        )

    def __iter__(self) -> Iterator[EpisodeBatch]:
        for _ in range(self.num_episodes):
            yield self.sample_episode()

    def __len__(self) -> int:
        return self.num_episodes
