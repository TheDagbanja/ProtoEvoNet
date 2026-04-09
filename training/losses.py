"""
Training losses for ProtoEvoNet.

AGCP Loss — AIS-Guided Contrastive Pretraining (Phase 1)
---------------------------------------------------------
An InfoNCE / NT-Xent contrastive loss modified to include physics-guided
negative mining.  For each anchor, standard negatives are drawn randomly;
*hard physics negatives* are drawn from the batch using Fisher–Rao distance
as a selector — embeddings that are *physically similar* (small Fisher–Rao
distance) but from *different classes* are treated as hard negatives and
receive an extra loss penalty.

    L_AGCP = L_InfoNCE + λ_hard · L_hard_negative

PIAM Loss — Physics-Informed Adversarial Meta-learning (Phase 2)
----------------------------------------------------------------
Combines three objectives:

1. Prototypical loss (cross-entropy over query embeddings vs. prototypes):
        L_proto = CE( -d(query, prototype) / T, episode_labels )

2. Physics consistency loss (encourage embeddings to preserve Fisher–Rao
   ordering):
        L_physics = MSE( d_embed(z_i, z_j), d_FR(p_i, p_j) )
        where d_FR is the Fisher–Rao distance between the Gamma params
        of the two image patches.

3. Novelty calibration loss (binary CE for Platt calibration):
        L_novelty = BCE( P(novel | z), is_novel_label )

    L_PIAM = λ_proto · L_proto + λ_physics · L_physics + λ_novelty · L_novelty
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics.fisher_rao import FisherRaoDistance


# ---------------------------------------------------------------------------
# AGCP Loss
# ---------------------------------------------------------------------------

class AGCPLoss(nn.Module):
    """
    AIS-Guided Contrastive Pretraining loss.

    Parameters
    ----------
    temperature:
        InfoNCE temperature τ.
    hard_negative_weight:
        λ_hard — extra weight on physics-guided hard negatives.
    margin:
        Minimum Fisher–Rao distance for a pair to be considered a
        hard negative (pairs closer than *margin* that are different-class
        contribute to the hard negative loss).
    use_fisher_rao:
        If ``False``, hard negatives are selected by embedding distance only.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 0.5,
        margin: float = 0.5,
        use_fisher_rao: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin
        self.fisher_rao = FisherRaoDistance(
            method="approx", use_fisher_rao=use_fisher_rao
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        k_params: Optional[torch.Tensor] = None,
        theta_params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute AGCP loss.

        Parameters
        ----------
        embeddings:
            Shape ``(B, D)`` — L2-normalised embeddings.
        labels:
            Shape ``(B,)`` — integer class labels.
        k_params, theta_params:
            Shape ``(B,)`` — Gamma parameters for each sample.  Required for
            physics-guided hard negatives; if ``None``, only InfoNCE is used.

        Returns
        -------
        loss:
            Scalar loss tensor.
        info:
            Dict with component losses for logging.
        """
        B = embeddings.size(0)
        device = embeddings.device

        # ---- InfoNCE loss -----------------------------------------------
        # Pairwise cosine similarity (since embeddings are L2-normalised,
        # dot product = cosine similarity)
        sim_matrix = embeddings @ embeddings.T  # (B, B)
        sim_matrix = sim_matrix / self.temperature

        # Build positive mask: (B, B) True where labels[i] == labels[j] and i != j
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & diag_mask                         # (B, B)
        neg_mask = ~label_eq & diag_mask

        # For each anchor, sum positives / sum all-but-self (NT-Xent style)
        # Mask out self-similarities on the diagonal
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))

        # Log-sum-exp over all non-self entries
        log_denom = torch.logsumexp(sim_matrix, dim=1)  # (B,)

        # Mean of log-similarities over positive pairs
        pos_sim = sim_matrix.masked_fill(~pos_mask, float("-inf"))
        pos_log_numerator = torch.logsumexp(pos_sim, dim=1)  # (B,)

        # Ignore anchors with no positives in the batch
        has_positive = pos_mask.any(dim=1)
        if has_positive.sum() == 0:
            infonce = torch.tensor(0.0, device=device)
        else:
            infonce = -(pos_log_numerator - log_denom)
            infonce = infonce[has_positive].mean()

        # ---- Physics-guided hard negative loss --------------------------
        hard_loss = torch.tensor(0.0, device=device)

        if (
            k_params is not None
            and theta_params is not None
            and self.hard_negative_weight > 0
        ):
            # Fisher–Rao distances: (B, B)
            k1 = k_params.unsqueeze(1).expand(B, B)       # (B, B)
            k2 = k_params.unsqueeze(0).expand(B, B)
            t1 = theta_params.unsqueeze(1).expand(B, B)
            t2 = theta_params.unsqueeze(0).expand(B, B)

            fr_dist = self.fisher_rao(k1.reshape(-1), t1.reshape(-1),
                                      k2.reshape(-1), t2.reshape(-1))
            fr_dist = fr_dist.view(B, B)  # (B, B)

            # Hard negatives: different class AND physically similar (FR < margin)
            hard_neg_mask = neg_mask & (fr_dist < self.margin)

            if hard_neg_mask.any():
                # These pairs should have large embedding distance: push apart
                hard_sim = (embeddings.unsqueeze(1) * embeddings.unsqueeze(0)).sum(dim=-1)
                # Loss: hinge on cosine similarity — penalise if too similar
                hard_loss = F.relu(hard_sim[hard_neg_mask] - 0.0).mean()

        total = infonce + self.hard_negative_weight * hard_loss

        info = {
            "agcp/infonce": infonce.item(),
            "agcp/hard_negative": hard_loss.item(),
            "agcp/total": total.item(),
        }
        return total, info


# ---------------------------------------------------------------------------
# PIAM Loss
# ---------------------------------------------------------------------------

class PIAMLoss(nn.Module):
    """
    Physics-Informed Adversarial Meta-learning loss for Phase 2.

    Parameters
    ----------
    lambda_proto:
        Weight on the prototypical classification loss.
    lambda_physics:
        Weight on the physics consistency loss.
    lambda_novelty:
        Weight on the novelty calibration loss.
    temperature:
        Temperature for the prototypical softmax.
    use_fisher_rao:
        Whether to use Fisher–Rao distance in the physics consistency term.
    """

    def __init__(
        self,
        lambda_proto: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_novelty: float = 0.1,
        temperature: float = 0.5,
        use_fisher_rao: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_proto = lambda_proto
        self.lambda_physics = lambda_physics
        self.lambda_novelty = lambda_novelty
        self.temperature = temperature
        self.fisher_rao = FisherRaoDistance(
            method="approx", use_fisher_rao=use_fisher_rao
        )

    def prototypical_loss(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard prototypical network cross-entropy loss.

        Parameters
        ----------
        query_embeddings:
            Shape ``(Q, D)``.
        prototypes:
            Shape ``(N, D)`` — one prototype per class.
        query_labels:
            Shape ``(Q,)`` — episode-local class indices in [0, N).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Euclidean distances (in L2-normalised space these equal sqrt(2(1-cos)))
        dists = torch.cdist(query_embeddings, prototypes)     # (Q, N)
        logits = -dists / self.temperature                    # (Q, N)
        return F.cross_entropy(logits, query_labels)

    def physics_consistency_loss(
        self,
        embeddings: torch.Tensor,
        k_params: torch.Tensor,
        theta_params: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Encourage the embedding-space distance to be monotone with the
        Fisher–Rao distance between Gamma parameters.

        We use a ranking loss: for randomly sampled triplets (a, p, n),
        if FR(a, p) < FR(a, n) then ||z_a - z_p||² < ||z_a - z_n||² + margin.

        Parameters
        ----------
        embeddings:
            Shape ``(B, D)``.
        k_params, theta_params:
            Shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar physics consistency loss.
        """
        B = embeddings.size(0)
        if B < 3:
            return torch.tensor(0.0, device=embeddings.device)

        # Pairwise FR distances (B, B)
        k1 = k_params.unsqueeze(1).expand(B, B).reshape(-1)
        k2 = k_params.unsqueeze(0).expand(B, B).reshape(-1)
        t1 = theta_params.unsqueeze(1).expand(B, B).reshape(-1)
        t2 = theta_params.unsqueeze(0).expand(B, B).reshape(-1)
        fr_dists = self.fisher_rao(k1, t1, k2, t2).view(B, B)  # (B, B)

        # Pairwise embedding distances (B, B)
        emb_dists = torch.cdist(embeddings, embeddings)  # (B, B)

        # Triplet sampling: sample B random triplets
        n_triplets = min(B * 4, B * (B - 1))
        idx = torch.randperm(B * B, device=embeddings.device)[:n_triplets]
        row = idx // B
        col = idx % B

        # Anchor is 'row', positive/negative pair is (row, col) vs random third
        third = torch.randint(0, B, (len(row),), device=embeddings.device)

        fr_a_p = fr_dists[row, col]
        fr_a_n = fr_dists[row, third]

        emb_a_p = emb_dists[row, col]
        emb_a_n = emb_dists[row, third]

        # For pairs where FR(a,p) < FR(a,n), emb(a,p) should < emb(a,n)
        should_be_closer = fr_a_p < fr_a_n
        if should_be_closer.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        margin = 0.1
        loss = F.relu(emb_a_p[should_be_closer] - emb_a_n[should_be_closer] + margin).mean()
        return loss

    def novelty_loss(
        self,
        novelty_probs: torch.Tensor,
        is_novel_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy for novelty probability calibration.

        Parameters
        ----------
        novelty_probs:
            Shape ``(N,)`` — P(novel | z).
        is_novel_labels:
            Shape ``(N,)`` — binary labels (1 = novel, 0 = known).
        """
        return F.binary_cross_entropy(
            novelty_probs.clamp(1e-6, 1 - 1e-6),
            is_novel_labels.float(),
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        query_labels: torch.Tensor,
        k_params: Optional[torch.Tensor] = None,
        theta_params: Optional[torch.Tensor] = None,
        novelty_probs: Optional[torch.Tensor] = None,
        is_novel_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute PIAM loss.

        Parameters
        ----------
        query_embeddings:
            Shape ``(Q, D)``.
        prototypes:
            Shape ``(N, D)``.
        query_labels:
            Shape ``(Q,)`` — episode-local.
        k_params, theta_params:
            Shape ``(Q,)`` — optional physics descriptors for query images.
        novelty_probs:
            Shape ``(M,)`` — novelty probabilities for a separate batch.
        is_novel_labels:
            Shape ``(M,)`` — ground truth novelty labels.

        Returns
        -------
        loss, info_dict
        """
        # 1. Prototypical loss
        l_proto = self.prototypical_loss(query_embeddings, prototypes, query_labels)

        # 2. Physics consistency loss
        l_physics = torch.tensor(0.0, device=query_embeddings.device)
        if k_params is not None and theta_params is not None and self.lambda_physics > 0:
            l_physics = self.physics_consistency_loss(
                query_embeddings, k_params, theta_params
            )

        # 3. Novelty calibration loss
        l_novelty = torch.tensor(0.0, device=query_embeddings.device)
        if (
            novelty_probs is not None
            and is_novel_labels is not None
            and self.lambda_novelty > 0
        ):
            l_novelty = self.novelty_loss(novelty_probs, is_novel_labels)

        total = (
            self.lambda_proto * l_proto
            + self.lambda_physics * l_physics
            + self.lambda_novelty * l_novelty
        )

        info = {
            "piam/proto": l_proto.item(),
            "piam/physics": l_physics.item(),
            "piam/novelty": l_novelty.item(),
            "piam/total": total.item(),
        }
        return total, info
