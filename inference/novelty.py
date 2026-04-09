"""
Novelty Detection via Platt Scaling and Mahalanobis Distance.

A query embedding that does not match any known class prototype should be
flagged as *novel* rather than forced into an existing category.

Two complementary scoring strategies are implemented:

1. **d₁/d₂ ratio (NoveltyDetector.score)**
   The ratio of the nearest to the second-nearest prototype distance.
   Low ratio → query exclusively close to one class → known.
   High ratio → ambiguously close to multiple classes → novel.
   Fast; does not require covariance estimation.

2. **Mahalanobis distance (MahalanobisScorer)**
   Based on Lee et al. (NeurIPS 2018).  Accounts for the *shape* of each
   class distribution in embedding space, not just the prototype mean.
   score(z) = min_c  (z − μ_c)ᵀ Σ⁻¹ (z − μ_c)
   where Σ is the tied (pooled) regularised covariance estimated from a
   reference set of known-class embeddings.  MSTAR images that lie in
   directions that FUSAR prototypes never occupy get large scores even
   if they are Euclidean-close to a mean.

   Also implements **Relative Mahalanobis Distance (RMD)**
   (Ren et al., 2021):
   score_rmd(z) = min_c M(z,c) − M(z, background)
   which normalises against the overall distribution to suppress false
   positives from hard-to-classify in-distribution samples.

Platt scaling maps any raw score to a calibrated probability:

    P(novel | z) = σ(temperature · score(z) + bias)

where (temperature, bias) are fitted by minimising binary cross-entropy
with L-BFGS on a held-out calibration set.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memory.owms import OnlineWorkingMemoryStore


# ---------------------------------------------------------------------------
# Platt scaler
# ---------------------------------------------------------------------------

class PlattCalibrator(nn.Module):
    """
    Platt scaling for novelty detection.

    Parameters
    ----------
    temperature_init:
        Initial value of the learnable temperature parameter.
    bias_init:
        Initial value of the learnable bias parameter.
    use_novelty:
        If ``False``, always returns P(novel) = 0.0 (ablation).
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        bias_init: float = 0.0,
        use_novelty: bool = True,
    ) -> None:
        super().__init__()
        self.use_novelty = use_novelty

        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """
        Convert raw novelty scores to calibrated novelty probabilities.

        Parameters
        ----------
        raw_scores:
            Shape ``(N,)`` — minimum distances to stored prototypes.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` — P(novel | query) in [0, 1].
        """
        if not self.use_novelty:
            return torch.zeros_like(raw_scores)

        return torch.sigmoid(self.temperature * raw_scores + self.bias)

    def fit(
        self,
        raw_scores: torch.Tensor,
        labels: torch.Tensor,
        n_iter: int = 200,
        lr: float = 0.01,
    ) -> None:
        """
        Fit Platt parameters to calibration data via L-BFGS.

        Parameters
        ----------
        raw_scores:
            Shape ``(N,)`` — raw novelty scores for calibration samples.
        labels:
            Shape ``(N,)`` — binary labels: 1 = novel, 0 = known.
        n_iter:
            Maximum L-BFGS iterations.
        lr:
            Learning rate for L-BFGS.
        """
        raw_scores = raw_scores.detach().float()
        labels = labels.detach().float().to(raw_scores.device)

        optimizer = optim.LBFGS(
            [self.temperature, self.bias],
            lr=lr,
            max_iter=n_iter,
        )

        def closure():
            optimizer.zero_grad()
            probs = torch.sigmoid(self.temperature * raw_scores + self.bias)
            loss = F.binary_cross_entropy(probs, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

    def extra_repr(self) -> str:
        t = self.temperature.item()
        b = self.bias.item()
        return f"temperature={t:.4f}, bias={b:.4f}, use_novelty={self.use_novelty}"


# ---------------------------------------------------------------------------
# Novelty detector (wraps OWMS + Platt calibrator)
# ---------------------------------------------------------------------------

class NoveltyDetector:
    """
    End-to-end novelty detector: computes minimum-distance score and
    applies Platt scaling to produce a calibrated probability.

    Parameters
    ----------
    calibrator:
        Trained (or default) ``PlattCalibrator``.
    threshold:
        P(novel) threshold above which a query is declared novel.
    """

    def __init__(
        self,
        calibrator: PlattCalibrator,
        threshold: float = 0.5,
    ) -> None:
        self.calibrator = calibrator
        self.threshold = threshold

    def _get_dists(
        self,
        query: torch.Tensor,
        owms: OnlineWorkingMemoryStore,
    ) -> Optional[torch.Tensor]:
        """
        Return ``(N, C)`` Euclidean distance matrix, or ``None`` if OWMS empty.
        Guarantees ``query`` is 2-D and on the correct device.
        """
        protos, _ = owms.all_prototypes()   # (C, D)
        if protos.size(0) == 0:
            return None
        if query.dim() == 1:
            query = query.unsqueeze(0)
        return torch.cdist(query.detach().to(protos.device), protos.detach())  # (N, C)

    def score(
        self,
        query: torch.Tensor,
        owms: OnlineWorkingMemoryStore,
    ) -> torch.Tensor:
        """
        Compute d₁/d₂ novelty score — nearest-to-second-nearest distance ratio.

        **NOTE — use-case limitation**: This score measures *within-domain class
        ambiguity*.  It performs poorly for cross-domain OOD (e.g. MSTAR vs
        FUSAR) because out-of-domain images are often mapped near a specific
        known prototype by the SAR-trained backbone, yielding a low ratio that
        looks "known".  Use ``energy_score()`` or ``min_dist_score()`` for
        cross-domain novelty evaluation.

            score(z) = d₁ / d₂

        where d₁ = distance to nearest prototype, d₂ = second-nearest.

        * Low  → exclusively close to ONE prototype → likely known
        * High → ambiguous between classes → likely novel (within-domain)

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` — scores in (0, 1].
        """
        dists = self._get_dists(query, owms)   # (N, C)
        if dists is None:
            N = query.size(0) if query.dim() == 2 else 1
            return torch.full((N,), 1.0, device=query.device)

        if dists.size(1) == 1:
            return dists.squeeze(1)

        d_sorted = dists.sort(dim=1).values    # (N, C)
        d1 = d_sorted[:, 0]
        d2 = d_sorted[:, 1]
        return d1 / (d2 + 1e-8)               # (N,) ∈ (0, 1]

    def min_dist_score(
        self,
        query: torch.Tensor,
        owms: OnlineWorkingMemoryStore,
    ) -> torch.Tensor:
        """
        Raw minimum Euclidean distance to the nearest prototype.

        Unlike d₁/d₂, this preserves **absolute** distance information.
        It is the correct baseline for cross-domain OOD: a known image is
        geometrically close to its true class prototype (small d₁); an
        out-of-domain image that the backbone cannot match to any known class
        will have a larger d₁ even if it is closer to one prototype than
        to others.

            score(z) = min_c ‖z − μ_c‖₂

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` — scores in [0, 2] for L2-normalised embeddings.
        """
        dists = self._get_dists(query, owms)
        if dists is None:
            N = query.size(0) if query.dim() == 2 else 1
            return torch.full((N,), 2.0, device=query.device)
        return dists.min(dim=1).values         # (N,)

    def energy_score(
        self,
        query: torch.Tensor,
        owms: OnlineWorkingMemoryStore,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Energy-based OOD score (Liu et al., NeurIPS 2020).

        Treats the negative squared distances to all prototypes as unnormalised
        log-probabilities.  The free energy of the query under the prototype
        distribution is:

            E(z) = −T · log Σ_c exp(−‖z − μ_c‖² / T)

        **Higher energy = more OOD / novel.**

        This is strictly better than d₁/d₂ for cross-domain OOD because:
        * It is an ABSOLUTE score — the scale of distances is preserved.
        * It aggregates evidence from ALL prototypes, not just the two nearest.
        * When a known image is very close to its prototype, the exp term for
          that class dominates the sum → very low energy.
        * When an OOD image is at moderate distance from ALL prototypes, no
          single term dominates → high energy.

        Parameters
        ----------
        temperature:
            Softmax temperature T.  Smaller = sharper.  0.1 works well for
            L2-normalised 256-D embeddings where typical distances are 0.2–1.5.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` — energy scores (higher = more novel).
        """
        dists = self._get_dists(query, owms)
        if dists is None:
            N = query.size(0) if query.dim() == 2 else 1
            return torch.full((N,), 100.0, device=query.device)

        dists_sq = dists.pow(2)                                        # (N, C)
        log_sum  = torch.logsumexp(-dists_sq / temperature, dim=1)    # (N,)
        return -temperature * log_sum                                  # (N,)

    def novelty_probability(
        self,
        query: torch.Tensor,
        owms: OnlineWorkingMemoryStore,
    ) -> torch.Tensor:
        """
        Compute calibrated P(novel) for query embedding(s).

        Parameters
        ----------
        query:
            Shape ``(D,)`` or ``(N, D)``.
        owms:
            Prototype store.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` — novelty probabilities in [0, 1].
        """
        scores = self.score(query, owms)
        return self.calibrator(scores)

    def is_novel(
        self,
        query: torch.Tensor,
        owms: OnlineWorkingMemoryStore,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Binary novelty decision.

        Returns
        -------
        decisions:
            Shape ``(N,)`` boolean tensor — True = novel.
        probs:
            Shape ``(N,)`` calibrated novelty probabilities.
        """
        probs = self.novelty_probability(query, owms)
        decisions = probs > self.threshold
        return decisions, probs

    def collect_calibration_data(
        self,
        known_queries: torch.Tensor,
        novel_queries: torch.Tensor,
        owms: OnlineWorkingMemoryStore,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to build a calibration dataset from known and novel queries.

        Returns
        -------
        scores, labels:
            Each shape ``(N_known + N_novel,)``.
        """
        known_scores = self.score(known_queries, owms)
        novel_scores = self.score(novel_queries, owms)

        scores = torch.cat([known_scores, novel_scores], dim=0)
        labels = torch.cat([
            torch.zeros(len(known_scores)),
            torch.ones(len(novel_scores)),
        ], dim=0)
        return scores, labels

    def recalibrate_from_owms(
        self,
        backbone: "torch.nn.Module",
        owms: OnlineWorkingMemoryStore,
        dataset,
        n_known: int = 300,
        n_novel: int = 300,
        device: str = "cuda",
    ) -> None:
        """
        Recalibrate the Platt scaler using the *actual* enrolled OWMS.

        This must be called after all classes have been enrolled so that the
        prototype locations used for scoring match those at inference time.

        Parameters
        ----------
        backbone:
            Trained backbone (used to embed images and synthetic noise).
        owms:
            The populated working-memory store to measure distances against.
        dataset:
            A FUSARShipDataset instance to sample known images from.
        n_known, n_novel:
            Number of known / novel samples to use for calibration.
        device:
            Torch device string.
        """
        import torch
        import torch.nn.functional as F

        dev = torch.device(device)
        backbone.eval()

        # Sample known images and embed them
        n_known = min(n_known, len(dataset))
        indices = torch.randperm(len(dataset))[:n_known].tolist()
        with torch.no_grad():
            images = torch.stack([dataset[i][0] for i in indices]).to(dev)
            known_emb = backbone(images)  # (n_known, D)

        # Synthetic novel embeddings: random unit-sphere points
        D = known_emb.size(1)
        with torch.no_grad():
            novel_emb = F.normalize(
                torch.randn(n_novel, D, device=dev), p=2, dim=1
            )

        known_scores = self.score(known_emb, owms)   # (n_known,)
        novel_scores = self.score(novel_emb, owms)   # (n_novel,)

        scores = torch.cat([known_scores, novel_scores], dim=0)
        labels = torch.cat([
            torch.zeros(n_known, device=dev),
            torch.ones(n_novel, device=dev),
        ], dim=0)

        self.calibrator.fit(scores, labels)

        import logging
        logging.getLogger(__name__).info(
            "Novelty recalibrated against live OWMS (%d classes).  "
            "Known score mean=%.3f, Novel score mean=%.3f",
            owms.num_classes,
            known_scores.mean().item(),
            novel_scores.mean().item(),
        )


# ---------------------------------------------------------------------------
# Physics-based novelty scorer
# ---------------------------------------------------------------------------

class PhysicsNoveltyScorer:
    """
    Novelty detection using SAR physics statistics.

    FUSAR ships and MSTAR ground vehicles differ fundamentally in their SAR
    physics signatures:

    * Gamma shape k:  Ships are distributed scatterers (moderate k ≈ 1–3).
                      Ground vehicles are bright compact point scatterers
                      (high k, often 5–20+).
    * SNR:            Vehicles on land have very high SNR; ships at sea have
                      moderate SNR with clutter.
    * Speckle index:  Differs systematically between maritime and land scenes.

    This scorer fits an axis-aligned Gaussian on a reference set of FUSAR
    physics feature vectors [k, θ, SNR, SI] and computes the squared
    Mahalanobis distance (under diagonal covariance) for query images.

    This operates in 4-D physics space (not 256-D embedding space), so
    there is no rank-deficiency problem and no need for PCA.

    Parameters
    ----------
    eps:
        Minimum standard deviation to prevent division by zero.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self._mean: Optional[torch.Tensor] = None   # (P,)
        self._std:  Optional[torch.Tensor] = None   # (P,)
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, physics_features: torch.Tensor) -> None:
        """
        Fit the reference Gaussian from FUSAR physics feature vectors.

        Parameters
        ----------
        physics_features:
            ``(N, P)`` — physics descriptors [k, θ, SNR, SI] for N
            known-class (FUSAR) images.
        """
        self._mean   = physics_features.mean(dim=0)                   # (P,)
        self._std    = physics_features.std(dim=0).clamp(min=self.eps) # (P,)
        self._fitted = True

    # ------------------------------------------------------------------
    def score(self, query_physics: torch.Tensor) -> torch.Tensor:
        """
        Squared Mahalanobis distance in physics space.

        Parameters
        ----------
        query_physics:
            ``(P,)`` or ``(N, P)`` — per-image physics descriptors.

        Returns
        -------
        torch.Tensor
            ``(N,)`` — higher = more OOD.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")

        if query_physics.dim() == 1:
            query_physics = query_physics.unsqueeze(0)

        query_physics = query_physics.to(self._mean.device)
        z = (query_physics - self._mean) / self._std  # (N, P) z-scores
        return z.pow(2).sum(dim=1)                    # (N,) squared Mahal distance

    # ------------------------------------------------------------------
    def fit_per_class(
        self,
        class_physics_list: List[torch.Tensor],
        min_samples: int = 3,
    ) -> None:
        """
        Fit one axis-aligned Gaussian *per enrolled class* instead of a
        single global Gaussian over all FUSAR images.

        This is the physics-space analogue of per-class prototype matching.
        A FUSAR Cargo ship is scored against the Cargo class physics mean;
        an MSTAR vehicle must be far from *every* class physics mean to score
        highly.  This eliminates the inflated-variance problem that arises
        when 12 different ship types (with very different physics) are pooled
        into a single Gaussian.

        Parameters
        ----------
        class_physics_list:
            List of ``(N_i, P)`` tensors, one per enrolled class.
            ``N_i`` is the number of support images for class ``i``.
        min_samples:
            Classes with fewer than ``min_samples`` images fall back to the
            global Gaussian (mean/std from ``fit()`` must be called first).
        """
        if not self._fitted:
            raise RuntimeError(
                "Call fit() with the global reference set first, then "
                "call fit_per_class() for per-class refinement."
            )
        P = self._mean.shape[0]
        device = self._mean.device

        self._class_means: List[torch.Tensor] = []
        self._class_stds:  List[torch.Tensor] = []

        for phys in class_physics_list:
            phys = phys.to(device)
            if phys.shape[0] >= min_samples:
                self._class_means.append(phys.mean(dim=0))
                self._class_stds.append(
                    phys.std(dim=0).clamp(min=self.eps)
                )
            else:
                # Fall back to global statistics for data-scarce classes
                self._class_means.append(self._mean)
                self._class_stds.append(self._std)

        self._per_class_fitted = True

    # ------------------------------------------------------------------
    def min_class_score(self, query_physics: torch.Tensor) -> torch.Tensor:
        """
        Minimum squared Mahalanobis distance to any class physics Gaussian.

        For each query image this computes the distance to every enrolled
        class's physics distribution and returns the minimum — so a FUSAR
        image that matches *any* class is scored as known (low score), while
        an MSTAR image that doesn't match *any* class is scored as novel
        (high score).

            score(z) = min_c  Σ_p  ((z_p − μ_{c,p}) / σ_{c,p})²

        Parameters
        ----------
        query_physics:
            ``(P,)`` or ``(N, P)``.

        Returns
        -------
        torch.Tensor
            ``(N,)`` — min over classes of squared physics Mahalanobis.
        """
        if not getattr(self, "_per_class_fitted", False):
            raise RuntimeError("Call fit_per_class() before min_class_score().")

        if query_physics.dim() == 1:
            query_physics = query_physics.unsqueeze(0)

        device = self._mean.device
        query_physics = query_physics.to(device)   # (N, P)
        N, P = query_physics.shape
        C = len(self._class_means)

        # Stack class means/stds: (C, P)
        means = torch.stack(self._class_means, dim=0)   # (C, P)
        stds  = torch.stack(self._class_stds,  dim=0)   # (C, P)

        # (N, 1, P) - (1, C, P) → (N, C, P) z-scores per class
        z = (query_physics.unsqueeze(1) - means.unsqueeze(0)) / stds.unsqueeze(0)
        dists = z.pow(2).sum(dim=2)   # (N, C) — squared Mahal to each class
        return dists.min(dim=1).values # (N,) — minimum over classes

    # ------------------------------------------------------------------
    def fit_per_class_pooled(
        self,
        class_physics_list: List[torch.Tensor],
        reg: float = 0.01,
        min_samples: int = 3,
    ) -> None:
        """
        Fit per-class means with a **shared (pooled) full covariance** matrix.

        This is the LDA-style Mahalanobis approach computed in a
        **feature-standardised space** (zero mean, unit variance per feature
        using the global reference statistics from ``fit()``):

            z_std = (z − μ_global) / σ_global
            score(z) = min_c  (z_std − μ_c_std)ᵀ Σ_std_pooled⁻¹ (z_std − μ_c_std)

        Standardising before computing the scatter matrix ensures that
        regularisation (reg × I) is proportionally the same for every feature
        dimension.  Without standardisation, small-variance features (e.g. SNR
        range 0.21–0.47) would be overwhelmed by the reg term while
        large-variance features (e.g. k range 0.5–3.4) are barely affected —
        destroying the discriminative SNR signal that separates MSTAR from FUSAR.

        Parameters
        ----------
        class_physics_list:
            List of ``(N_i, P)`` tensors, one per enrolled class.
        reg:
            Regularisation added as reg × I in the *standardised* space
            (i.e. as a fraction of unit variance per feature).  Default 0.01.
        min_samples:
            Classes with fewer images fall back to global statistics for mean;
            their samples are excluded from the pooled scatter computation.
        """
        if not self._fitted:
            raise RuntimeError(
                "Call fit() with the global reference set first, then "
                "call fit_per_class_pooled()."
            )
        P      = self._mean.shape[0]
        device = self._mean.device

        # Global normalisation constants (from fit())
        g_mean = self._mean          # (P,)
        g_std  = self._std           # (P,)  clamp(min=eps) already applied in fit()

        # Helper: standardise raw physics to zero-mean / unit-variance
        def _std(x: torch.Tensor) -> torch.Tensor:
            return (x.to(device) - g_mean) / g_std

        # 1. Per-class means in standardised space
        self._lda_means: List[torch.Tensor] = []
        valid_classes: List[int] = []
        global_mean_std = _std(g_mean.unsqueeze(0)).squeeze(0)   # zero vector

        for i, phys in enumerate(class_physics_list):
            phys_s = _std(phys)
            if phys_s.shape[0] >= min_samples:
                self._lda_means.append(phys_s.mean(dim=0))
                valid_classes.append(i)
            else:
                self._lda_means.append(global_mean_std)   # global fallback → zero

        # 2. Pooled within-class scatter in standardised space
        scatter   = torch.zeros(P, P, device=device)
        n_total   = 0
        n_classes = 0

        for i in valid_classes:
            phys_s = _std(class_physics_list[i])          # (N_i, P)
            mu_c   = self._lda_means[i]                   # (P,)
            centered = phys_s - mu_c.unsqueeze(0)         # (N_i, P)
            scatter += centered.t().mm(centered)
            n_total   += phys_s.shape[0]
            n_classes += 1

        dof = max(n_total - n_classes, 1)
        sigma_pooled = scatter / dof + reg * torch.eye(P, device=device)

        # 3. Invert — P≤5 so direct inverse is numerically fine
        try:
            self._lda_sigma_inv = torch.linalg.inv(sigma_pooled)
        except RuntimeError:
            self._lda_sigma_inv = torch.diag(1.0 / sigma_pooled.diagonal().clamp(min=self.eps))

        self._lda_fitted = True

    # ------------------------------------------------------------------
    def min_class_score_pooled(self, query_physics: torch.Tensor) -> torch.Tensor:
        """
        Minimum LDA Mahalanobis distance to any class physics distribution.

        Uses the pooled covariance fitted by ``fit_per_class_pooled()``.

            score(z) = min_c  (z − μ_c)ᵀ Σ_pooled⁻¹ (z − μ_c)

        Parameters
        ----------
        query_physics:
            ``(P,)`` or ``(N, P)``.

        Returns
        -------
        torch.Tensor
            ``(N,)`` — min over classes of LDA Mahalanobis distance.
        """
        if not getattr(self, "_lda_fitted", False):
            raise RuntimeError("Call fit_per_class_pooled() before min_class_score_pooled().")

        if query_physics.dim() == 1:
            query_physics = query_physics.unsqueeze(0)

        device = self._mean.device
        # Standardise using the same global stats used during fit_per_class_pooled
        query_std = (query_physics.to(device) - self._mean) / self._std  # (N, P)
        N, P = query_std.shape
        C    = len(self._lda_means)

        means     = torch.stack(self._lda_means, dim=0)   # (C, P) — already standardised
        sigma_inv = self._lda_sigma_inv                    # (P, P)

        # delta: (N, C, P) — both query and means are in standardised space
        delta = query_std.unsqueeze(1) - means.unsqueeze(0)

        # Mahalanobis: for each (n, c): delta[n,c] @ Σ⁻¹ @ delta[n,c]
        # Use einsum for clarity: 'ncp,pq,ncq->nc'
        tmp    = torch.einsum("ncp,pq->ncq", delta, sigma_inv)   # (N, C, P)
        scores = (tmp * delta).sum(dim=2)                        # (N, C)
        return scores.min(dim=1).values                          # (N,)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        s = f"fitted={self._fitted}"
        if self._fitted and self._mean is not None:
            s += f", P={self._mean.shape[0]}"
        if getattr(self, "_per_class_fitted", False):
            s += f", C={len(self._class_means)}"
        if getattr(self, "_lda_fitted", False):
            s += f", LDA_C={len(self._lda_means)}"
        return f"PhysicsNoveltyScorer({s})"


# ---------------------------------------------------------------------------
# Mahalanobis-distance novelty scorer
# ---------------------------------------------------------------------------

class MahalanobisScorer:
    """
    Tied-covariance Mahalanobis distance for OOD novelty detection.

    Based on Lee et al. (NeurIPS 2018) "A Simple Unified Framework for
    Detecting Out-of-Distribution Samples and Adversarial Attacks".

    For each query z the novelty score is the **minimum** Mahalanobis
    distance to any known-class prototype mean under the shared covariance:

        score_M(z) = min_c  (z − μ_c)ᵀ  Σ⁻¹  (z − μ_c)

    The tied covariance Σ is estimated from a reference set of known-class
    backbone embeddings by:
      1. Assigning each reference embedding to its nearest prototype mean.
      2. Centering by subtracting the class mean.
      3. Computing the pooled outer-product covariance.
      4. Adding λI for numerical stability.

    Also implements **Relative Mahalanobis Distance (RMD)**
    (Ren et al., 2021):

        score_RMD(z) = score_M(z) − M(z, background)

    where the background mean is the centroid of all reference embeddings.
    RMD reduces false positives from hard-to-classify in-distribution
    samples by subtracting out the "distance from the overall distribution".

    Parameters
    ----------
    reg_coeff:
        Regularisation coefficient λ added to the diagonal of Σ before
        inversion.  Larger values = more robust but less discriminative.
    """

    def __init__(self, reg_coeff: float = 1e-3) -> None:
        self.reg_coeff = reg_coeff
        self._class_means:      Optional[torch.Tensor] = None   # (C, D)
        self._precision:        Optional[torch.Tensor] = None   # (D, D)
        self._background_mean:  Optional[torch.Tensor] = None   # (D,)
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        class_means: torch.Tensor,
        reference_emb: torch.Tensor,
    ) -> None:
        """
        Estimate the tied precision matrix from reference embeddings.

        Parameters
        ----------
        class_means:
            ``(C, D)`` — prototype means for each enrolled class (e.g.
            from ``OnlineWorkingMemoryStore.all_prototypes()``).
        reference_emb:
            ``(N, D)`` — backbone embeddings of *known-class* reference
            images used to estimate the within-class covariance.
        """
        C, D = class_means.shape
        device = class_means.device
        ref = reference_emb.to(device)          # (N, D)

        self._class_means     = class_means
        self._background_mean = ref.mean(dim=0)  # (D,)

        # Assign each reference embedding to nearest class prototype
        # dists: (N, C) — Euclidean distances
        dists   = torch.cdist(ref, class_means)  # (N, C)
        nearest = dists.argmin(dim=1)            # (N,)

        # Center embeddings by subtracting their assigned class mean
        centered = ref - class_means[nearest]    # (N, D)

        # Tied (pooled) covariance: (D, D)
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)

        # Regularise and invert
        cov = cov + self.reg_coeff * torch.eye(D, device=device)
        self._precision = torch.linalg.inv(cov)  # (D, D)
        self._fitted = True

    # ------------------------------------------------------------------
    def _mahal_to_means(self, query: torch.Tensor) -> torch.Tensor:
        """
        Return ``(N, C)`` Mahalanobis distances from each query to each
        class mean under the shared precision matrix.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before scoring.")

        if query.dim() == 1:
            query = query.unsqueeze(0)

        query = query.to(self._class_means.device)
        N, D  = query.shape
        C     = self._class_means.shape[0]

        # diff: (N, C, D)
        diff = query.unsqueeze(1) - self._class_means.unsqueeze(0)
        diff_flat = diff.reshape(N * C, D)             # (N*C, D)

        # (N*C, D) @ (D, D) = (N*C, D), then dot with diff_flat
        diff_prec = diff_flat @ self._precision        # (N*C, D)
        mahal     = (diff_prec * diff_flat).sum(dim=1) # (N*C,)
        return mahal.reshape(N, C)                     # (N, C)

    # ------------------------------------------------------------------
    def score(self, query: torch.Tensor) -> torch.Tensor:
        """
        Minimum Mahalanobis distance to any class prototype.

        Parameters
        ----------
        query:
            ``(D,)`` or ``(N, D)``.

        Returns
        -------
        torch.Tensor
            ``(N,)`` — high values = OOD / novel.
        """
        mahal = self._mahal_to_means(query)   # (N, C)
        return mahal.min(dim=1).values        # (N,)

    # ------------------------------------------------------------------
    def relative_score(self, query: torch.Tensor) -> torch.Tensor:
        """
        Relative Mahalanobis Distance (RMD):

            score_RMD(z) = min_c M(z, c) − M(z, background)

        Subtracting the background distance normalises for regions of
        embedding space that are intrinsically far from any known-class
        cluster (e.g. due to imbalanced class density), which reduces
        false-positive novelty detections for hard in-distribution samples.

        Parameters
        ----------
        query:
            ``(D,)`` or ``(N, D)``.

        Returns
        -------
        torch.Tensor
            ``(N,)`` — high values = OOD / novel.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before scoring.")

        if query.dim() == 1:
            query = query.unsqueeze(0)

        query = query.to(self._class_means.device)

        min_class_mahal = self.score(query)                          # (N,)

        diff_bg   = query - self._background_mean.unsqueeze(0)       # (N, D)
        diff_bg_p = diff_bg @ self._precision                        # (N, D)
        bg_mahal  = (diff_bg_p * diff_bg).sum(dim=1)                 # (N,)

        return min_class_mahal - bg_mahal                            # (N,)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = f"fitted={self._fitted}"
        if self._fitted:
            C = self._class_means.shape[0]
            D = self._class_means.shape[1]
            status += f", C={C}, D={D}"
            if self._pca_components is not None:
                status += f", pca_dim={self._pca_components.shape[0]}"
        return f"MahalanobisScorer(reg={self.reg_coeff}, {status})"


# ---------------------------------------------------------------------------
# PCA-projected Mahalanobis scorer (fixes singular covariance in high-D)
# ---------------------------------------------------------------------------

class PCAMahalanobisScorer:
    """
    Mahalanobis distance with PCA pre-projection to avoid singular covariance.

    When N_ref << D (e.g. 300 reference embeddings in 256-D space) the
    pooled covariance is rank-deficient and regularisation alone cannot save
    it — precision eigenvalues blow up, amplifying noise.

    The fix is to project embeddings to the top-``pca_dim`` principal
    components of the reference set *before* estimating covariance.  With
    ``pca_dim=64`` and ``N_ref=300`` the ratio is 4.7×, making the
    covariance estimate well-conditioned.

    All other scoring logic is identical to ``MahalanobisScorer``.

    Parameters
    ----------
    pca_dim:
        Number of PCA components to retain (must be < N_ref).
    reg_coeff:
        Regularisation added to the diagonal of the projected covariance.
    """

    def __init__(self, pca_dim: int = 64, reg_coeff: float = 1e-3) -> None:
        self.pca_dim   = pca_dim
        self.reg_coeff = reg_coeff

        self._pca_mean:       Optional[torch.Tensor] = None   # (D,)
        self._pca_components: Optional[torch.Tensor] = None   # (pca_dim, D)
        self._class_means_p:  Optional[torch.Tensor] = None   # (C, pca_dim)
        self._background_p:   Optional[torch.Tensor] = None   # (pca_dim,)
        self._precision:      Optional[torch.Tensor] = None   # (pca_dim, pca_dim)
        self._fitted = False

    # ------------------------------------------------------------------
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Project (N, D) → (N, pca_dim)."""
        return (x - self._pca_mean) @ self._pca_components.T

    # ------------------------------------------------------------------
    def fit(
        self,
        class_means: torch.Tensor,
        reference_emb: torch.Tensor,
    ) -> None:
        """
        Fit PCA on reference embeddings, then estimate tied covariance in
        the projected space.

        Parameters
        ----------
        class_means:
            ``(C, D)`` prototype means from the OWMS.
        reference_emb:
            ``(N, D)`` known-class backbone embeddings.
        """
        device = class_means.device
        ref    = reference_emb.to(device)       # (N, D)
        N, D   = ref.shape

        # ---- PCA via truncated SVD on mean-centred reference embeddings ----
        self._pca_mean = ref.mean(dim=0)        # (D,)
        centred        = ref - self._pca_mean   # (N, D)

        # torch.linalg.svd returns U (N,N), S (min(N,D),), Vh (D,D)
        # We only need the top pca_dim right singular vectors
        dim = min(self.pca_dim, N, D)
        _, _, Vh = torch.linalg.svd(centred, full_matrices=False)
        self._pca_components = Vh[:dim]         # (pca_dim, D)

        # ---- Project everything into PCA subspace --------------------------
        ref_p         = self._project(ref)                          # (N, pca_dim)
        class_means_p = self._project(class_means)                  # (C, pca_dim)
        self._class_means_p  = class_means_p
        self._background_p   = ref_p.mean(dim=0)                    # (pca_dim,)

        # ---- Tied covariance in projected space ----------------------------
        dists   = torch.cdist(ref_p, class_means_p)                 # (N, C)
        nearest = dists.argmin(dim=1)                               # (N,)
        centred_p = ref_p - class_means_p[nearest]                  # (N, pca_dim)

        cov = (centred_p.T @ centred_p) / max(N - 1, 1)            # (pca_dim, pca_dim)
        cov = cov + self.reg_coeff * torch.eye(dim, device=device)
        self._precision = torch.linalg.inv(cov)                     # (pca_dim, pca_dim)
        self._fitted    = True

    # ------------------------------------------------------------------
    def _mahal(self, query_p: torch.Tensor) -> torch.Tensor:
        """``(N, C)`` Mahalanobis distances from projected queries."""
        N = query_p.shape[0]
        C = self._class_means_p.shape[0]
        diff      = query_p.unsqueeze(1) - self._class_means_p.unsqueeze(0)  # (N, C, p)
        diff_flat = diff.reshape(N * C, -1)
        prec_diff = diff_flat @ self._precision                               # (N*C, p)
        mahal     = (prec_diff * diff_flat).sum(dim=1).reshape(N, C)         # (N, C)
        return mahal

    # ------------------------------------------------------------------
    def score(self, query: torch.Tensor) -> torch.Tensor:
        """Minimum PCA-Mahalanobis distance to any prototype. ``(N,)``."""
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = query.to(self._pca_mean.device)
        query_p = self._project(query)
        return self._mahal(query_p).min(dim=1).values

    # ------------------------------------------------------------------
    def relative_score(self, query: torch.Tensor) -> torch.Tensor:
        """RMD in PCA subspace. ``(N,)``."""
        if not self._fitted:
            raise RuntimeError("Call fit() before relative_score().")
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query   = query.to(self._pca_mean.device)
        query_p = self._project(query)

        min_mahal = self._mahal(query_p).min(dim=1).values           # (N,)

        diff_bg   = query_p - self._background_p.unsqueeze(0)        # (N, p)
        bg_mahal  = (diff_bg @ self._precision * diff_bg).sum(dim=1) # (N,)
        return min_mahal - bg_mahal

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        s = f"fitted={self._fitted}, pca_dim={self.pca_dim}"
        if self._fitted and self._class_means_p is not None:
            s += f", C={self._class_means_p.shape[0]}"
        return f"PCAMahalanobisScorer(reg={self.reg_coeff}, {s})"


# ---------------------------------------------------------------------------
# KNN novelty scorer (Sun et al., NeurIPS 2022)
# ---------------------------------------------------------------------------

class KNNScorer:
    """
    k-Nearest-Neighbour novelty scoring.

    Based on Sun et al. (NeurIPS 2022) "Out-of-Distribution Detection with
    Deep Nearest Neighbors".

    For each query z the novelty score is the Euclidean distance to its
    k-th nearest neighbour in the reference set of known-class embeddings:

        score_KNN(z) = ‖z − z_{(k)}‖₂

    where z_{(k)} is the k-th closest reference embedding.

    * Known queries  → small k-NN distance (lie inside the known manifold)
    * Novel queries  → large k-NN distance (outside the known manifold)

    No covariance estimation is needed, so this is robust to high-
    dimensional embeddings with limited reference data.

    Parameters
    ----------
    k:
        Number of neighbours.  Typical values: 5–50.  Larger k gives
        smoother, more conservative boundaries.
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self._reference: Optional[torch.Tensor] = None   # (N, D)
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, reference_emb: torch.Tensor) -> None:
        """
        Store reference embeddings.

        Parameters
        ----------
        reference_emb:
            ``(N, D)`` known-class backbone embeddings.  These are the
            neighbours queried at inference time.
        """
        self._reference = reference_emb.detach()
        self._fitted    = True

    # ------------------------------------------------------------------
    def score(self, query: torch.Tensor) -> torch.Tensor:
        """
        k-th nearest-neighbour distance for each query.

        Parameters
        ----------
        query:
            ``(D,)`` or ``(N, D)``.

        Returns
        -------
        torch.Tensor
            ``(N,)`` — high values = OOD / novel.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")

        if query.dim() == 1:
            query = query.unsqueeze(0)

        query = query.to(self._reference.device)

        # (Q, N_ref) pairwise Euclidean distances
        dists = torch.cdist(query, self._reference)                  # (Q, N_ref)

        k = min(self.k, dists.shape[1])
        # topk with largest=False → k smallest distances; take the k-th (last)
        kth_dist = dists.topk(k, dim=1, largest=False).values[:, -1]  # (Q,)
        return kth_dist

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        s = f"k={self.k}, fitted={self._fitted}"
        if self._fitted and self._reference is not None:
            s += f", N_ref={self._reference.shape[0]}"
        return f"KNNScorer({s})"
