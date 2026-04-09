"""
ProtoEvoNet physics package.

Provides differentiable SAR physics primitives:
  * Gamma distribution statistics (MoM / MLE)
  * Dimension-wise SAR Reliability (DSAR)
  * Fisher–Rao geodesic distance on the Gamma manifold
  * Signal-to-Noise Ratio estimation
"""

from .gamma_stats import GammaStatisticsModule, mom_estimate, mle_estimate, patch_gamma_stats
from .dsar import DSARModule
from .fisher_rao import FisherRaoDistance, fisher_rao_distance_approx, fisher_info_matrix
from .snr import SNREstimationModule, local_snr, speckle_index


class PhysicsLayer(object):
    """
    Thin namespace that bundles all physics sub-modules.

    Instantiate via ``PhysicsLayer.from_config(cfg)`` and call
    ``compute(image, embedding)`` to get a ``PhysicsDescriptor``.
    """

    def __init__(
        self,
        gamma_module: GammaStatisticsModule,
        dsar_module: DSARModule,
        fisher_rao_module: FisherRaoDistance,
        snr_module: SNREstimationModule,
    ) -> None:
        self.gamma = gamma_module
        self.dsar = dsar_module
        self.fisher_rao = fisher_rao_module
        self.snr = snr_module

    @classmethod
    def from_config(cls, cfg) -> "PhysicsLayer":
        """Build from a ``PhysicsConfig``."""
        from utils.config import PhysicsConfig
        assert isinstance(cfg, PhysicsConfig)
        return cls(
            gamma_module=GammaStatisticsModule(
                patch_size=cfg.snr_patch_size,
                estimator="mom",
                eps=cfg.gamma_eps,
                max_iter=cfg.gamma_mle_iters,
            ),
            dsar_module=DSARModule(
                embedding_dim=cfg.dsar_dim,
                use_dsar=cfg.use_dsar,
            ),
            fisher_rao_module=FisherRaoDistance(
                method="approx",
                eps=cfg.fisher_rao_eps,
                use_fisher_rao=cfg.use_fisher_rao,
            ),
            snr_module=SNREstimationModule(
                patch_size=cfg.snr_patch_size,
                use_snr=cfg.use_snr,
            ),
        )

    def modules(self):
        """Yield all nn.Module children (for parameter registration)."""
        yield self.gamma
        yield self.dsar
        yield self.fisher_rao
        yield self.snr


__all__ = [
    "GammaStatisticsModule",
    "DSARModule",
    "FisherRaoDistance",
    "SNREstimationModule",
    "PhysicsLayer",
    "mom_estimate",
    "mle_estimate",
    "patch_gamma_stats",
    "fisher_rao_distance_approx",
    "fisher_info_matrix",
    "local_snr",
    "speckle_index",
]
