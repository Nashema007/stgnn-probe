"""STGNN-Probe: analysis framework for validating spatial dependencies in STGNNs.

Six lenses. The paper reports Lenses 0-3 (MAE, SGS, the Granger reference and
AAS); Lenses 4-5 are additional diagnostics that the paper does not use.
  0. Performance Benchmark — raw forecasting accuracy across models and horizons
  1. Spatial Utility   — baseline-relative gain (SGS) over the per-sensor TCN
  2. Granger Reference — pairwise Granger predictive reference (not a ground truth)
  3. Structural Align  — overlap of learned top-k edges with the reference (AAS)
  4. Community Coher.  — do learned communities match geographic structure?
  5. Horizon Degrad.   — how do spatial utility and alignment change with horizon?
"""

from .config import (
    AlignmentConfig,
    CommunityConfig,
    DatasetConfig,
    GrangerConfig,
    ModelsConfig,
    PerformanceConfig,
    ProbeConfig,
    load_config,
)
from .explore import explore
from .lens0_performance import Lens0Result, run_lens0, standardise_predictions
from .lens1_spatial_utility import SpatialUtilityResult, run_lens1
from .lens2_granger import GrangerResult, gcg_at_horizon, run_lens2
from .lens3_alignment import AlignmentResult, run_lens3
from .lens4_community import CommunityResult, run_lens4
from .lens5_degradation import (
    DegradationResult,
    compute_cross_model_correlation,
    run_lens5,
)
from .probe import ProbeResult, ProbeRunner, normalize_adjacency

__all__ = [
    # Config
    "AlignmentConfig",
    "CommunityConfig",
    "DatasetConfig",
    "GrangerConfig",
    "ModelsConfig",
    "PerformanceConfig",
    "ProbeConfig",
    "load_config",
    "explore",
    # Lens 0
    "Lens0Result",
    "run_lens0",
    "standardise_predictions",
    # Lens 1
    "SpatialUtilityResult",
    "run_lens1",
    # Lens 2
    "GrangerResult",
    "gcg_at_horizon",
    "run_lens2",
    # Lens 3
    "AlignmentResult",
    "run_lens3",
    # Lens 4
    "CommunityResult",
    "run_lens4",
    # Lens 5
    "DegradationResult",
    "compute_cross_model_correlation",
    "run_lens5",
    # Runner
    "ProbeResult",
    "ProbeRunner",
    "normalize_adjacency",
]
