"""Model implementations and registries for migrated STGNN architectures.

GWN and TCN are no longer custom implementations here — they are built
directly from ``tsl.nn.models`` in ``scripts/run_training.py``.
"""

from .arima import RollingARIMA
from .bigst import BigST, make_bigst
from .d2stgnn import D2STGNN, make_d2stgnn
from .dssa_tcn import DSSATCN
from .gwn_v2 import GWNv2
from .staeformer import STAEformer
from .stawnet import STAWnet

__all__ = [
    "RollingARIMA",
    "BigST",
    "make_bigst",
    "D2STGNN",
    "make_d2stgnn",
    "DSSATCN",
    "GWNv2",
    "STAEformer",
    "STAWnet",
]
