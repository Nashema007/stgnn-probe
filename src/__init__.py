"""PyTorch-based framework for STGNN experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stgnn-framework")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
