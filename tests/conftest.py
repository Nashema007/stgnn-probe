"""Repo-wide pytest configuration."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Fail fast with an actionable message if PyG's binary extensions are missing.

    torch_scatter/torch_sparse are a hard prerequisite for this test suite, not
    just for tsl-pipeline-specific tests: src/evaluation/metrics.py imports
    tsl.metrics.torch at module level, and that's pulled in by nearly every
    other module (src/training/__init__.py -> adapters.py -> torch_geometric
    -> tsl). Without this check, collection fails deep inside third-party
    internals with a cryptic ModuleNotFoundError on whichever test file
    happens to import first. See README.md's "PyTorch Backend Support"
    section.
    """
    try:
        import torch_scatter  # noqa: F401
        import torch_sparse  # noqa: F401
    except ImportError as exc:
        pytest.exit(
            "torch_scatter/torch_sparse are required to run this test suite "
            "(see README.md's 'PyTorch Backend Support' section). Install them with:\n\n"
            "    uv pip install torch-scatter torch-sparse "
            "-f https://data.pyg.org/whl/torch-2.2.2+cpu.html\n\n"
            "or run the suite inside the stgnn-tsl-dev Docker image (see Dockerfile).\n\n"
            f"Underlying error: {exc}",
            returncode=1,
        )
