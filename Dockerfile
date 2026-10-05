# Linux dev/test environment for the GWN/TCN tsl migration.
#
# torch_scatter / torch_sparse (required by torch-spatiotemporal's
# GraphWaveNetModel and TCNModel) have no prebuilt wheel for macOS x86_64 +
# Python 3.12 + torch 2.8.0, and fail to compile from source against the
# current Xcode/clang toolchain. Linux has prebuilt wheels for this combo.
#
# This isn't limited to GWN/TCN-specific code: src/evaluation/metrics.py
# imports tsl.metrics.torch at module level, and src/training/__init__.py
# imports it unconditionally, so torch_scatter/torch_sparse block test
# collection for nearly the entire suite on macOS, not just tsl-pipeline
# tests. Running the test suite therefore requires this container (or CI).
#
# torch==2.8.0 (not 2.2.2) so that the matching CUDA 12.8/12.9 wheel
# (installed separately, on top of this same pin, on GPU machines — see
# README) has NVIDIA Blackwell (sm_120) kernels. This CPU image only needs
# to prove the rest of the stack (PyG/tsl/Lightning) still works at that
# torch version; it can't validate GPU/Blackwell behavior itself.
#
# Build:  docker build -t stgnn-tsl-dev .
# Run:    docker run --rm -v "$(pwd)":/app -w /app stgnn-tsl-dev <command>

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "numpy>=1.26,<2"
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "torch-geometric>=2.8,<2.9"
RUN pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
RUN pip install --no-cache-dir torch-spatiotemporal==0.9.5 "pytorch-lightning>=2.4,<2.7" hydra-core==1.3.3

COPY pyproject.toml .
# scipy>=1.18 hard-requires numpy>=2, which breaks torch==2.2.2's numpy ABI
# (RuntimeError: Numpy is not available). uv resolves scipy==1.17.1 against
# this project's numpy<2 pin on macOS, so pin that exact version here too.
# pandas<3 because torch-spatiotemporal==0.9.5 hardcodes the "5T" offset
# alias (tsl/datasets/metr_la.py, pems_bay.py), which pandas 3.0 removed
# (deprecated since 2.2, dropped in 3.0 — "did you mean 'min'?").
RUN pip install --no-cache-dir \
    "matplotlib>=3.10" "pandas>=2.2,<3" "pyyaml>=6.0" "scipy==1.17.1" \
    "statsmodels>=0.14" "threadpoolctl>=3.0" "tqdm>=4.67" "wandb>=0.16" "psutil>=5.9" \
    "mypy==2.1.0" "pytest>=8.3" \
    "ruff>=0.11" "types-PyYAML>=6.0" "networkx>=3.3" "python-louvain>=0.16" \
    "plotly>=5.20" "altair>=5.3" "kaleido>=0.2.1" "vl-convert-python>=1.0" \
    "pygwalker>=0.4" "tables>=3.9"

# Several deps above can still pull in numpy>=2 transitively; re-pin last to
# win the install-order race.
RUN pip install --no-cache-dir "numpy>=1.26,<2" --force-reinstall

# pytorch-lightning 2.2.5 imports pkg_resources unconditionally at startup;
# setuptools>=81 dropped it by default, breaking `import pytorch_lightning`
# (and therefore tsl.data, which imports it) on any recent setuptools.
RUN pip install --no-cache-dir "setuptools<81"

ENV PYTHONPATH=/app/src:/app
