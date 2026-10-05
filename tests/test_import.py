import torch

import src


def test_package_imports() -> None:
    assert src.__version__
    assert torch.__version__
