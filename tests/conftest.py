"""Shared pytest fixtures for pipeline tests."""
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent

# Ensure the repository root is on the path so namespace-style imports like
# ``data.knowledge_base...`` resolve during local pytest runs.
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Ensure src/ is on the path so openfinai_pipeline / openfinai_harbor resolve.
_src_root = str(_project_root / "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

# Ensure the reward bank (data/knowledge_base/rewards/) is importable.
_eval_root = str(_project_root / "data" / "knowledge_base" / "rewards")
if _eval_root not in sys.path:
    sys.path.insert(0, _eval_root)


def pytest_addoption(parser):
    parser.addoption(
        "--reward-module",
        default=None,
        help="Absolute path to a generated reward_fn .py file to validate.",
    )
    parser.addoption(
        "--loader-module",
        default=None,
        help="Absolute path to a generated dataset load.py file to validate.",
    )
    parser.addoption(
        "--loader-data-dir",
        default=None,
        help="Absolute path to the dataset's data/ subdirectory the loader reads from.",
    )


@pytest.fixture
def reward_module_path(request):
    """Path to the reward_fn module under test (None in CI mode)."""
    return request.config.getoption("--reward-module")


@pytest.fixture
def loader_module_path(request):
    """Path to the dataset load.py module under test (None in CI mode)."""
    return request.config.getoption("--loader-module")


@pytest.fixture
def loader_data_dir(request):
    """Path to the dataset's data/ subdir the loader reads from (None in CI mode)."""
    return request.config.getoption("--loader-data-dir")


@pytest.fixture
def sample_forecasting_tensors():
    """Pair of (pred, gt) tensors for forecasting reward tests."""
    import torch

    torch.manual_seed(42)
    return torch.randn(32, 10), torch.randn(32, 10)


@pytest.fixture
def sample_generative_tensors():
    """Pair of (gt, pred) 3D tensors for generative reward tests."""
    import torch

    torch.manual_seed(42)
    return torch.randn(64, 24, 5), torch.randn(64, 24, 5)
