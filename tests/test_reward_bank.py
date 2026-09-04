"""Tests for the hand-written torch reward bank (reward_bank.py).

Validates that all Loss subclasses pass the same interface and behavioral
contract used for LLM-generated reward_fn rewards.
"""
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# Ensure the reward bank directory is importable.
_eval_root = str(
    Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "rewards"
)
if _eval_root not in sys.path:
    sys.path.insert(0, _eval_root)

import reward_bank  # noqa: E402

# Forecasting rewards: __init__(y_true) + forward(y_pred) -> scalar
FORECASTING_CLASSES = [
    reward_bank.MSELoss,
    reward_bank.RMSELoss,
    reward_bank.MAELoss,
    reward_bank.MAPELoss,
    reward_bank.DirectionalAccuracy,
    reward_bank.SignLoss,
    reward_bank.PearsonCorrelation,
    reward_bank.SpearmanCorrelation,
    reward_bank.R2Score,
    reward_bank.HuberLoss,
    reward_bank.SharpeRatioLoss,
    reward_bank.SortinoRatio,
]


@pytest.mark.parametrize(
    "cls",
    FORECASTING_CLASSES,
    ids=lambda c: c.__name__,
)
class TestForecastingRewards:
    """Forecasting rewards capture y_true at init, take y_pred at forward."""

    def test_instantiation(self, cls, sample_forecasting_tensors):
        _, y_true = sample_forecasting_tensors
        instance = cls(y_true, name="test")
        assert instance is not None

    def test_forward_signature(self, cls, sample_forecasting_tensors):
        y_pred, y_true = sample_forecasting_tensors
        instance = cls(y_true, name="test")
        result = instance(y_pred)
        assert isinstance(result, torch.Tensor)

    def test_returns_scalar(self, cls, sample_forecasting_tensors):
        y_pred, y_true = sample_forecasting_tensors
        instance = cls(y_true, name="test")
        result = instance(y_pred)
        assert (
            result.dim() == 0 or result.numel() == 1
        ), f"{cls.__name__} returned shape {result.shape}, expected scalar"

    def test_returns_finite(self, cls, sample_forecasting_tensors):
        y_pred, y_true = sample_forecasting_tensors
        instance = cls(y_true, name="test")
        result = instance(y_pred)
        assert torch.isfinite(
            result
        ).all(), f"{cls.__name__} returned non-finite: {result}"


# Forecasting rewards with extra constructor args
class TestQuantileLoss:
    def test_default_quantile(self, sample_forecasting_tensors):
        y_pred, y_true = sample_forecasting_tensors
        instance = reward_bank.QuantileLoss(y_true, name="test")
        result = instance(y_pred)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0 or result.numel() == 1
        assert torch.isfinite(result).all()

    def test_custom_quantile(self, sample_forecasting_tensors):
        y_pred, y_true = sample_forecasting_tensors
        instance = reward_bank.QuantileLoss(y_true, quantile=0.9, name="test")
        result = instance(y_pred)
        assert torch.isfinite(result).all()


class TestCRPSLoss:
    def test_forward(self):
        torch.manual_seed(42)
        # CRPSLoss: y_true at init [N, D], y_pred_samples at forward [N, S, D].
        y_pred_samples = torch.randn(16, 10, 5)
        y_true = torch.randn(16, 5)
        instance = reward_bank.CRPSLoss(y_true, name="test")
        result = instance(y_pred_samples)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0 or result.numel() == 1
        assert torch.isfinite(result).all()


class TestF1Score:
    def test_default_threshold(self, sample_forecasting_tensors):
        y_pred, y_true = sample_forecasting_tensors
        instance = reward_bank.F1Score(y_true, name="test")
        result = instance(y_pred)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0 or result.numel() == 1
        assert torch.isfinite(result).all()
        assert 0.0 <= result.item() <= 1.0

    def test_custom_threshold(self, sample_forecasting_tensors):
        y_pred, y_true = sample_forecasting_tensors
        instance = reward_bank.F1Score(y_true, threshold=0.5, name="test")
        result = instance(y_pred)
        assert torch.isfinite(result).all()
        assert 0.0 <= result.item() <= 1.0


class TestNewMetricsEdgeCases:
    """Targeted tests for eps-fallback branches that the random fixture won't hit."""

    def test_pearson_constant_pred(self):
        y_true = torch.randn(16, 5)
        y_pred = torch.zeros(16, 5)
        result = reward_bank.PearsonCorrelation(y_true, name="test")(y_pred)
        assert torch.isfinite(result).all()

    def test_spearman_constant_pred(self):
        y_true = torch.randn(16, 5)
        y_pred = torch.zeros(16, 5)
        result = reward_bank.SpearmanCorrelation(y_true, name="test")(y_pred)
        assert torch.isfinite(result).all()

    def test_r2_constant_target(self):
        y_true = torch.full((16, 5), 0.42)
        y_pred = torch.randn(16, 5)
        result = reward_bank.R2Score(y_true, name="test")(y_pred)
        assert torch.isfinite(result).all()

    def test_sharpe_single_element(self):
        y_true = torch.tensor([0.01])
        y_pred = torch.tensor([0.005])
        result = reward_bank.SharpeRatioLoss(y_true, name="test")(y_pred)
        assert torch.isfinite(result).all()
        assert result.item() == 0.0

    def test_sortino_no_downside(self):
        y_true = torch.ones(16, 5)
        y_pred = torch.ones(16, 5)
        result = reward_bank.SortinoRatio(y_true, name="test")(y_pred)
        assert torch.isfinite(result).all()
        assert result.item() > 0

    def test_f1_all_positive_class(self):
        y_true = torch.ones(16, 5)
        y_pred = torch.ones(16, 5)
        result = reward_bank.F1Score(y_true, name="test")(y_pred)
        assert torch.isfinite(result).all()
        assert 0.0 <= result.item() <= 1.0


# Generative rewards with __init__(x_real) + compute(x_fake)
SIMPLE_GENERATIVE_CLASSES = [
    reward_bank.MeanLoss,
    reward_bank.StdLoss,
    reward_bank.SkewnessLoss,
    reward_bank.KurtosisLoss,
]


@pytest.mark.parametrize(
    "cls",
    SIMPLE_GENERATIVE_CLASSES,
    ids=lambda c: c.__name__,
)
class TestSimpleGenerativeRewards:
    """Generative rewards that take x_real in __init__ and x_fake in compute."""

    def test_instantiation(self, cls, sample_generative_tensors):
        x_real, _ = sample_generative_tensors
        instance = cls(x_real, name="test")
        assert instance is not None

    def test_forward_returns_tensor(self, cls, sample_generative_tensors):
        x_real, x_fake = sample_generative_tensors
        instance = cls(x_real, name="test")
        result = instance(x_fake)
        assert isinstance(result, torch.Tensor)

    def test_forward_returns_scalar(self, cls, sample_generative_tensors):
        x_real, x_fake = sample_generative_tensors
        instance = cls(x_real, name="test")
        result = instance(x_fake)
        assert (
            result.dim() == 0 or result.numel() == 1
        ), f"{cls.__name__} returned shape {result.shape}, expected scalar"

    def test_forward_returns_finite(self, cls, sample_generative_tensors):
        x_real, x_fake = sample_generative_tensors
        instance = cls(x_real, name="test")
        result = instance(x_fake)
        assert torch.isfinite(
            result
        ).all(), f"{cls.__name__} returned non-finite: {result}"


# Distance-style generative rewards: x_real at init, x_fake at forward
# (formerly two-arg forward(x_real, x_fake); migrated per RULE A).
DISTANCE_GENERATIVE_CLASSES = [
    reward_bank.VARMetricLoss,
    reward_bank.ONNDMetricLoss,
]


@pytest.mark.parametrize(
    "cls",
    DISTANCE_GENERATIVE_CLASSES,
    ids=lambda c: c.__name__,
)
class TestDistanceGenerativeRewards:
    """Generative rewards that capture x_real at init and take x_fake at forward."""

    def test_instantiation(self, cls, sample_generative_tensors):
        x_real, _ = sample_generative_tensors
        instance = cls(x_real, name="test")
        assert instance is not None

    def test_forward_returns_tensor(self, cls, sample_generative_tensors):
        x_real, x_fake = sample_generative_tensors
        instance = cls(x_real, name="test")
        result = instance(x_fake)
        assert isinstance(result, torch.Tensor)

    def test_forward_returns_scalar(self, cls, sample_generative_tensors):
        x_real, x_fake = sample_generative_tensors
        instance = cls(x_real, name="test")
        result = instance(x_fake)
        assert (
            result.dim() == 0 or result.numel() == 1
        ), f"{cls.__name__} returned shape {result.shape}, expected scalar"

    def test_forward_returns_finite(self, cls, sample_generative_tensors):
        x_real, x_fake = sample_generative_tensors
        instance = cls(x_real, name="test")
        result = instance(x_fake)
        assert torch.isfinite(
            result
        ).all(), f"{cls.__name__} returned non-finite: {result}"


# Single-argument generative rewards: forward(x_fake) → scalar (no x_real in init)
class TestICDMetricLoss:
    """ICDMetricLoss uses base Loss forward(x_fake) pattern."""

    def test_forward(self, sample_generative_tensors):
        _, x_fake = sample_generative_tensors
        instance = reward_bank.ICDMetricLoss(name="test")
        result = instance(x_fake)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0 or result.numel() == 1
        assert torch.isfinite(result).all()


# Shared signature-introspecting helpers (used by the unified contract).
# Names of __init__ params that bind to ground-truth / reference tensors.
# Under the canonical-name contract the param IS the ctx key, so the dict
# is keyed by canonical names (gt / real_emb / gt_samples).
_INIT_REF_PARAM_TO_SHAPE: dict[str, str] = {
    "gt": "2d",          # default for forecasting / pair-distance metrics
    "real_emb": "2d",
    "gt_samples": "3d",  # reserved
}

# Per-class override for classes whose `gt` is actually a 3D distributional
# tensor (path-space generative metrics). Without this override the harness
# would feed a 2D sample and the class would crash on shape mismatch.
# Names refer to ``cls.__name__`` so the override survives renames.
_INIT_GT_3D_OVERRIDES: frozenset[str] = frozenset(
    {
        "ACFLoss",
        "MeanLoss",
        "StdLoss",
        "SkewnessLoss",
        "KurtosisLoss",
        "CrossCorrelLoss",
        "CovLoss",
        "Sig_MMD_loss",
        "cross_correlation",
        "SigW1Loss",
        "HistogramLoss",
        "VARMetricLoss",
        "ONNDMetricLoss",
    }
)

# Names of forward() params that bind to agent-output tensors.
_FORWARD_PRED_PARAM_TO_SHAPE: dict[str, str] = {
    "pred": "2d",          # default for forecasting metrics
    "pred_samples": "3d",  # probabilistic / multi-sample forecasts
    "fake_emb": "2d",
}

# Per-class override for classes whose `pred` is actually a 3D path-space
# tensor (matches their 3D `gt`).
_FORWARD_PRED_3D_OVERRIDES: frozenset[str] = frozenset(
    {
        "ACFLoss",
        "MeanLoss",
        "StdLoss",
        "SkewnessLoss",
        "KurtosisLoss",
        "CrossCorrelLoss",
        "CovLoss",
        "Sig_MMD_loss",
        "cross_correlation",
        "SigW1Loss",
        "HistogramLoss",
        "VARMetricLoss",
        "ONNDMetricLoss",
        "ICDMetricLoss",
    }
)


def _required_params(fn) -> list[inspect.Parameter]:
    """Return non-self parameters of *fn* that have no default value.

    Skips ``name`` (always optional via the Loss base) and varargs.
    """
    sig = inspect.signature(fn)
    return [
        p
        for p in sig.parameters.values()
        if p.name not in ("self", "name")
        and p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


def _make_instance(cls, sample_2d, sample_3d=None):
    """Build an instance of *cls* by inspecting its ``__init__`` signature.

    The picker is deterministic: if ``__init__`` declares any required
    reference parameter (per :data:`_INIT_REF_PARAM_TO_SHAPE`), the
    corresponding sample tensor is bound positionally; otherwise the
    no-arg ``cls(name="test")`` is used. There is no order-dependent
    try/except fallback — required references must be declared without
    a default value (RULE B), so introspection finds them deterministically.

    The harness also peeks at ``forward`` to keep init/forward shapes
    consistent: if ``forward`` takes ``y_pred_samples`` (3D probabilistic
    forecasts), the y_true reference is projected to a matching 2D
    ``[N, D]`` shape derived from ``sample_3d``.
    """
    required = _required_params(cls.__init__)
    if not required:
        return cls(name="test")
    # Pick the sample tensor for the first required reference parameter.
    first = required[0]
    shape_hint = _INIT_REF_PARAM_TO_SHAPE.get(first.name)
    # Class-level override: some Loss subclasses use canonical name ``gt``
    # but expect a 3D path-space tensor (generative / signature-based metrics).
    if first.name == "gt" and cls.__name__ in _INIT_GT_3D_OVERRIDES and sample_3d is not None:
        sample = sample_3d
    elif shape_hint == "3d" and sample_3d is not None:
        sample = sample_3d
    elif shape_hint == "2d":
        sample = sample_2d
    else:
        # Unknown reference name — fall back to 2D, but surface clearly if
        # instantiation fails so the error message names the offending
        # parameter (rather than a generic "could not instantiate").
        sample = sample_2d

    # Probabilistic forecasting case: gt at init + pred_samples at forward
    # needs aligned [N, D] vs [N, S, D]. Project sample_3d to 2D.
    if first.name == "gt" and cls.__name__ not in _INIT_GT_3D_OVERRIDES and sample_3d is not None:
        forward_param_names = [
            p.name
            for p in inspect.signature(cls.forward).parameters.values()
            if p.name != "self"
        ]
        if "pred_samples" in forward_param_names:
            sample = sample_3d[:, 0, :]  # [N, D] derived from [N, S, D]

    try:
        return cls(sample, name="test")
    except TypeError:
        return cls(**{first.name: sample}, name="test")


def _call_forward(instance, y_pred, y_true, x_real_3d=None, x_fake_3d=None):
    """Call ``instance.forward()`` with the right agent-output tensor.

    Per RULE R3, ``forward`` MUST NOT take any reference tensor — only
    agent output. Pick the input by introspecting ``forward``'s parameter
    names against :data:`_FORWARD_PRED_PARAM_TO_SHAPE` and the per-class
    3D override set.
    """
    # ``y_true`` and ``x_real_3d`` are kept in the signature so existing
    # call sites do not need to change, but they are intentionally unused:
    # under the canonical-name contract reference tensors are bound at
    # init time.
    del y_true, x_real_3d
    sig = inspect.signature(instance.forward)
    forward_params = [p for p in sig.parameters.values() if p.name != "self"]

    if not forward_params:
        pytest.fail(
            f"{type(instance).__name__}.forward() takes no parameters; "
            "expected exactly one agent-output tensor."
        )

    # Reject the legacy 2-arg shape early with a clear message.
    if len(forward_params) >= 2:
        names = [p.name for p in forward_params]
        pytest.fail(
            f"{type(instance).__name__}.forward({', '.join(names)}) violates "
            "RULE R3: forward must take exactly one agent-output tensor. "
            "Capture the reference in __init__."
        )

    arg_name = forward_params[0].name
    cls_name = type(instance).__name__
    shape_hint = _FORWARD_PRED_PARAM_TO_SHAPE.get(arg_name)
    # Class-level override: Loss subclasses where canonical ``pred`` carries
    # a 3D path-space tensor.
    if arg_name == "pred" and cls_name in _FORWARD_PRED_3D_OVERRIDES and x_fake_3d is not None:
        return instance(x_fake_3d)
    if shape_hint == "3d" and x_fake_3d is not None:
        return instance(x_fake_3d)
    return instance(y_pred)


# Unified contract: same tests for ALL rewards (hand-written + generated)
# All hand-written reward classes that follow the standard Loss pattern
# (i.e. those tested by the explicit test classes above).
# Excludes special-purpose rewards that require extra args like a trained model
# (e.g. Predictive_FID, Predictive_KID) or non-Loss nn.Module subclasses
# (e.g. HistogramLoss).
_REWARD_BANK_CLASSES = (
    FORECASTING_CLASSES
    + [reward_bank.QuantileLoss, reward_bank.CRPSLoss]
    + SIMPLE_GENERATIVE_CLASSES
    + DISTANCE_GENERATIVE_CLASSES
    + [reward_bank.ICDMetricLoss]
)


def _collect_all_reward_classes(reward_module_path=None):
    """Gather standard reward classes + optional reward_fn module classes."""
    from reward_bank import Loss

    classes = list(_REWARD_BANK_CLASSES)
    if reward_module_path:
        spec = importlib.util.spec_from_file_location(
            "reward_fn_under_test", reward_module_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, Loss) and obj is not Loss:
                classes.append(obj)
    return classes


@pytest.fixture
def unified_reward_classes(reward_module_path):
    classes = _collect_all_reward_classes(reward_module_path)
    assert len(classes) >= 1
    return classes


class TestUnifiedRewardContract:
    """Every reward — hand-written or generated — must pass these tests.

    Always runs in CI against all hand-written rewards from reward_bank.py.
    When --reward-module is provided, generated reward_fn classes are added
    to the same list and tested through the same assertions.
    """

    def test_is_nn_module(self, unified_reward_classes):
        for cls in unified_reward_classes:
            assert issubclass(cls, nn.Module), f"{cls.__name__} not nn.Module"

    def test_has_forward(self, unified_reward_classes):
        for cls in unified_reward_classes:
            assert hasattr(cls, "forward"), f"{cls.__name__} missing forward()"

    def test_instantiation(
        self,
        unified_reward_classes,
        sample_forecasting_tensors,
        sample_generative_tensors,
    ):
        y_pred, _ = sample_forecasting_tensors
        x_real, _ = sample_generative_tensors
        for cls in unified_reward_classes:
            _make_instance(cls, y_pred, x_real)

    def test_forward_returns_tensor(
        self,
        unified_reward_classes,
        sample_forecasting_tensors,
        sample_generative_tensors,
    ):
        y_pred, y_true = sample_forecasting_tensors
        x_real, x_fake = sample_generative_tensors
        for cls in unified_reward_classes:
            instance = _make_instance(cls, y_pred, x_real)
            result = _call_forward(
                instance, y_pred, y_true, x_real_3d=x_real, x_fake_3d=x_fake
            )
            assert isinstance(result, torch.Tensor), (
                f"{cls.__name__}.forward() returned {type(result).__name__}, "
                "expected Tensor"
            )

    def test_forward_returns_scalar(
        self,
        unified_reward_classes,
        sample_forecasting_tensors,
        sample_generative_tensors,
    ):
        y_pred, y_true = sample_forecasting_tensors
        x_real, x_fake = sample_generative_tensors
        for cls in unified_reward_classes:
            instance = _make_instance(cls, y_pred, x_real)
            result = _call_forward(
                instance, y_pred, y_true, x_real_3d=x_real, x_fake_3d=x_fake
            )
            assert (
                result.dim() == 0 or result.numel() == 1
            ), f"{cls.__name__} returned shape {result.shape}, expected scalar"

    def test_forward_returns_finite(
        self,
        unified_reward_classes,
        sample_forecasting_tensors,
        sample_generative_tensors,
    ):
        y_pred, y_true = sample_forecasting_tensors
        x_real, x_fake = sample_generative_tensors
        for cls in unified_reward_classes:
            instance = _make_instance(cls, y_pred, x_real)
            result = _call_forward(
                instance, y_pred, y_true, x_real_3d=x_real, x_fake_3d=x_fake
            )
            assert torch.isfinite(
                result
            ).all(), f"{cls.__name__} returned non-finite: {result}"
