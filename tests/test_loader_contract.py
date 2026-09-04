"""Contract tests for LLM-generated staging loaders.

A candidate loader exposes::

    def load(data_dir: Path) -> dict[str, np.ndarray | None]:
        return {"features": <ndarray>, "ground_truth": <ndarray | None>}

Installation runs the approved loader to materialize HDF5 artifacts, then
ships the deterministic runtime loader.

These tests are invoked by ``benchmark.loader.executor.run_loader_script``
via::

    pytest tests/test_loader_contract.py
        --loader-module <abs path to load.py>
        --loader-data-dir <abs path to dataset data/ dir>

Without either CLI option the tests skip. Unique module names prevent stale
imports across review rounds.

Ground-truth assertions follow ``interaction_model`` rather than the source
artifact's storage shape.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from openfinai_pipeline.benchmark.loader import LOADER_REQUIRES_GT_MODELS


def _resolve_data_dir(loader_data_dir: str | None) -> Path | None:
    if loader_data_dir is None:
        return None
    path = Path(loader_data_dir).resolve()
    if not path.exists() or not path.is_dir():
        return None
    return path


def _read_manifest(data_dir: Path) -> dict:
    """Read the dataset manifest sitting at ``<data_dir>/../manifest.json``.

    Returns an empty dict if the manifest is missing or unparseable —
    the contract tests degrade to the safest assumptions in that case.
    """
    manifest_path = data_dir.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@pytest.fixture
def loader_module(loader_module_path):
    """Import the load.py module under test by absolute path.

    Skip the entire suite when ``--loader-module`` is not provided.
    """
    if loader_module_path is None:
        pytest.skip("--loader-module not provided")
    path = Path(loader_module_path).resolve()
    if not path.exists() or not path.is_file():
        pytest.skip(f"loader path does not exist: {path}")
    # Unique module name keeps subsequent rounds from hitting stale imports.
    mod_name = f"_dataset_loader_under_test_{abs(hash(str(path)))}"
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        pytest.fail(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def resolved_data_dir(loader_data_dir):
    """Return the resolved data dir or skip when not provided / missing."""
    path = _resolve_data_dir(loader_data_dir)
    if path is None:
        pytest.skip("--loader-data-dir not provided or does not exist")
    return path


@pytest.fixture
def manifest_payload(resolved_data_dir):
    return _read_manifest(resolved_data_dir)


@pytest.fixture
def interaction_model(manifest_payload):
    return str(manifest_payload.get("interaction_model", "")).strip().lower()


@pytest.fixture
def requires_ground_truth(interaction_model):
    """True when this task family expects ``ground_truth`` to be a non-None
    array at evaluation time.

    Mirrors the gate ``benchmark/loader/executor.py`` uses to decide whether
    to invoke loader codegen at all. Forecasting/generative tasks always need
    a target array — whether it's loaded from a disk artifact (case A:
    ``has_ground_truth=true``) or derived from a feature column (case B:
    ``has_ground_truth=false``, deferred-GT) is the loader's internal choice
    and does not change the contract on its output.
    """
    return interaction_model in LOADER_REQUIRES_GT_MODELS


class TestLoaderContract:
    """Every per-dataset ``load.py`` must pass these assertions.

    The Phase 4 evaluator template imports ``load.py`` and reads its
    ``ground_truth`` value — these tests guarantee that import doesn't
    explode and that the dict shape is exactly what downstream code relies
    on.
    """

    def test_module_imports_cleanly(self, loader_module):
        # Reaching here means the import in the fixture succeeded.
        assert loader_module is not None

    def test_load_symbol_exists_and_is_callable(self, loader_module):
        assert hasattr(loader_module, "load"), (
            "load.py must export a top-level `load` symbol"
        )
        assert callable(loader_module.load), "`load` must be callable"

    def test_load_signature_takes_data_dir(self, loader_module):
        sig = inspect.signature(loader_module.load)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        assert len(params) >= 1, (
            f"load() must accept a positional data_dir argument; got {sig}"
        )

    # Helpers — used by the B-shape / reference-shape assertions

    @staticmethod
    def _is_b_shape(out: dict) -> bool:
        return "train" in out and "test" in out

    @staticmethod
    def _is_reference_shape(out: dict) -> bool:
        return "reference" in out and "train" not in out and "test" not in out

    @staticmethod
    def _flatten_leaf(arr) -> np.ndarray:
        """Coerce a single-ndarray-or-panel-dict leaf to a single 2D array.

        For panel data (``{sym: ndarray}``), concatenates per-symbol
        arrays along axis 0. The result is shape (sum_of_lengths,
        n_features) for 2D inputs or (sum_of_lengths,) for 1D.
        """
        if isinstance(arr, dict):
            parts = [np.asarray(v, dtype=np.float64) for v in arr.values()]
            return np.concatenate(parts, axis=0)
        return np.asarray(arr, dtype=np.float64)

    def _assert_leaf_finite_float(self, label: str, leaf):
        flat = self._flatten_leaf(leaf)
        assert flat.size > 0, f"{label}: must not be empty"
        assert np.issubdtype(flat.dtype, np.floating), (
            f"{label}: dtype must be float; got {flat.dtype}"
        )
        assert np.all(np.isfinite(flat)), (
            f"{label}: contains non-finite values"
        )

    # Shape assertions

    def test_load_returns_expected_dict_shape(
        self, loader_module, resolved_data_dir
    ):
        out = loader_module.load(resolved_data_dir)
        assert isinstance(out, dict), (
            f"load() must return a dict; got {type(out).__name__}"
        )
        is_b = self._is_b_shape(out)
        is_ref = self._is_reference_shape(out)
        assert is_b or is_ref, (
            f"load() must return either B-shape (with 'train'+'test' keys) "
            f"or reference-shape (with only 'reference'); got keys "
            f"{sorted(out.keys())}"
        )
        if is_b:
            for split in ("train", "test"):
                assert isinstance(out[split], dict), (
                    f"load()[{split!r}] must be a dict; got "
                    f"{type(out[split]).__name__}"
                )
                for leaf_key in ("features", "ground_truth"):
                    assert leaf_key in out[split], (
                        f"load()[{split!r}] missing required leaf {leaf_key!r}"
                    )

    def test_split_policy_present_and_valid(
        self, loader_module, resolved_data_dir
    ):
        out = loader_module.load(resolved_data_dir)
        sp = out.get("split_policy")
        valid = (
            "chronological_80_20",
            "random_80_20",
            "paper_dates",
            "no_split",
        )
        assert sp in valid, (
            f"split_policy must be one of {valid}; got {sp!r}"
        )
        if "reference" in out:
            assert sp == "no_split", (
                f"reference-shape requires split_policy='no_split'; got {sp!r}"
            )
        else:
            assert sp != "no_split", (
                "B-shape forbids split_policy='no_split'"
            )

    def test_features_array_shape(self, loader_module, resolved_data_dir):
        """features (per split for B-shape) is non-empty finite float."""
        out = loader_module.load(resolved_data_dir)
        if self._is_reference_shape(out):
            self._assert_leaf_finite_float("reference", out["reference"])
            return
        for split in ("train", "test"):
            self._assert_leaf_finite_float(
                f"{split}.features", out[split]["features"]
            )

    def test_ground_truth_present_when_required(
        self, loader_module, resolved_data_dir, requires_ground_truth
    ):
        """B-shape: train + test ground_truth are both non-None floats.

        The install slicer omits ``test["ground_truth"]`` from the
        agent-readable HDF5, but the loader's PRE-INSTALL output here
        must still carry it — that's what the slicer demuxes.
        """
        out = loader_module.load(resolved_data_dir)
        if self._is_reference_shape(out):
            # Reference-shape has no train/test; the reference ndarray
            # is exercised by test_features_array_shape above.
            return
        if not requires_ground_truth:
            pytest.skip(
                "interaction model does not require a ground_truth array"
            )
        for split in ("train", "test"):
            gt = out[split].get("ground_truth")
            assert gt is not None, (
                f"loader output {split}.ground_truth is None; B-shape "
                "requires non-None test+train targets so the install "
                "slicer can demux."
            )
            self._assert_leaf_finite_float(f"{split}.ground_truth", gt)

    def test_forecasting_per_split_lengths_align(
        self,
        loader_module,
        resolved_data_dir,
        interaction_model,
    ):
        """Forecasting requires per-row alignment between features and GT
        within EACH split (train and test independently)."""
        if interaction_model != "forecasting":
            pytest.skip(
                f"length-alignment is forecasting-only; got {interaction_model!r}"
            )
        out = loader_module.load(resolved_data_dir)
        if self._is_reference_shape(out):
            pytest.skip("reference-shape has no train/test split")
        for split in ("train", "test"):
            features = self._flatten_leaf(out[split]["features"])
            gt = self._flatten_leaf(out[split]["ground_truth"])
            assert len(features) == len(gt), (
                f"forecasting {split} features/ground_truth length mismatch: "
                f"len(features)={len(features)} vs len(ground_truth)={len(gt)}"
            )

    def test_train_test_have_no_zero_length(
        self, loader_module, resolved_data_dir
    ):
        """Both splits must have at least one row."""
        out = loader_module.load(resolved_data_dir)
        if self._is_reference_shape(out):
            pytest.skip("reference-shape has no train/test split")
        for split in ("train", "test"):
            features = self._flatten_leaf(out[split]["features"])
            assert len(features) > 0, (
                f"{split} split is empty (zero rows); split policy "
                f"{out.get('split_policy')!r} must keep at least one row "
                "per split"
            )

    def test_no_obvious_feature_target_leakage(
        self,
        loader_module,
        resolved_data_dir,
        interaction_model,
    ):
        """Reject loaders where a feature column linearly reveals the GT.

        For forecasting tasks, any feature column that is a deterministic
        affine transform of a ground-truth column (literal copy, min-max
        scaled, z-score normalized, sign flipped) shows
        ``|Pearson corr| ~= 1.0``. We fail at >= 0.9999 so these jump out
        immediately.

        Runs on BOTH disk-loaded GT and derived GT (deferred-GT case) —
        derivation is exactly when leakage is most likely, because the
        LLM has to remember to drop the source column AND its derivatives
        (``EMA_close``, ``MACD_close``, ``scaled_close``, ...) from
        features after computing ``sign(close[t+h] - close[t])``.

        What this DOES catch:
          - the literal target column kept as a feature
          - ``scaled_<target>`` (min-max), ``z_<target>`` (z-score), and
            other affine transforms — corr is exactly +/-1.0
          - any sign-flipped or rescaled copy

        What this does NOT catch (these rely on the codegen + review
        prompt leakage rules instead):
          - lagged copies (``X[t-1]`` predicting ``X[t]`` is a legitimate
            feature for some tasks; corr is high but < 0.9999 for noisy
            series)
          - same-day OHLC vs same-day close (high/low have corr < 0.9999
            with close on volatile assets, but still leak structurally)

        Nonlinear monotone transforms (``log``, ``sqrt``, ``pctdiff``,
        ``tanh``) are NOT caught here because Pearson is sensitive to
        curvature — they're covered by
        :meth:`test_no_monotone_nonlinear_feature_target_leakage` below
        via Spearman rank correlation at the same threshold.
        """
        if interaction_model != "forecasting":
            pytest.skip(
                f"leakage check is forecasting-only; got {interaction_model!r}"
            )
        out = loader_module.load(resolved_data_dir)
        if self._is_reference_shape(out):
            pytest.skip("reference-shape has no train/test split")
        # Concatenate train + test rows for higher statistical power on
        # the Pearson correlation check; structural leakage (the literal
        # target column or an affine transform of it) is the same shape
        # in train and test, so the union is the natural axis.
        features = np.concatenate(
            [
                self._flatten_leaf(out["train"]["features"]),
                self._flatten_leaf(out["test"]["features"]),
            ],
            axis=0,
        )
        gt = np.concatenate(
            [
                self._flatten_leaf(out["train"]["ground_truth"]),
                self._flatten_leaf(out["test"]["ground_truth"]),
            ],
            axis=0,
        )
        if len(features) != len(gt):
            pytest.skip(
                "features/gt length mismatch (failed by "
                "test_forecasting_per_split_lengths_align)"
            )
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        gt_2d = gt.reshape(-1, 1) if gt.ndim == 1 else gt
        n = features.shape[0]
        if n < 5:
            pytest.skip(
                f"series too short for meaningful correlation check: n={n}"
            )

        leakage_threshold = 0.9999
        findings: list[str] = []
        for j in range(features.shape[1]):
            fj = features[:, j]
            sf = float(np.std(fj))
            if not np.isfinite(sf) or sf == 0.0:
                continue
            for k in range(gt_2d.shape[1]):
                gk = gt_2d[:, k]
                sg = float(np.std(gk))
                if not np.isfinite(sg) or sg == 0.0:
                    continue
                corr = float(np.corrcoef(fj, gk)[0, 1])
                if not np.isfinite(corr):
                    continue
                if abs(corr) >= leakage_threshold:
                    findings.append(
                        f"  - features[:, {j}] vs ground_truth[:, {k}]: "
                        f"|corr|={abs(corr):.6f}"
                    )

        assert not findings, (
            "Probable target leakage detected (Pearson |corr| >= "
            f"{leakage_threshold} between feature and ground_truth columns):\n"
            + "\n".join(findings)
            + "\n\nFor forecasting tasks, feature columns must not be "
            "deterministic transforms of the ground-truth target. Common "
            "offenders to drop: the literal target column itself, plus any "
            "``scaled_X``, ``norm_X``, ``z_X``, ``log_X``, ``pctdiff_X``, "
            "or ``ret_X`` derivative (X = the target column)."
        )

    def test_no_monotone_nonlinear_feature_target_leakage(
        self,
        loader_module,
        resolved_data_dir,
        interaction_model,
    ):
        """Reject loaders where a feature column is a monotone (possibly
        nonlinear) function of a ground-truth column.

        Spearman rank correlation = Pearson on rank-transformed values,
        so it equals +/-1.0 exactly for any monotone transformation
        regardless of curvature: ``log(x)``, ``sqrt(x)``, ``exp(x)``,
        ``tanh(x)``, and any other rank-preserving function. Pearson is
        sensitive to the curvature and may fall short of the 0.9999
        threshold even when the feature perfectly recovers the target's
        ordering.

        What this ADDS over the Pearson check at the same threshold:
          - ``log_<target>``, ``sqrt_<target>``, ``exp_<target>``,
            ``tanh_<target>``, ``z_score(log(<target>))``-style
            compositions kept under cryptic feature names that bypass
            the codegen / reviewer substring scan.
          - Any feature engineered as ``f(target)`` where ``f`` is
            strictly monotone — same-row alignment only.

        What this does NOT add (still relies on the codegen / review
        prompt rules):
          - Lagged copies (different time indices — Spearman is also
            < 1 for noisy series).
          - Multivariate / interaction leakage where two features
            jointly reconstruct the target.
          - Look-ahead bias from data-prep performed inside ``load.py``.
        """
        if interaction_model != "forecasting":
            pytest.skip(
                f"leakage check is forecasting-only; got {interaction_model!r}"
            )
        out = loader_module.load(resolved_data_dir)
        if self._is_reference_shape(out):
            pytest.skip("reference-shape has no train/test split")
        # Concatenate train + test rows for higher statistical power on
        # the rank-correlation check; structural monotone leakage is the
        # same shape in train and test, so the union is the natural axis.
        features = np.concatenate(
            [
                self._flatten_leaf(out["train"]["features"]),
                self._flatten_leaf(out["test"]["features"]),
            ],
            axis=0,
        )
        gt = np.concatenate(
            [
                self._flatten_leaf(out["train"]["ground_truth"]),
                self._flatten_leaf(out["test"]["ground_truth"]),
            ],
            axis=0,
        )
        if len(features) != len(gt):
            pytest.skip(
                "features/gt length mismatch (failed by "
                "test_forecasting_per_split_lengths_align)"
            )
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        gt_2d = gt.reshape(-1, 1) if gt.ndim == 1 else gt
        n = features.shape[0]
        if n < 5:
            pytest.skip(
                f"series too short for meaningful rank correlation: n={n}"
            )

        leakage_threshold = 0.9999
        findings: list[str] = []
        for j in range(features.shape[1]):
            fj = features[:, j]
            sf = float(np.std(fj))
            if not np.isfinite(sf) or sf == 0.0:
                continue
            for k in range(gt_2d.shape[1]):
                gk = gt_2d[:, k]
                sg = float(np.std(gk))
                if not np.isfinite(sg) or sg == 0.0:
                    continue
                rho, _ = spearmanr(fj, gk)
                rho_f = float(rho)
                if not np.isfinite(rho_f):
                    continue
                if abs(rho_f) >= leakage_threshold:
                    findings.append(
                        f"  - features[:, {j}] vs ground_truth[:, {k}]: "
                        f"|spearman|={abs(rho_f):.6f}"
                    )

        assert not findings, (
            "Probable monotone-nonlinear target leakage (Spearman "
            f"|rho| >= {leakage_threshold} between feature and "
            "ground_truth columns):\n"
            + "\n".join(findings)
            + "\n\nA Spearman |rho| of 1.0 means the feature column "
            "ranks perfectly with a ground_truth column — i.e. some "
            "monotone function ``f`` satisfies "
            "``feature[i] = f(ground_truth[i])`` for all rows. Common "
            "causes: ``log(target)``, ``sqrt(target)``, ``tanh(target)``, "
            "z-score of any of those, or any rank-preserving transform "
            "kept under a feature name that doesn't contain the target "
            "name as a substring. Drop the offending column. The "
            "Pearson check at the same threshold covers literal copies "
            "and affine transforms; this test covers the strictly-"
            "monotone-but-nonlinear family."
        )

    def test_load_is_idempotent(self, loader_module, resolved_data_dir):
        """Calling load() twice should produce equivalent arrays.

        Catches loaders that mutate state, consume generators, or close
        file handles after first read. Cheap to run on small fixtures.
        Walks the B-shape / reference-shape recursively to compare
        every ndarray leaf.
        """
        first = loader_module.load(resolved_data_dir)
        second = loader_module.load(resolved_data_dir)

        def _compare(a, b, label: str):
            if isinstance(a, np.ndarray):
                assert isinstance(b, np.ndarray), (
                    f"{label}: shape changed across calls"
                )
                np.testing.assert_array_equal(
                    a, b, err_msg=f"{label} not idempotent"
                )
                return
            if isinstance(a, dict) and isinstance(b, dict):
                assert set(a.keys()) == set(b.keys()), (
                    f"{label}: dict keys changed across calls"
                )
                for k in a:
                    _compare(a[k], b[k], f"{label}[{k!r}]")
                return
            assert a == b, f"{label}: scalar/None mismatch ({a!r} vs {b!r})"

        # Compare top-level keys we care about.
        for key in ("train", "test", "reference"):
            if key in first or key in second:
                _compare(first.get(key), second.get(key), f"load()[{key!r}]")
