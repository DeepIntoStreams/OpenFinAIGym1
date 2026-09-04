"""Unit tests for ``benchmark.install.slicer`` + the runtime loader read path.

These tests are pipeline-internal — they don't require a real LLM or any
on-disk dataset. Each test constructs a synthetic loader-output dict,
runs it through the slicer, writes the HDF5 artifacts to a tmp_path,
and asserts the round-trip produces the expected structure with
``test["ground_truth"]`` correctly held out from the agent payload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openfinai_pipeline.benchmark.install.runtime_loader import (
    read_dataset_h5,
    read_test_gt_h5,
)
from openfinai_pipeline.benchmark.install.slicer import (
    SliceShapeError,
    slice_b_shape,
    write_dataset_h5,
    write_test_gt_h5,
)


# Fixtures


def _b_shape_single_symbol() -> dict:
    """Standard B-shape with single ndarray leaves."""
    rng = np.random.default_rng(0)
    return {
        "train": {
            "features": rng.normal(size=(80, 5)).astype(np.float64),
            "ground_truth": rng.normal(size=(80,)).astype(np.float64),
        },
        "test": {
            "features": rng.normal(size=(20, 5)).astype(np.float64),
            "ground_truth": rng.normal(size=(20,)).astype(np.float64),
        },
        "split_policy": "chronological_80_20",
        "split_metadata": {"paper_train_end": "2018-12-31"},
    }


def _b_shape_panel() -> dict:
    """B-shape with panel data: features/gt are {sym: ndarray}."""
    rng = np.random.default_rng(1)
    return {
        "train": {
            "features": {
                "AAPL": rng.normal(size=(80, 4)).astype(np.float64),
                "MSFT": rng.normal(size=(60, 4)).astype(np.float64),
            },
            "ground_truth": {
                "AAPL": rng.normal(size=(80,)).astype(np.float64),
                "MSFT": rng.normal(size=(60,)).astype(np.float64),
            },
        },
        "test": {
            "features": {
                "AAPL": rng.normal(size=(20, 4)).astype(np.float64),
                "MSFT": rng.normal(size=(15, 4)).astype(np.float64),
            },
            "ground_truth": {
                "AAPL": rng.normal(size=(20,)).astype(np.float64),
                "MSFT": rng.normal(size=(15,)).astype(np.float64),
            },
        },
        "split_policy": "chronological_80_20",
        "split_metadata": None,
    }


def _reference_shape() -> dict:
    """Unconditional generative: just a reference set, no train/test."""
    rng = np.random.default_rng(2)
    return {
        "reference": rng.normal(size=(200, 8)).astype(np.float64),
        "split_policy": "no_split",
        "split_metadata": None,
    }


# slice_b_shape — happy paths


def test_slice_b_shape_single_symbol_extracts_test_gt():
    raw = _b_shape_single_symbol()
    result = slice_b_shape(raw)

    assert result.shape == "b_shape"
    assert result.split_policy == "chronological_80_20"
    assert result.split_metadata == {"paper_train_end": "2018-12-31"}
    # Agent payload has train fully and test features but NO test ground_truth
    assert set(result.agent_payload.keys()) == {"train", "test"}
    assert set(result.agent_payload["train"].keys()) == {"features", "ground_truth"}
    assert set(result.agent_payload["test"].keys()) == {"features"}
    # Held-out is exactly the test ground_truth ndarray
    np.testing.assert_array_equal(result.held_out, raw["test"]["ground_truth"])


def test_slice_b_shape_panel_preserves_dict_leaves():
    raw = _b_shape_panel()
    result = slice_b_shape(raw)

    assert result.shape == "b_shape"
    assert isinstance(result.agent_payload["train"]["features"], dict)
    assert isinstance(result.held_out, dict)
    assert set(result.held_out.keys()) == {"AAPL", "MSFT"}
    np.testing.assert_array_equal(
        result.held_out["AAPL"], raw["test"]["ground_truth"]["AAPL"]
    )
    np.testing.assert_array_equal(
        result.held_out["MSFT"], raw["test"]["ground_truth"]["MSFT"]
    )


def test_slice_reference_shape_no_split():
    raw = _reference_shape()
    result = slice_b_shape(raw)

    assert result.shape == "reference"
    assert result.split_policy == "no_split"
    assert result.split_metadata is None
    assert set(result.agent_payload.keys()) == {"reference"}
    # For unconditional, agent and verifier see the same reference data.
    np.testing.assert_array_equal(result.held_out, raw["reference"])


# slice_b_shape — error paths


def test_slice_rejects_missing_split_policy():
    raw = _b_shape_single_symbol()
    raw.pop("split_policy")
    with pytest.raises(SliceShapeError, match="split_policy"):
        slice_b_shape(raw)


def test_slice_rejects_missing_train_bundle():
    raw = _b_shape_single_symbol()
    raw.pop("train")
    with pytest.raises(SliceShapeError, match="missing the required 'train' bundle"):
        slice_b_shape(raw)


def test_slice_rejects_missing_test_bundle():
    raw = _b_shape_single_symbol()
    raw.pop("test")
    with pytest.raises(SliceShapeError, match="missing the required 'test' bundle"):
        slice_b_shape(raw)


def test_slice_rejects_missing_ground_truth_leaf():
    raw = _b_shape_single_symbol()
    raw["test"].pop("ground_truth")
    with pytest.raises(SliceShapeError, match="missing leaf 'ground_truth'"):
        slice_b_shape(raw)


def test_slice_rejects_empty_ndarray():
    raw = _b_shape_single_symbol()
    raw["train"]["features"] = np.empty((0, 5))
    with pytest.raises(SliceShapeError, match="ndarray is empty"):
        slice_b_shape(raw)


def test_slice_rejects_non_ndarray_leaf():
    raw = _b_shape_single_symbol()
    raw["train"]["features"] = [[1.0, 2.0]]
    with pytest.raises(SliceShapeError, match="must be np.ndarray"):
        slice_b_shape(raw)


def test_slice_rejects_panel_with_non_str_keys():
    raw = _b_shape_single_symbol()
    raw["train"]["features"] = {1: np.ones((10, 2))}
    with pytest.raises(SliceShapeError, match="panel keys must be str"):
        slice_b_shape(raw)


def test_slice_rejects_mixed_reference_and_train():
    raw = _b_shape_single_symbol()
    raw["reference"] = np.ones((10,))
    with pytest.raises(SliceShapeError, match="mixes 'reference' with 'train'"):
        slice_b_shape(raw)


def test_slice_rejects_b_shape_with_no_split_policy():
    raw = _b_shape_single_symbol()
    raw["split_policy"] = "no_split"
    with pytest.raises(SliceShapeError, match="reserved for reference-shape"):
        slice_b_shape(raw)


def test_slice_rejects_reference_shape_with_real_split_policy():
    raw = _reference_shape()
    raw["split_policy"] = "chronological_80_20"
    with pytest.raises(SliceShapeError, match="requires split_policy='no_split'"):
        slice_b_shape(raw)


# HDF5 round-trip: dataset.h5


def test_write_dataset_h5_round_trip_single_symbol(tmp_path: Path):
    raw = _b_shape_single_symbol()
    result = slice_b_shape(raw)

    h5_path = tmp_path / "dataset.h5"
    write_dataset_h5(h5_path, result)
    assert h5_path.exists()

    out = read_dataset_h5(h5_path)
    # Train round-trips fully.
    np.testing.assert_array_equal(out["train"]["features"], raw["train"]["features"])
    np.testing.assert_array_equal(
        out["train"]["ground_truth"], raw["train"]["ground_truth"]
    )
    # Test features round-trip; ground_truth is None (held out).
    np.testing.assert_array_equal(out["test"]["features"], raw["test"]["features"])
    assert out["test"]["ground_truth"] is None
    # Metadata attrs survive.
    assert out["split_policy"] == "chronological_80_20"
    assert out["split_metadata"] == {"paper_train_end": "2018-12-31"}


def test_write_dataset_h5_round_trip_panel(tmp_path: Path):
    raw = _b_shape_panel()
    result = slice_b_shape(raw)

    h5_path = tmp_path / "dataset.h5"
    write_dataset_h5(h5_path, result)
    out = read_dataset_h5(h5_path)

    assert isinstance(out["train"]["features"], dict)
    for sym in ("AAPL", "MSFT"):
        np.testing.assert_array_equal(
            out["train"]["features"][sym], raw["train"]["features"][sym]
        )
        np.testing.assert_array_equal(
            out["train"]["ground_truth"][sym], raw["train"]["ground_truth"][sym]
        )
        np.testing.assert_array_equal(
            out["test"]["features"][sym], raw["test"]["features"][sym]
        )
    # test/ground_truth held out — None at top of test bundle.
    assert out["test"]["ground_truth"] is None


def test_write_dataset_h5_round_trip_reference_shape(tmp_path: Path):
    raw = _reference_shape()
    result = slice_b_shape(raw)

    h5_path = tmp_path / "dataset.h5"
    write_dataset_h5(h5_path, result)
    out = read_dataset_h5(h5_path)

    np.testing.assert_array_equal(out["reference"], raw["reference"])
    # No train/test keys.
    assert "train" not in out
    assert "test" not in out
    assert out["split_policy"] == "no_split"


# HDF5 round-trip: test_ground_truth.h5


def test_write_test_gt_h5_round_trip_single_symbol(tmp_path: Path):
    raw = _b_shape_single_symbol()
    result = slice_b_shape(raw)

    gt_path = tmp_path / "test_ground_truth.h5"
    write_test_gt_h5(gt_path, result.held_out)
    assert gt_path.exists()

    held_out = read_test_gt_h5(gt_path)
    np.testing.assert_array_equal(held_out, raw["test"]["ground_truth"])


def test_write_test_gt_h5_round_trip_panel(tmp_path: Path):
    raw = _b_shape_panel()
    result = slice_b_shape(raw)

    gt_path = tmp_path / "test_ground_truth.h5"
    write_test_gt_h5(gt_path, result.held_out)
    held_out = read_test_gt_h5(gt_path)

    assert isinstance(held_out, dict)
    for sym in ("AAPL", "MSFT"):
        np.testing.assert_array_equal(
            held_out[sym], raw["test"]["ground_truth"][sym]
        )


def test_test_gt_h5_missing_group_raises(tmp_path: Path):
    """If somehow a test_ground_truth.h5 is malformed, the reader fails loudly."""
    import h5py

    bad_path = tmp_path / "bad.h5"
    with h5py.File(bad_path, "w") as f:
        f.create_dataset("not_ground_truth", data=np.ones((4,)))
    with pytest.raises(KeyError, match="missing required '/ground_truth'"):
        read_test_gt_h5(bad_path)


# Held-out-target disk-leak check


def test_dataset_h5_does_not_contain_test_ground_truth(tmp_path: Path):
    """The agent-readable artifact must not contain the held-out target."""
    import h5py

    raw = _b_shape_single_symbol()
    result = slice_b_shape(raw)
    h5_path = tmp_path / "dataset.h5"
    write_dataset_h5(h5_path, result)

    with h5py.File(h5_path, "r") as f:
        assert "test" in f
        assert "features" in f["test"]
        assert "ground_truth" not in f["test"], (
            "test/ground_truth must not be present in the agent-readable "
            "dataset.h5; it is held out and lives only in test_ground_truth.h5"
        )
