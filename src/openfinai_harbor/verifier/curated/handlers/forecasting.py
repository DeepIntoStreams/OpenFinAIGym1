"""Synchronous scoring for curated offline forecasting tasks.

The verifier caches held-out targets, exposes only train targets and test
features, and accepts base64-encoded ndarray or per-symbol ndarray mappings.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from openfinai_harbor.verifier.curated._coerce import coerce_scores
from openfinai_harbor.verifier.curated._envelope import (
    SubmitOutcome,
    run_submit,
)
from openfinai_harbor.verifier.curated.models import (
    CuratedForecastingSubmitRequest,
    CuratedSubmitResponse,
)


# Surfaced via /info; actual reward.json keys are whatever the task returns.

REWARD_NAMES = [
    "mse",
    "rmse",
    "mae",
    "mape",
    "r2",
    "pearson",
    "directional_accuracy",
]

#: Fallback advertised via /info when the task isn't instantiated yet;
#: instantiated tasks emit ``"reward"`` matching their ``headline_metric``.
HEADLINE_METRIC = "mape"


_HANDLER_KEY = "forecasting"


def on_startup(state) -> None:  # type: CuratedVerifierState
    """Build the cached task instance and the held-out gt at spawn time.

    Failures here propagate up through ``create_curated_app`` and abort
    the spawn — better to surface "task config invalid" or "data fetch
    failed" before uvicorn binds the port than at first submission.

    What we cache:

    * ``task`` — the curated task instance (sees the same config across
      all submissions for this spawn).
    * ``y_test_dict`` — held-out target, kept in memory only. A
      ``{symbol -> ndarray}`` for multi-symbol curated tasks; a single
      ndarray for legacy single-output cases.
    * ``X_test_dict`` — features the agent sees (we don't strictly need
      them server-side but caching helps when the agent submits an
      empty payload by mistake — we can still reject with a precise
      shape mismatch instead of an opaque KeyError).
    """
    from openfinai_harbor.verifier.curated.task_loader import instantiate_task

    cfg = state.config
    spec = cfg.spec

    task = instantiate_task(
        state.task_cls,
        default_config=spec.default_config,
    )
    # Force load so a broken data source (Binance unreachable, corrupt CSV)
    # fails the spawn rather than the first submission. load_data is idempotent.
    task.load_data()

    # get_features / get_ground_truth return either a single ndarray or
    # {symbol -> ndarray}; dispatch at score-time handles both.
    X_test = task.get_features()
    y_test = task.get_ground_truth()

    handler_state: Dict[str, Any] = {
        "task": task,
        "X_test": X_test,
        "y_test": y_test,
        "is_dict": isinstance(y_test, dict),
    }
    state.handler_state[_HANDLER_KEY] = handler_state

    # Dry-run scoring with zeros to surface actual reward keys for /info.
    try:
        zeros_pred = _zeros_like_predictions(y_test)
        dry_scores = task.predict_and_evaluate(zeros_pred)
        if isinstance(dry_scores, dict) and dry_scores:
            # Preserve task-defined insertion order; alphabetical sort would
            # mis-order the headline aggregate vs per-symbol variants.
            state.reward_names = [
                k for k in dry_scores.keys() if not k.startswith("_")
            ]
    except Exception:
        state.reward_names = list(REWARD_NAMES)

    # Persist (X_train, y_train, X_test) for the agent container; held-out
    # y_test stays in memory + environment/eval-data/ on the host.
    _persist_agent_data(state, task)
    # Persist an auditable verifier-only copy outside the agent data mount.
    _persist_held_out_gt(state, y_test)


def _persist_agent_data(state, task: Any) -> None:  # type: CuratedVerifierState
    """Write agent-visible train targets and test features to dataset.h5.

    Multi-symbol tasks use per-symbol groups; single outputs use root datasets.
    Optional references and task metadata are included when available. A file
    with matching dataset shapes is reused.
    """
    import h5py  # local import: keeps the module-level cost down

    cfg = state.config
    data_dir = cfg.task_dir / "environment" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = data_dir / "dataset.h5"

    X_train = task.get_train_features()
    y_train = task.get_train_ground_truth()
    X_test = task.get_features()

    # Reference arrays are optional — only price-prediction tasks
    # expose them. Detect via ``hasattr`` so subclasses without the
    # accessor keep working.
    ref_train_fn = getattr(task, "get_reference_train", None)
    ref_test_fn = getattr(task, "get_reference_test", None)
    ref_train = ref_train_fn() if callable(ref_train_fn) else None
    ref_test = ref_test_fn() if callable(ref_test_fn) else None
    has_reference = (
        isinstance(ref_train, dict)
        and isinstance(ref_test, dict)
        and len(ref_train) > 0
        and len(ref_test) > 0
    )

    is_dict = (
        isinstance(X_train, dict)
        and isinstance(y_train, dict)
        and isinstance(X_test, dict)
    )

    if is_dict:
        layout = "per_symbol"
        symbols = sorted(set(X_train.keys()) & set(y_train.keys()) & set(X_test.keys()))
        # Flat shape map so idempotent-skip compares without re-opening twice.
        shape_map: Dict[str, tuple] = {}
        for sym in symbols:
            shape_map[f"{sym}/X_train"] = np.asarray(X_train[sym]).shape
            shape_map[f"{sym}/y_train"] = np.asarray(y_train[sym]).shape
            shape_map[f"{sym}/X_test"] = np.asarray(X_test[sym]).shape
            if has_reference and sym in ref_train and sym in ref_test:
                shape_map[f"{sym}/reference_train"] = np.asarray(
                    ref_train[sym]
                ).shape
                shape_map[f"{sym}/reference_test"] = np.asarray(
                    ref_test[sym]
                ).shape
    else:
        layout = "single"
        shape_map = {
            "X_train": np.asarray(X_train).shape,
            "y_train": np.asarray(y_train).shape,
            "X_test": np.asarray(X_test).shape,
        }
        if has_reference:
            # Defensive: has_reference requires dicts, so single-output
            # tasks with references aren't reachable in current code paths.
            pass

    # Optional scalars surfaced as h5 attrs: drives idempotent-skip check
    # and acts as agent-facing hints.
    horizon = getattr(task, "_forecast_horizon_bars", None)
    if horizon is None:
        horizon = getattr(task, "forecast_horizon_bars", None)
    headline = getattr(task, "_headline_metric", None)
    if headline is None:
        headline = getattr(task, "headline_metric", None)

    extra_attrs: Dict[str, Any] = {}
    if isinstance(horizon, int) and horizon >= 1:
        extra_attrs["forecast_horizon_bars"] = int(horizon)
    if isinstance(headline, str) and headline:
        extra_attrs["headline_metric"] = headline

    if h5_path.exists() and _h5_matches(
        h5_path,
        layout=layout,
        shape_map=shape_map,
        attrs=extra_attrs,
    ):
        return

    feature_names = _extract_feature_names(task)
    # Rewriting is cheap relative to a Binance fetch; no in-place patching.
    with h5py.File(h5_path, "w") as f:
        f.attrs["layout"] = layout
        if feature_names:
            f.attrs["feature_names"] = json.dumps(feature_names)
        for k, v in extra_attrs.items():
            f.attrs[k] = v
        if is_dict:
            for sym in symbols:
                grp = f.create_group(sym)
                grp.create_dataset("X_train", data=np.asarray(X_train[sym]))
                grp.create_dataset("y_train", data=np.asarray(y_train[sym]))
                grp.create_dataset("X_test", data=np.asarray(X_test[sym]))
                if has_reference and sym in ref_train and sym in ref_test:
                    grp.create_dataset(
                        "reference_train", data=np.asarray(ref_train[sym])
                    )
                    grp.create_dataset(
                        "reference_test", data=np.asarray(ref_test[sym])
                    )
        else:
            f.create_dataset("X_train", data=np.asarray(X_train))
            f.create_dataset("y_train", data=np.asarray(y_train))
            f.create_dataset("X_test", data=np.asarray(X_test))


def _persist_held_out_gt(state, y_test: Any) -> None:  # type: CuratedVerifierState
    """Write verifier-only targets in the runtime-loader HDF5 format.

    The agent mount excludes environment/eval-data. Panel targets use one
    dataset per symbol; single outputs use ``ground_truth/_value``.
    """
    import h5py

    cfg = state.config
    eval_dir = cfg.task_dir / "environment" / "eval-data"
    eval_dir.mkdir(parents=True, exist_ok=True)
    h5_path = eval_dir / "test_ground_truth.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("ground_truth")
        if isinstance(y_test, dict):
            for sym, arr in y_test.items():
                grp.create_dataset(str(sym), data=np.asarray(arr))
        else:
            grp.create_dataset("_value", data=np.asarray(y_test))


def _h5_matches(
    h5_path: Path,
    *,
    layout: str,
    shape_map: Dict[str, tuple],
    attrs: Optional[Dict[str, Any]] = None,
) -> bool:
    """True iff *h5_path* has the same layout + dataset shapes as *shape_map*.

    Used by ``_persist_agent_data`` to skip rewrites on re-spawn. Any
    structural deviation (missing path, wrong layout attr, mismatched
    shape, mismatched scalar attr) returns False and the caller
    overwrites.

    The optional ``attrs`` mapping is checked against ``f.attrs`` —
    used by the price-prediction overhaul to detect e.g. a
    ``forecast_horizon_bars`` change between spawns (the file is on
    disk but its target column is now misaligned with the configured
    horizon, so we MUST rewrite).
    """
    import h5py

    try:
        with h5py.File(h5_path, "r") as f:
            existing_layout = f.attrs.get("layout", None)
            if existing_layout is None or str(existing_layout) != layout:
                return False
            for path, expected_shape in shape_map.items():
                if path not in f:
                    return False
                ds = f[path]
                if not isinstance(ds, h5py.Dataset):
                    return False
                if tuple(ds.shape) != tuple(expected_shape):
                    return False
            # Reject extras: stale per-symbol groups would silently leak.
            actual_paths: list[str] = []
            f.visit(actual_paths.append)
            actual_dataset_paths = {
                p for p in actual_paths if isinstance(f[p], h5py.Dataset)
            }
            if actual_dataset_paths != set(shape_map.keys()):
                return False
            # Scalar attrs: every requested key must exist with expected value.
            # Extra attrs (e.g. feature_names) don't fail the match.
            if attrs:
                for k, expected in attrs.items():
                    if k not in f.attrs:
                        return False
                    actual = f.attrs[k]
                    # h5py returns numpy scalars; coerce via expected's type
                    # for a safe ==.
                    try:
                        if type(expected)(actual) != expected:
                            return False
                    except (TypeError, ValueError):
                        return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def _extract_feature_names(task: Any) -> Optional[list[str]]:
    """Pull ``FEATURE_NAMES`` off the task module if available.

    Curated forecasting task modules export a module-level
    ``FEATURE_NAMES`` list documenting the column order of ``X_*``
    arrays. We surface it via ``f.attrs["feature_names"]`` so an
    agent reading ``dataset.h5`` can verify column semantics without
    a separate manifest.
    """
    try:
        mod = type(task).__module__
        feature_mod = __import__(mod, fromlist=["FEATURE_NAMES"])
        names = getattr(feature_mod, "FEATURE_NAMES", None)
        if isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names):
            return list(names)
    except (ImportError, AttributeError):
        pass
    return None


def _zeros_like_predictions(y_test: Any) -> Any:
    """Build a dict-or-ndarray of zeros matching ``y_test``'s shape(s).

    Used by on_startup's dry-run probe to discover the reward names.
    """
    if isinstance(y_test, dict):
        return {k: np.zeros_like(np.asarray(v)) for k, v in y_test.items()}
    return np.zeros_like(np.asarray(y_test))


def _decode_predictions(payload_b64: str, fmt: str) -> Any:
    """Decode base64 → ndarray (fmt='ndarray') or dict-of-arrays (fmt='ndarray_dict')."""
    raw = base64.b64decode(payload_b64.encode("ascii"))
    buf = io.BytesIO(raw)
    if fmt == "ndarray":
        return np.load(buf, allow_pickle=False)
    if fmt == "ndarray_dict":
        with np.load(buf, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}
    raise ValueError(
        f"unknown predictions_format={fmt!r}; expected 'ndarray' or 'ndarray_dict'"
    )


def register_routes(app: FastAPI, state, *, bearer) -> None:  # type: CuratedVerifierState
    """Mount ``POST /submit/predictions`` on ``app``.

    Also exposes a back-compat ``POST /submit`` alias so clients that
    were written against the auto-pipe verifier's endpoint still work
    when targeting a curated forecasting bundle (by passing a flat
    ndarray; multi-symbol clients must hit /submit/predictions
    explicitly with predictions_format='ndarray_dict').
    """

    @app.post("/submit/predictions", response_model=CuratedSubmitResponse)
    async def submit_predictions(
        req: CuratedForecastingSubmitRequest,
        request: Request,
        token: str = Depends(bearer),
    ) -> Union[CuratedSubmitResponse, JSONResponse]:
        def _score() -> SubmitOutcome:
            try:
                predictions = _decode_predictions(
                    req.predictions_b64, req.predictions_format
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid predictions payload: {exc}",
                ) from exc

            handler_state = state.handler_state[_HANDLER_KEY]
            task = handler_state["task"]
            scores = task.predict_and_evaluate(predictions)
            flat, weighted_total = coerce_scores(scores, HEADLINE_METRIC)
            return SubmitOutcome(
                status="scored",
                scores=flat,
                weighted_total=weighted_total,
            )

        return await run_submit(
            state=state,
            agent_id=req.agent_id,
            submission_id=req.submission_id,
            token=token,
            score_fn=_score,
            error_detail_prefix="task.predict_and_evaluate raised",
        )

    # Auto-pipe wire-shape alias: ndarray (single-output) only. Dict-shape
    # clients must hit /submit/predictions explicitly.
    @app.post("/submit", response_model=CuratedSubmitResponse)
    async def submit_compat(
        request: Request,
        token: str = Depends(bearer),
    ) -> Union[CuratedSubmitResponse, JSONResponse]:
        # Hand-parse the auto-pipe SubmitRequest body to avoid importing its
        # pydantic class (fake_emb_b64 etc. don't apply here).
        body = await request.json()
        translated = CuratedForecastingSubmitRequest(
            agent_id=body.get("agent_id", ""),
            submission_id=body.get("submission_id", ""),
            predictions_b64=body.get("predictions_b64", ""),
            predictions_format="ndarray",
            extra=body.get("extra") or {},
        )
        return await submit_predictions(translated, request, token=token)
