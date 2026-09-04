"""Pydantic request/response models for the curated verifier RPC service.

Curated submissions don't all fit the auto-pipe ``predictions_b64 ->
scores`` shape: trading is an action stream, realtime forecasting is
deferred via a ledger. We keep one common ``CuratedSubmitResponse`` so
clients have a uniform return contract across families, and use family-
specific request models for the body.

Forecasting submissions reuse the ``predictions_b64`` envelope from the
auto-pipe verifier but extend it with a ``predictions_format`` field
(``"ndarray"`` for ``np.save`` bytes, ``"ndarray_dict"`` for ``np.savez``
bytes). Curated forecasting tasks like :class:`OfflineCryptoForecasting`
emit ``{symbol -> ndarray}`` via :meth:`predict_and_evaluate` and the
auto-pipe ``np.save`` envelope cannot represent that shape.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# Common response (shared across all families)


class CuratedSubmitResponse(BaseModel):
    """Uniform response shape across all curated families.

    ``status`` is ``"scored"`` for synchronous families (forecasting,
    offline trading, realtime trading) and ``"deferred"`` for realtime
    forecasting (ground truth resolves later via the resolver service).

    ``scores`` is empty for ``"deferred"`` submissions; use ``deferred``
    metadata to look up final scores after the resolver runs.
    """

    submission_id: str
    status: Literal["scored", "deferred"]
    scores: Dict[str, float] = Field(default_factory=dict)
    weighted_total: float = 0.0
    elapsed_sec: float = 0.0
    cached: bool = Field(
        default=False,
        description="True if this submission_id was served from the LRU cache.",
    )
    deferred: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Set when status='deferred'. Carries session_id, "
            "n_predictions, expected_resolve_at, ledger_path so the "
            "host-side caller can write the trial sidecar."
        ),
    )


# Forecasting (offline + realtime, but realtime uses a different endpoint)


class CuratedForecastingSubmitRequest(BaseModel):
    """``POST /submit/predictions`` body.

    ``predictions_format`` distinguishes ``np.save`` (single array) from
    ``np.savez`` (dict of arrays). Curated multi-symbol forecasting
    tasks return a ``{symbol -> ndarray}`` dict, which only fits npz.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    submission_id: str = Field(..., min_length=1, max_length=128)
    predictions_b64: str = Field(..., min_length=1)
    predictions_format: Literal["ndarray", "ndarray_dict"] = Field(
        default="ndarray",
        description=(
            "How predictions_b64 was encoded. "
            "'ndarray' = np.save bytes; "
            "'ndarray_dict' = np.savez bytes (one array per symbol)."
        ),
    )
    extra: Dict[str, Any] = Field(default_factory=dict)


# Interactive offline trading
# The verifier advances only after an action, keeping future bars hidden.
# Realtime trading uses an in-container loop instead of these endpoints.


class CuratedSessionResetRequest(BaseModel):
    """``POST /session/reset`` — open an interactive offline-trading episode.

    The verifier builds a fresh task, resets it, and returns the first
    causal observation. ``session_id`` is optional; the verifier mints one
    if absent and echoes it back so subsequent ``/session/step`` calls
    address the same episode.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    session_id: Optional[str] = Field(default=None, max_length=128)
    extra: Dict[str, Any] = Field(default_factory=dict)


class CuratedSessionStepRequest(BaseModel):
    """``POST /session/step`` — submit one action, advance one bar.

    Returns the next causal observation + the realized reward. When the
    episode ends the response also carries the server-computed ``scores``
    (scoring stays verifier-side, so it can't be tampered with).
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    action: Any = Field(
        ...,
        description=(
            "The JSON-serialisable action for this step — the same shape "
            "the task's step() expects (a transactional order intent, e.g. "
            '{"action": "buy", "symbol": "BTCUSDT", "quantity": 0.1}, or '
            '{"orders": [...]}).'
        ),
    )


class CuratedSessionResetResponse(BaseModel):
    session_id: str
    observation: Optional[Dict[str, Any]] = None
    step: int = 0
    done: bool = False


class CuratedSessionStepResponse(BaseModel):
    session_id: str
    observation: Optional[Dict[str, Any]] = None  # None on the terminal step
    reward: float = 0.0
    done: bool = False
    step: int = 0
    info: Dict[str, Any] = Field(default_factory=dict)
    #: Populated only on the terminal step (done=True); server-authoritative.
    scores: Optional[Dict[str, float]] = None
    weighted_total: Optional[float] = None


# Realtime forecasting (deferred via ledger)


class CuratedDeferredPrediction(BaseModel):
    """One prediction record in a deferred-forecasting batch submission.

    The canonical agent-supplied signal is ``predicted_price`` — the
    absolute future price the agent expects ``horizon_bars`` ahead (the
    close of the bar that many bars after submission, at the task's
    ``data_resolution`` granularity).
    Direction is *derived* server-side from
    ``sign(predicted_price - entry_price)`` at scoring time, which means
    a single price prediction feeds both the price-error metric panel
    (mse / rmse / mae / mape / r2 / pearson) and the directional
    metric. Agents that prefer to submit a direction-only long/short
    call still can — those submissions score on directional metrics
    only and NaN-out of price metrics.

    Note: ``entry_price`` is intentionally NOT a field — the verifier
    always captures it server-side via ``provider.get_current_price()``
    at ``ledger.submit()`` time. Allowing the agent to supply it
    previously enabled a fake-price exploit (submit ``direction="long"``
    with ``entry_price=0.01`` to guarantee a positive return after
    resolution). Pydantic's default behavior strips unknown fields, so
    an agent that smuggles entry_price in via extra is silently ignored.
    The same fake-price logic does NOT apply to ``predicted_price``: it
    is the agent's *forward-looking* prediction and is compared against
    the server-captured exit price, so an agent submitting an absurd
    value just gets a bad score, not free PnL.
    """

    symbol: str = Field(..., min_length=1, max_length=64)
    predicted_price: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Absolute predicted future price. Direction is derived "
            "server-side from sign(predicted_price - entry_price); if "
            "the agent also supplies an explicit `direction`, it must "
            "agree with the derived value or the submission is rejected "
            "with 422."
        ),
    )
    direction: Optional[Literal["long", "short"]] = Field(
        default=None,
        description=(
            "Optional. Either supply this OR `predicted_price` "
            "(or both — they must agree). Direction-only submissions "
            "score on directional metrics only; price metrics report "
            "NaN."
        ),
    )
    submitted_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp; defaults to verifier-side now.",
    )

    @model_validator(mode="after")
    def _at_least_one_signal(self) -> "CuratedDeferredPrediction":
        """At least one of price / direction must be supplied.

        ``entry_price`` is captured server-side, so the price↔direction
        consistency check (when both are supplied) cannot run here — it
        moves into the curated handler.
        """
        if self.predicted_price is None and self.direction is None:
            raise ValueError(
                "must supply at least one of predicted_price | direction"
            )
        return self


class CuratedDeferredSubmitRequest(BaseModel):
    """``POST /submit/predictions_async`` body for realtime forecasting.

    Each prediction is appended to the curated task's prediction
    ledger. The submission returns synchronously with
    ``status="deferred"`` and metadata pointing at the ledger; ground
    truth is filled in later by the resolver service (or by the
    ``resolve_deferred`` CLI if running outside the realtime CLI loop).

    ``trial_dir`` is an optional back-pointer: harbor sets
    ``HARBOR_TRIAL_DIR`` on the agent container at trial spawn, and the
    in-container submit helper forwards it here so the handler can
    stamp it onto each ledger row. Inspection / cleanup tools then use
    that pointer to detect orphaned ledgers (trial dir deleted) without
    scanning the ``examples/`` tree. Absent when running outside harbor.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    submission_id: str = Field(..., min_length=1, max_length=128)
    predictions: list[CuratedDeferredPrediction] = Field(..., min_length=1)
    trial_dir: Optional[str] = Field(default=None, max_length=1024)
    extra: Dict[str, Any] = Field(default_factory=dict)


# Curated /info shape — mirrors auto-pipe but adds family + reward names


class CuratedEventPrediction(BaseModel):
    """One probability prediction for a binary-event market.

    ``symbol`` is the provider's market identifier — for Polymarket
    this is the ``conditionId`` (a 0x-prefixed hex hash). ``entry_price``
    is captured server-side; the agent never supplies it. Direction is
    implicitly "long YES" since the probability is the YES-side
    probability — the opposite is just ``1 - p``.
    """

    symbol: str = Field(..., min_length=1, max_length=128)
    predicted_yes_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the market resolves YES, in [0, 1].",
    )
    submitted_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp; defaults to verifier-side now.",
    )


class CuratedEventSubmitRequest(BaseModel):
    """``POST /submit/event_predictions_async`` body for realtime polymarket.

    Same deferred-ledger pattern as realtime forecasting but with a
    binary-outcome contract: each market resolves to YES (1.0) / NO
    (0.0) at its own absolute resolution timestamp (carried in the
    discovered universe, NOT supplied per submission). Resolver fetches
    outcomes via ``EventDataProvider.get_event_outcome`` and computes
    ``EventReward`` metrics (Brier / log-loss / ECE).
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    submission_id: str = Field(..., min_length=1, max_length=128)
    predictions: list[CuratedEventPrediction] = Field(..., min_length=1)
    trial_dir: Optional[str] = Field(default=None, max_length=1024)
    extra: Dict[str, Any] = Field(default_factory=dict)


class CuratedInfoResponse(BaseModel):
    """Returned by ``GET /info`` so clients can format submissions correctly."""

    task_id: str
    family: Literal[
        "offline_forecasting",
        "offline_trading",
        "realtime_forecasting",
        "realtime_trading",
        "realtime_polymarket",
    ]
    reward_names: list[str] = Field(default_factory=list)
    n_active_agents: int = 0
    started_at_utc: str
    despawn_grace_min: float
