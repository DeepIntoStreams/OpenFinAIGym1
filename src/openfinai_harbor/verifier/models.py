"""Shared Pydantic request/response models for the verifier RPC service.

These models are used by both the FastAPI server and the host-side Python
client. The single-file in-container helper (``submit.py``) does NOT
import them — it builds its JSON payloads by hand to keep its dependency
surface at stdlib + ``requests`` only.

Prediction arrays cross the wire as base64-encoded numpy bytes (the bytes
produced by ``numpy.save(BytesIO(), arr)``). This format preserves
shape/dtype without requiring a side-channel and round-trips losslessly
for any ``np.ndarray``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# Register / heartbeat / deregister


class RegisterRequest(BaseModel):
    """Agent declares it is running for this task.

    ``agent_id`` is client-generated (UUID4 recommended). Re-registering
    the same ``agent_id`` is idempotent and refreshes the heartbeat clock.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    notes: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Free-form note recorded in the audit log; not interpreted.",
    )


class RegisterResponse(BaseModel):
    agent_id: str
    task_id: str
    registered_at_utc: str
    n_active_agents: int


class HeartbeatRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=128)


class HeartbeatResponse(BaseModel):
    agent_id: str
    last_heartbeat_utc: str
    n_active_agents: int


# Submit


class SubmitRequest(BaseModel):
    """Agent submits predictions for scoring.

    ``predictions_b64`` is the base64 encoding of bytes produced by
    ``numpy.save(BytesIO(), predictions)`` — preserves shape/dtype exactly.

    ``submission_id`` is client-generated (UUID4 recommended). The verifier
    caches a bounded LRU of ``submission_id`` -> response pairs so a network
    blip retry returns the same scores rather than re-running the metric.

    ``fake_emb_b64`` is for tasks with embedding-pair generative metrics
    (FID-style); ignored for forecasting / non-embedding-pair tasks.

    ``extra`` is forwarded to ``evaluator.score(**kwargs)`` for
    forward-compatibility — additional per-task hooks (e.g. custom
    weighting) can be added without changing the wire format.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    submission_id: str = Field(..., min_length=1, max_length=128)
    predictions_b64: str = Field(..., min_length=1)
    fake_emb_b64: Optional[str] = Field(default=None)
    extra: Dict[str, Any] = Field(default_factory=dict)


class SubmitResponse(BaseModel):
    submission_id: str
    scores: Dict[str, float]
    weighted_total: float
    elapsed_sec: float
    cached: bool = Field(
        default=False,
        description="True if this submission_id was served from the LRU cache.",
    )


# Info (debug / client introspection)


class InfoResponse(BaseModel):
    """Returned by GET /info — helps clients format submissions correctly."""

    task_id: str
    interaction_model: str
    reward_names: list[str]
    has_held_out_test_gt: bool
    expects_fake_emb: bool = Field(
        description="True if any reward is an embedding-pair metric requiring fake_emb_b64.",
    )
    n_active_agents: int
    started_at_utc: str
    despawn_grace_min: float
