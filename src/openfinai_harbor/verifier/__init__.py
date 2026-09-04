"""Per-task verifier RPC service for OpenFinGym tasks.

Holds a task's held-out test ground truth in memory and serves prediction
submissions over HTTP. Replaces the legacy file-IO mount-split protocol
where the agent's container writes ``predictions.npy`` to disk and the
evaluator reads it from a sibling mount.

Submodules:

- :mod:`models`   - shared Pydantic request/response models.
- :mod:`auth`     - bearer token generation/validation + sliding-window rate limiter.
- :mod:`registry` - per-task state directory layout helpers.
- :mod:`server`   - FastAPI application (run via ``python -m openfinai_harbor.verifier``).
- :mod:`spawn`    - lazy-spawn coordinator (filelock + DCL + wait-for-ready).
- :mod:`client`   - host-side Python client used by ``openfinai_harbor.run_trial``.
- :mod:`submit`   - single-file in-container helper bundled into agent task images.
"""

from openfinai_harbor.verifier.models import (
    HeartbeatRequest,
    HeartbeatResponse,
    InfoResponse,
    RegisterRequest,
    RegisterResponse,
    SubmitRequest,
    SubmitResponse,
)

__all__ = [
    "HeartbeatRequest",
    "HeartbeatResponse",
    "InfoResponse",
    "RegisterRequest",
    "RegisterResponse",
    "SubmitRequest",
    "SubmitResponse",
]
