"""Verifier service that owns and dispatches hand-written curated tasks.

The spawn coordinator selects this service from task metadata; family handlers
provide request models and scoring routes.
"""

from openfinai_harbor.verifier.curated.server import (
    CuratedVerifierConfig,
    create_curated_app,
)


__all__ = [
    "CuratedVerifierConfig",
    "create_curated_app",
]
