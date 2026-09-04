"""Family handler dispatch for the curated verifier.

Each handler module exposes:

* ``REWARD_NAMES: list[str]`` — surfaced via ``GET /info``.
* ``on_startup(state)`` (optional) — eager init at app construction time
  (e.g. extract held-out gt, load cached price arrays).
* ``register_routes(app, state, *, bearer)`` — mount the family-specific
  submit endpoint(s) on the FastAPI app.

Registry lookup is by family name (matches ``task.toml [metadata]
.family``). Adding a new handler is two lines: import + entry.
"""

from __future__ import annotations

from typing import Any, Dict


# Lazy-imported handler modules. Each entry is set to the module object
# the first time `handler_for` is called for that family — keeping the
# top of this module import-light so the test suite doesn't pay the
# import cost of every handler when only one is exercised.
FAMILY_HANDLERS: Dict[str, Any] = {
    "offline_forecasting": None,
    "offline_trading": None,
    "realtime_forecasting": None,
    "realtime_polymarket": None,
    # Realtime trading scores inside the agent container and has no host handler.
}


def handler_for(family: str) -> Any:
    """Return the module that handles ``family``, importing on first use.

    Returns ``None`` for families that don't have a handler. The server
    raises a clear error in that case.
    """
    if family not in FAMILY_HANDLERS:
        return None

    cached = FAMILY_HANDLERS[family]
    if cached is not None:
        return cached

    if family == "offline_forecasting":
        from openfinai_harbor.verifier.curated.handlers import forecasting as mod
    elif family == "offline_trading":
        from openfinai_harbor.verifier.curated.handlers import trading as mod
    elif family == "realtime_forecasting":
        from openfinai_harbor.verifier.curated.handlers import (
            realtime_forecasting as mod,
        )
    elif family == "realtime_polymarket":
        from openfinai_harbor.verifier.curated.handlers import (
            realtime_polymarket as mod,
        )
    else:  # pragma: no cover — guarded by the membership check above
        return None

    FAMILY_HANDLERS[family] = mod
    return mod


__all__ = ["FAMILY_HANDLERS", "handler_for"]
