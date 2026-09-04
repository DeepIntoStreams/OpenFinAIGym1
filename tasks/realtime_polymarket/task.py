"""Deferred probability forecasting on binary Polymarket events.

The verifier freezes a discovery-filtered market universe for each trial.
Market-data access uses public Polymarket endpoints.
"""

from typing import Any, Dict, Optional

from openfinai_pipeline.realtime.tasks.realtime_polymarket_task import (
    RealtimePolymarketTask,
)


class RealtimePolymarket(RealtimePolymarketTask):
    """Curated entry point for :class:`RealtimePolymarketTask`.

    ``config`` controls market discovery, the prediction ledger, and the
    headline metric.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
