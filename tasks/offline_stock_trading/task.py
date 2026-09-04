"""Historical equity trading with cached Alpaca bars.

The first download requires ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY``;
cached data can be replayed offline. ``start`` and ``end`` are interpreted in
UTC, so intraday windows should use full ISO timestamps.
"""

from openfinai_pipeline.benchmark.offline_trading_base import _OfflineTradingTask
from openfinai_pipeline.realtime.data_providers.alpaca import AlpacaProvider


class OfflineStockTrading(_OfflineTradingTask):
    """Historical Alpaca trading task; see the base class for its contract."""

    _CACHE_SUBDIR = "alpaca_stock_hourly_ohlcv"
    _DEFAULT_SYMBOLS = ("SPY",)
    _SOURCE_LABEL = "Alpaca"
    _ASSET_TAG = "stock"
    _TASK_ID_PREFIX = "offline_stock_trading"
    _TITLE = "Offline Stock Trading"
    _VERSION = "1.0.0"

    def _make_default_provider(self):
        return AlpacaProvider()
