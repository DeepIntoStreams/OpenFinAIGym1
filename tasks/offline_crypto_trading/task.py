"""Curated task: Offline Crypto Trading (Binance hourly replay).

Multi-symbol sequential trading on historical Binance bars. Data is
fetched on first use via ``BinanceProvider.get_bars()`` and cached as CSV
under ``data/pipeline_output/datasets/binance_crypto_hourly_ohlcv/``.

All replay machinery lives in
:class:`~openfinai_pipeline.benchmark.offline_trading_base._OfflineTradingTask`
(shared with :class:`OfflineStockTrading`); this class only pins the
Binance provider, cache directory, default symbol, and metadata.

Interaction pattern (gym loop)::

    obs = task.reset()
    while not done:
        action = agent.act(obs)   # {"action": "buy", "symbol": ..., "quantity": ...}
        obs, reward, done, info = task.step(action)
    rewards = task.evaluate(actions)
"""

from openfinai_pipeline.benchmark.offline_trading_base import _OfflineTradingTask
from openfinai_pipeline.realtime.data_providers.binance import BinanceProvider


class OfflineCryptoTrading(_OfflineTradingTask):
    """Multi-symbol historical Binance trading task. See base class for the
    observation/action contract and config keys (``provider`` kwarg
    overrides the default :class:`BinanceProvider`)."""

    _CACHE_SUBDIR = "binance_crypto_hourly_ohlcv"
    _DEFAULT_SYMBOLS = ("BTCUSDT",)
    _SOURCE_LABEL = "Binance"
    _ASSET_TAG = "crypto"
    _TASK_ID_PREFIX = "offline_crypto_trading"
    _TITLE = "Offline Crypto Trading"
    _VERSION = "1.1.0"

    def _make_default_provider(self):
        return BinanceProvider()
