"""Agent-side accessor for the curated offline crypto forecast dataset.

The sibling loader reads the provisioned HDF5 file. Training targets are
available for fitting; held-out targets remain verifier-only and
``get_ground_truth()`` raises ``PermissionError``.
"""
from typing import Any, Dict, List

import numpy as np

from openfinai_pipeline.benchmark.contracts import ForecastingTask, TaskMetadata

#: Column order of the per-symbol ``X_*`` feature matrices (also surfaced
#: as the ``dataset.h5`` attribute ``feature_names``).
FEATURE_NAMES: List[str] = [
    "return_1h",
    "log_return_1h",
    "rolling_mean_5h",
    "rolling_std_5h",
    "rolling_mean_20h",
    "rolling_std_20h",
    "volume_change",
    "high_low_range",
    "close_open_range",
    "momentum_20h",
]


class OfflineCryptoForecasting(ForecastingTask):
    """Read-only per-symbol features and training targets."""

    def _symbols(self) -> List[str]:
        self.load_data()
        return list(self._data.keys())

    def metadata(self) -> TaskMetadata:
        syms = self._symbols()
        return TaskMetadata(
            task_id="offline_crypto_forecasting_" + "_".join(s.lower() for s in syms),
            title="Offline Crypto Forecasting (" + ", ".join(syms) + ")",
            description=(
                "Predict the absolute future close price per symbol from "
                "OHLCV-derived features (Binance hourly bars)."
            ),
            interaction_model="forecasting",
            data_requirements=["Binance hourly OHLCV"],
            tags=["crypto", "forecasting"],
            difficulty="medium",
            version="2.0.0",
        )

    def get_features(self) -> Dict[str, np.ndarray]:
        """Held-out **test** features; predictions align 1:1 per symbol."""
        self.load_data()
        return {s: self._data[s]["X_test"] for s in self._data}

    def get_train_features(self) -> Dict[str, np.ndarray]:
        self.load_data()
        return {s: self._data[s]["X_train"] for s in self._data}

    def get_train_ground_truth(self) -> Dict[str, np.ndarray]:
        self.load_data()
        return {s: self._data[s]["y_train"] for s in self._data}

    def get_reference_train(self) -> Dict[str, np.ndarray]:
        """Current-bar close prices for the train split (per symbol)."""
        self.load_data()
        return {
            s: self._data[s]["reference_train"]
            for s in self._data
            if "reference_train" in self._data[s]
        }

    def get_reference_test(self) -> Dict[str, np.ndarray]:
        """Current-bar close prices for the test split (per symbol).

        Direction at scoring time is ``sign(predicted_price - reference)``.
        """
        self.load_data()
        return {
            s: self._data[s]["reference_test"]
            for s in self._data
            if "reference_test" in self._data[s]
        }

    # get_ground_truth() is inherited from ForecastingTask and raises
    # PermissionError by design — do NOT override it here.

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "per_symbol": {
                "shape": (len(FEATURE_NAMES),),
                "dtype": "float32",
                "features": list(FEATURE_NAMES),
            },
            "symbols": self._symbols(),
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "per_symbol": {
                "type": "continuous",
                "shape": (1,),
                "description": "predicted absolute future close price",
            },
            "symbols": self._symbols(),
        }
