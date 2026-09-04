from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from data.knowledge_base.rewards import (
    MaxDrawdown,
    PnL,
    SharpeRatio,
    TradingReward,
    WinRate,
)


def _import_bundled_loader(load_path: Path) -> Any:
    """Import a bundled loader under a path-specific module name.

    Dynamic imports stay in framework code because generated task code may
    not use ``importlib``.
    """
    import importlib.util
    import sys

    module_name = f"_dataset_loader_{abs(hash(str(load_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(load_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {load_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_task_data_dir(task_obj: "BaseTask") -> Path:
    """Resolve bundled data from ``config['data_dir']`` or the task module.

    An unresolved directory is a broken task package and raises.
    """
    import sys

    cfg_dir = task_obj.config.get("data_dir") if isinstance(task_obj.config, dict) else None
    if cfg_dir:
        candidate = Path(cfg_dir).resolve()
        if candidate.is_dir():
            return candidate

    cls_module = sys.modules.get(type(task_obj).__module__)
    module_file = getattr(cls_module, "__file__", None) if cls_module else None
    if module_file is None:
        raise RuntimeError(
            f"cannot resolve data directory for {type(task_obj).__name__}: "
            "the subclass module has no __file__ attribute. Pass "
            "config={'data_dir': '/path/to/dir'} when instantiating."
        )
    return Path(module_file).resolve().parent


@dataclass
class TaskMetadata:
    task_id: str
    title: str
    description: str
    task_type: str = "offline"  # "offline" | "realtime"
    interaction_model: str = "gym"  # "forecasting" | "trading" | "gym" (legacy default)
    source_papers: list[str] | None = None
    tags: list[str] | None = None
    data_requirements: list[str] | None = None
    difficulty: str = "medium"
    version: str = "1.0.0"


class BaseTask(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._data: Any = None

    @abstractmethod
    def metadata(self) -> TaskMetadata:
        raise NotImplementedError

    def load_data(self) -> Any:
        """Load and cache data through the bundled ``load.py``.

        Streaming or otherwise bespoke tasks may override this method.
        """
        if self._data is not None:
            return self._data
        data_dir = _resolve_task_data_dir(self)
        load_py = data_dir / "load.py"
        if not load_py.exists():
            raise FileNotFoundError(
                f"bundled load.py not found at {load_py}. Phase 4 must "
                "generate and install a per-task loader for this task "
                "(see openfinai_pipeline.benchmark.loader). Subclasses "
                "with bespoke data flows should override load_data."
            )
        module = _import_bundled_loader(load_py)
        loader_fn = getattr(module, "load", None)
        if not callable(loader_fn):
            raise ImportError(
                f"{load_py} does not expose a callable load(data_dir) "
                "function — regenerate the per-task loader."
            )
        self._data = loader_fn(data_dir)
        return self._data

    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError

    def prepare_eval_context(
        self,
        predictions: Any,
        ground_truth: Any,
    ) -> Dict[str, Any]:
        """Return optional context merged into the evaluator call."""
        return {}


class ForecastingTask(BaseTask):
    """Prediction task supporting batch and streaming execution.

    Batch tasks call :meth:`predict_and_evaluate`; streaming subclasses set
    :attr:`batch_mode` false and use the gym loop. Default training accessors
    read the installed train/test bundle. Held-out test targets remain
    verifier-only: :meth:`get_ground_truth` raises unless a curated task
    explicitly overrides it.
    """

    batch_mode: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._evaluator: Optional["BaseEvaluator"] = None
        self._current_idx: int = 0
        self._done: bool = False
        self._predictions: List[Any] = []

    @abstractmethod
    def get_features(self) -> Any:
        """Return features on which the agent must predict."""
        raise NotImplementedError

    def get_train_features(self) -> Any:
        """Return ``self._data['train']['features']`` by default."""
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict) or "train" not in self._data:
            raise NotImplementedError(
                "default get_train_features expects self._data to be the "
                "B-shape bundle {'train': {...}, 'test': {...}}; this task "
                "should override get_train_features"
            )
        return self._data["train"]["features"]

    def get_train_ground_truth(self) -> Any:
        """Return agent-visible training labels."""
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict) or "train" not in self._data:
            raise NotImplementedError(
                "default get_train_ground_truth expects self._data to be "
                "the B-shape bundle {'train': {...}, 'test': {...}}; this "
                "task should override get_train_ground_truth"
            )
        return self._data["train"]["ground_truth"]

    def get_ground_truth(self) -> Any:
        """Reject agent access to held-out test targets.

        Curated tasks with a separate evaluation path may override this.
        """
        raise PermissionError(
            "ForecastingTask.get_ground_truth() must not be called from "
            "agent-side code: the held-out test target is reserved for "
            "the verifier. Use get_train_ground_truth() to fit a model. "
            "If you are writing a curated task, override this method on "
            "your subclass."
        )

    def predict_and_evaluate(
        self,
        predictions: Any,
        *,
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Score predictions on an allowed split.

        ``train`` uses agent-visible labels, ``test`` is forbidden, and
        ``None`` delegates to the curated-task compatibility accessor.
        Agent calls may not write to Harbor's ``/logs/verifier/`` channel.
        """
        if split == "test":
            raise PermissionError(
                "split='test' is verifier-only; use split='train' for "
                "agent-side self-evaluation"
            )
        reward_output = kwargs.get("reward_output")
        if reward_output is not None:
            ro_str = str(reward_output)
            if "/logs/verifier/" in ro_str.replace("\\", "/"):
                raise PermissionError(
                    "predict_and_evaluate cannot write into /logs/verifier/ "
                    "— that path is reserved for the verifier's canonical "
                    "reward.json. Pick a path under your workspace."
                )

        if split is None:
            ground_truth = self.get_ground_truth()
        elif split == "train":
            ground_truth = self.get_train_ground_truth()
        else:
            raise ValueError(
                f"unknown split {split!r}; valid: 'train' (agent-side) "
                "or omit for legacy curated-task behaviour"
            )

        if self._evaluator is not None:
            extra_ctx = self.prepare_eval_context(predictions, ground_truth) or {}
            merged = {**extra_ctx, **kwargs}  # explicit kwargs win on collision
            return self._evaluator.score(predictions, ground_truth, **merged)
        return self._default_score(predictions, ground_truth)

    def reset(self) -> Any:
        if self._data is None:
            self.load_data()
        self._current_idx = 0
        self._done = False
        self._predictions = []
        return self._get_observation_at(0)

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        """Record one prediction; scoring is deferred until evaluation."""
        if self._done:
            raise RuntimeError("step() called on a finished episode. Call reset() first.")
        self._predictions.append(action)
        self._current_idx += 1
        total = self._get_num_samples()
        self._done = self._current_idx >= total
        obs_idx = min(self._current_idx, max(total - 1, 0))
        obs = self._get_observation_at(obs_idx)
        return obs, 0.0, self._done, {"prediction_idx": self._current_idx - 1}

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Score actions through the configured evaluator or fallback."""
        return self.predict_and_evaluate(agent_actions, **kwargs)

    def _get_num_samples(self) -> int:
        """Number of samples in the dataset. Default: ``len(features)``."""
        try:
            features = self.get_features()
        except Exception:
            return 1
        if features is None:
            return 1
        if hasattr(features, "__len__"):
            try:
                return int(len(features))
            except TypeError:
                pass
        return 1

    def _get_observation_at(self, idx: int) -> Any:
        """Observation at index ``idx``. Default: ``features[idx]``."""
        try:
            features = self.get_features()
        except Exception:
            return None
        if features is None:
            return None
        if hasattr(features, "__getitem__"):
            try:
                return features[idx]
            except (IndexError, KeyError):
                return None
        return features

    def _default_score(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Compute fallback directional accuracy on flattened arrays."""
        if predictions is None or ground_truth is None:
            return {"directional_accuracy": 0.0}
        try:
            import numpy as np

            pred = np.asarray(predictions).astype(float).flatten()
            gt = np.asarray(ground_truth).astype(float).flatten()
        except Exception:
            return {"directional_accuracy": 0.0}
        n = min(len(pred), len(gt))
        if n == 0:
            return {"directional_accuracy": 0.0}
        pred, gt = pred[:n], gt[:n]
        correct = float(((pred > 0) == (gt > 0)).sum())
        return {"directional_accuracy": correct / n}


class GenerativeTask(BaseTask):
    """Sample-generation task scored against reference data.

    Conditional tasks expose training pairs while withholding the test
    reference. Unconditional tasks may expose their sole ``reference``
    distribution. Curated tasks can override :meth:`get_reference_data`.
    """

    batch_mode: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._evaluator: Optional["BaseEvaluator"] = None
        self._generated_samples: list[Any] = []

    def get_reference_data(self) -> Any:
        """Reject access to a conditional task's verifier-only reference."""
        raise PermissionError(
            "GenerativeTask.get_reference_data() must not be called from "
            "agent-side code in the conditional case: the held-out test "
            "reference is reserved for the verifier. Use "
            "get_train_reference_data() to fit a generator. Unconditional "
            "generative tasks override this method on their subclass."
        )

    def get_train_features(self) -> Any:
        """Training conditioning inputs (or training reference for unconditional)."""
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict):
            raise NotImplementedError(
                "default get_train_features expects self._data to be a "
                "B-shape or reference-shape bundle dict"
            )
        if "train" in self._data:
            return self._data["train"]["features"]
        if "reference" in self._data:
            return self._data["reference"]
        raise NotImplementedError(
            "default get_train_features cannot find a 'train' bundle or "
            "'reference' key in self._data"
        )

    def get_train_reference_data(self) -> Any:
        """Training real reference samples for the agent to fit a generator on."""
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict):
            raise NotImplementedError(
                "default get_train_reference_data expects self._data to be a "
                "B-shape or reference-shape bundle dict"
            )
        if "train" in self._data:
            return self._data["train"]["ground_truth"]
        if "reference" in self._data:
            return self._data["reference"]
        raise NotImplementedError(
            "default get_train_reference_data cannot find a 'train' bundle "
            "or 'reference' key in self._data"
        )

    def get_conditioning_data(self) -> Any:
        """Return optional conditioning inputs for conditional generation."""
        return None

    def generate_and_evaluate(
        self,
        generated_samples: Any,
        *,
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Score samples using the forecasting split and output safeguards."""
        if split == "test":
            raise PermissionError(
                "split='test' is verifier-only; use split='train' for "
                "agent-side self-evaluation"
            )
        reward_output = kwargs.get("reward_output")
        if reward_output is not None:
            ro_str = str(reward_output)
            if "/logs/verifier/" in ro_str.replace("\\", "/"):
                raise PermissionError(
                    "generate_and_evaluate cannot write into /logs/verifier/ "
                    "— that path is reserved for the verifier's canonical "
                    "reward.json. Pick a path under your workspace."
                )

        if split is None:
            reference_data = self.get_reference_data()
        elif split == "train":
            reference_data = self.get_train_reference_data()
        else:
            raise ValueError(
                f"unknown split {split!r}; valid: 'train' (agent-side) "
                "or omit for legacy curated-task behaviour"
            )

        if self._evaluator is not None:
            extra_ctx = self.prepare_eval_context(generated_samples, reference_data) or {}
            merged = {**extra_ctx, **kwargs}  # explicit kwargs win on collision
            return self._evaluator.score(generated_samples, reference_data, **merged)
        return self._default_score(generated_samples, reference_data)

    def reset(self) -> Any:
        if self._data is None:
            self.load_data()
        self._generated_samples = []
        conditioning = self.get_conditioning_data()
        if conditioning is not None:
            return conditioning
        try:
            return self.get_train_reference_data()
        except (NotImplementedError, PermissionError):
            return self.get_reference_data()

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        self._generated_samples.append(action)
        return None, 0.0, True, {"generated_batches": len(self._generated_samples)}

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        if len(agent_actions) == 1:
            payload = agent_actions[0]
        else:
            payload = agent_actions
        return self.generate_and_evaluate(payload, **kwargs)

    def _default_score(self, generated_samples: Any, reference_data: Any) -> Dict[str, float]:
        if generated_samples is None or reference_data is None:
            return {"reward": 0.0}
        try:
            import numpy as np

            fake = np.asarray(generated_samples, dtype=float)
            real = np.asarray(reference_data, dtype=float)
        except Exception:
            return {"reward": 0.0}

        if fake.size == 0 or real.size == 0:
            return {"reward": 0.0}

        fake_mean = float(fake.mean())
        real_mean = float(real.mean())
        mean_gap = abs(fake_mean - real_mean)
        return {"mean_gap": mean_gap, "reward": 1.0 / (1.0 + mean_gap)}


@dataclass
class PortfolioState:
    """Portfolio cash, signed positions, history, and pending orders.

    ``reserved_cash`` prevents pending buys from double-spending funds.
    """

    cash: float = 10000.0
    positions: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    pending_orders: Dict[str, Any] = field(default_factory=dict)
    reserved_cash: float = 0.0

    def market_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Return cash plus holdings; absent prices value holdings at zero."""
        prices = prices or {}
        holdings_value = sum(
            prices.get(symbol, 0.0) * qty for symbol, qty in self.positions.items()
        )
        return self.cash + holdings_value


@dataclass
class TradingAction:
    """Structured action used by realtime and transactional trading tasks."""

    action: str  # "buy" | "sell" | "hold"
    symbol: str
    quantity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradingAction":
        """Parse a dict (typically from agent output) into a TradingAction."""
        return cls(
            action=d["action"],
            symbol=d["symbol"],
            quantity=float(d.get("quantity", 0.0)),
            metadata={k: v for k, v in d.items() if k not in ("action", "symbol", "quantity")},
        )

def dispatch_orders_via_executor(
    executor: Any,
    orders: List[Dict[str, Any]],
    quotes: Dict[str, Any],
    *,
    known_symbols: set,
    step: int,
    timestamp: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Tick pending orders, then dispatch this step's orders.

    Quotes may be realtime scalar prices or offline OHLC bars. Results contain
    ``fills``, ``accepted``, ``rejections``, and ``expired`` lists.
    """
    from openfinai_pipeline.realtime.execution import (
        ActionVerb,
        OrderIntent,
        OrderStatus,
        Rejection,
        RejectionCode,
    )

    fills: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    expired: List[Dict[str, Any]] = []

    # Match orders carried from prior steps before new submissions.
    tick_res = executor.tick(quotes, step=step, timestamp=timestamp)
    for f in tick_res.fills:
        fills.append(f.to_dict())
    for e in tick_res.expirations:
        expired.append(e.to_dict())

    for od in orders:
        verb = od.get("action")
        if verb == ActionVerb.CANCEL:
            order_id = od.get("order_id")
            if not isinstance(order_id, str) or not order_id:
                rejections.append(
                    {
                        "order_intent": od,
                        "reason_code": RejectionCode.MISSING_FIELD,
                        "reason": "cancel requires a string order_id",
                    }
                )
                continue
            cres = executor.cancel(order_id)
            if cres.is_cancelled and cres.cancelled is not None:
                expired.append(
                    {**cres.cancelled.to_dict(), "status": OrderStatus.CANCELLED}
                )
            elif cres.rejection is not None:
                rejections.append(cres.rejection.to_dict())
            continue

        symbol = od.get("symbol")
        if (
            verb in (ActionVerb.BUY, ActionVerb.SELL)
            and isinstance(symbol, str)
            and symbol not in known_symbols
        ):
            rejections.append(
                {
                    "order_intent": od,
                    "reason_code": RejectionCode.UNKNOWN_SYMBOL,
                    "reason": (
                        f"symbol {symbol!r} is not in this task's "
                        f"symbols={sorted(known_symbols)}"
                    ),
                }
            )
            continue

        parsed = OrderIntent.from_dict(od)
        if isinstance(parsed, Rejection):
            rejections.append(parsed.to_dict())
            continue

        quote = quotes.get(parsed.symbol)
        if quote is None:
            rejections.append(
                {
                    "order_intent": od,
                    "reason_code": RejectionCode.UNKNOWN_SYMBOL,
                    "reason": f"no quote for symbol {parsed.symbol!r}",
                }
            )
            continue

        sub = executor.submit(
            parsed, market_price=quote, step=step, timestamp=timestamp
        )
        if sub.is_filled and sub.fill is not None:
            fills.append(sub.fill.to_dict())
        elif sub.is_accepted and sub.accepted is not None:
            accepted.append(sub.accepted.to_dict())
        elif sub.is_rejected and sub.rejection is not None:
            rejections.append(sub.rejection.to_dict())

    return {
        "fills": fills,
        "accepted": accepted,
        "rejections": rejections,
        "expired": expired,
    }


def _trade_history_to_pairs(
    history: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Adapt already-signed rewards to ``TradingReward`` input pairs.

    Fixed long/unit predictions preserve the reward sign instead of applying
    the action direction twice.
    """
    predictions: List[Dict[str, Any]] = []
    ground_truths: List[Dict[str, Any]] = []
    for entry in history:
        predictions.append({"direction": "long", "quantity": 1.0})
        ground_truths.append({"actual_return": float(entry["reward"])})
    return predictions, ground_truths


def _compute_trading_metrics_from_history(
    history: List[Dict[str, Any]],
    reward_classes: Sequence[type[TradingReward]],
    *,
    count_label: str = "num_trades",
) -> Dict[str, float]:
    """Compute scalar and reward-bank metrics from aggregate history rows."""
    aggregate_history = [e for e in history if "kind" not in e]
    returns = [float(entry["reward"]) for entry in aggregate_history]
    metrics: Dict[str, float] = {
        "total_return": float(sum(returns)),
        count_label: float(len(returns)),
    }
    if not returns:
        return metrics

    import numpy as np

    arr = np.asarray(returns, dtype=float)
    metrics["mean_return"] = float(arr.mean())
    metrics["std_return"] = float(arr.std())

    predictions, ground_truths = _trade_history_to_pairs(aggregate_history)
    for cls in reward_classes:
        try:
            metric = cls()
            value = float(metric.compute_aggregate(predictions, ground_truths))
            metrics[metric.name] = value
        except Exception:
            metrics[cls.__name__.lower()] = float("nan")
    return metrics


class TradingTask(BaseTask):
    """Sequential trading task with persistent portfolio state.

    Subclasses provide market prices, execution quotes, observations, and
    episode bounds. Evaluation reads accumulated history without replaying it.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._portfolio: PortfolioState = PortfolioState()
        self._step_count: int = 0
        self._done: bool = False
        self._trade_history: List[Dict[str, Any]] = []
        # Per-step rewards are changes from each symbol's PnL watermark.
        self._prev_pnl_per_sym: Dict[str, float] = {}

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        return list(self._trade_history)

    @abstractmethod
    def _current_prices(self) -> Dict[str, float]:
        """Latest scalar price per symbol (drives PnL + portfolio value)."""
        raise NotImplementedError

    @abstractmethod
    def _execution_quotes(self) -> Dict[str, Any]:
        """Return OHLC bars offline or scalar ticks in realtime."""
        raise NotImplementedError

    @abstractmethod
    def _market_observation_block(self) -> Dict[str, Dict[str, Any]]:
        """Return each symbol's agent-visible market data."""
        raise NotImplementedError

    @abstractmethod
    def _episode_done(self) -> bool:
        """Whether this step terminates the episode."""
        raise NotImplementedError

    @abstractmethod
    def _steps_remaining(self) -> int:
        """Steps left in the episode (sentinel ``-1`` if unbounded)."""
        raise NotImplementedError

    def _market_timestamp(self) -> Any:
        """Execution timestamp passed to the executor. Default ``None``."""
        return None

    def close(self) -> None:
        """Release task resources. Default no-op (realtime overrides)."""

    @staticmethod
    def _normalize_action(action: Any) -> List[Dict[str, Any]]:
        """Normalize a structured action, dict, or sequence into order dicts.

        Unsupported values become one malformed order for clean rejection.
        """
        if isinstance(action, TradingAction):
            return [
                {
                    "action": action.action,
                    "symbol": action.symbol,
                    "quantity": action.quantity,
                    **action.metadata,
                }
            ]
        if isinstance(action, dict):
            if "orders" in action and isinstance(action["orders"], list):
                return [o for o in action["orders"] if isinstance(o, dict)]
            return [action]
        if isinstance(action, (list, tuple)):
            return [o for o in action if isinstance(o, dict)]
        return [{"action": "<malformed>", "raw": action}]

    def _execute_trade(self, action: Any) -> tuple[float, Dict[str, Any]]:
        """Dispatch orders and return the change in per-symbol PnL."""
        orders = self._normalize_action(action)
        quotes = self._execution_quotes()
        prices = self._current_prices()
        step_index = self._step_count + 1
        ts = self._market_timestamp()

        disp = dispatch_orders_via_executor(
            self._executor,
            orders,
            quotes,
            known_symbols=set(self._symbols),
            step=step_index,
            timestamp=ts,
        )

        pnl = self._executor.compute_pnl(prices)
        per_symbol_rewards: Dict[str, float] = {}
        for sym in self._symbols:
            cur = float(pnl.get(sym, 0.0))
            per_symbol_rewards[sym] = cur - self._prev_pnl_per_sym.get(sym, 0.0)
            self._prev_pnl_per_sym[sym] = cur
        total_reward = float(sum(per_symbol_rewards.values()))

        return total_reward, {
            "done": self._episode_done(),
            "prices": prices,
            "total_pnl": float(pnl.get("__total", 0.0)),
            "per_symbol_rewards": per_symbol_rewards,
            "fills": disp["fills"],
            "accepted": disp["accepted"],
            "rejections": disp["rejections"],
            "expired": disp["expired"],
        }

    def _get_market_observation(self) -> Dict[str, Any]:
        """Return per-symbol market data plus executor portfolio state."""
        prices = self._current_prices()
        positions = self._executor.get_positions()
        cash = self._executor.get_cash()
        value: Optional[float]
        if cash is None:
            value = None
        else:
            value = float(cash) + sum(
                float(qty) * prices.get(sym, 0.0)
                for sym, qty in positions.items()
            )
        return {
            "step": self._step_count,
            "steps_remaining": self._steps_remaining(),
            "symbols": self._market_observation_block(),
            "portfolio": {
                "cash": cash,
                "reserved_cash": self._executor.get_reserved_cash(),
                "positions": positions,
                "pnl": self._executor.compute_pnl(prices),
                "value": value,
                "pending_orders": [
                    o.to_dict() for o in self._executor.get_pending_orders()
                ],
            },
        }

    def reset(self) -> Any:
        initial_cash = float(self.config.get("initial_cash", 10000.0))
        self._portfolio = PortfolioState(cash=initial_cash)
        self._step_count = 0
        self._done = False
        self._trade_history = []
        self._prev_pnl_per_sym = {}
        if self._data is None:
            self.load_data()
        return self._get_market_observation()

    def _sync_portfolio_from_executor(self) -> None:
        """Mirror executor state when the subclass owns an executor."""
        ex = getattr(self, "_executor", None)
        if ex is None:
            return
        self._portfolio.cash = float(ex.get_cash() or 0.0)
        self._portfolio.positions = dict(ex.get_positions())
        self._portfolio.reserved_cash = float(ex.get_reserved_cash())
        self._portfolio.pending_orders = {
            o.order_id: o for o in ex.get_pending_orders()
        }

    def _on_step_start(self) -> None:
        """Pre-trade hook; realtime tasks use it to refresh market data."""

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode. Call reset() first.")
        self._on_step_start()
        reward, info = self._execute_trade(action)
        info = dict(info)  # copy to avoid mutating caller-supplied dicts
        self._step_count += 1
        self._trade_history.append(
            {
                "step": self._step_count,
                "action": action,
                "reward": float(reward),
                **info,
            }
        )
        # Kind-tagged audit rows are excluded from aggregate metrics.
        for fill in info.get("fills", []):
            self._trade_history.append(
                {"step": self._step_count, "kind": "fill", **fill, "reward": 0.0}
            )
        for entry in info.get("rejections", []):
            self._trade_history.append(
                {
                    "step": self._step_count,
                    "kind": "rejection",
                    **entry,
                    "reward": 0.0,
                }
            )
        for entry in info.get("expired", []):
            self._trade_history.append(
                {
                    "step": self._step_count,
                    "kind": "expiration",
                    **entry,
                    "reward": 0.0,
                }
            )
        self._done = bool(info.get("done", False))
        # Cancel any GTC orders still queued at episode end.
        if self._done:
            executor = getattr(self, "_executor", None)
            if executor is not None:
                prices = (
                    self._current_prices()
                    if hasattr(self, "_current_prices")
                    else {}
                )
                extra_expired = executor.expire_all(prices, step=self._step_count)
                if extra_expired:
                    self._sync_portfolio_from_executor()
                    expired_dicts = [r.to_dict() for r in extra_expired]
                    info.setdefault("expired", []).extend(expired_dicts)
                    for entry in expired_dicts:
                        self._trade_history.append(
                            {
                                "step": self._step_count,
                                "kind": "episode_end_expiration",
                                **entry,
                                "reward": 0.0,
                            }
                        )
        obs = self._get_market_observation() if not self._done else None
        return obs, float(reward), self._done, info

    #: Reward classes used by :meth:`_compute_trading_metrics`.
    DEFAULT_TRADING_REWARDS: tuple[type[TradingReward], ...] = (
        PnL,
        SharpeRatio,
        MaxDrawdown,
        WinRate,
    )

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Score accumulated history; ``total_pnl`` covers all symbols."""
        metrics = self._compute_trading_metrics(self._trade_history)
        metrics["total_pnl"] = float(
            self._executor.compute_pnl(self._current_prices()).get("__total", 0.0)
        )
        return metrics

    def _compute_trading_metrics(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute metrics, restricting rewards to ``target_symbols`` if set."""
        target = getattr(self, "_target_symbols", None)
        symbols = getattr(self, "_symbols", None)
        if not target or not symbols or set(target) == set(symbols):
            return _compute_trading_metrics_from_history(
                history,
                self.DEFAULT_TRADING_REWARDS,
                count_label="num_trades",
            )
        target_set = set(target)
        filtered: List[Dict[str, Any]] = []
        for entry in history:
            per_sym = entry.get("per_symbol_rewards") or {}
            tgt_reward = float(
                sum(v for s, v in per_sym.items() if s in target_set)
            )
            filtered.append({**entry, "reward": tgt_reward})
        return _compute_trading_metrics_from_history(
            filtered,
            self.DEFAULT_TRADING_REWARDS,
            count_label="num_trades",
        )


class BaseEvaluator(ABC):
    REWARD_NAMES: list[str] = []

    @abstractmethod
    def score(
        self,
        predictions: Any,
        ground_truth: Any,
        weights: list[float] | None = None,
        reward_output: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        raise NotImplementedError


class TaskEnvironmentAdapter:
    def __init__(self, task: BaseTask) -> None:
        self._task = task
        self._episode_actions: list[Any] = []

    def reset(self) -> Any:
        self._episode_actions = []
        return self._task.reset()

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        self._episode_actions.append(action)
        return self._task.step(action)

    def get_rewards(self) -> Dict[str, float]:
        return self._task.evaluate(self._episode_actions)

    @property
    def episode_actions(self) -> list[Any]:
        return list(self._episode_actions)


class BaseAgent(ABC):
    """Minimal gym-agent interface with optional lifecycle hooks."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Given an observation, return an action."""
        raise NotImplementedError

    def on_episode_start(self, task_metadata: TaskMetadata) -> None:
        """Called before ``task.reset()``."""

    def on_episode_end(self, rewards: Dict[str, float]) -> None:
        """Called after ``task.evaluate()``."""
