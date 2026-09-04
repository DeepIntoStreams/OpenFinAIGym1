"""Run curated realtime trading policies inside agent containers.

Bundles provide the data-provider factory and configuration. This module owns
the gym loop, scoring, and reward artifacts; agent ``train.py`` controls only
the action policy::

    def step(observation: dict) -> dict:
        # observation: see RealtimeTradingTask.get_observation_space()
        # return: {"action": "buy"|"sell"|"hold", "symbol": str, "quantity": float}
        ...

"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# reward.json + error.txt artifact helpers (mirrors run_evaluation_curated.py
# numeric-only filter so harbor.VerifierResult.rewards passes pydantic)


def _numeric_only(scores: Dict[str, Any]) -> Dict[str, float]:
    return {
        k: v
        for k, v in scores.items()
        if not isinstance(v, bool) and isinstance(v, (int, float))
    }


def _write_reward_artifacts(
    scores: Dict[str, Any],
    reward_path: Path,
    *,
    error: Optional[str] = None,
) -> None:
    """Write the harbor-schema-safe ``reward.json`` + the rich
    ``metrics.json`` sidecar (+ ``error.txt`` diagnostic when needed).

    Layout written next to *reward_path*::

        reward.json   {"reward": <float>}   (single-key — Mean.compute-safe)
        metrics.json  multi-key numeric payload (every reward channel)

    The richer per-channel dict used to live in reward.json itself but
    harbor's ``Mean.compute`` requires the parsed reward dict to be
    exactly one entry. The rename keeps both schemas: harbor reads
    reward.json (single-key), tooling reads metrics.json.
    """
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = _numeric_only(scores)
    if "reward" not in numeric:
        # Convention: harbor's VerifierResult.rewards picks up `reward` as
        # the headline. Mirror the curated handler's HEADLINE_METRIC =
        # "pnl" choice for trading bundles.
        if "pnl" in numeric:
            numeric["reward"] = numeric["pnl"]
        else:
            numeric["reward"] = 0.0

    # metrics.json — multi-key numeric payload.
    metrics_path = reward_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(numeric, indent=2), encoding="utf-8")

    # reward.json — strict single-key {"reward": <float>}.
    try:
        reward_value = float(numeric.get("reward", 0.0))
    except (TypeError, ValueError):
        reward_value = 0.0
    reward_path.write_text(
        json.dumps({"reward": reward_value}, indent=2), encoding="utf-8"
    )

    stripped = {k: v for k, v in scores.items() if k not in numeric}
    if not error and not stripped:
        return
    lines: list[str] = []
    if error:
        lines.append(error)
    if stripped:
        if lines:
            lines.append("")
        lines.append(
            "Non-numeric fields filtered from metrics.json "
            "(harbor.VerifierResult.rewards rejects non-numeric values):"
        )
        lines.append(json.dumps(stripped, indent=2, default=str))
    error_txt = reward_path.parent / "error.txt"
    error_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Loading the agent's train.py — must define a ``step(obs) -> action``
# callable. Anything else is the agent's business.


def _load_agent_step(train_py: Path) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("agent_train", str(train_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {train_py} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_train"] = module
    spec.loader.exec_module(module)
    step_fn = getattr(module, "step", None)
    if not callable(step_fn):
        raise RuntimeError(
            f"{train_py} must define a top-level callable `step(observation) "
            "-> action_dict`. See instruction.md for the contract."
        )
    return step_fn


def _check_market_staleness(
    obs: Dict[str, Any],
    now: datetime,
    max_age_seconds: float,
) -> Dict[str, float]:
    """Return symbol→age_seconds for symbols whose latest bar is too old.

    Empty dict means every symbol's most recent bar is within the
    threshold (i.e. market is producing new bars right now).

    The check is data-driven, not calendar-based: rather than encoding
    NYSE hours + holidays + DST + half-days, we ask "did the provider
    actually return a fresh bar?". This naturally handles weekends,
    federal holidays, early closes, DST transitions, and Alpaca outages
    with one signal — and it's safe to run on crypto bundles too where
    it should never trip (Binance bars are continuous).

    Symbols whose ``timestamp`` field is missing / None / unparseable
    are skipped — those represent provider-fetch failures, which are
    diagnosed elsewhere by the runner's exception handling.
    """
    stale: Dict[str, float] = {}
    for sym, sym_obs in (obs.get("symbols") or {}).items():
        if not isinstance(sym_obs, dict):
            continue
        ts_str = sym_obs.get("timestamp")
        if not isinstance(ts_str, str) or not ts_str:
            continue
        try:
            bar_ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.replace(tzinfo=timezone.utc)
        age = (now - bar_ts).total_seconds()
        if age > max_age_seconds:
            stale[sym] = age
    return stale


def _provider_staleness_check(
    provider: Any,
    symbols: list[str],
    now: datetime,
    max_age_seconds: float | None,
) -> Dict[str, float]:
    """Pre-flight staleness check that queries the provider directly.

    Runs BEFORE task / executor construction so a closed-market trial
    is aborted before any broker-state mutations (e.g.
    :class:`AlpacaPaperExecutor` dirty-start flatten) are issued.
    Reuses :func:`_check_market_staleness` for the timestamp math by
    building a synthetic observation from per-symbol
    :meth:`DataProvider.get_current_price` calls.

    A symbol whose ``get_current_price`` raises is silently dropped
    from the staleness map — provider-fetch errors will resurface
    later when the buffer tries to backfill, so we don't want to
    short-circuit the trial on a transient blip here.

    Returns ``{}`` when ``max_age_seconds`` is ``None`` (crypto
    bundles disable the check), short-circuiting all provider calls.
    """
    if max_age_seconds is None:
        return {}
    synthetic_obs: Dict[str, Any] = {"symbols": {}}
    for sym in symbols:
        try:
            snap = provider.get_current_price(sym)
        except Exception:
            logger.debug(
                "pre-flight get_current_price failed for %s", sym, exc_info=True
            )
            continue
        ts = getattr(snap, "timestamp", None)
        if ts is None:
            continue
        try:
            ts_str = ts.isoformat()
        except AttributeError:
            ts_str = str(ts)
        synthetic_obs["symbols"][sym] = {"symbol": sym, "timestamp": ts_str}
    return _check_market_staleness(synthetic_obs, now, max_age_seconds)


def _to_jsonable(value: Any) -> Any:
    """Convert MarketSnapshot / OrderBookSnapshot / datetime to plain dict.

    LLM-generated agent code uniformly assumes dict-style access on
    observations (``b.get("close")``, ``b["price"]``, etc.). The
    underlying task class returns dataclass instances, which break those
    natural patterns. We normalise to plain Python primitives + dicts
    before handing the observation to ``step()`` so the LLM doesn't
    have to learn the dataclass attribute API in addition to everything
    else. This is a one-way conversion — actions returned by ``step()``
    are JSON-compatible already (per the contract in instruction.md).

    MarketSnapshot / OrderBookSnapshot conversions partially memoize on
    the instance (via a ``_jsonable_cache`` attribute stashed in the
    dataclass's ``__dict__``). These snapshots live in the bar buffer
    and are re-walked on every observation; their scalar fields are
    immutable once constructed, so caching the ISO-formatted timestamp
    + the scalar skeleton dict is safe. To preserve the original
    deep-copy semantics of ``asdict`` (so agent code mutating the
    returned obs cannot poison the cache for the next step), every
    call returns a fresh shallow copy of the cached skeleton plus
    freshly-built ``extra`` / ``bids`` / ``asks`` containers. The
    output is bit-equivalent to the previous ``_to_jsonable(asdict(...))``
    pipeline.
    """
    from dataclasses import asdict, is_dataclass
    from datetime import datetime as _dt
    from openfinai_pipeline.realtime.data_providers.base import (
        MarketSnapshot,
        OrderBookSnapshot,
    )

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, MarketSnapshot):
        cached = value.__dict__.get("_jsonable_cache")
        if cached is None:
            cached = {
                "symbol": value.symbol,
                "timestamp": value.timestamp.isoformat(),
                "price": value.price,
                "open": value.open,
                "high": value.high,
                "low": value.low,
                "close": value.close,
                "volume": value.volume,
            }
            value.__dict__["_jsonable_cache"] = cached
        result = dict(cached)
        result["extra"] = _to_jsonable(value.extra)
        return result
    if isinstance(value, OrderBookSnapshot):
        cached = value.__dict__.get("_jsonable_cache")
        if cached is None:
            cached = {
                "symbol": value.symbol,
                "timestamp": value.timestamp.isoformat(),
                "best_bid": value.best_bid,
                "best_bid_qty": value.best_bid_qty,
                "best_ask": value.best_ask,
                "best_ask_qty": value.best_ask_qty,
            }
            value.__dict__["_jsonable_cache"] = cached
        result = dict(cached)
        # bids/asks: original ``_to_jsonable(asdict(...))`` path emits
        # list[list[T]] (tuples promoted to lists, inner numerics passed
        # through unchanged). Replicate exactly — no numeric coercion,
        # so int inputs stay int.
        result["bids"] = [[p, q] for p, q in value.bids]
        result["asks"] = [[p, q] for p, q in value.asks]
        result["extra"] = _to_jsonable(value.extra)
        return result
    if isinstance(value, _dt):
        return value.isoformat()
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    # Fall through: anything else gets stringified so the agent at least
    # sees something instead of a TypeError when it tries to serialise.
    return str(value)


# Public entry point


def run_realtime_trading_trial(
    *,
    provider_factory: Callable[[], Any],
    config: Dict[str, Any],
    reward_output_path: str | os.PathLike[str] = "/logs/verifier/reward.json",
    train_py: str | os.PathLike[str] = "/workspace/train.py",
    max_bar_age_seconds: Optional[float] = None,
) -> None:
    """Run a curated realtime trading task from an agent ``step(obs)`` script.

    ``max_bar_age_seconds`` rejects stale markets before constructing the
    executor, which avoids account-reset side effects. Expected failures write
    a schema-valid zero or partial reward plus ``error.txt``.
    """
    # Keep host-side imports independent of the realtime dependency tree.
    from openfinai_pipeline.realtime.tasks.realtime_trading_task import (
        RealtimeTradingTask,
    )
    from openfinai_pipeline.settings import TradingConfig

    reward_path = Path(reward_output_path)
    train_path = Path(train_py)

    if not train_path.is_file():
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error=f"agent train.py not found at {train_path}",
        )
        sys.stderr.write(f"[agent-runtime] no train.py at {train_path}\n")
        return

    print(f"[agent-runtime] loading agent step() from {train_path}")
    try:
        step_fn = _load_agent_step(train_path)
    except Exception as exc:
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error=f"failed to import train.py: {exc}\n{traceback.format_exc()}",
        )
        return

    # Build the trading config — execution_mode comes from the bundle
    # config, defaulting to TradingConfig's default ("internal_paper" after
    # the rename).
    trading_config = TradingConfig(
        slippage_pct=float(config.get("slippage_pct", 0.001)),
        transaction_cost_pct=float(config.get("transaction_cost_pct", 0.0)),
        execution_mode=str(
            config.get("execution_mode", TradingConfig().execution_mode)
        ),
    )

    # Build provider. Provider failures (no creds, unreachable
    # endpoint) surface here as a clear error in reward.json.
    try:
        provider = provider_factory()
    except Exception as exc:
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error=f"provider construction failed: {exc}",
        )
        return

    # Pre-flight market-staleness guard. Runs BEFORE task construction
    # so a closed-market trial can't trigger an Alpaca dirty-start
    # flatten (which would issue DELETE /v2/positions outside trading
    # hours and queue liquidations that wouldn't actually settle until
    # the next session). Queries provider.get_current_price directly;
    # bypassed for crypto bundles by max_bar_age_seconds=None.
    if max_bar_age_seconds is not None:
        symbols_for_check = list(config.get("symbols") or [])
        if symbols_for_check:
            now = datetime.now(timezone.utc)
            stale = _provider_staleness_check(
                provider, symbols_for_check, now, max_bar_age_seconds
            )
            if stale:
                worst_age = max(stale.values())
                stale_lines = "\n".join(
                    f"  - {sym}: {age:.0f}s old (~{age / 3600:.1f}h)"
                    for sym, age in sorted(stale.items())
                )
                warning_msg = (
                    f"market appears closed or feed stale — at least one "
                    f"symbol's latest bar is older than the "
                    f"{max_bar_age_seconds:.0f}s threshold. Per-symbol "
                    f"staleness:\n{stale_lines}\n\n"
                    "Trial aborted with reward=0.0; reward.json carries "
                    "market_closed=1.0 + max_observed_staleness_seconds for "
                    "downstream tooling. For US equities, the next IEX "
                    "session opens Mon–Fri at 13:30 UTC (09:30 ET); "
                    "regular hours close at 20:00 UTC (16:00 ET)."
                )
                sys.stderr.write(
                    f"[agent-runtime] WARNING: {warning_msg.splitlines()[0]} "
                    f"Worst staleness {worst_age:.0f}s. Skipping gym loop, "
                    f"writing reward.json with reward=0.0.\n"
                )
                _write_reward_artifacts(
                    {
                        "reward": 0.0,
                        "market_closed": 1.0,
                        "max_observed_staleness_seconds": float(worst_age),
                        "steps_executed": 0.0,
                        "execution_mode_internal_paper": (
                            1.0
                            if trading_config.execution_mode == "internal_paper"
                            else 0.0
                        ),
                    },
                    reward_path,
                    error=warning_msg,
                )
                return

    # RealtimeTradingTask uses the unified context_resolutions +
    # data_resolution API (single buffer, sidecar resolutions
    # downsampled). The task.toml of every realtime trading bundle is
    # already authored against this shape, so we forward as-is.
    try:
        task = RealtimeTradingTask(
            config=config,
            provider=provider,
            symbols=list(config["symbols"]),
            trading_config=trading_config,
            context_resolutions=config.get("context_resolutions"),
            data_resolution=config.get("data_resolution"),
            max_steps=int(config.get("max_steps", 10)),
            target_symbols=config.get("target_symbols"),
        )
    except Exception as exc:
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error=f"RealtimeTradingTask construction failed: {exc}",
        )
        return

    print(
        f"[agent-runtime] task ready: provider={provider.name} "
        f"symbols={config['symbols']} max_steps={config.get('max_steps', 10)} "
        f"execution_mode={trading_config.execution_mode}"
    )

    # Drive the gym loop. Per-step exceptions are caught so a malformed
    # action only forfeits the trailing steps, not the whole trial.
    actions: list[Any] = []
    failure_index: Optional[int] = None
    failure_msg: Optional[str] = None

    try:
        obs = task.reset()
    except Exception as exc:
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error=f"task.reset() failed: {exc}\n{traceback.format_exc()}",
        )
        return

    started = time.time()
    for i in range(int(config.get("max_steps", 10))):
        if task._done:
            break
        try:
            action = step_fn(_to_jsonable(obs))
        except Exception as exc:
            failure_index = i
            failure_msg = f"agent step() raised: {exc}"
            sys.stderr.write(f"[agent-runtime] {failure_msg}\n")
            break
        try:
            obs, reward, done, info = task.step(action)
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            # Genuine errors only (malformed step, bad data). Trading-business
            # rejections (insufficient buying power, invalid quantity) never
            # raise — they land in info["rejections"]. Surface a partial-replay
            # failure index without killing the trial.
            failure_index = i
            failure_msg = (
                f"task.step(action[{i}]={action!r}) raised: {exc}"
            )
            sys.stderr.write(f"[agent-runtime] {failure_msg}\n")
            break
        actions.append(action)
        if done:
            break

    elapsed = time.time() - started
    print(
        f"[agent-runtime] gym loop done in {elapsed:.1f}s "
        f"(steps_completed={len(actions)})"
    )

    # Aggregate metrics via the curated reward bank. task.evaluate()
    # filters to target_symbols and dispatches through
    # _compute_trading_metrics_from_history.
    try:
        scores = task.evaluate(actions)
    except Exception as exc:
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error=(
                f"task.evaluate() failed: {exc}\n{traceback.format_exc()}"
            ),
        )
        return

    if not isinstance(scores, dict):
        scores = {"reward": float(scores)}
    scores = dict(scores)
    scores["steps_executed"] = float(len(actions))
    scores["execution_mode_internal_paper"] = (
        1.0 if trading_config.execution_mode == "internal_paper" else 0.0
    )
    if failure_index is not None:
        scores["replay_failure_at"] = float(failure_index)
        scores["_replay_failure_msg"] = failure_msg or ""

    _write_reward_artifacts(scores, reward_path)
    print(
        f"[agent-runtime] wrote reward.json -> {reward_path} "
        f"(headline={scores.get('pnl', scores.get('reward'))!r})"
    )
