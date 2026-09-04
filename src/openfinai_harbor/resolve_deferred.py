"""Resolve deferred predictions after their horizons elapse.

The command reads a trial sidecar, scores ready ledger rows, writes a
single-key ``reward.json`` and detailed ``metrics.json``. Exit codes are 0 for
scored rows, 2 when nothing is ready, and 1 for an unrecoverable error.
``--wait`` polls up to ``--max-wait-sec``; the default is one-shot.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_SIDECAR_NAME = "deferred_session.json"
_REWARD_NAME = "reward.json"


def _find_sidecar(trial_dir: Path) -> Optional[Path]:
    """Locate the deferred sidecar inside ``trial_dir``.

    Search order (matches how harbor lays out trial output):

    1. ``trial_dir/verifier/deferred_session.json`` — canonical.
    2. ``trial_dir/deferred_session.json`` — direct (uncommon).
    3. recursive — for unusual harbor layouts (e.g. step-based trials).
    """
    candidates = [
        trial_dir / "verifier" / _SIDECAR_NAME,
        trial_dir / _SIDECAR_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(trial_dir.rglob(_SIDECAR_NAME))
    return matches[0] if matches else None


def _find_reward_json(sidecar_path: Path) -> Path:
    """The reward.json sits next to the sidecar — use the same directory."""
    return sidecar_path.parent / _REWARD_NAME


def _build_provider(name: str) -> Any:
    """Construct a DataProvider by name. Same naming as the realtime CLI.

    We keep this small and explicit (no plugin discovery) so the
    failure mode is "unknown provider" rather than a lazy ImportError.
    """
    if name in ("binance", ""):
        from openfinai_pipeline.realtime.data_providers.binance import (
            BinanceProvider,
        )

        return BinanceProvider()
    if name == "alpaca":
        from openfinai_pipeline.realtime.data_providers.alpaca import (
            AlpacaProvider,
        )

        return AlpacaProvider()
    if name == "polymarket":
        from openfinai_pipeline.realtime.data_providers.polymarket import (
            PolymarketProvider,
        )

        return PolymarketProvider()
    raise RuntimeError(
        f"unknown provider {name!r}; expected 'binance' | 'alpaca' | 'polymarket'"
    )


def _reward_classes_for_provider(name: str) -> Optional[list]:
    """Pick the reward set the resolver should use for *name*.

    Event-resolution providers (Polymarket etc.) use ``ALL_EVENT_REWARDS``
    — Brier / log-loss / ECE on probability vs binary outcome. All
    other providers fall back to the resolver's default
    (``ALL_TRADING_REWARDS``) by returning ``None``.
    """
    if name == "polymarket":
        from data.knowledge_base.rewards.reward_bank import (
            ALL_EVENT_REWARDS,
        )

        return [cls() for cls in ALL_EVENT_REWARDS]
    return None


def _run_resolver_for_session(
    ledger_path: str,
    session_id: str,
    provider_name: str,
    *,
    max_iterations: int = 4,
) -> Dict[str, Any]:
    """Resolve + score predictions for a single session.

    Loops the resolver up to ``max_iterations`` times to drain
    everything in ``status='pending'`` -> ``'resolved'`` -> ``'scored'``.
    Returns the full set of scored predictions for this session.
    """
    from openfinai_pipeline.realtime.ledger import PredictionLedger
    from openfinai_pipeline.realtime.resolver import ResolverService

    ledger = PredictionLedger(ledger_path)
    try:
        provider = _build_provider(provider_name)
        reward_classes = _reward_classes_for_provider(provider_name)
        resolver = ResolverService(
            ledger,
            {provider_name: provider},
            reward_classes=reward_classes,
        )

        for _ in range(max_iterations):
            stats = resolver.run_once()
            logger.info("resolver pass: %s", stats)
            if stats.get("resolved", 0) == 0 and stats.get("scored", 0) == 0:
                break

        rows = ledger.get_session(session_id)
    finally:
        ledger.close()
    return {"rows": rows}


def _aggregate_session_metrics(rows: list[dict]) -> Dict[str, float]:
    """Compute headline metrics from a list of scored predictions.

    Per-prediction ``rewards`` is a JSON-encoded dict (one entry per
    ``TradingReward.name`` or ``EventReward.name``). We average across
    the session.

    Headline rule:
      - **Event-shape sessions** (polymarket et al., where rows have
        ``actual_return`` NULL and a ``ground_truth`` JSON carrying
        ``outcome``) → headline = ``brier_score``.
      - **Trading-shape sessions** (everything else) → headline =
        ``directional_accuracy``.
    """
    if not rows:
        return {
            "reward": 0.0,
            "directional_accuracy": 0.0,
            "n_resolved": 0.0,
            "n_total": 0.0,
        }

    # Scan ALL resolved/scored rows for an event-shape match: actual_return
    # NULL + ground_truth JSON carrying ``outcome``. Breaking on the first
    # qualifying-status row (without gt validity check) would mislabel a
    # mostly-event session as trading-shape on a leading non-event row.
    is_event_session = False
    for row in rows:
        if row.get("status") not in ("resolved", "scored"):
            continue
        gt_raw = row.get("ground_truth")
        if not gt_raw:
            continue
        try:
            gt = json.loads(gt_raw)
        except (ValueError, TypeError):
            continue
        if (
            row.get("actual_return") is None
            and isinstance(gt, dict)
            and gt.get("outcome") is not None
        ):
            is_event_session = True
            break

    n_total = len(rows)
    n_scored = 0
    n_resolved = 0
    n_correct = 0
    return_errors: list[float] = []
    sums: Dict[str, float] = {}
    for row in rows:
        status = row.get("status")
        if status == "resolved":
            n_resolved += 1
        if status == "scored":
            n_resolved += 1
            n_scored += 1
            if not is_event_session:
                actual = row.get("actual_return")
                direction = row.get("direction")
                if actual is not None and direction in ("long", "short"):
                    actual_sign = 1 if actual > 0 else (-1 if actual < 0 else 0)
                    pred_sign = 1 if direction == "long" else -1
                    if actual_sign == pred_sign:
                        n_correct += 1
                    # Derive predicted_return from price (no longer a wire
                    # field). Direction-only rows lack predicted/entry price
                    # and contribute no per-row return error to MAE.
                    pp = row.get("predicted_price")
                    ep = row.get("entry_price")
                    if pp is not None and ep not in (None, 0):
                        predicted_return = (float(pp) - float(ep)) / float(ep)
                        return_errors.append(
                            abs(predicted_return - float(actual))
                        )
            rewards_blob = row.get("rewards")
            if rewards_blob:
                try:
                    parsed = json.loads(rewards_blob)
                except (ValueError, TypeError):
                    parsed = {}
                for k, v in parsed.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        sums[k] = sums.get(k, 0.0) + float(v)

    averaged = {k: v / n_scored for k, v in sums.items()} if n_scored else {}

    metrics: Dict[str, float] = dict(averaged)
    metrics["n_resolved"] = float(n_resolved)
    metrics["n_total"] = float(n_total)

    if is_event_session:
        # Polymarket headline: Brier is lower-is-better; surfaced as-is.
        metrics["reward"] = float(metrics.get("brier_score", 0.0))
    else:
        directional_accuracy = (n_correct / n_scored) if n_scored else 0.0
        return_mae = (
            sum(return_errors) / len(return_errors) if return_errors else 0.0
        )
        metrics["directional_accuracy"] = directional_accuracy
        metrics["return_mae"] = return_mae
        metrics["reward"] = directional_accuracy

    return metrics


def _write_resolved_reward(
    reward_path: Path, metrics: Dict[str, float], *, session_id: str
) -> None:
    """Overwrite ``reward.json`` (single-key) + ``metrics.json``
    (multi-key) with resolved metrics. Numeric-only payload.

    Schema split mirrors the rest of the harbor stack (see
    ``openfinai_harbor.eval.run_evaluation_explicit._write_reward_artifacts``):
    the rich per-channel payload goes to metrics.json, the harbor-safe
    headline goes to reward.json as ``{"reward": <float>}``.
    """
    metrics_payload: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            metrics_payload[k] = v
    # Numeric status flags so harbor consumers can branch without re-parsing.
    metrics_payload["status_resolved"] = 1.0
    metrics_payload["status_deferred"] = 0.0
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = reward_path.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    try:
        reward_value = float(metrics_payload.get("reward", 0.0))
    except (TypeError, ValueError):
        reward_value = 0.0
    reward_path.write_text(
        json.dumps({"reward": reward_value}, indent=2), encoding="utf-8"
    )
    logger.info(
        "wrote resolved reward.json session=%s reward=%.4f n_resolved=%d/%d",
        session_id,
        reward_value,
        int(metrics_payload.get("n_resolved", 0)),
        int(metrics_payload.get("n_total", 0)),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="openfinai_harbor.resolve_deferred",
        description=(
            "Resolve a deferred curated forecasting trial: read the "
            "deferred_session sidecar, run the ledger resolver, "
            "overwrite reward.json with the headline metric."
        ),
    )
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument(
        "--wait",
        action="store_true",
        help=(
            "Poll until at least one prediction resolves. Without "
            "--wait, exit code 2 means 'horizon not yet elapsed'."
        ),
    )
    parser.add_argument(
        "--max-wait-sec",
        type=int,
        default=900,
        help="Max polling time in seconds when --wait is set (default 15min).",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=30,
        help="Polling interval when --wait is set (default 30s).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show resolver-pass logs at INFO level.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    trial_dir = args.trial_dir.resolve()
    if not trial_dir.exists() or not trial_dir.is_dir():
        sys.stderr.write(f"[resolve_deferred] not a directory: {trial_dir}\n")
        return 1

    sidecar_path = _find_sidecar(trial_dir)
    if sidecar_path is None:
        sys.stderr.write(
            f"[resolve_deferred] no {_SIDECAR_NAME} found under {trial_dir} "
            f"— this trial may not be deferred (offline forecasting / "
            f"trading don't write the sidecar)\n"
        )
        return 1

    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"[resolve_deferred] failed to read sidecar: {exc}\n")
        return 1

    session_id = sidecar.get("session_id")
    ledger_path = sidecar.get("ledger_path")
    provider_name = sidecar.get("provider") or "binance"
    resolve_at_last_str = sidecar.get("resolve_at_last")

    if not session_id or not ledger_path:
        sys.stderr.write(
            f"[resolve_deferred] sidecar missing required fields "
            f"session_id/ledger_path: {sidecar}\n"
        )
        return 1

    reward_path = _find_reward_json(sidecar_path)

    print(
        f"[resolve_deferred] trial={trial_dir.name} "
        f"session={session_id} ledger={ledger_path}"
    )
    print(
        f"[resolve_deferred] resolve_at_last={resolve_at_last_str} "
        f"now={datetime.now(timezone.utc).isoformat()}"
    )

    deadline = time.time() + max(args.max_wait_sec, 0) if args.wait else 0
    while True:
        result = _run_resolver_for_session(
            ledger_path, session_id, provider_name
        )
        rows = result["rows"]
        n_scored = sum(1 for r in rows if r.get("status") == "scored")
        n_total = len(rows)

        if n_scored > 0:
            metrics = _aggregate_session_metrics(rows)
            _write_resolved_reward(
                reward_path, metrics, session_id=session_id
            )
            print(
                f"[resolve_deferred] resolved {n_scored}/{n_total} predictions "
                f"directional_accuracy={metrics.get('directional_accuracy', 0.0):.4f} "
                f"reward={metrics.get('reward', 0.0):.4f}"
            )
            return 0

        if not args.wait or time.time() >= deadline:
            statuses: Dict[str, int] = {}
            for r in rows:
                statuses[r.get("status", "?")] = statuses.get(
                    r.get("status", "?"), 0
                ) + 1
            sys.stderr.write(
                f"[resolve_deferred] no scored predictions yet "
                f"(statuses={statuses}); horizon may not have elapsed. "
                f"Re-run later or use --wait.\n"
            )
            return 2

        print(
            f"[resolve_deferred] {n_scored}/{n_total} scored; "
            f"sleeping {args.poll_interval_sec}s"
        )
        time.sleep(args.poll_interval_sec)


if __name__ == "__main__":
    # When invoked as a script, sys.path[0] is the script's directory, not
    # the cwd; insert the repo root so the polymarket branch can import
    # data.knowledge_base.rewards.reward_bank. Inert as a library import.
    _REPO = Path(__file__).resolve().parents[2]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    raise SystemExit(main())
