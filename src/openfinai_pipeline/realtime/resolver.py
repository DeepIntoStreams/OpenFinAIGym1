"""ResolverService — poll data providers for ground truth, then score.

Two-phase lifecycle for each pending prediction:

1. **Resolve**: once ``resolve_at`` has passed, fetch the actual market
   price at that time from the data provider and compute the realised
   return.  Store ``exit_price`` / ``actual_return`` in the ledger.

2. **Score**: for each resolved-but-unscored prediction, compute all
   configured :class:`TradingReward` metrics and store the result.
"""

import itertools
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from openfinai_pipeline.settings import ResolverConfig
from openfinai_pipeline.realtime.data_providers.base import (
    DataProvider,
    EventDataProvider,
    interval_to_seconds,
)
from openfinai_pipeline.realtime.ledger import PredictionLedger

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ground_truth(raw: Any) -> dict[str, Any]:
    """Coerce the ledger's ``ground_truth`` TEXT column to a dict.

    The column stores ``json.dumps(...)`` so it can be either ``None``,
    an empty string, or a JSON-encoded dict. Returns ``{}`` for
    anything that doesn't decode to a dict.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


class ResolverService:
    """Periodically resolves and scores pending predictions."""

    def __init__(
        self,
        ledger: PredictionLedger,
        providers: dict[str, DataProvider],
        config: ResolverConfig | None = None,
        reward_classes: list[Any] | None = None,
    ) -> None:
        self.ledger = ledger
        self.providers = providers
        self.config = config or ResolverConfig()
        # Import here to avoid circular deps; default to all trading rewards.
        if reward_classes is None:
            from data.knowledge_base.rewards.reward_bank import ALL_TRADING_REWARDS

            self._reward_classes = [cls() for cls in ALL_TRADING_REWARDS]
        else:
            self._reward_classes = reward_classes

    # Phase 1: Resolve (fill in ground truth)

    def resolve_pending(self) -> dict[str, int]:
        """One pass: resolve all predictions past their ``resolve_at``.

        Returns counts ``{"resolved": N, "failed": N}``.
        """
        grace = timedelta(seconds=self.config.grace_period_sec)
        cutoff = _utcnow() - grace
        pending = self.ledger.get_pending(before=cutoff, limit=self.config.batch_size)

        if not pending:
            return {"resolved": 0, "failed": 0}

        # Group by (provider, symbol) so we can batch API calls.
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for pred in pending:
            groups[(pred["provider"], pred["symbol"])].append(pred)

        resolved = failed = 0
        for (provider_name, symbol), preds in groups.items():
            provider = self.providers.get(provider_name)
            if provider is None:
                for pred in preds:
                    self.ledger.mark_failed(
                        pred["id"], f"Unknown provider: {provider_name}"
                    )
                    failed += 1
                continue

            is_event_provider = isinstance(provider, EventDataProvider)

            for pred in preds:
                try:
                    if is_event_provider:
                        # Event-resolution flow: outcome is binary (or
                        # 0.5 for ambiguous UMA disputes). Pending →
                        # None: leave the row pending, don't mark failed.
                        outcome = provider.get_event_outcome(symbol)
                        if outcome is None:
                            continue
                        self.ledger.mark_resolved(
                            pred["id"],
                            exit_price=None,
                            actual_return=None,
                            ground_truth={
                                "outcome": float(outcome),
                                "resolved_at": _utcnow().isoformat(),
                            },
                        )
                        resolved += 1
                    else:
                        # Continuous-asset flow. resolve_at is the close
                        # timestamp of the target bar; step back one bar to
                        # the target bar's open and fetch that bar at the
                        # task's resolution interval — its close is the exit
                        # price. Legacy rows (NULL interval) default to 1m.
                        interval = pred.get("resolution_interval") or "1m"
                        resolve_at = datetime.fromisoformat(pred["resolve_at"])
                        fetch_at = resolve_at - timedelta(
                            seconds=interval_to_seconds(interval)
                        )
                        snap = provider.get_price_at(symbol, fetch_at, interval)
                        entry_price = pred["entry_price"]
                        exit_price = snap.price
                        actual_return = (exit_price - entry_price) / entry_price if entry_price else 0.0
                        self.ledger.mark_resolved(
                            pred["id"],
                            exit_price=exit_price,
                            actual_return=actual_return,
                            ground_truth={
                                "exit_price": exit_price,
                                "actual_return": actual_return,
                                "resolve_timestamp": snap.timestamp.isoformat(),
                            },
                        )
                        resolved += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to resolve %s/%s pred=%s: %s",
                        provider_name,
                        symbol,
                        pred["id"][:8],
                        exc,
                    )
                    self.ledger.mark_failed(pred["id"], str(exc))
                    failed += 1

        return {"resolved": resolved, "failed": failed}

    # Phase 2: Score (compute rewards from resolved ground truth)

    def score_resolved(self) -> dict[str, int]:
        """Score all sessions whose pending predictions have resolved.

        **Per-session aggregation**: predictions belonging to the same
        ``session_id`` are scored
        together via a single ``compute_aggregate`` call so cross-trade
        metrics like ``price_r2`` and ``price_pearson`` see the full
        session series. Each row in the session receives the same
        rewards dict — for series-level metrics this is the only
        meaningful value; for per-trade-aggregable metrics (MSE, MAE,
        MAPE, …) the row-stamped value is the session mean.

        Sessions with any remaining ``pending`` rows are deferred to
        the next pass. ``failed`` rows are excluded from the scoring
        set but do not block the session — they're treated as "won't
        ever resolve" and the rest of the session can score.

        Returns ``{"scored": N, "failed": N}`` counting predictions,
        not sessions.
        """
        unscored = self.ledger.get_resolved_unscored(limit=self.config.batch_size)
        if not unscored:
            return {"scored": 0, "failed": 0}

        # Group unscored predictions by session_id so we can decide on
        # a per-session basis whether the session is complete enough
        # to score.
        by_session: dict[str, list[dict]] = defaultdict(list)
        for pred in unscored:
            by_session[pred["session_id"]].append(pred)

        scored = failed = 0
        for sid, candidate_preds in by_session.items():
            try:
                full = self.ledger.get_session(sid)
                statuses = {r["status"] for r in full}
                if "pending" in statuses:
                    # Defer until all resolved/failed. Session metrics
                    # (R², Pearson) need full support, and per-trade
                    # means only converge when support is stable.
                    continue

                session_resolved_all = [
                    r for r in full if r["status"] == "resolved"
                ]
                if not session_resolved_all:
                    continue

                # Event-shape sessions have actual_return NULL +
                # ground_truth.outcome. Sessions are uniform per family,
                # so the first resolved row is a reliable probe.
                first = session_resolved_all[0]
                first_gt = _parse_ground_truth(first.get("ground_truth"))
                is_event_session = (
                    first.get("actual_return") is None
                    and first_gt.get("outcome") is not None
                )

                predictions: list[dict] = []
                ground_truths: list[dict] = []
                n_dropped_ambiguous = 0

                if is_event_session:
                    for r in session_resolved_all:
                        gt = _parse_ground_truth(r.get("ground_truth"))
                        outcome = gt.get("outcome")
                        if outcome is None:
                            continue
                        try:
                            outcome_f = float(outcome)
                        except (TypeError, ValueError):
                            continue
                        # Drop UMA-disputed 0.5 outcomes: unscoreable,
                        # would penalise calibrated predictions.
                        if abs(outcome_f - 0.5) < 1e-9:
                            n_dropped_ambiguous += 1
                            continue
                        predictions.append({
                            "predicted_yes_probability": r.get("predicted_price"),
                        })
                        ground_truths.append({"outcome": outcome_f})
                else:
                    # Continuous-asset (existing) flow. Filter to rows
                    # with non-null actual_return — sessions can have
                    # mixed-status rows when some predictions failed to
                    # resolve.
                    for r in session_resolved_all:
                        if r.get("actual_return") is None:
                            continue
                # Repair direction for manually written or older price records.
                        direction = r["direction"]
                        pp = r.get("predicted_price")
                        ep = r["entry_price"]
                        if (direction is None or direction == "") and pp is not None:
                            direction = "long" if float(pp) >= float(ep) else "short"
                        predictions.append({
                            "direction": direction,
                            "predicted_price": pp,
                        })
                        ground_truths.append({
                            "actual_return": r["actual_return"],
                            "entry_price": ep,
                            "exit_price": r.get("exit_price"),
                        })

                if not predictions:
                    # Nothing scoreable (e.g. event session where every
                    # outcome was 0.5). Stamp a sentinel so the trial
                    # reward.json shows the drop count and zero reward
                    # rather than staying status=resolved forever.
                    rewards = {
                        rc.name: float("nan") for rc in self._reward_classes
                    }
                    if n_dropped_ambiguous > 0:
                        rewards["n_dropped_ambiguous"] = float(n_dropped_ambiguous)
                    for cand in candidate_preds:
                        self.ledger.mark_scored(cand["id"], rewards)
                        scored += 1
                    continue

                rewards: dict[str, float] = {}
                for reward_fn in self._reward_classes:
                    try:
                        rewards[reward_fn.name] = float(
                            reward_fn.compute_aggregate(predictions, ground_truths)
                        )
                    except Exception:
                        rewards[reward_fn.name] = float("nan")
                if n_dropped_ambiguous > 0:
                    rewards["n_dropped_ambiguous"] = float(n_dropped_ambiguous)

                # Stamp the same rewards dict on every newly-resolved
                # row in the candidate set. Already-scored rows are
                # untouched (mark_scored's WHERE clause filters by
                # status='resolved').
                for cand in candidate_preds:
                    self.ledger.mark_scored(cand["id"], rewards)
                    scored += 1
            except Exception as exc:
                logger.warning(
                    "Failed to score session=%s: %s", sid[:8], exc
                )
                for cand in candidate_preds:
                    self.ledger.mark_failed(cand["id"], f"scoring: {exc}")
                    failed += 1

        return {"scored": scored, "failed": failed}

    # Combined pass

    def run_once(self) -> dict[str, Any]:
        """Resolve + score in one pass."""
        resolve_result = self.resolve_pending()
        score_result = self.score_resolved()
        return {**resolve_result, **score_result}

    def run_loop(self, poll_interval_sec: int | None = None) -> None:
        """Blocking loop: resolve + score every *poll_interval_sec*.

        Press Ctrl-C to stop.
        """
        interval = poll_interval_sec or self.config.poll_interval_sec
        logger.info("Resolver loop starting (interval=%ds)", interval)
        try:
            while True:
                result = self.run_once()
                logger.info("Resolver pass: %s", result)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Resolver loop stopped by user.")
