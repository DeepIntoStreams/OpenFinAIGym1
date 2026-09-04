"""Sanitize engineered OHLCV features shared by curated forecasting tasks.

Non-finite feature values are reported with their source bars, converted to
NaN, and removed before model input. Warning previews are capped.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cap per-row warnings so a pathological dataset doesn't flood logs.
_PREVIEW_LIMIT = 5


def sanitize_engineered_features(
    df: pd.DataFrame,
    raw: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    head_envelope: int,
    tail_envelope: int,
    symbol: Optional[str] = None,
    ts_col: str = "timestamp",
) -> Tuple[pd.DataFrame, int]:
    """Replace ±inf with NaN, ``dropna()``, and warn on body-row anomalies.

    ``head_envelope`` and ``tail_envelope`` are the row counts the caller
    expects to drop due to rolling-window warmup and forecast-horizon
    shift respectively — NaN there is normal and silently consumed.

    "Anomaly" = any non-finite cell (``inf`` or ``NaN``) in the scanned
    columns outside that envelope. For each anomalous row we emit one
    ``logging.warning`` naming the offending column(s), the engineered
    row's timestamp, and the upstream raw bar. If the bar (or its
    predecessor) has ``volume == 0`` with frozen OHLC the warning flags
    that explicitly as the canonical exchange-gap signature.

    Returns ``(cleaned_df, n_anomalies)``. ``n_anomalies`` is zero on
    clean data.

    Note: ``df`` is expected to have integer-aligned indexing with
    ``raw`` (i.e. ``df`` was produced via ``raw.copy()`` and no reindex
    has occurred). This holds for the current curated forecasting
    feature pipelines.
    """
    scan_cols = list(feature_cols)
    for extra in ("target", "reference"):
        if extra in df.columns and extra not in scan_cols:
            scan_cols.append(extra)

    n = len(df)
    body_lo = min(head_envelope, n)
    body_hi = max(n - tail_envelope, body_lo)

    n_anomalies = 0
    if body_hi > body_lo:
        body = df.iloc[body_lo:body_hi]
        try:
            vals = body[scan_cols].to_numpy(dtype=float, copy=False)
        except (TypeError, ValueError):
            # Non-numeric column slipped in — fall back to per-column dtype-safe scan.
            vals = np.column_stack(
                [pd.to_numeric(body[c], errors="coerce").to_numpy() for c in scan_cols]
            )
        bad_rows_mask = ~np.isfinite(vals).all(axis=1)
        bad_idx = np.where(bad_rows_mask)[0] + body_lo
        n_anomalies = int(bad_idx.size)

        if n_anomalies > 0:
            tag = f"features:{symbol}" if symbol else "features"
            for k, df_idx in enumerate(bad_idx):
                if k >= _PREVIEW_LIMIT:
                    logger.warning(
                        "[%s] ... and %d more anomalous row(s) (truncated).",
                        tag, n_anomalies - _PREVIEW_LIMIT,
                    )
                    break
                _emit_anomaly_warning(df, raw, int(df_idx), scan_cols, tag, ts_col)

    cleaned = (
        df.replace([np.inf, -np.inf], np.nan)
          .dropna()
          .reset_index(drop=True)
    )
    return cleaned, n_anomalies


def _emit_anomaly_warning(
    df: pd.DataFrame,
    raw: pd.DataFrame,
    df_idx: int,
    scan_cols: list[str],
    tag: str,
    ts_col: str,
) -> None:
    row = df.iloc[df_idx]
    ts = row[ts_col] if ts_col in df.columns else f"row {df_idx}"
    offenders: list[str] = []
    for c in scan_cols:
        v = row[c]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isinf(fv):
            offenders.append(f"{c}={'+inf' if fv > 0 else '-inf'}")
        elif np.isnan(fv):
            offenders.append(f"{c}=NaN")
    upstream = _upstream_context(raw, df_idx, ts_col)
    logger.warning(
        "[%s] anomalous engineered row idx=%d ts=%s: %s. %s "
        "Sanitizing (inf -> NaN -> dropna).",
        tag, df_idx, ts, ", ".join(offenders) or "(no detail)", upstream,
    )


def _upstream_context(raw: pd.DataFrame, df_idx: int, ts_col: str) -> str:
    """Format the raw OHLCV bar(s) most likely behind the anomaly.

    Engineered row indexing matches raw row indexing because features
    are computed in-place on ``raw.copy()`` and ``dropna()`` runs *after*
    this helper. We surface the same-index bar and the previous bar —
    the predecessor is the typical culprit for ``pct_change``-style
    features (zero divisor lands one row earlier).
    """
    parts: list[str] = []
    flag = ""
    for offset, label in ((0, "raw"), (-1, "prev_raw")):
        cand = df_idx + offset
        if not (0 <= cand < len(raw)):
            continue
        r = raw.iloc[cand]
        try:
            o, h, l, c, v = (
                float(r["open"]), float(r["high"]), float(r["low"]),
                float(r["close"]), float(r["volume"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        ts = r[ts_col] if ts_col in raw.columns else f"row {cand}"
        parts.append(f"{label}[{ts}]: O={o} H={h} L={l} C={c} V={v}")
        if v == 0.0 and not flag:
            if o == h == l == c:
                flag = (
                    f"; likely cause: exchange-gap bar at {ts} "
                    f"(volume=0, OHLC frozen at {o})"
                )
            else:
                flag = f"; likely cause: zero-volume bar at {ts}"
    if not parts:
        return "(no upstream raw context available)"
    return "upstream " + " | ".join(parts) + flag
