"""Create diagnostic loss, reward, and baseline-comparison plots for SFT runs.

The CLI scans variant manifests, training metrics, and evaluation summaries,
then writes plots and a manifest under the experiment report directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


# Shared plot palette and typography.
_PALETTE = {
    "base": "#7f7f7f",          # neutral grey for the base-LLM bar
    "trained": "#1f77b4",       # accent blue for the SFT'd variant
    "improve": "#2ca02c",       # green when SFT beats the base
    "degrade": "#d62728",       # red when SFT loses to the base
    "grid": "#cccccc",
    "axis": "#444444",
}

_FONT = {
    "title": 14,
    "axis": 12,
    "tick": 10,
    "legend": 10,
    "annot": 9,
}


def _apply_style() -> None:
    """Apply a single, predictable matplotlib style to every figure."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": _FONT["tick"],
        "axes.titlesize": _FONT["title"],
        "axes.titleweight": "medium",
        "axes.labelsize": _FONT["axis"],
        "axes.edgecolor": _PALETTE["axis"],
        "axes.labelcolor": _PALETTE["axis"],
        "xtick.labelsize": _FONT["tick"],
        "ytick.labelsize": _FONT["tick"],
        "xtick.color": _PALETTE["axis"],
        "ytick.color": _PALETTE["axis"],
        "legend.fontsize": _FONT["legend"],
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": _PALETTE["grid"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.6,
    })


def _annotate_bars(ax, bars, values, *, fmt: str = "{:.2f}", color: str | None = None) -> None:
    """Stick a small value label just above (or below, for negatives) each bar."""
    for rect, val in zip(bars, values):
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v == 0.0:
            label = "0"
        else:
            label = fmt.format(v)
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 1.02 if h >= 0 else 0.98
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2.0, h * offset if h != 0 else 0.0),
            ha="center", va=va,
            fontsize=_FONT["annot"],
            color=color or _PALETTE["axis"],
            xytext=(0, 2 if h >= 0 else -2),
            textcoords="offset points",
        )


def _safe_task_label(name: str) -> str:
    """Shrink ``1_commodity_logreturn_forecasting`` → ``commodity (logret)``
    so x-tick labels fit without rotating to 90°."""
    n = name
    if n.startswith("1_"):
        n = n[2:]
    n = n.replace("_logreturn_forecasting", " (logret)")
    n = n.replace("_forecasting", "")
    n = n.replace("_variant", " v")
    return n.replace("_", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="SFT experiment root produced by openfinai_skyrl.train.driver.train.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override plots output dir. Defaults to <root>/report/plots.",
    )
    parser.add_argument(
        "--eval-subdir-name",
        default=None,
        help=(
            "Eval subdir to visualise under each run_dir (e.g. ``eval``, "
            "``eval_FC``). When omitted, prefers ``eval`` then the newest "
            "non-historical ``eval_*`` dir."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# Per-run artefacts (called by train/{single,multi}_turn.py + eval/trained.py)


def write_train_loss_artifacts(
    step_rows: Sequence[dict[str, Any]],
    *,
    run_dir: Path | str,
    variant: str | None = None,
) -> dict[str, str]:
    """Write ``train_loss.png`` for a single run.

    ``step_rows`` is the list of per-step dicts produced by
    :func:`openfinai_skyrl.train.loop.run_sft_training_loop`. Best-effort:
    if matplotlib is unavailable the function returns an empty dict (the
    per-step rows are still in ``train_metrics.jsonl`` next to the run dir).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is best-effort.
        return out

    if not step_rows:
        return out

    _apply_style()
    steps = [int(row.get("step", i + 1)) for i, row in enumerate(step_rows)]
    losses = [float(row.get("loss", float("nan"))) for row in step_rows]
    title_variant = variant or run_dir.name
    png_path = run_dir / "train_loss.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, losses, color=_PALETTE["trained"], linewidth=1.4, alpha=0.85)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"SFT training loss — {title_variant}  ·  {len(steps)} steps")
    # Mark the final loss value so readers don't squint at the right edge.
    if losses and losses[-1] == losses[-1]:  # not NaN
        ax.annotate(
            f"final = {losses[-1]:.3f}",
            xy=(steps[-1], losses[-1]),
            xytext=(-65, 18),
            textcoords="offset points",
            fontsize=_FONT["annot"], color=_PALETTE["axis"],
            arrowprops=dict(arrowstyle="-", color=_PALETTE["axis"], lw=0.6),
        )
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    out["train_loss_png"] = str(png_path)
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _aggregate_tokens_from_trial_dirs(eval_dir: Path) -> dict[str, int]:
    """Sum ``n_input_tokens`` / ``n_output_tokens`` across every per-task trial.

    ``eval/trained.py`` and ``eval/baseline.py`` write per-task trials under
    ``<eval_dir>/trials/<task>/<utc>/agent/answer.json``. Each ``answer.json``
    carries the agent's chat totals (``n_input_tokens``,
    ``n_output_tokens``, ``n_cache_tokens``).
    """
    totals = {"n_input_tokens": 0, "n_output_tokens": 0, "n_cache_tokens": 0, "n_trials": 0}
    if not eval_dir.exists():
        return totals
    for answer_path in eval_dir.rglob("agent/answer.json"):
        payload = _load_json(answer_path)
        if not isinstance(payload, dict):
            continue
        totals["n_trials"] += 1
        for k in ("n_input_tokens", "n_output_tokens", "n_cache_tokens"):
            v = payload.get(k)
            if isinstance(v, (int, float)):
                totals[k] += int(v)
    return totals


_HISTORICAL_EVAL_HINTS = ("old", "inscript", "stale", "buggy", "backup")


def _resolve_eval_dir(run_dir: Path, explicit_name: str | None = None) -> Path | None:
    """Pick the eval dir to visualise for ``run_dir``.

    When ``explicit_name`` is set, use ``<run_dir>/<explicit_name>/``
    (returns None if it has no ``summary.json``).

    Otherwise the resolution order is:
      1. ``<run_dir>/eval/`` if its ``summary.json`` exists.
      2. ``<run_dir>/eval_*/`` candidates, with directories whose name
         contains an obvious historical hint (``old``, ``inscript``, …)
         filtered out; among the survivors, the most recently modified
         wins. This lets operators preserve historical evals alongside
         the live one without poisoning the report.
    Returns None when no usable eval dir is found.
    """
    if explicit_name:
        explicit = run_dir / explicit_name
        if (explicit / "summary.json").exists():
            return explicit
        return None
    canonical = run_dir / "eval"
    if (canonical / "summary.json").exists():
        return canonical
    candidates = []
    for p in run_dir.glob("eval_*"):
        if not p.is_dir():
            continue
        if not (p / "summary.json").exists():
            continue
        lname = p.name.lower()
        if any(hint in lname for hint in _HISTORICAL_EVAL_HINTS):
            continue
        candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _collect_variants(
    experiment_root: Path,
    *,
    eval_subdir_name: str | None = None,
) -> list[dict[str, Any]]:
    runs_root = experiment_root / "runs"
    if not runs_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        eval_dir = _resolve_eval_dir(run_dir, eval_subdir_name)
        exec_summary: dict[str, Any] = {}
        token_totals: dict[str, Any] = {}
        if eval_dir is not None:
            exec_summary = _load_json(eval_dir / "summary.json")
            token_totals = _aggregate_tokens_from_trial_dirs(eval_dir)
        train_metrics = _load_jsonl(run_dir / "train_metrics.jsonl")
        rows.append(
            {
                "variant": manifest.get("variant", run_dir.name),
                "steps": manifest.get("steps"),
                "status": manifest.get("status"),
                "exec_summary": exec_summary,
                "exec_token_totals": token_totals,
                "eval_dir_name": eval_dir.name if eval_dir is not None else None,
                "train_metrics": train_metrics,
                "kind": "trained",
            }
        )
    return rows


def _collect_baseline(experiment_root: Path) -> dict[str, Any] | None:
    """Pick up the pre-trained-LLM baseline output, if present.

    Convention: ``eval/baseline.py`` is run with
    ``--output-dir <experiment_root>/baseline/<name>``; the script writes
    ``summary.json`` + per-task trial dirs there. The first such
    directory found is treated as THE baseline for lift comparisons.
    Returns ``None`` if no baseline directory exists.
    """
    base_root = experiment_root / "baseline"
    if not base_root.exists():
        return None
    for candidate in sorted(p for p in base_root.iterdir() if p.is_dir()):
        summary_path = candidate / "summary.json"
        if not summary_path.is_file():
            continue
        return {
            "variant": candidate.name,
            "kind": "baseline",
            "exec_summary": _load_json(summary_path),
            "exec_token_totals": _aggregate_tokens_from_trial_dirs(candidate),
        }
    return None


def _plot_train_loss(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    out: dict[str, str] = {}
    for row in rows:
        metrics = row.get("train_metrics") or []
        if not metrics:
            continue
        _apply_style()
        steps = [int(m.get("step", i + 1)) for i, m in enumerate(metrics)]
        losses = [float(m.get("loss", float("nan"))) for m in metrics]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(steps, losses, color=_PALETTE["trained"], linewidth=1.4, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss (log scale)")
        ax.set_title(f"SFT training loss — {row['variant']}  ·  {len(steps)} steps")
        if losses and losses[-1] == losses[-1]:
            ax.annotate(
                f"final = {losses[-1]:.3f}",
                xy=(steps[-1], losses[-1]),
                xytext=(-65, 18), textcoords="offset points",
                fontsize=_FONT["annot"], color=_PALETTE["axis"],
                arrowprops=dict(arrowstyle="-", color=_PALETTE["axis"], lw=0.6),
            )
        fig.tight_layout()
        path = output_dir / f"train_loss__{row['variant']}.png"
        fig.savefig(path)
        plt.close(fig)
        out[path.stem] = str(path)
    return out


def _plot_per_task_reward(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    """Bar plot of per-task reward with per-seed scatter overlay.

    Bar height = **median** of per-seed rewards (robust to a single bad
    seed); individual seed scores are scattered on top so the reader
    sees the spread directly. Reward is loss-scaled (lower=better); 0 is
    the verifier-failure sentinel.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import statistics

    out: dict[str, str] = {}
    for row in rows:
        summary = row.get("exec_summary") or {}
        tasks = summary.get("tasks") or []
        if not tasks:
            continue
        _apply_style()
        n_seeds = int(summary.get("n_seeds") or 1)
        labels: list[str] = []
        medians: list[float] = []
        all_seed_rewards: list[list[float]] = []
        succ_rates: list[float] = []
        for t in tasks:
            seeds = t.get("seeds") or []
            seed_rewards = [
                float(s.get("reward")) for s in seeds
                if isinstance(s, dict) and s.get("reward") is not None
            ]
            if not seed_rewards:
                # fall back to whatever summary mean we have
                fallback = t.get("reward_mean") or t.get("reward") or 0.0
                seed_rewards = [float(fallback)]
            base_label = _safe_task_label(str(t.get("task", "?")))
            sr = float(t.get("success_rate") or 0.0)
            labels.append(f"{base_label}\n{int(round(sr * 100))}% success")
            medians.append(float(statistics.median(seed_rewards)))
            all_seed_rewards.append(seed_rewards)
            succ_rates.append(sr)

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5.2))
        bars = ax.bar(
            x, medians,
            color=_PALETTE["trained"], edgecolor="white", linewidth=0.5,
            alpha=0.85, zorder=2,
        )
        # Pick a y-range that keeps the bars readable. Outlier seed points
        # are clipped to the top edge with their value annotated.
        max_bar = max(medians) if medians else 1.0
        in_range_seeds: list[float] = []
        for rewards in all_seed_rewards:
            in_range_seeds.extend(rewards)
        sane_top = max(2.5 * max_bar, 1.0)
        natural_top = max(in_range_seeds) if in_range_seeds else 1.0
        # If the natural y-max is reasonable (< 4× max bar), don't clip.
        ymax = natural_top if natural_top <= 4 * max_bar else sane_top

        # Per-seed scatter — dots in the bar's accent colour with a dark edge
        # so they're legible against the bar; any value above the cap gets
        # pinned to the top edge with a red arrow + value annotation.
        rng = np.random.default_rng(0)
        for xi, seed_rewards in zip(x, all_seed_rewards):
            jitter = rng.uniform(-0.12, 0.12, size=len(seed_rewards))
            in_y = np.array([r if r <= ymax else ymax for r in seed_rewards])
            ax.scatter(
                xi + jitter, in_y,
                s=28, color=_PALETTE["trained"], alpha=0.85,
                edgecolors=_PALETTE["axis"], linewidths=0.6, zorder=3,
            )
            for r in seed_rewards:
                if r > ymax:
                    ax.annotate(
                        f"↑ {r:.1f}",
                        xy=(xi, ymax),
                        xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top",
                        fontsize=_FONT["annot"] - 1,
                        color=_PALETTE["degrade"], weight="bold",
                    )
        ax.set_ylim(top=ymax * 1.08, bottom=min(0.0, ax.get_ylim()[0]))
        _annotate_bars(ax, bars, medians, fmt="{:.2f}")

        seed_tag = f"  ·  n_seeds={n_seeds}" if n_seeds > 1 else ""
        ax.set_title(f"Per-task reward — {row['variant']}{seed_tag}")
        ax.set_ylabel("reward (loss-scaled, lower is better)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("")
        ax.set_axisbelow(True)
        ax.margins(y=0.18)
        # Caption inside the figure so the meaning of bar/dot/0 is unambiguous.
        ax.text(
            0.01, 0.99,
            "bar = median across seeds   ·   dot = individual seed   ·   0 = verifier failed",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=_FONT["annot"],
            color=_PALETTE["axis"],
        )
        fig.tight_layout()
        path = output_dir / f"per_task_reward__{row['variant']}.png"
        fig.savefig(path)
        plt.close(fig)
        out[path.stem] = str(path)
    return out


def _plot_training_curve(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    out: dict[str, str] = {}
    successful = sorted(
        [r for r in rows if r.get("status") == "success" and r.get("steps") is not None],
        key=lambda r: int(r["steps"]),
    )
    if not successful:
        return out
    _apply_style()
    steps = [int(r["steps"]) for r in successful]
    rewards = [float((r.get("exec_summary") or {}).get("avg_reward", 0.0)) for r in successful]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        steps, rewards, marker="o", color=_PALETTE["trained"],
        linewidth=1.6, markersize=7,
    )
    for x, y in zip(steps, rewards):
        ax.annotate(
            f"{y:.2f}", (x, y), xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=_FONT["annot"], color=_PALETTE["axis"],
        )
    ax.set_xlabel("SFT steps trained")
    ax.set_ylabel("held-out avg reward  (loss-scaled · lower is better)")
    ax.set_title("Held-out reward vs. SFT steps")
    ax.margins(y=0.2)
    fig.tight_layout()
    path = output_dir / "training_curve_avg_reward.png"
    fig.savefig(path)
    plt.close(fig)
    out["training_curve_avg_reward"] = str(path)
    return out


def _per_task_reward_map(exec_summary: dict[str, Any]) -> dict[str, float]:
    """Flatten ``summary.json`` → ``{task_name: reward}``. Missing → 0.0."""
    out: dict[str, float] = {}
    for task_row in (exec_summary.get("tasks") or []):
        name = str(task_row.get("task", ""))
        if not name:
            continue
        try:
            out[name] = float(task_row.get("reward") or 0.0)
        except (TypeError, ValueError):
            out[name] = 0.0
    return out


def _per_task_seed_rewards(exec_summary: dict[str, Any]) -> dict[str, list[float]]:
    """Flatten ``summary.json`` → ``{task_name: [seed reward, ...]}``."""
    out: dict[str, list[float]] = {}
    for task_row in (exec_summary.get("tasks") or []):
        name = str(task_row.get("task", ""))
        if not name:
            continue
        seeds = task_row.get("seeds") or []
        rewards: list[float] = []
        for s in seeds:
            if not isinstance(s, dict):
                continue
            r = s.get("reward")
            if r is None:
                continue
            try:
                rewards.append(float(r))
            except (TypeError, ValueError):
                continue
        if not rewards:
            # Fall back to summary mean so we always have at least one point.
            fallback = task_row.get("reward_mean") or task_row.get("reward")
            if fallback is not None:
                try:
                    rewards.append(float(fallback))
                except (TypeError, ValueError):
                    pass
        out[name] = rewards
    return out


def _per_task_success_rate(exec_summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for task_row in (exec_summary.get("tasks") or []):
        name = str(task_row.get("task", ""))
        if name:
            try:
                out[name] = float(task_row.get("success_rate") or 0.0)
            except (TypeError, ValueError):
                out[name] = 0.0
    return out


def _plot_lift_over_base_llm(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, str]:
    """Per-task SFT lift as **success-rate delta** (trained − base).

    The previous numerical-reward formulation was misleading whenever the
    base failed the verifier (reward=0 sentinel) — the subtraction made
    SFT look like a degradation even though the base produced nothing
    runnable. Success-rate delta is well-defined in that regime: the
    trained variant either lifts off the verifier-failure floor or it
    does not.

    Skipped when no baseline run is present under
    ``<experiment_root>/baseline/<name>/``.
    """
    if baseline is None:
        return {}
    import matplotlib.pyplot as plt

    base_succ = _per_task_success_rate(baseline.get("exec_summary") or {})
    if not base_succ:
        return {}
    out: dict[str, str] = {}
    for row in rows:
        trained_succ = _per_task_success_rate(row.get("exec_summary") or {})
        common = sorted(set(base_succ) & set(trained_succ))
        if not common:
            continue
        _apply_style()
        deltas_pct = [
            (trained_succ[t] - base_succ[t]) * 100.0 for t in common
        ]
        labels = [_safe_task_label(t) for t in common]
        colors = [_PALETTE["improve"] if v >= 0 else _PALETTE["degrade"] for v in deltas_pct]
        fig, ax = plt.subplots(
            figsize=(max(9, len(common) * 1.3), 5.2),
            layout="constrained",
        )
        bars = ax.bar(labels, deltas_pct, color=colors, edgecolor="white", linewidth=0.5)
        _annotate_bars(ax, bars, deltas_pct, fmt="{:+.0f}pp")
        ax.axhline(0.0, color=_PALETTE["axis"], linewidth=0.8)
        ax.set_title(
            f"Per-task SFT success-rate lift\n"
            f"trained: {row['variant']}   vs   base: {baseline['variant']}"
        )
        ax.set_ylabel("trained success rate − base success rate (pp)")
        # Pad so the per-task percentage labels don't overlap the title.
        ax.set_ylim(top=max(deltas_pct + [0]) + 25, bottom=min(deltas_pct + [0]) - 15)
        ax.set_yticks(range(-100, 101, 25))
        ax.set_yticklabels([f"{v:+d}pp" if v != 0 else "0" for v in range(-100, 101, 25)])
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
            tick.set_ha("right")
        ax.text(
            0.99, 0.97,
            "▲ green = SFT lifts success rate above base",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=_FONT["annot"], color=_PALETTE["improve"], weight="bold",
        )
        ax.text(
            0.99, 0.03,
            "▼ red = SFT loses ground to base",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=_FONT["annot"], color=_PALETTE["degrade"], weight="bold",
        )
        path = output_dir / f"lift_over_base__{row['variant']}.png"
        fig.savefig(path)
        plt.close(fig)
        out[path.stem] = str(path)
    return out


def _plot_base_vs_trained_per_task(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, str]:
    """Side-by-side base-LLM vs each trained variant, per task (grouped bars)."""
    if baseline is None or not rows:
        return {}
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_style()
    import statistics

    base_summary = baseline.get("exec_summary") or {}
    base_seeds = _per_task_seed_rewards(base_summary)
    base_succ = _per_task_success_rate(base_summary)
    if not base_seeds:
        return {}
    common_tasks = sorted(base_seeds)
    labels = [_safe_task_label(t) for t in common_tasks]
    out: dict[str, str] = {}

    n_variants = 1 + len(rows)
    width = 0.8 / n_variants
    x = np.arange(len(common_tasks))
    trained_palette = [
        _PALETTE["trained"], "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
    ]

    # Two stacked panels: median reward (top) + success rate (bottom). The
    # success-rate panel is the cleanest signal when verifier failure
    # collapses real differences in the reward axis.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(max(10, len(common_tasks) * 1.45), 8.2),
        gridspec_kw={"height_ratios": [2.6, 1.2]},
        layout="constrained",
    )

    def _median(arr: list[float]) -> float:
        return float(statistics.median(arr)) if arr else 0.0

    # Top panel — median bars + per-seed scatter, capped y.
    base_medians = [_median(base_seeds.get(t, [])) for t in common_tasks]
    variant_medians: list[list[float]] = []
    variant_seeds: list[dict[str, list[float]]] = []
    variant_succ: list[dict[str, float]] = []
    for r in rows:
        smap = _per_task_seed_rewards(r.get("exec_summary") or {})
        variant_seeds.append(smap)
        variant_succ.append(_per_task_success_rate(r.get("exec_summary") or {}))
        variant_medians.append([_median(smap.get(t, [])) for t in common_tasks])

    all_medians = base_medians + [v for vm in variant_medians for v in vm]
    max_bar = max(all_medians) if any(m > 0 for m in all_medians) else 1.0
    all_seeds = []
    for t in common_tasks:
        all_seeds.extend(base_seeds.get(t, []))
        for smap in variant_seeds:
            all_seeds.extend(smap.get(t, []))
    natural_top = max(all_seeds) if all_seeds else 1.0
    sane_top = max(2.5 * max_bar, 1.0)
    ymax = natural_top if natural_top <= 4 * max_bar else sane_top

    rng = np.random.default_rng(1)

    def _draw_group(ax, xs, vals, color, label):
        bars = ax.bar(
            xs, vals, width,
            color=color, edgecolor="white", linewidth=0.4,
            label=label, zorder=2,
        )
        _annotate_bars(ax, bars, vals, fmt="{:.2f}")
        return bars

    def _scatter_group(xs, seed_map, fill_color):
        """Overlay per-seed dots in fill_color; pin out-of-range values to the top."""
        for xi, t in zip(xs, common_tasks):
            seeds = seed_map.get(t, [])
            if not seeds:
                continue
            jitter = rng.uniform(-width * 0.35, width * 0.35, size=len(seeds))
            clipped = [min(r, ymax) for r in seeds]
            ax_top.scatter(
                xi + jitter, clipped,
                s=22,
                color=fill_color, alpha=0.75,
                edgecolors=_PALETTE["axis"], linewidths=0.5,
                zorder=4,
            )
            for r in seeds:
                if r > ymax:
                    ax_top.annotate(
                        f"↑{r:.1f}", xy=(xi, ymax),
                        xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top", fontsize=_FONT["annot"] - 1,
                        color=_PALETTE["degrade"], weight="bold",
                    )

    base_x = x - 0.4 + width / 2
    _draw_group(ax_top, base_x, base_medians, _PALETTE["base"], f"base · {baseline['variant']}")
    _scatter_group(base_x, base_seeds, _PALETTE["base"])

    for i, row in enumerate(rows, start=1):
        color = trained_palette[(i - 1) % len(trained_palette)]
        var_x = x - 0.4 + (i + 0.5) * width
        _draw_group(ax_top, var_x, variant_medians[i - 1], color, f"trained · {row['variant']}")
        _scatter_group(var_x, variant_seeds[i - 1], color)

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(labels, rotation=15, ha="right")
    ax_top.set_ylabel("reward (loss-scaled, lower is better)")
    ax_top.set_ylim(top=ymax * 1.18, bottom=0.0)
    ax_top.set_title("Base LLM vs SFT — per-task held-out reward")
    ax_top.legend(loc="upper right", framealpha=0.9)

    # Bottom panel — success rate per task, same x-axis.
    base_sr = [base_succ.get(t, 0.0) * 100 for t in common_tasks]
    bars = ax_bot.bar(
        base_x, base_sr, width,
        color=_PALETTE["base"], edgecolor="white", linewidth=0.4,
    )
    _annotate_bars(ax_bot, bars, base_sr, fmt="{:.0f}%")
    for i, row in enumerate(rows, start=1):
        color = trained_palette[(i - 1) % len(trained_palette)]
        var_x = x - 0.4 + (i + 0.5) * width
        srs = [variant_succ[i - 1].get(t, 0.0) * 100 for t in common_tasks]
        b = ax_bot.bar(
            var_x, srs, width,
            color=color, edgecolor="white", linewidth=0.4,
        )
        _annotate_bars(ax_bot, b, srs, fmt="{:.0f}%")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=15, ha="right")
    ax_bot.set_ylim(0, 110)
    ax_bot.set_yticks([0, 25, 50, 75, 100])
    ax_bot.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_bot.set_ylabel("success rate")
    ax_bot.set_title("Per-task success rate  (fraction of seeds with reward > 0)",
                     fontsize=_FONT["axis"])
    fig.suptitle("", y=0.999)  # ensure constrained_layout reserves top space
    fig.supxlabel(
        "bar = median across seeds   ·   dot = individual seed   ·   "
        "0 = verifier failed (sentinel)   ·   ↑N = seed value clipped above top panel cap",
        fontsize=_FONT["annot"],
        color=_PALETTE["axis"],
    )

    path = output_dir / "base_vs_trained_per_task.png"
    fig.savefig(path)
    plt.close(fig)
    out["base_vs_trained_per_task"] = str(path)
    return out


def _plot_token_dashboard(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, str]:
    """Total input / output / cache tokens per variant (incl. baseline).

    Pure tokens — no cost / $-rate. Lets the operator see how much
    inference each variant burned against the 5-hour Claude Max budget,
    decoupled from prices that change monthly.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    bundle: list[dict[str, Any]] = []
    if baseline is not None:
        bundle.append({"variant": f"base: {baseline['variant']}",
                       "totals": baseline.get("exec_token_totals") or {}})
    bundle.extend(
        {"variant": r["variant"], "totals": r.get("exec_token_totals") or {}}
        for r in rows
    )
    bundle = [b for b in bundle if (b["totals"] or {}).get("n_trials", 0) > 0]
    if not bundle:
        return {}
    _apply_style()
    labels = [b["variant"] for b in bundle]
    n_in = [b["totals"].get("n_input_tokens", 0) for b in bundle]
    n_out = [b["totals"].get("n_output_tokens", 0) for b in bundle]
    n_cache = [b["totals"].get("n_cache_tokens", 0) for b in bundle]
    has_cache = any(c > 0 for c in n_cache)
    x = np.arange(len(labels))
    n_groups = 3 if has_cache else 2
    w = 0.8 / n_groups
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.0), 5))

    def _fmt(v: int | float) -> str:
        v = float(v)
        if v >= 1e6: return f"{v/1e6:.2f}M"
        if v >= 1e3: return f"{v/1e3:.1f}k"
        return f"{int(v)}"

    offsets = [-w] if n_groups == 2 else [-w, 0, w]
    # When cache is empty, plot input + output as the two groups instead.
    series = [("input tokens", n_in, "#1f77b4"), ("output tokens", "#ff7f0e", n_out)] if False else None
    # Simpler: lay out the bars based on whether cache exists.
    if has_cache:
        b1 = ax.bar(x - w, n_in,  w, label="input tokens",   color="#1f77b4", edgecolor="white", linewidth=0.4)
        b2 = ax.bar(x,     n_out, w, label="output tokens",  color="#ff7f0e", edgecolor="white", linewidth=0.4)
        b3 = ax.bar(x + w, n_cache, w, label="cache tokens", color="#9467bd", edgecolor="white", linewidth=0.4)
        groups = [(b1, n_in), (b2, n_out), (b3, n_cache)]
    else:
        b1 = ax.bar(x - w/2, n_in,  w, label="input tokens",  color="#1f77b4", edgecolor="white", linewidth=0.4)
        b2 = ax.bar(x + w/2, n_out, w, label="output tokens", color="#ff7f0e", edgecolor="white", linewidth=0.4)
        groups = [(b1, n_in), (b2, n_out)]
    for bars, vals in groups:
        for rect, v in zip(bars, vals):
            if v <= 0: continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                _fmt(v),
                ha="center", va="bottom",
                fontsize=_FONT["annot"], color=_PALETTE["axis"],
            )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("tokens — sum over held-out trials  (log scale)")
    ax.set_title("Token usage per variant — base LLM vs SFT")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "token_dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    return {"token_dashboard": str(path)}


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    if not experiment_root.exists():
        raise FileNotFoundError(f"experiment-root does not exist: {experiment_root}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else experiment_root / "report" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_variants(experiment_root, eval_subdir_name=args.eval_subdir_name)
    baseline = _collect_baseline(experiment_root)
    plots: dict[str, str] = {}
    plots.update(_plot_train_loss(rows, output_dir))
    plots.update(_plot_per_task_reward(rows, output_dir))
    plots.update(_plot_training_curve(rows, output_dir))
    # Baseline-aware plots — only emitted when a base-LLM run exists at
    # ``<experiment_root>/baseline/<name>/``.
    plots.update(_plot_lift_over_base_llm(rows, baseline, output_dir))
    plots.update(_plot_base_vs_trained_per_task(rows, baseline, output_dir))
    plots.update(_plot_token_dashboard(rows, baseline, output_dir))

    manifest_path = output_dir / "plots_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_root": str(experiment_root),
                "n_variants": len(rows),
                "baseline": (baseline or {}).get("variant"),
                "plots": plots,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[visualize] Wrote {len(plots)} plots under {output_dir}")


if __name__ == "__main__":
    main()
