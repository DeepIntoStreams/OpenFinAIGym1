"""Unified SFT driver for single-turn and multi-turn modes.

Both modes share the same SkyRL training loop; they differ only in the
agent class and response-parser format recorded in the run manifest
(used downstream by ``eval/trained.py`` to pick the right agent + parser).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
from loguru import logger

from openfinai_skyrl.data.visualize import write_train_loss_artifacts
from openfinai_skyrl.train.config import build_train_config, uses_local_backend
from openfinai_skyrl.train.dataset import build_sft_examples
from openfinai_skyrl.train.loop import (
    build_policy_dispatch,
    load_tokenizer,
    run_sft_training_loop,
)
from openfinai_skyrl.train.manifest import (
    hash_config,
    resolve_git_sha,
    snapshot_dataset_system_prompt,
)
from openfinai_skyrl.train.ray_setup import initialize_local_ray


@dataclass(frozen=True)
class ModeSpec:
    """Per-mode constants recorded in the run manifest."""

    agent_class_path: str
    expected_output_format: str


MODE_SPECS: dict[str, ModeSpec] = {
    "single": ModeSpec(
        agent_class_path="openfinai_harbor.agents.single_shot_llm.SingleShotLLMAgent",
        expected_output_format="fenced_python",
    ),
    "multi": ModeSpec(
        agent_class_path="openfinai_harbor.agents.react_fin_agent.ReActFinAgent",
        expected_output_format="python_tag_with_answer",
    ),
}


def train(config: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Run an SFT job for ``mode`` and return a manifest dict.

    ``messages_to_sft_examples`` yields one example per assistant turn,
    so a multi-turn row naturally produces N examples while a single-turn
    row produces one — no per-mode code path is needed inside the loop.
    """
    if mode not in MODE_SPECS:
        raise ValueError(f"Unknown mode: {mode!r}. Expected one of {sorted(MODE_SPECS)}.")
    spec = MODE_SPECS[mode]

    cfg, runtime = build_train_config(config)
    tokenizer = load_tokenizer(cfg.trainer.policy.model.path)
    examples = build_sft_examples(
        dataset_name=runtime.dataset_name,
        dataset_split=runtime.dataset_split,
        messages_key=runtime.messages_key,
        tokenizer=tokenizer,
        max_length=runtime.max_length,
    )

    use_ray = not uses_local_backend(cfg.trainer.strategy)
    if use_ray:
        initialize_local_ray(cfg)
    dispatch = build_policy_dispatch(cfg, runtime.num_steps)
    try:
        step_rows = run_sft_training_loop(cfg, runtime, tokenizer, examples, dispatch)
    finally:
        if use_ray:
            ray.shutdown()

    ckpt_root = Path(cfg.trainer.ckpt_path)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": mode,
        "variant": ckpt_root.name,
        "agent_class_path": spec.agent_class_path,
        "expected_output_format": spec.expected_output_format,
        "model_path": cfg.trainer.policy.model.path,
        "dataset_name": runtime.dataset_name,
        "dataset_split": runtime.dataset_split,
        "num_steps": runtime.num_steps,
        "batch_size": runtime.batch_size,
        "max_length": runtime.max_length,
        "ckpt_path": str(ckpt_root),
        "hf_final_dir": str(ckpt_root / "hf_final"),
        "checkpoints_dir": str(ckpt_root / "checkpoints"),
        "n_examples": len(examples),
        "strategy": cfg.trainer.strategy,
        "git_sha": resolve_git_sha(),
        "config_sha256": hash_config(config),
        "system_prompt": snapshot_dataset_system_prompt(
            runtime.dataset_name, runtime.dataset_split
        ),
        "status": "success",
    }
    (ckpt_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path = ckpt_root / "train_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in step_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    plot_artefacts = write_train_loss_artifacts(
        step_rows, run_dir=ckpt_root, variant=manifest["variant"]
    )
    manifest.update(plot_artefacts)
    (ckpt_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "{}-turn SFT done. manifest={} metrics={} train_loss_png={}",
        mode,
        ckpt_root / "manifest.json",
        metrics_path,
        plot_artefacts.get("train_loss_png", "<skipped>"),
    )
    return manifest


__all__ = ["MODE_SPECS", "ModeSpec", "train"]
