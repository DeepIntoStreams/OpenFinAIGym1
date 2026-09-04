"""Tokenizer, chat-template, and config-parsing helpers for the SFT loop."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import scan_cache_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer


def _snapshot_has_model_files(snapshot_path: Path) -> bool:
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        return False

    weight_candidates = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    return any((snapshot_path / candidate).exists() for candidate in weight_candidates)


def parse_cli(argv: list[str]) -> dict[str, Any]:
    """Parse ``[--config foo.yaml] KEY=VALUE`` style CLI args via OmegaConf."""
    args = list(argv)
    config_path: str | None = None
    if args[:1] == ["--config"]:
        if len(args) < 2:
            raise ValueError("--config requires a path")
        config_path = args[1]
        args = args[2:]

    base = OmegaConf.load(config_path) if config_path else OmegaConf.create({})
    overrides = OmegaConf.from_cli(args)
    merged = OmegaConf.merge(base, overrides)
    raw = OmegaConf.to_container(merged, resolve=True)
    return raw if isinstance(raw, dict) else {}


def get_nested(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    """Return ``config["a"]["b"]["c"]`` for ``dotted_key="a.b.c"`` or ``default``."""
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_messages_dataset(dataset_name: str, dataset_split: str) -> Dataset:
    """Load a Dataset split — works for local ``save_to_disk`` dirs and Hub ids.

    Loads the requested split's sub-directory directly when it exists on disk;
    this avoids a ``datasets`` bug where ``DatasetDict.load_from_disk`` fails
    with ``IndexError`` if any sibling split has zero shards (e.g. an empty
    test split when the collect stage hasn't reached the held-out tasks yet).
    """
    path = Path(dataset_name)
    if path.exists():
        split_path = path / dataset_split
        if (split_path / "dataset_info.json").exists():
            return load_from_disk(str(split_path))
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            if dataset_split not in ds:
                raise KeyError(f"Dataset split '{dataset_split}' not found in {dataset_name}")
            return ds[dataset_split]
        return ds
    return load_dataset(dataset_name, split=dataset_split)


def has_local_hf_model(model_name_or_path: str) -> bool:
    """Return True if a model with usable weight files is locally cached."""
    path = Path(model_name_or_path)
    if path.exists():
        return True

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False

    for repo in cache_info.repos:
        if repo.repo_id == model_name_or_path and repo.repo_type == "model":
            for revision in repo.revisions:
                snapshot_path = Path(revision.snapshot_path)
                if _snapshot_has_model_files(snapshot_path):
                    return True
    return False


def resolve_local_hf_path(model_name_or_path: str) -> str:
    """Return a local snapshot path for ``model_name_or_path`` (or the input itself)."""
    path = Path(model_name_or_path)
    if path.exists():
        return str(path)

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return model_name_or_path

    for repo in cache_info.repos:
        if repo.repo_id != model_name_or_path or repo.repo_type != "model":
            continue
        revisions = list(repo.revisions)
        if not revisions:
            return model_name_or_path
        revisions.sort(key=lambda rev: rev.last_modified or 0, reverse=True)
        for revision in revisions:
            snapshot_path = Path(revision.snapshot_path)
            if _snapshot_has_model_files(snapshot_path):
                return str(snapshot_path)

    return model_name_or_path


def hf_load_kwargs(model_name_or_path: str) -> dict[str, Any]:
    """Standard kwargs for HF ``from_pretrained`` (respects local-only env)."""
    local_only_env = os.environ.get("OPENFINAI_HF_LOCAL_ONLY")
    if local_only_env is not None:
        local_files_only = local_only_env.strip().lower() not in {"0", "false", "no"}
    else:
        local_files_only = has_local_hf_model(model_name_or_path)

    return {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }


def render_chat_template(
    tokenizer: AutoTokenizer, messages: list[dict[str, Any]], *, add_generation_prompt: bool
) -> str:
    """Apply the tokenizer's chat template; falls back to ``ROLE:\\ncontent\\n``."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )

    text = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        text.append(f"{role.upper()}:\n{content}\n")
    if add_generation_prompt:
        text.append("ASSISTANT:\n")
    return "".join(text)


def tokenize_rendered_text(tokenizer: AutoTokenizer, text: str, *, add_bos: bool) -> list[int]:
    """Tokenise pre-rendered chat text; the BOS-dedup branch prevents double-BOS
    for templates (e.g. Llama 3.x) that already bake ``<|begin_of_text|>`` in.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if add_bos:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            ids_list = list(ids)
            if ids_list and ids_list[0] == int(bos_token_id):
                return ids_list
            return [int(bos_token_id)] + ids_list
    return list(ids)
