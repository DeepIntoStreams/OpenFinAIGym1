"""Chat-history → SFT example construction and per-batch collation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from transformers import AutoTokenizer

# ray_setup must precede skyrl.* imports (see openfinai_skyrl.train.config).
from openfinai_skyrl.train import ray_setup as _ray_setup  # noqa: F401

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

from openfinai_skyrl.train.tokenization import (
    load_messages_dataset,
    render_chat_template,
    tokenize_rendered_text,
)


@dataclass
class SFTExample:
    sequences: list[int]
    attention_mask: list[int]
    response_mask: list[int]
    loss_mask: list[int]


def messages_to_sft_examples(
    messages: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> list[SFTExample]:
    """Render one SFTExample per assistant turn from a chat history.

    Truncates prompt against max_length so response always fits; if response
    alone exceeds max_length, uses a single fallback BOS/EOS/PAD token as
    prompt so the SkyRL worker has at least one conditioning position.
    Skips (with a warning) any assistant turn where rendering
    ``history + assistant`` is not a string-prefix extension of rendering
    ``history`` alone — that mismatch would misalign the loss mask.
    """
    examples: list[SFTExample] = []
    history: list[dict[str, Any]] = []
    fallback_prefix_token_id = next(
        (
            int(token_id)
            for token_id in (
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
            )
            if token_id is not None
        ),
        None,
    )

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue

        if role == "assistant" and history:
            prompt_text = render_chat_template(tokenizer, history, add_generation_prompt=True)
            full_text = render_chat_template(tokenizer, history + [message], add_generation_prompt=False)

            if not full_text.startswith(prompt_text):
                logger.warning(
                    "Skipping assistant turn (idx={}): chat-template prefix mismatch "
                    "(prompt_len={}, full_len={}). Loss mask would be misaligned.",
                    len(history),
                    len(prompt_text),
                    len(full_text),
                )
            else:
                response_text = full_text[len(prompt_text) :]
                prompt_ids = tokenize_rendered_text(tokenizer, prompt_text, add_bos=True)
                response_ids = tokenize_rendered_text(tokenizer, response_text, add_bos=False)
                if response_ids:
                    # SkyRL slices log_probs by the last max_response_len positions
                    # and requires >=1 conditioning prompt token before the response.
                    fallback_token = (
                        fallback_prefix_token_id if fallback_prefix_token_id is not None else 0
                    )
                    if len(response_ids) >= max_length:
                        prompt_ids = [fallback_token]
                        max_response_len = max_length - 1
                        response_ids = response_ids[:max_response_len]
                    else:
                        prompt_budget = max_length - len(response_ids)
                        prompt_ids = prompt_ids[-prompt_budget:]
                        if not prompt_ids:
                            prompt_ids = [fallback_token]
                            response_ids = response_ids[: max_length - 1]

                    sequences = prompt_ids + response_ids
                    response_mask = [1] * len(response_ids)
                    loss_mask = [1] * len(response_ids)
                    if sequences and any(loss_mask):
                        examples.append(
                            SFTExample(
                                sequences=sequences,
                                attention_mask=[1] * len(sequences),
                                response_mask=response_mask,
                                loss_mask=loss_mask,
                            )
                        )

        history.append({"role": role, "content": content})

    return examples


def build_sft_examples(
    dataset_name: str,
    dataset_split: str,
    messages_key: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> list[SFTExample]:
    """Read the split and render N examples per row (one per assistant turn)."""
    logger.info("Loading SFT dataset from {}", dataset_name)
    dataset = load_messages_dataset(dataset_name, dataset_split)
    all_examples: list[SFTExample] = []
    skipped_rows = 0

    for row in dataset:
        messages = row.get(messages_key)
        if not isinstance(messages, list):
            skipped_rows += 1
            continue
        examples = messages_to_sft_examples(messages, tokenizer, max_length=max_length)
        if not examples:
            skipped_rows += 1
            continue
        all_examples.extend(examples)

    logger.info(
        "Prepared {} SFT examples from {} rows; skipped {} rows",
        len(all_examples),
        len(dataset),
        skipped_rows,
    )
    if not all_examples:
        raise RuntimeError("No usable SFT examples were built from the dataset")
    return all_examples


def collate_sft_batch(examples: list[SFTExample], tokenizer: AutoTokenizer) -> TrainingInputBatch:
    """Stack ``examples`` into a left-padded batch ready for SkyRL's slicing."""
    max_seq_len = max(len(ex.sequences) for ex in examples)
    max_response_len = max(len(ex.response_mask) for ex in examples)

    sequences = []
    attention_masks = []
    response_masks = []
    loss_masks = []

    for example in examples:
        left_pad = max_seq_len - len(example.sequences)
        right_pad = max_response_len - len(example.response_mask)
        sequences.append([tokenizer.pad_token_id] * left_pad + example.sequences)
        attention_masks.append([0] * left_pad + example.attention_mask)
        # Mask must be LEFT-padded so the 1s align with the right-edge response
        # positions sliced by SkyRL's `log_probs[:, -num_actions-1:-1]`.
        response_masks.append([0] * right_pad + example.response_mask)
        loss_masks.append([0] * right_pad + example.loss_mask)

    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "response_mask": torch.tensor(response_masks, dtype=torch.long),
            "loss_mask": torch.tensor(loss_masks, dtype=torch.long),
        }
    )
    batch.metadata = {"response_length": max_response_len}
    return batch


__all__ = ["SFTExample", "build_sft_examples", "collate_sft_batch", "messages_to_sft_examples"]
