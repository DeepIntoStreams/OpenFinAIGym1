from __future__ import annotations

import importlib

import pytest

pytest.importorskip("skyrl")  # SFT runtime tests require skyrl-train installed

from openfinai_skyrl.train.config import build_train_config
from openfinai_skyrl.train.dataset import collate_sft_batch, messages_to_sft_examples
from openfinai_skyrl.train.loop import _resolve_policy_worker
from openfinai_skyrl.train.tokenization import tokenize_rendered_text


class _ToyTokenizer:
    pad_token_id = 0
    bos_token_id = 101
    eos_token = "<eos>"
    pad_token = "<pad>"
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}


class _PrependedBOSTokenizer:
    """Mimics tokenizers (e.g. Llama 3.x) whose chat template bakes BOS into
    the rendered text, so ``tokenizer(text)`` already emits the BOS id first."""

    pad_token_id = 0
    bos_token_id = 7
    eos_token = "<eos>"
    pad_token = "<pad>"
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        # Pretend that any rendered text starts with the BOS token id when
        # tokenized (Llama-style behaviour).
        return {"input_ids": [self.bos_token_id] + [ord(ch) for ch in text]}


def test_build_train_config_keeps_native_strategy_and_placement() -> None:
    cfg, runtime = build_train_config(
        {
            "strategy": "megatron",
            "batch_size": 8,
            "micro_train_batch_size_per_gpu": 1,
            "num_steps": 7,
            "max_length": 512,
            "placement": {"num_nodes": 2, "num_gpus_per_node": 4},
            "megatron_config": {"tensor_model_parallel_size": 2},
        }
    )

    assert cfg.trainer.strategy == "megatron"
    assert cfg.trainer.placement.policy_num_nodes == 2
    assert cfg.trainer.placement.policy_num_gpus_per_node == 4
    assert cfg.trainer.policy.megatron_config.tensor_model_parallel_size == 2
    assert runtime.num_steps == 7


def test_messages_to_sft_examples_builds_assistant_targets() -> None:
    tokenizer = _ToyTokenizer()
    examples = messages_to_sft_examples(
        [
            {"role": "system", "content": "Follow instructions."},
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        tokenizer,
        max_length=256,
    )

    assert len(examples) == 1
    assert len(examples[0].sequences) >= len(examples[0].response_mask)
    assert all(token == 1 for token in examples[0].loss_mask)


def test_collate_sft_batch_sets_response_length_metadata() -> None:
    tokenizer = _ToyTokenizer()
    examples = messages_to_sft_examples(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "assistant", "content": "C"},
        ],
        tokenizer,
        max_length=256,
    )

    batch = collate_sft_batch(examples, tokenizer)

    assert batch["sequences"].shape[0] == len(examples)
    assert batch.metadata["response_length"] == max(len(example.response_mask) for example in examples)


def test_tokenize_rendered_text_does_not_double_prepend_bos() -> None:
    """A template-provided BOS is not duplicated or allowed to shift the mask."""
    tokenizer = _PrependedBOSTokenizer()
    ids = tokenize_rendered_text(tokenizer, "abc", add_bos=True)
    assert ids[0] == 7
    assert ids[1] != 7
    assert ids == [7, ord("a"), ord("b"), ord("c")]


def test_tokenize_rendered_text_still_prepends_bos_when_missing() -> None:
    tokenizer = _ToyTokenizer()  # tokenizer() returns no BOS by itself
    ids = tokenize_rendered_text(tokenizer, "abc", add_bos=True)
    assert ids[0] == tokenizer.bos_token_id
    assert ids[1:] == [ord("a"), ord("b"), ord("c")]


def test_collate_sft_batch_right_aligns_response_tokens() -> None:
    """Loss mask must mark exactly the response tokens at the right edge of
    each row (left-padded sequences, right-aligned response). Verify across
    examples of differing lengths."""
    tokenizer = _ToyTokenizer()
    short = messages_to_sft_examples(
        [{"role": "user", "content": "A"}, {"role": "assistant", "content": "X"}],
        tokenizer,
        max_length=256,
    )
    long = messages_to_sft_examples(
        [
            {"role": "user", "content": "Longer question here"},
            {"role": "assistant", "content": "Longer multi-word answer"},
        ],
        tokenizer,
        max_length=256,
    )
    assert short and long
    batch = collate_sft_batch(short + long, tokenizer)

    seq = batch["sequences"]
    attn = batch["attention_mask"]
    loss = batch["loss_mask"]
    resp = batch["response_mask"]

    max_seq_len = seq.shape[1]
    max_resp_len = loss.shape[1]
    # response_mask / loss_mask are sized to max_resp_len and represent the
    # LAST max_resp_len positions of `sequences`.
    assert resp.shape == loss.shape

    for row_idx, example in enumerate(short + long):
        n_seq = len(example.sequences)
        n_resp = len(example.response_mask)
        left_pad = max_seq_len - n_seq
        right_pad_in_mask = max_resp_len - n_resp

        # Attention mask: 0 on the left pads, 1 on every actual sequence token.
        assert attn[row_idx, :left_pad].sum().item() == 0
        assert attn[row_idx, left_pad:].sum().item() == n_seq

        # Loss / response mask: leading positions zeroed for short examples,
        # the last n_resp positions set to 1.
        assert loss[row_idx, :right_pad_in_mask].sum().item() == 0
        assert loss[row_idx, right_pad_in_mask:].sum().item() == n_resp
        assert resp[row_idx, :right_pad_in_mask].sum().item() == 0
        assert resp[row_idx, right_pad_in_mask:].sum().item() == n_resp

        # The last `n_resp` sequence positions should be the response_ids
        # (because both sides are right-aligned).
        response_ids = example.sequences[-n_resp:]
        assert seq[row_idx, -n_resp:].tolist() == response_ids

        # And the left-most `left_pad` sequence positions are pad tokens.
        if left_pad > 0:
            assert (seq[row_idx, :left_pad] == tokenizer.pad_token_id).all().item()


def test_messages_to_sft_examples_skips_when_chat_template_breaks_prefix() -> None:
    """If a custom chat template breaks the ``full.startswith(prompt)`` invariant
    we must skip the assistant turn (otherwise the loss mask would land on the
    wrong tokens) and log a warning so the operator notices."""
    from loguru import logger as loguru_logger

    tokenizer = _ToyTokenizer()

    # Monkey-patch render_chat_template AT THE CALL SITE
    # (openfinai_skyrl.train.dataset) — that's where messages_to_sft_examples
    # actually reads the name from, after the runtime/ flatten.
    import openfinai_skyrl.train.dataset as dataset_mod

    real_render = dataset_mod.render_chat_template

    def _bad_render(_tok, _msgs, *, add_generation_prompt):
        if add_generation_prompt:
            return "PREFIX_A:\n"
        return "PREFIX_B:\nresponse\n"  # does NOT start with "PREFIX_A:\n"

    dataset_mod.render_chat_template = _bad_render
    captured: list[str] = []
    handler_id = loguru_logger.add(lambda m: captured.append(str(m)), level="WARNING")
    try:
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        examples = messages_to_sft_examples(messages, tokenizer, max_length=256)
    finally:
        loguru_logger.remove(handler_id)
        dataset_mod.render_chat_template = real_render

    assert examples == []
    assert any("prefix mismatch" in entry for entry in captured)


def test_resolve_policy_worker_uses_fsdp_module() -> None:
    worker = _resolve_policy_worker("fsdp2")
    module = importlib.import_module("skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker")
    assert worker is module.PolicyWorker


def test_resolve_policy_worker_errors_cleanly_without_megatron(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "skyrl.backends.skyrl_train.workers.megatron.megatron_worker":
            raise ModuleNotFoundError("megatron")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    with pytest.raises(RuntimeError, match="strategy=megatron requires SkyRL Megatron dependencies"):
        _resolve_policy_worker("megatron")
