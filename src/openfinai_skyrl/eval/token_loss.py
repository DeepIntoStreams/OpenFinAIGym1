"""Offline per-token cross-entropy loss evaluator for base or LoRA checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM

from openfinai_skyrl.train.dataset import build_sft_examples, collate_sft_batch
from openfinai_skyrl.train.loop import load_tokenizer
from openfinai_skyrl.train.tokenization import hf_load_kwargs, resolve_local_hf_path
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base model path or HF repo id.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="LoRA adapter dir from hf_final export; mutually exclusive with --checkpoints-root.",
    )
    parser.add_argument(
        "--checkpoints-root",
        default=None,
        help="Sweep <root>/step_<N>/ subdirs; results written to --output-dir/step_<N>.json.",
    )
    parser.add_argument("--output-json", default=None, help="Single-adapter output.")
    parser.add_argument("--output-dir", default=None, help="Sweep-mode output dir.")
    return parser.parse_args()


def load_wrapped_model(model_path: str, adapter_dir: str | None, device: torch.device) -> HFModelWrapper:
    resolved_model_path = resolve_local_hf_path(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        **hf_load_kwargs(model_path),
    )
    if adapter_dir:
        base_model = PeftModel.from_pretrained(base_model, adapter_dir)
    base_model.to(device)
    base_model.eval()
    return HFModelWrapper(base_model, use_flash_attention_2=False, bf16=device.type == "cuda")


def _score_one(
    model_path: str,
    adapter_dir: str | None,
    dataset_dir: str,
    dataset_split: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    tokenizer,
    examples,
) -> dict[str, Any]:
    wrapper = load_wrapped_model(model_path, adapter_dir, device)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_examples = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = collate_sft_batch(batch_examples, tokenizer)
            sequences = batch["sequences"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            response_length = batch.metadata["response_length"]
            action_log_probs = wrapper(
                sequences,
                response_length,
                attention_mask=attention_mask,
                temperature=1.0,
                return_output=False,
                compute_entropy=False,
            )
            batch_loss = -(action_log_probs * loss_mask).sum()
            batch_tokens = loss_mask.sum()
            total_loss_sum += float(batch_loss.item())
            total_tokens += float(batch_tokens.item())
            total_examples += len(batch_examples)
    return {
        "model_path": model_path,
        "adapter_dir": adapter_dir,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "dataset_split": dataset_split,
        "max_length": max_length,
        "batch_size": batch_size,
        "n_examples": total_examples,
        "n_batches": ceil(total_examples / batch_size) if total_examples else 0,
        "n_tokens": total_tokens,
        "loss_sum": total_loss_sum,
        "avg_token_loss": (total_loss_sum / total_tokens) if total_tokens else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if bool(args.adapter_dir) and bool(args.checkpoints_root):
        raise SystemExit("--adapter-dir and --checkpoints-root are mutually exclusive.")
    if args.checkpoints_root and not args.output_dir:
        raise SystemExit("--checkpoints-root requires --output-dir.")
    if not args.checkpoints_root and not args.output_json:
        raise SystemExit("Either --output-json (single mode) or --output-dir (sweep mode) is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Token-loss eval ===", flush=True)
    print(f"Model:      {args.model_path}", flush=True)
    print(f"Dataset:    {Path(args.dataset_dir).resolve()} [split={args.dataset_split}]", flush=True)
    print(f"Batch:      {args.batch_size}", flush=True)
    print(f"MaxLength:  {args.max_length}", flush=True)
    print(f"Device:     {device}", flush=True)

    tokenizer = load_tokenizer(args.model_path)
    examples = build_sft_examples(
        dataset_name=args.dataset_dir,
        dataset_split=args.dataset_split,
        messages_key="messages",
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if args.checkpoints_root:
        ckpt_root = Path(args.checkpoints_root)
        step_dirs = sorted(p for p in ckpt_root.glob("step_*") if p.is_dir())
        if not step_dirs:
            raise SystemExit(f"No step_* dirs under {ckpt_root}.")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for step_dir in step_dirs:
            try:
                step = int(step_dir.name.replace("step_", ""))
            except ValueError:
                continue
            print(f"\n--- step={step} adapter={step_dir} ---", flush=True)
            metrics = _score_one(
                args.model_path, str(step_dir),
                args.dataset_dir, args.dataset_split,
                args.max_length, args.batch_size,
                device, tokenizer, examples,
            )
            metrics["step"] = step
            out_path = out_dir / f"step_{step:06d}.json"
            out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Result:     avg_token_loss={metrics['avg_token_loss']:.6f} -> {out_path}", flush=True)
            rows.append(metrics)
        import csv as _csv
        with (out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as h:
            w = _csv.DictWriter(h, fieldnames=["step", "avg_token_loss", "n_tokens", "n_examples"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        logger.info("Sweep done; wrote {} per-step JSONs + sweep.csv under {}", len(rows), out_dir)
        return

    print(f"Adapter:    {args.adapter_dir or 'base_model'}", flush=True)
    metrics = _score_one(
        args.model_path, args.adapter_dir,
        args.dataset_dir, args.dataset_split,
        args.max_length, args.batch_size,
        device, tokenizer, examples,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Result:     examples={metrics['n_examples']} batches={metrics['n_batches']} "
        f"tokens={int(metrics['n_tokens'])} avg_token_loss={metrics['avg_token_loss']:.6f}",
        flush=True,
    )
    print(f"Output:     {args.output_json}", flush=True)
    logger.info("Saved token-loss metrics to {}", args.output_json)


if __name__ == "__main__":
    main()
