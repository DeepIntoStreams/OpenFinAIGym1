# `openfinai_skyrl/` — SFT (and RL) integration for OpenFinGym

This package is the training half of OpenFinGym. It takes
trajectories produced by `openfinai_harbor` (the inference / scoring
half) and turns them into a fine-tuned model that can be slotted back
into the same agents at inference time. Online RL is integrated via
the same building blocks but its surface area is intentionally smaller
in this iteration — the supported, end-to-end audited path is
**supervised fine-tuning (SFT)**.

The shape of the package:

```
src/openfinai_skyrl/
├── README.md                       # this file
├── __init__.py
├── data/                           # trajectory + dataset prep + plots
│   ├── prepare.py                  # CLI: harbor jobs → DatasetDict (single|multi)
│   ├── splitting.py                # task-level train/test split (5 held-out tasks)
│   └── visualize.py                # diagnostic plots (train loss, per-task reward)
├── train/                          # SFT training (unified driver + flat submodules)
│   ├── driver.py                   # train(config, mode={single|multi})
│   ├── config.py                   # SkyRL train-config builder + SFTRuntimeConfig
│   ├── dataset.py                  # chat-history → SFTExample + collation
│   ├── loop.py                     # training loop + policy dispatch factory
│   ├── manifest.py                 # git SHA / config hash / system-prompt snapshot
│   ├── ray_setup.py                # Ray init + SkyRL/Ray compatibility patch
│   ├── tokenization.py             # chat-template + loss-mask helpers
│   └── policy.py                   # single-process LocalPolicyDispatch
├── eval/                           # checkpoint evaluators
│   ├── common.py                   # shared helpers (read_reward, AGENT_BY_MODE, ...)
│   ├── token_loss.py               # CLI: avg per-token CE loss on a split
│   ├── baseline.py                 # zero-shot: base model inside target agent on 5 held-out tasks
│   └── trained.py                  # SFT'd ckpt inside target agent on 5 held-out tasks
├── entrypoints/                    # CLI entrypoints
│   ├── main_sft.py                 # SFT trainer dispatcher (--mode {single|multi})
│   ├── main_sft_pipeline.py        # end-to-end SFT lifecycle orchestrator
│   └── main_rl.py                  # online RL trainer (PPO/GRPO)
├── registry/
│   └── build_task_registry.py      # CLI: emit a Harbor registry.json from task globs
├── openfinai_dataset.py            # task dataset wrapper for online RL
└── openfinai_generator.py          # ReActFinAgent-compatible HarborGenerator (online RL)
```

---

## Architecture overview

Everything in this package is glue between three external systems —
**Harbor** (trial orchestration + per-task verifier), **HuggingFace
Datasets** (the SFT corpus carrier), and **SkyRL** (the LoRA / FSDP
SFT training loop). The data flow is intentionally one-way and the
package keeps the train-time prompt format identical to the
inference-time prompt format, so a checkpoint trained here can be
slotted directly back into the agents under `openfinai_harbor/`.

```
                        ┌─────────────────────────────────────────┐
                        │   openfinai_harbor.run_trial            │
                        │   (host verifier RPC + Docker agent)    │
                        └────────────────┬────────────────────────┘
                                         │
              jobs/<batch>/<task>__<uid>/agent/trajectory.json
              jobs/<batch>/<task>__<uid>/verifier/reward.json
                                         │
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │  data/prepare.py --mode {single|multi}  │
                        │  + system prompt (auto-detected from    │
                        │    trajectory → examples/prompts/*.txt) │
                        │  + filter rows whose task ∈             │
                        │    DEFAULT_TEST_TASKS (train-only)      │
                        └────────────────┬────────────────────────┘
                                         │
                          DatasetDict({"train": ...})  (no test split)
                          on disk: dataset_<mode>_task_split/
                                         │
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │  entrypoints/main_sft.py --mode <m>     │
                        │     → train/driver.train(cfg, mode=<m>) │
                        │       (train/loop.run_sft_training_loop)│
                        └────────────────┬────────────────────────┘
                                         │
                       <ckpt_path>/hf_final/ (LoRA adapter or full)
                       <ckpt_path>/train_loss.csv + .png
                       <ckpt_path>/checkpoints/step_<N>/ (interim)
                                         │
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │  eval/trained.py --mode <m>             │
                        │     for task in DEFAULT_TEST_TASKS:     │
                        │       python -m openfinai_harbor.run_trial\│
                        │         --task-dir tasks/<task>         │
                        │         --agent-import-path <agent>     │
                        └────────────────┬────────────────────────┘
                                         │
                       <eval_dir>/summary.json
                       <eval_dir>/trials/<task>/<utc>/...
```

The arrow from `eval/trained.py` back to `run_trial` is real — the
trained model is served by an external endpoint (vLLM, Ollama, OpenAI),
and `eval/trained.py` shells out to the same `run_trial` entry point
that produced the original trajectories. This is deliberate: it
guarantees the train-time and eval-time agent code paths are
byte-identical, so any gap between offline metrics and real reward is
attributable to the model alone.

---

## Per-module purpose

### `data/prepare.py`
The single CLI for turning Harbor trial output (`agent/trajectory.json`
or the legacy `agent/conversation.json`) into a HuggingFace
`DatasetDict({"train": ...})` — train-only. Rows whose task is in
`DEFAULT_TEST_TASKS` are dropped at split time, so the held-out set is
filtered out implicitly rather than written as a `test` split.
Parameterised by `--mode {single, multi}`:
- `single` extracts the last accepted `<python>` block per trial and
  emits one `(system, user, assistant<python>)` row.
- `multi` preserves the entire ReAct history (system + N user/assistant
  turns) so the model sees the back-and-forth.

The system prompt is sourced from each trajectory's recorded `system`
step verbatim (typically the contents of a file under
`examples/prompts/` composed with the agent's `MODE_PROMPT_ADDENDUM` at
collection time). When every trajectory shares the same recorded
prompt and it matches a file under `examples/prompts/`, `prepare.py`
auto-detects and records that path in `dataset_manifest.json` so
`eval-baseline` and `eval-trained` pick it up without an extra flag.
Pass `--system-prompt-path file.txt` to relabel the corpus with a
different domain prompt. The CLI also writes a `dataset_manifest.json`
and a small `reports/dataset_sanity.{json,md}` for human inspection.

Per-trial filters: rows with `reward is None` are dropped
(`n_missing_reward`); rows with `reward <= 0.0` are dropped as the
verifier failure sentinel (`n_zero_reward_sentinel`); optional
`--reward-max` and `--top-k-lowest-reward N` further trim the corpus.

### `data/splitting.py`
Enforces the train-only contract: `build_train_only_corpus(rows, ...)`
drops every row whose task is in `DEFAULT_TEST_TASKS` (the canonical 5
held-out tasks — see below), normalises the remaining task names via
`normalize_exported_task_name` (Harbor sometimes truncates them), and
returns `DatasetDict({"train": ...})`. The dropped count is recorded
in the split manifest so an accidental collection on a test task is
visible. There is no on-disk test split; `eval/baseline.py` and
`eval/trained.py` consume `DEFAULT_TEST_TASKS` directly to know which
tasks to score against.

### `data/visualize.py`
Plotting helpers. Two surfaces:
1. **Per-run helper** — `write_train_loss_artifacts(...)` is called
   from `train/loop.py` so every training run drops its
   `train_loss.csv` + `train_loss.png` right next to its manifest, no
   follow-up command required.
2. **Multi-variant CLI / pipeline stage** — `python -m
   openfinai_skyrl.data.visualize --experiment-root <root>` (also
   wired into the orchestrator's `visualize` stage) scans
   `<root>/runs/<variant>/...` plus `<root>/baseline/...` and emits
   comparison plots across variants: training curve, per-task reward,
   lift over base LLM, base vs trained per task, and the token
   dashboard. Outputs land under `<root>/report/plots/`.

### `train/driver.py`
Unified SFT driver. `train(config, mode={single|multi})` records the
per-mode constants from `MODE_SPECS` (agent class +
expected_output_format) into the per-run `manifest.json` so
`eval/trained.py` can pick the correct agent class without re-asking
the operator. `messages_to_sft_examples` in `train/dataset.py` treats
per-row data as opaque: it iterates messages and emits one
`SFTExample` per assistant turn, so a multi-turn row yields N
examples vs. 1 for a single-turn row.

### `train/config.py`
SkyRL training-config construction. `build_train_config(raw)` maps an
OmegaConf-parsed dict into `(SkyRLTrainConfig, SFTRuntimeConfig)`.

### `train/dataset.py`
Chat-history → SFT example construction. `build_sft_examples(...)`
loads the DatasetDict split, renders the chat template per row, and
emits one `SFTExample` per assistant turn.  `collate_sft_batch(...)`
stacks examples into a left-padded `TrainingInputBatch` (right-aligned
response tokens — see the "Loss masking" section below).

### `train/loop.py`
The SFT training loop itself. `build_policy_dispatch(...)` instantiates
`LocalPolicyDispatch` (when `strategy=local`) or the Ray/FSDP backend;
`run_sft_training_loop(...)` runs the loop, logs per-step metrics, and
calls `save_checkpoint()` / `save_final_hf_export()` at the right
moments.

### `train/manifest.py`
Run-metadata helpers: `resolve_git_sha()`, `hash_config()`, and
`snapshot_dataset_system_prompt()`. Used to populate the per-run
`manifest.json`.

### `train/ray_setup.py`
`initialize_local_ray(...)` plus a SkyRL 0.1.0 / new-Ray compatibility
patch that runs at import time. Imported first by `config.py`,
`dataset.py`, `loop.py`, and `policy.py` so the patch is in place
before any `skyrl.*` import resolves.

### `train/tokenization.py`
Tokenizer + chat-template + config-parsing helpers. The two
non-trivial pieces:
- `render_chat_template(...)` falls back to a tiny `ROLE:\ncontent\n`
  template when the tokenizer has none, so unit tests can run with
  toy tokenizers.
- `tokenize_rendered_text(...)` handles the BOS-already-prepended
  case for templates (notably Llama 3.x) that bake
  `<|begin_of_text|>` into the rendered string as a literal special
  token. See the loss-masking section below for why this matters.

### `train/policy.py`
Single-process (non-Ray, non-FSDP) policy dispatch. `LocalPolicyDispatch`
is a drop-in for the `SFTDispatch` protocol, useful on a single GPU
where you want to skip Ray/FSDP entirely. The PEFT save path patches
`base_model_name_or_path` to the canonical HuggingFace repo id rather
than the (host-local) HF cache snapshot, so the saved adapter is
portable across machines.

### `eval/baseline.py`
Zero-shot evaluator. Boots the **base** model (no SFT adapter) inside
the target agent against the 5 held-out tasks via `openfinai_harbor.run_trial`.
Mostly a thin wrapper around `eval/trained.py::run_eval` so the
baseline and trained numbers are produced by the exact same code path.

### `eval/trained.py`
The fine-tuned-checkpoint evaluator. For each of the 5 held-out test
tasks, shells out to:

```
python -m openfinai_harbor.run_trial \
    --task-dir tasks/<task> \
    --agent-import-path openfinai_harbor.agents:{SingleShotLLMAgent,ReActFinAgent} \
    --provider <vllm|ollama|openai> \
    --model <served-model-id> \
    --trials-dir <eval_dir>/trials/<task>
```

`run_trial` itself pre-spawns the per-task host verifier RPC server.
The trained adapter is served behind that endpoint by the operator
(vLLM with `--enable-lora`, by an externally launched vLLM, or by the
orchestrator's `--manage-vllm` mode that runs vLLM around the eval
stage). Per-task reward is read from `verifier/reward.json`; a single
top-level `summary.json` is written (carrying per-task `reward_mean`
/ `reward_std` / `success_rate` / `seeds[]` when `--n-seeds > 1`).
Per-trial agent + verifier dirs under
`output_dir/trials/<task>/<utc>/` remain the source of truth.

### `eval/token_loss.py`
Offline token-level loss evaluator. Loads a DatasetDict split,
tokenises with the shared helpers, and reports average per-token
cross-entropy loss. Useful as a fast, GPU-cheap sanity check — does
NOT require running the full agent.

### `entrypoints/main_sft.py`
Top-level SFT CLI. Pops `--mode {single,multi}` out of argv, then
forwards the rest verbatim into `openfinai_skyrl.train.tokenization.parse_cli`
(OmegaConf-style `KEY=value` overrides) and dispatches to
`openfinai_skyrl.train.driver.train(config, mode=...)`.

### `entrypoints/main_sft_pipeline.py`
End-to-end SFT pipeline orchestrator. Runs
`collect → prepare → eval-baseline → train → auto-merge → eval-trained
→ visualize` in sequence. Each stage can be skipped via
`--skip-stages`, or selected via `--only-stages`. With
`--manage-vllm`, the orchestrator launches a vLLM server for the base
model around `eval-baseline` and for the merged checkpoint around
`eval-trained`, tearing it down between stages. `auto-merge`
materialises `<ckpt>/hf_merged/` from base + LoRA adapter (no-op for
full-param checkpoints). Token-loss evaluation lives in
`eval/token_loss.py` and is run separately when wanted, not as a
pipeline stage.

### `entrypoints/main_rl.py`
Online RL trainer for PPO/GRPO. Less heavily exercised than SFT in
this iteration. Uses `OpenFinAIGenerator` (wrapping SkyRL's
`HarborGenerator`) + `OpenFinAITaskDataset` to drive Harbor trials as
RL environments.

---

## CLI cheat sheet

End-to-end commands. Default experiment root is
`data/run_output/experiments-sft`; substitute your own model id where
the examples use `Qwen/Qwen3-4B`. For the fully orchestrated path use
`scripts/sft/run_sft_pipeline.sh` (see `README.SFT.md`).

| Step | Command |
|---|---|
| 1. Collect trajectories | `bash scripts/sft/collect_trajectories.sh` (or `python -m openfinai_harbor.run_trial --task-dir tasks/<task> --agent-import-path openfinai_harbor.agents:SingleShotLLMAgent --provider openrouter --model openai/gpt-5.4-mini --system-prompt-path examples/prompts/financial_competitive.txt --trials-dir data/run_output/jobs/<batch>`) |
| 2a. Prep dataset (single) | `python -m openfinai_skyrl.data.prepare --mode single --jobs-dir data/run_output/jobs/<batch> --tasks-root tasks --output-root data/run_output/experiments-sft` |
| 2b. Prep dataset (multi)  | `python -m openfinai_skyrl.data.prepare --mode multi  --jobs-dir data/run_output/jobs/<batch> --tasks-root tasks --output-root data/run_output/experiments-sft` |
| 3a. SFT train (single) | `python -m openfinai_skyrl.entrypoints.main_sft --mode single --config config/sft.yaml dataset_name=data/run_output/experiments-sft/data/dataset_single_task_split model.path=Qwen/Qwen3-4B ckpt_path=data/run_output/experiments-sft/runs/sft_single` |
| 3b. SFT train (multi)  | `python -m openfinai_skyrl.entrypoints.main_sft --mode multi  --config config/sft.yaml dataset_name=data/run_output/experiments-sft/data/dataset_multi_task_split  model.path=Qwen/Qwen3-4B ckpt_path=data/run_output/experiments-sft/runs/sft_multi` |
| 4a. Baseline eval | `python -m openfinai_skyrl.eval.baseline --mode single --provider openai --model qwen3_4b_base --tasks-root tasks --dataset-dir data/run_output/experiments-sft/data --output-dir data/run_output/experiments-sft/baseline/base_qwen3_4b_base --n-seeds 5` |
| 4b. Token-loss eval | `python -m openfinai_skyrl.eval.token_loss --model-path Qwen/Qwen3-4B --adapter-dir data/run_output/experiments-sft/runs/sft_single/hf_final --dataset-dir data/run_output/experiments-sft/data/dataset_single_task_split --dataset-split train --max-length 10240 --output-json data/run_output/experiments-sft/runs/sft_single/eval/token_loss/train.json` |
| 4c. Trained eval | `python -m openfinai_skyrl.eval.trained --mode single --checkpoint-dir data/run_output/experiments-sft/runs/sft_single --provider openai --model qwen3_4b_lora_r16 --tasks-root tasks --output-dir data/run_output/experiments-sft/runs/sft_single/eval --n-seeds 5` |
| 5. Multi-variant plots | `python -m openfinai_skyrl.data.visualize --experiment-root data/run_output/experiments-sft` |
| Orchestrator | `MANAGE_VLLM=true bash scripts/sft/run_sft_pipeline.sh` (drives every stage end-to-end; reads `config/sft.yaml`) |
| Shell wrapper | `MODE=single bash scripts/sft/train_sft.sh` (single-stage train, forwards env vars into the entrypoint) |

Each per-run dir auto-emits `train_loss.{csv,png}` during step 3 and a
`summary.json` during step 4c, so step 5 is only needed when comparing
multiple variants under a common experiment root.

---

## Output artefacts layout

After `train_sft.sh` + `eval/trained.py` finish, each per-run dir is
flat and self-describing:

```
<output-root>/<experiment-name>/                       # = ckpt_path argument
├── manifest.json                  # config + variant + git_sha + model + dataset + system_prompt sha
├── train_metrics.jsonl            # one JSON row per training step (raw)
├── train_loss.csv                 # step,loss,grad_norm,response_length,lr
├── train_loss.png                 # the training-loss curve
├── hf_final/                      # final HF / PEFT export (load this for inference)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer.json + tokenizer_config.json
├── hf_merged/                     # base + adapter merged (LoRA only; written by auto-merge stage)
├── checkpoints/                   # mid-training snapshots (only if ckpt_interval>0)
│   └── step_000050/
│       └── hf_model/...
└── eval/                          # produced by eval/trained.py (--output-dir <root>/eval)
    ├── summary.json               # n_tasks, n_seeds, success_rate, avg_reward, per-task seeds
    ├── <task-name>.log            # stdout/stderr of run_trial for each task (n_seeds==1)
    ├── <task-name>__seed<S>.log   # one log per seed when n_seeds>1
    └── trials/<task-name>/<utc>/  # Harbor trial dir (agent/, verifier/, result.json)
        ├── agent/trajectory.json
        ├── agent/conversation.json
        ├── verifier/reward.json
        └── result.json
```

The dataset-prep artefacts live under `<output-root>/data/` and
`<output-root>/reports/`:

```
<output-root>/
├── data/
│   ├── dataset_single_task_split/         # HuggingFace DatasetDict on disk (train only)
│   │   ├── train/                         # arrow-format split
│   │   └── train.jsonl                    # human-readable preview
│   ├── dataset_multi_task_split/          # same shape, multi-turn rows
│   └── dataset_manifest.json              # mode, prompt path, counts, dropped-test-task tally
└── reports/
    ├── dataset_sanity.json
    └── dataset_sanity.md
```

---

## Single-turn vs multi-turn semantics

**Single-turn** = one `(system, user, assistant<python>)` row per
trial. The assistant content is the LAST `<python>...</python>` block
the agent successfully had accepted by the in-container verifier
during its trial. Targets `SingleShotLLMAgent`, which parses one
`<python>` block from the model response and atomically writes it to
`/workspace/train.py` inside the Harbor environment container. The
agent never iterates — one prompt, one response, the response IS the
deliverable.

**Multi-turn** = one row per trial that carries the FULL ReAct
history: `system + (user instruction → assistant<python>/<answer>) ×
N`. Targets `ReActFinAgent`, which runs the multi-step ReAct loop
with `<python>` / `<answer>` tags and an in-container exec server,
gating the next turn on the observation from the previous one.

The split is enforced in two places:
- `data/prepare.py::load_messages_from_trial` decides the message
  layout based on `--mode`. Single-turn collapses to two messages
  (user + assistant), prepends system; multi-turn preserves all
  rounds, prepends system.
- `train/dataset.py::messages_to_sft_examples` emits **one example
  per assistant turn** regardless of mode. Single-turn rows therefore
  produce 1 example each; multi-turn rows produce N (one per
  assistant turn in the row).

`SingleShotLLMAgent` and `ReActFinAgent` are the matching
inference-time agents because they install the same on-disk system
prompt that the dataset rows were rendered with, parse the same
`<python>` tag format, and (for multi-turn) gate exactly the same way
on observation text — so a fine-tuned checkpoint round-trips
losslessly between train and inference.

---

## 5 held-out test tasks

`DEFAULT_TEST_TASKS` in `data/splitting.py`:

| task | asset family |
|---|---|
| `1_commodity_logreturn_forecasting` | Commodity panel log-return |
| `1_crypto_logreturn_forecasting` | Crypto panel log-return |
| `1_equity_logreturn_forecasting` | Equity panel log-return |
| `1_treasury_variant1_forecasting` | Treasury / rates |
| `1_fx_variant1_logreturn_forecasting` | FX log-return |

One representative task per asset family (commodity / crypto / equity
/ treasury / fx) so the held-out signal isn't dominated by any single
family. All five are forecasting tasks under the current contract —
generation tasks live under `tasks/2_*` but are not held out by
default; if you want a generation task in the eval set, override with
`--test-tasks` (and accept that avg-reward numbers stop being directly
comparable across runs that picked different sets).

---

## Loss masking + chat template assumptions

The SFT loss is supervised ONLY on assistant tokens — system + user
prompts are zeroed out of the loss mask. Two non-obvious things make
this work, both regression-tested in `tests/test_sft_runtime.py`:

**1. BOS token deduplication** (`train/tokenization.py:184-191`).
Llama 3.x's chat template bakes `<|begin_of_text|>` into the
rendered string as a literal special token, so when we tokenise the
rendered text and then naively prepend `bos_token_id`, every example
starts with TWO BOS tokens and the loss mask shifts by one. Fix:
inspect the first token of `tokenizer(text)` and skip the prepend
when it already matches. This branch is preserved exactly as Agent A
left it and is covered by `test_tokenize_rendered_text_does_not_double_prepend_bos`.

```python
ids = tokenizer(text, add_special_tokens=False)["input_ids"]
if add_bos:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        ids_list = list(ids)
        if ids_list and ids_list[0] == int(bos_token_id):
            return ids_list
        return [int(bos_token_id)] + ids_list
return list(ids)
```

**2. Right-aligned response mask** (`train/dataset.py::collate_sft_batch`,
`collate_sft_batch`). Sequences are LEFT-padded (so the response
tokens land at the right edge of every row); SkyRL's `HFModelWrapper`
then slices `log_probs[:, -num_actions-1:-1]` — i.e. the right-most
`num_actions = max_response_len` action positions. That means
`response_mask` / `loss_mask` must ALSO be left-padded with the 1s
flush to the RIGHT. If they were right-padded with 0s (the original
bug Agent A fixed), the mask 1s would end up over prompt-tail tokens
for any example shorter than `max_response_len`, and loss would be
applied to the wrong tokens. Preserved verbatim:

```python
# SkyRL's HFModelWrapper slices `log_probs[:, -num_actions-1:-1]`
# — i.e. the RIGHT-most `num_actions = max_response_len` action
# positions. So response_mask / loss_mask must also be LEFT-padded
# with the 1s flush to the RIGHT. If instead we right-pad with 0s
# (the original bug), the mask 1s end up over prompt-tail tokens
# for any example shorter than max_response_len → loss applied to
# wrong tokens.
response_masks.append([0] * right_pad + example.response_mask)
loss_masks.append([0] * right_pad + example.loss_mask)
```

There is also a **chat-template prefix-mismatch guard** in
`messages_to_sft_examples`: if rendering `history + assistant` does
NOT extend the rendering of `history` alone as a strict string prefix
(custom Jinja templates that reorder turns or inject whitespace
differently), we log a `WARNING` and skip the assistant turn rather
than emit a misaligned mask. See
`test_messages_to_sft_examples_skips_when_chat_template_breaks_prefix`.

---

## Pitfalls to avoid

- **Do not change the system prompt path** without also re-running
  `data/prepare.py`. The trained model sees the prompt baked into
  every row, so a swap at inference time silently shifts the
  distribution.
- **vLLM LoRA serving** requires `--enable-lora` and matching
  `--max-loras` / `--max-lora-rank`. The adapter PEFT writes is
  portable (we patch `base_model_name_or_path` in
  `train/policy.py::LocalTrainingStrategy.save_hf_model`), but the
  served-model id you pass via `--model` to `eval/trained.py` MUST
  match what the server has registered.
- **batch_size must be divisible by total GPUs** for the Ray / FSDP
  backends (see the validation in
  `runtime.build_train_config`). `strategy=local` only supports
  single GPU.
- **5 held-out tasks** is the contract. If you override
  `DEFAULT_TEST_TASKS` (via the `--test-tasks` flag), be aware that
  any task you move INTO the test set is removed from training, and
  the avg-reward numbers across runs are no longer apples-to-apples.
