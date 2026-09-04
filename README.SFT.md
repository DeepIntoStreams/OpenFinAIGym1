# SFT Workflow in OpenFinGym

Supervised fine-tuning workflow: Harbor trajectory collection -> SkyRL SFT
artifacts under `data/run_output/experiments-sft/`.

Flow:

1. Collect Harbor trajectories
2. Convert trajectories into a DatasetDict
3. Train an SFT checkpoint with SkyRL
4. Optionally evaluate the trained checkpoint

## Scope

Two SFT targets:

- `single` mode: trains a model for `SingleShotLLMAgent`
- `multi` mode: trains a model for `ReActFinAgent`

Both share the trajectory source and training runtime; they differ only in
how each Harbor trial is converted into messages.

## Important Paths

Core entrypoints:

- [scripts/sft/collect_trajectories.sh](scripts/sft/collect_trajectories.sh)
- [scripts/sft/train_sft.sh](scripts/sft/train_sft.sh)
- [config/sft.yaml](config/sft.yaml)
- [src/openfinai_skyrl/data/prepare.py](src/openfinai_skyrl/data/prepare.py)
- [src/openfinai_skyrl/entrypoints/main_sft.py](src/openfinai_skyrl/entrypoints/main_sft.py)

Training drivers:

- [src/openfinai_skyrl/train/driver.py](src/openfinai_skyrl/train/driver.py)
- [src/openfinai_skyrl/train/config.py](src/openfinai_skyrl/train/config.py)
- [src/openfinai_skyrl/train/dataset.py](src/openfinai_skyrl/train/dataset.py)
- [src/openfinai_skyrl/train/loop.py](src/openfinai_skyrl/train/loop.py)
- [src/openfinai_skyrl/train/manifest.py](src/openfinai_skyrl/train/manifest.py)
- [src/openfinai_skyrl/train/ray_setup.py](src/openfinai_skyrl/train/ray_setup.py)
- [src/openfinai_skyrl/train/policy.py](src/openfinai_skyrl/train/policy.py)
- [src/openfinai_skyrl/train/tokenization.py](src/openfinai_skyrl/train/tokenization.py)

Harbor execution path:

- [src/openfinai_harbor/run_trial.py](src/openfinai_harbor/run_trial.py)
- [config/harbor_trial.yaml](config/harbor_trial.yaml)

## Output Layout

Default experiment root: `data/run_output/experiments-sft/`.

```text
data/run_output/experiments-sft/
├── data/
│   ├── dataset_single_task_split/
│   ├── dataset_multi_task_split/
│   └── dataset_manifest.json
├── reports/
│   ├── dataset_sanity.json
│   └── dataset_sanity.md
└── runs/
    ├── sft_single/
    │   ├── manifest.json
    │   ├── train_metrics.jsonl
    │   ├── train_loss.csv
    │   ├── train_loss.png
    │   ├── checkpoints/
    │   └── hf_final/
    └── sft_multi/
```

`prepare.py` writes `data/` and `reports/`; `train_sft.sh` writes `runs/<variant>/`.

## Step 1: Collect Trajectories

Driven by [scripts/sft/collect_trajectories.sh](scripts/sft/collect_trajectories.sh),
which wraps `python -m openfinai_harbor.run_trial` (not bare `harbor run`)
so the host verifier is pre-spawned and stays alive for the attempt.

Defaults: tasks read from `registry.json`, filtered to `1_*`, multiple
attempts per task, outputs under `jobs/collect_trajectories/`.

Basic usage:

```bash
bash scripts/sft/collect_trajectories.sh
```

Common overrides:

```bash
MODEL=claude-cli/opus \
N_ATTEMPTS=8 \
N_CONCURRENT=2 \
MAX_TURNS=10 \
TEMPERATURE=0.8 \
TRIALS_DIR=jobs/my_sft_collection \
bash scripts/sft/collect_trajectories.sh
```

Important collection environment variables:

- `MODEL`: model identifier passed to the Harbor agent
- `AGENT_IMPORT_PATH`: agent class import path
- `N_ATTEMPTS`: number of trials per task
- `N_CONCURRENT`: per-task concurrency
- `MAX_TURNS`: only relevant for multi-turn agents like `ReActFinAgent`
- `TEMPERATURE`: collection-time sampling temperature
- `TIMEOUT_SEC`: max wall time for one agent step
- `TRIALS_DIR`: Harbor output root

Typical per-trial contents:

```text
jobs/<batch-or-run>/<trial-id>/
├── agent/
│   ├── trajectory.json
│   ├── conversation.json
│   └── llm_response.txt
├── verifier/
│   ├── reward.json
│   ├── test-stdout.txt
│   └── test-stderr.txt
├── config.json
├── result.json
└── trial.log
```

For SFT, the most important files are:

- `agent/trajectory.json`
- `agent/conversation.json`
- `verifier/reward.json`

## Step 2: Prepare the SFT Dataset

CLI: `python -m openfinai_skyrl.data.prepare --mode single ...`
(see [prepare.py](src/openfinai_skyrl/data/prepare.py))

Scans Harbor trial dirs, prefers `agent/trajectory.json` (falls back to
`agent/conversation.json`), reads reward from `verifier/reward.json` /
`reward.txt` / `result.json`, applies task-level train/test split, writes
a `DatasetDict`.

### Single-turn dataset

For `SingleShotLLMAgent`. Each trial becomes one row: system prompt + first
user message + final accepted assistant `<python>...</python>` answer.

```bash
python -m openfinai_skyrl.data.prepare \
  --mode single \
  --jobs-dir jobs/2026-05-18-claudecli_singleshot \
  --tasks-root tasks \
  --output-root data/run_output/experiments-sft \
  --reward-filter success
```

### Multi-turn dataset

For `ReActFinAgent`. Preserves full turn history.

```bash
python -m openfinai_skyrl.data.prepare \
  --mode multi \
  --jobs-dir jobs/collect_trajectories \
  --tasks-root tasks \
  --output-root data/run_output/experiments-sft \
  --reward-filter success
```

### Important prepare arguments

- `--mode`: `single` or `multi`
- `--jobs-dir`: Harbor jobs root, or a single trial directory
- `--job-prefix`: optional directory-name filter
- `--tasks-root`: task root used for task-name normalization
- `--output-root`: experiment root, now defaulting to `data/run_output/experiments-sft`
- `--reward-filter`: `success`, `failure`, or `all`
- `--test-tasks`: optional explicit held-out task set
- `--system-prompt-path`: optional override for the prompt baked into every row

### Produced dataset artifacts

Under `<output-root>/data/`:

- `dataset_single_task_split/` or `dataset_multi_task_split/`
- `dataset_manifest.json`

Under `<output-root>/reports/`:

- `dataset_sanity.json`
- `dataset_sanity.md`

## Step 3: Configure SFT

[config/sft.yaml](config/sft.yaml) is the source
of truth; [train_sft.sh](scripts/sft/train_sft.sh)
passes `--config config/sft.yaml` unless you override `CONFIG_PATH`.

### Keys in `sft.yaml`

- Model/LoRA: `model.path`, `model.lora.{rank,alpha,target_modules}`
- Dataset: `dataset_name`, `dataset_split`, `messages_key`
- Training: `strategy`, `num_steps`, `batch_size`, `micro_train_batch_size_per_gpu`, `max_length`, `seed`
- Optimizer: `optimizer_config.{lr,weight_decay,num_warmup_steps}`
- Output: `ckpt_path`, `ckpt_interval`, `max_ckpts_to_keep`, `logger`, `project_name`, `run_name`
- Placement: `placement.{num_nodes,num_gpus_per_node}`

`logger: console` keeps metadata in the run manifest under `ckpt_path`.

## Step 4: Run SFT

[train_sft.sh](scripts/sft/train_sft.sh) calls
[main_sft.py](src/openfinai_skyrl/entrypoints/main_sft.py),
which forwards `--mode {single|multi}` to the unified
[driver.train](src/openfinai_skyrl/train/driver.py).

### Default single-turn run

```bash
bash scripts/sft/train_sft.sh
```

### Multi-turn run

```bash
MODE=multi \
DATASET_DIR=data/run_output/experiments-sft/data/dataset_multi_task_split \
CKPT_DIR=data/run_output/experiments-sft/runs/sft_multi \
bash scripts/sft/train_sft.sh
```

### Example overrides

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct bash scripts/sft/train_sft.sh
LORA_RANK=16 LORA_ALPHA=32 LORA_TARGET=all-linear bash scripts/sft/train_sft.sh
STRATEGY=local NUM_STEPS=1 BATCH_SIZE=1 MICRO_BATCH=1 MAX_LENGTH=512 bash scripts/sft/train_sft.sh
CONFIG_PATH=config/sft.yaml bash scripts/sft/train_sft.sh
```

## How Training Uses the Config

[train/tokenization.py](src/openfinai_skyrl/train/tokenization.py)
`parse_cli()` merges `--config` YAML with CLI overrides.
[train/config.py](src/openfinai_skyrl/train/config.py)
`build_train_config()` maps the merged dict into `SkyRLTrainConfig` +
`SFTRuntimeConfig`. The loop in
[train/loop.py](src/openfinai_skyrl/train/loop.py)
tokenizes message rows, builds a local or distributed policy backend,
runs, and writes checkpoints + final HF export.

## Training Outputs

Under `data/run_output/experiments-sft/runs/sft_single`:

- `manifest.json`: run metadata
- `train_metrics.jsonl`: per-step metrics
- `train_loss.csv`: tabular training curve
- `train_loss.png`: loss plot
- `checkpoints/step_000050/`: intermediate snapshots
- `hf_final/`: final HF export (LoRA adapter artifacts)

`hf_final/` is the main output used for serving and evaluation.

## Optional Evaluation

- [eval/baseline.py](src/openfinai_skyrl/eval/baseline.py): zero-shot Harbor reward
- [eval/trained.py](src/openfinai_skyrl/eval/trained.py): end-to-end Harbor reward on trained checkpoint
- [eval/token_loss.py](src/openfinai_skyrl/eval/token_loss.py): offline token-loss

## Current Recommended Workflow

Single-turn:

1. Collect with `SingleShotLLMAgent`.
2. Prepare `--mode single` into `data/run_output/experiments-sft/`.
3. Point `config/sft.yaml` at the dataset + run root.
4. `bash scripts/sft/train_sft.sh`.
5. Inspect `manifest.json`, `train_metrics.jsonl`, `hf_final/`.

Multi-turn: same flow with `ReActFinAgent`, `--mode multi`, and
`MODE=multi` + dataset/ckpt overrides at launch.

## Notes and Caveats

- `MAX_TURNS` only matters for multi-turn collection.
- `prepare.py` splits at the task level, not the trajectory level.
- System prompt is sourced from each trajectory's recorded system step by
  default — i.e. whatever the collector wrote into the trial (typically the
  contents of a file under `examples/prompts/` composed with the
  `MODE_PROMPT_ADDENDUM` in force at collection time). The agent's
  `DEFAULT_DOMAIN_PROMPT` is only a last-resort fallback when no trial
  recorded a system step. If every trajectory carries the same prompt and
  it starts with a file under `examples/prompts/`, `prepare.py` records
  that path in `dataset_manifest.json` so eval-baseline / eval-trained pick
  it up automatically; otherwise the manifest records the
  `<trajectory-recorded>` sentinel and eval falls back to the agent's
  in-script prompt — pass `--system-prompt-path file.txt` to override
  either way.
- Reward is **loss-scaled** (smaller = better);
  `--reward-filter success/failure` are no-ops on this scale. Use
  `--top-k-lowest-reward N` or `--reward-filter below --reward-threshold T`.

### `OPENFINAI_VERIFIER_URL_MODE` — host-RPC URL strategy

`openfinai_harbor.verifier.client.container_url_for` picks how to make the
host verifier reachable from inside the agent container:

| Mode | Behaviour | When to use |
|---|---|---|
| `host` (default) | URL returned unchanged. Relies on `network_mode: host` (auto-written into each task's `environment/docker-compose.yaml` by `run_trial._ensure_host_network_compose`). `127.0.0.1` inside the container IS the host loopback. | Rootless podman, plain Linux docker, Docker Desktop — works everywhere |
| `rewrite` | `127.0.0.1` / `0.0.0.0` rewritten to `host.docker.internal`. | Legacy bridge-network setups, or hand-rolled compose overrides that disable host networking |

Override at run time:
```bash
OPENFINAI_VERIFIER_URL_MODE=rewrite bash scripts/sft/collect_trajectories.sh
```

### Multi-seed eval

`eval/trained.py --n-seeds 3` runs each task three times (seeds forwarded
as `--ak seed=N`). `summary.json` then carries `reward_mean`, `reward_std`,
`reward_n`, and `seeds[]` per task. Use `>= 3` for meaningful CIs.

### Manifest fields for reproducibility

Per-run `manifest.json`:
- `config_sha256` — hash of the merged yaml + CLI config dict.
- `system_prompt` = `{chars, sha256, preview}` of the first train-split
  system prompt, for system-prompt parity checks without the corpus.
