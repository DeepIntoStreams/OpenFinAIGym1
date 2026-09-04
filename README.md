# OpenFinGym

> This is the official code repository for the paper [OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents](https://arxiv.org/pdf/2606.26350).
>
> Further development of OpenFinGym is ongoing in the [OpenFinGym2 repository](https://github.com/DeepIntoStreams/OpenFinGym2). Please refer to that repository for the latest features, updates, and future releases.

OpenFinGym transforms research problems from academic finance papers into executable benchmark tasks for evaluating agents in offline and real-time environments. It supports time-series forecasting, time-series generation, paper trading and fraud detection through a unified task pipeline and reward bank, with two complementary modes of use.

## What's in the box

- **`openfinai_pipeline`** — a multi-phase LLM pipeline that scrapes papers,
  summarises them, builds datasets and reward functions, and assembles
  Harbor-ready benchmark task packages. Source: `src/openfinai_pipeline/`.
- **`openfinai_harbor`** — lightweight agent-evaluation runtime that
  consumes the task packages produced above. Source:
  `src/openfinai_harbor/` and `tasks/`.
- **`openfinai_skyrl`** — SFT and online-RL integration (SkyRL + Harbor)
  for training agents on the benchmark. Source: `src/openfinai_skyrl/`;
  see [`README.SFT.md`](README.SFT.md) and
  [`src/openfinai_skyrl/README.md`](src/openfinai_skyrl/README.md).

## Pre-built task bundles

Pre-built forecasting and generation task bundles — covering the SFT
training tasks, the held-out test tasks, and additional bundles — are
mirrored on Google Drive:

[OpenFinGym task bundles (Google Drive)](https://drive.google.com/drive/folders/1ggE1bX2YgQlqIcRqNc9PWsticrCvVxlo?usp=sharing)

Download and unpack into `tasks/` to skip the full pipeline run when
you only want to drive agents against existing bundles.

## Install

```bash
git clone https://github.com/DeepIntoStreams/OpenFinGym1.git
cd OpenFinGym1

# Create the conda env (Python 3.12+) and activate it
conda create -n openfingym python=3.12 -y
conda activate openfingym

# Editable install — registers openfinai_pipeline, openfinai_harbor and
# openfinai_skyrl as importable packages backed by src/, and pulls every
# dependency listed in pyproject.toml.
pip install -e .

# Copy the env template and fill in at least one LLM provider credential
cp .env.example .env
```

If you already have the heavy ML dependencies (`torch`, `pandas`,
`numpy`, ...) installed and just want to register the packages, swap the
install line for `pip install -e . --no-deps`.

Poetry users can run `poetry install` instead of `pip install -e .`
— `pyproject.toml` is Poetry-managed.

GPU policy serving for online RL (vLLM, xformers, flash-attn) is an
optional extra: `pip install -e ".[rl]"` or
`pip install -r requirements-rl.txt` (Linux x86_64 + CUDA 12 only).

## Task-construction pipeline

Full task-construction pipeline — one command per phase, run from the
project root. Each phase is idempotent: outputs land on disk, re-running
picks up where the previous left off. Artefacts land under
`data/pipeline_output/` (ephemeral) and `tasks/generated/` (versioned).
Run `python -m openfinai_pipeline --help` (or `<phase> --help`) for every
flag.

A few conventions that apply to every phase below:

- **Console output.** Default shows stage markers + warnings/errors only
  (compact, no timestamps). `--verbose` / `-v` adds per-item INFO detail
  (dedup / assembly / per-step outcome). All examples use `--verbose`.
- **Forensic log.** Regardless of `--verbose`, each run also writes a
  full timestamped record to
  `data/pipeline_output/logs/<run_tag>/<practice>.log` — every stage
  marker, INFO/WARN/ERROR line, internal detail dump, and LLM
  request/response body. LLM bodies never reach the terminal. Inspect
  this file for post-mortems; no shell redirection needed.
- **Re-run skip behaviour.** A candidate (dataset / reward / task) whose
  previous attempt left a staging directory at
  `data/pipeline_output/.staging/{datasets,rewards,tasks}/<name>/` but
  produced no success record is **skipped** on the rerun — re-running
  codegen against the same candidate just burns LLM credits + network.
  Pass `--retry-failed-attempts` to force retry, or `--overwrite` (which
  resets state across the board).
- **Models.** `--model` is a global override that forces every phase to
  use the same model. Omit it to let each phase pick its own via
  `llm.per_phase.<phase>.model` in `config/base.yaml`.

### Phase 1 — Paper Harvesting

```bash
python -m openfinai_pipeline scrape \
    --scopes alpha_trading \
    --provider openrouter \
    --model openai/gpt-5.4-mini \
    --limit 50 \
    --verbose
```

### Phase 1b — Paper Summarization

```bash
python -m openfinai_pipeline summarize-paper \
    --scopes alpha_trading \
    --overwrite \
    --provider openrouter \
    --model openai/gpt-5.4-mini \
    --verbose
```

### Phase 1c — Trading Paper Triage

Per-paper LLM triage that routes trading-family papers
(`task_family` in `{trading, realtime_trading, realtime_forecasting}`)
onto the curated bundles in `tasks/<curated_id>/triage_descriptor.toml`.
Writes a `trading_triage.json` sidecar next to each `paper.json` and,
for routed papers, a thin overlay under `tasks/routed/<paper_id>/`
(curated class re-export + paper-specific `task.toml` +
`instruction.md` + `triage_record.json`). Routed papers **skip
Phases 2-4** — the curated stack already owns dataset / reward /
task plumbing for these families. Forecasting and generative papers
fall through untouched.

```bash
python -m openfinai_pipeline triage-trading-papers \
    --scopes alpha_trading \
    --routes-written-cap 5 \
    --provider openrouter \
    --model openai/gpt-5.4 \
    --verbose
```

`--routes-written-cap` is the Phase 1c analogue of
`--datasets-written-cap` / `--rewards-written-cap` / `--tasks-written-cap`:
absolute total routed/partial_match papers across the corpus,
scope-aware against existing `paperN/trading_triage.json` sidecars
in each requested scope. `no_match` and `novel_required` outcomes
write no artifact and don't consume the budget. Add `--overwrite`
to re-triage papers that already have a `trading_triage.json`
sidecar, or `--overlay-root <path>` to write routed overlays
somewhere other than `tasks/routed/`.

### Phase 2 — Dataset Construction

```bash
python -m openfinai_pipeline construct-dataset \
    --scopes alpha_trading \
    --datasets-written-cap 5 \
    --provider openrouter \
    --model openai/gpt-5.4 \
    --verbose
```

Add `--retry-failed-attempts` to force retry of dataset candidates whose
previous run left a staging directory but no catalog entry. Without it
such candidates are skipped to save LLM credits + network roundtrips.

### Phase 3 — Reward Construction

```bash
python -m openfinai_pipeline construct-rewards \
    --scopes alpha_trading \
    --rewards-written-cap 10 \
    --provider openrouter \
    --model openai/gpt-5.4 \
    --verbose
```

### Phase 4 — Task Generation

Writes the full Harbor-canonical task layout (`instruction.md`,
`task.toml`, `tests/test.sh`, `environment/Dockerfile`,
`environment/data/{evaluator.py, task.py, test_task.py, load.py,
run_evaluation_explicit.py, __init__.py}`) plus dataset payloads into
`tasks/generated/<scope>/<task_id>/`. Each newly installed bundle is
then run through a non-blocking inline smoke gate (layout check +
zero-prediction smoke against the assembled evaluator); failures
sidecar to `<task_dir>/validation_failed.json` and bump the
`smoke_failed` count in the final summary print, but do not undo
install. Bundles are Harbor-ready as soon as this command completes —
there is no separate validation phase.

```bash
python -m openfinai_pipeline construct-benchmark-tasks \
    --scopes alpha_trading \
    --tasks-written-cap 3 \
    --provider openrouter \
    --model openai/gpt-5.4 \
    --verbose
```

### Run all phases end-to-end (Phases 1 to 4)

Triage runs as part of this chain between summarize and dataset
construction — trading-family papers are routed to curated overlays and
skip Phases 2-4 for that paper. The Phase 4 inline smoke gate also runs
here; smoke failures are surfaced via the `smoke_failed` field of the
final summary print but do not abort the run.

```bash
python -m openfinai_pipeline run-all \
    --scopes alpha_trading \
    --provider openrouter \
    --model openai/gpt-5.4 \
    --limit 40 \
    --routes-written-cap 3 \
    --datasets-written-cap 3 \
    --rewards-written-cap 3 \
    --tasks-written-cap 5 \
    --overwrite \
    --verbose
```

## Run an agent against a task

`python -m openfinai_harbor.run_trial` is the entry point for driving an
agent against any task bundle, generated or curated. The agent runs
under `harbor.Trial`: Harbor brings up the task's container, the agent
writes `train.py` to `/workspace`, and the in-container verifier
(`tests/test.sh`) executes it and submits the reward to the host-side
RPC service.

Requires an LLM provider key in `.env` (`OPENROUTER_API_KEY` in the
examples below), an installed task, and Docker running. The first run
pulls the shared sandbox base image, or builds it locally from
`docker/Dockerfile.base` if the pull fails (a few minutes, cached
thereafter). Model strings follow LiteLLM conventions
(`openrouter/<model>`, `anthropic/<model>`, ...); `claude-cli/<model>`
is an optional provider that shells out to a locally installed Claude
Code CLI.

### Smoke-test a generated task

```bash
python -m openfinai_harbor.run_trial \
    --task-dir tasks/generated/<scope>/<task_id> \
    --agent-import-path openfinai_harbor.agents:SingleShotLLMAgent \
    --model openrouter/openai/gpt-5.4-mini \
    --system-prompt-path examples/prompts/financial_simple.txt \
    --max-tokens 8000 \
    --temperature 0.0 \
    --timeout-sec 300 \
    --despawn-grace-min 0
```

Add `--rebuild-image` to force a fresh task image rebuild, or
`--rebuild-base` to rebuild the shared base image after editing
`docker/Dockerfile.base`. Trial output lands under
`data/run_output/examples/<task_name>/<utc_timestamp>/` by default;
override with `--trials-dir`.

### Smoke-test a curated task

The same driver runs against the 10 curated bundles under `tasks/` —
point `--task-dir` at any of them. Trust posture differs per family:
offline bundles use the host-side RPC verifier as oracle; realtime
trading runs the gym loop entirely in-container (no host verifier);
realtime forecasting and polymarket write a deferred ledger that you
resolve later via `python -m openfinai_harbor.resolve_deferred
<trial_dir>`.

```bash
python -m openfinai_harbor.run_trial \
    --task-dir tasks/realtime_stock_forecasting \
    --agent-import-path openfinai_harbor.agents:SingleShotLLMAgent \
    --model openrouter/openai/gpt-5.4-mini \
    --system-prompt-path examples/prompts/financial_simple.txt \
    --despawn-grace-min 0
```

The 10 curated task directories (drop any into `--task-dir`):

**Offline forecasting** — host RPC verifier scores against held-out data
- `tasks/offline_stock_forecasting`
- `tasks/offline_crypto_forecasting`

**Offline trading** — host RPC verifier scores submitted actions
- `tasks/offline_stock_trading`
- `tasks/offline_crypto_trading`

**Realtime trading** — in-container gym loop; bundled runner writes `reward.json` directly
- `tasks/realtime_stock_trading` *(Alpaca data + `internal_paper` bookkeeping)*
- `tasks/realtime_crypto_trading` *(Binance data + `internal_paper` bookkeeping)*
- `tasks/realtime_stock_trading_alpaca_paper` *(Alpaca data + orders routed to Alpaca's hosted paper-trading API)*

**Realtime forecasting** — deferred ledger; `entry_price` captured server-side
- `tasks/realtime_stock_forecasting`
- `tasks/realtime_crypto_forecasting`

**Realtime polymarket** — deferred event-resolution ledger over dynamically discovered binary markets
- `tasks/realtime_polymarket`

To author a new curated bundle, copy an existing one such as
`tasks/offline_crypto_forecasting/` and adjust its `task.toml`
(`[metadata].family` and `[curated].class_name`), `task.py`, and
`environment/data/`.

## Training agents

- **SFT**: trajectory collection, dataset preparation, SkyRL LoRA
  fine-tuning. See [`README.SFT.md`](README.SFT.md).
- **Online RL** (GRPO with SkyRL + Harbor + vLLM): `scripts/rl/train_rl.sh`
  and `config/rl.yaml`; package docs in
  [`src/openfinai_skyrl/README.md`](src/openfinai_skyrl/README.md).
- **Evaluation sweeps**: `scripts/run_harbor.sh <task_id> ...` wraps
  `python -m openfinai_harbor.run_trial` to run several curated tasks in
  sequence with shared model/agent flags.
- **Cluster launchers**: SLURM templates under `scripts/rl/slurm_*.sh` and
  `scripts/sft/slurm_*.sh` (Singularity sandbox via
  `config/harbor_trial_rl_singularity.yaml`), and a Kubernetes recipe
  under [`deploy/k8s/`](deploy/k8s/README.md) (rootless Podman sandbox via
  `config/harbor_trial_rl_podman.yaml`). Both need a few site values
  filled in; see the comments at the top of each file.

## Repository layout

```
config/                  Pipeline, trial, SFT and RL defaults; scope definitions
data/knowledge_base/     Reward bank (versioned)
data/pipeline_output/    Run artefacts — papers, datasets, rewards, logs (ephemeral)
src/openfinai_pipeline/  Pipeline CLI + phases + realtime tasks
src/openfinai_harbor/    Harbor-compatible agent runtime, verifier, environments
src/openfinai_skyrl/     SFT / RL training integration
tasks/                   Curated + generated task packages (+ _template)
examples/                System prompts and a verifier compose overlay
benchmarks/              Latency / data-feed measurement scripts
scripts/sft/             SFT drivers + SLURM launchers
scripts/rl/              RL drivers + SLURM launchers
scripts/maintenance/     Repo upkeep (registry sync, output cleanup, ledger inspection)
deploy/                  Container image, Singularity definition, Kubernetes manifests
docker/                  Shared sandbox base image (Dockerfile.base)
tests/                   Core tests + pipeline contract harnesses (pytest tests -q)
```

## Credentials

All read from the `.env` file at repo root (auto-loaded via `dotenv`).
Only the credentials you actually use are required.

| Variable | Required when |
|---|---|
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` | You invoke the pipeline or an agent with the corresponding provider |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | You run any `realtime_stock_*` or `offline_stock_*` bundle (Alpaca data feed / paper trading) |
| `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`, `FRED_API_KEY` | Phase 2 dataset construction hits the corresponding public data source |
| `FIN_PIPELINE_SUMMARY_EMBED_MODEL`, `FIN_PIPELINE_SUMMARY_RERANKER_MODEL` | You want to override the Phase 1b retrieval defaults |

## Citation

OpenFinGym was accepted to Findings of EMNLP 2026. If you use this
repository in your work, please cite:

```bibtex
@inproceedings{zhang2026openfingym,
  title     = {{OpenFinGym}: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents},
  author    = {Zhang, Kaicheng and Ge, Wen and Jiang, Lei and Yang, Weixin and Langham-Lopez, Jordan and Yu, Jialin and Szpruch, Lukasz and Ni, Hao},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.26350}
}
```

## Contributing

Issues and PRs welcome. See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md)
or open an issue to discuss before large changes.
