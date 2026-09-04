# [Task Title]

## Context

Describe the financial domain context and the agent's role.

## Objective

Clearly define what the agent must produce.

## Data (pre-installed in the container)

| File | Description |
|------|-------------|
| `/data/dataset.h5` | Train features / targets and test features (no test target). |
| `/data/harbor_submit.py` | Submit helper: `submit_and_persist(predictions)` sends predictions to the verifier and writes `reward.json`. |

## Instructions

1. Write `/workspace/train.py`.
2. Load the data through `/data/task.py` (or `/data/dataset.h5` directly).
3. Fit a model on the train split and predict the test split.
4. Call `harbor_submit.submit_and_persist(...)` with your predictions.

## Evaluation

Describe how the agent's output will be scored (headline metric, aggregation).

## Response Format

Use `<python>...</python>` tags for executable code and `<answer>...</answer>` for your final summary.
