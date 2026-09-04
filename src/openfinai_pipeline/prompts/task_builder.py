import json


def build_task_dedup_prompt(
    candidate_name: str,
    candidate_description: str,
    existing_tasks: list[dict[str, str]],
) -> str:
    catalog_json = json.dumps(existing_tasks, ensure_ascii=False)
    return f"""
You are deciding whether a newly extracted task is a duplicate of an already implemented task.

Candidate task:
- name: {candidate_name}
- description: {candidate_description or '(none)'}

Existing implemented tasks (task_id + title + description):
{catalog_json}

Decision policy:
1) is_duplicate=true only when there is clear semantic equivalence (same task, same dataset,
   same objective -- or an obvious rename/variant).
2) Similar but distinct tasks (e.g. different datasets or different objectives) are NOT duplicates.
3) If uncertain, choose is_duplicate=false.
4) matched_task_id must be one of the existing task IDs when is_duplicate=true, empty otherwise.

Output JSON only:
- reasoning: string. Walk through the closest existing tasks and explain whether the candidate is the same task / same dataset / same objective, or a similar-but-distinct task.
- matched_task_id: string (empty when is_duplicate=false)
- is_duplicate: boolean. Must be consistent with `reasoning`.
""".strip()


def task_dedup_schema() -> dict:
    return {
        "title": "TaskDedupResult",
        "description": "Deduplication check determining if a newly extracted task duplicates an existing benchmark task.",
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "matched_task_id": {"type": "string"},
            "is_duplicate": {"type": "boolean"},
        },
        "required": ["reasoning", "matched_task_id", "is_duplicate"],
        "additionalProperties": False,
    }


def build_implementation_prompt(
    task_info: str,
    base_task_source: str,
    example_task_source: str,
    paper_context: str,
    *,
    template_reference: str = "",
    interaction_model: str = "forecasting",
) -> str:
    if interaction_model not in ("forecasting", "generative"):
        raise ValueError(
            "build_implementation_prompt only supports interaction_model "
            "in {'forecasting','generative'}; the auto-pipeline does not "
            f"emit trading/gym/realtime tasks. Got: {interaction_model!r}"
        )
    # paper_context is sourced from task_builder._read_paper_context, which
    # already strips low-value sections and caps at 60K — this slice is a
    # defensive guard, not an active trim.
    paper_block = paper_context.strip()[:60_000] or "(paper text unavailable)"
    example_block = example_task_source.strip()[:20000] or "(no example available)"

    template_section = ""
    if template_reference:
        template_section = (
            "\n--- Template reference (use as structural guide) ---\n"
            "Below is a partially-filled template showing the expected file structure.\n"
            "Use this as your starting point. You MUST keep the same class structure,\n"
            "method signatures, imports, and metadata pattern. You MAY add helper\n"
            "methods and additional imports as needed.\n"
            "Replace all lines containing '<<< LLM:' with your implementation.\n\n"
            f"{template_reference}\n\n"
        )

    # Pick the base-class guidance string based on the interaction model
    if interaction_model == "forecasting":
        base_class_guidance = (
            "- task.py: A class inheriting ForecastingTask (from\n"
            "  openfinai_pipeline.benchmark.contracts). Use the template\n"
            "  reference below as your starting point — fill the LLM-marked\n"
            "  placeholders (observation/action spaces, get_features) and\n"
            "  customize the metadata fields; carry the rest of the template\n"
            "  through verbatim (notably __init__, imports, and the\n"
            "  inherited-method comments). Do NOT redefine load_data,\n"
            "  get_ground_truth, reset, step, or evaluate — they are\n"
            "  inherited by design. Train labels are reachable via the\n"
            "  inherited get_train_features() / get_train_ground_truth()\n"
            "  accessors; get_features() returns self._data['test']['features']\n"
            "  (the test inputs the agent predicts on). Set\n"
            "  metadata.interaction_model='forecasting'.\n"
        )
    else:  # generative
        base_class_guidance = (
            "- task.py: A class inheriting GenerativeTask (from\n"
            "  openfinai_pipeline.benchmark.contracts). Use the template\n"
            "  reference below as your starting point — fill the LLM-marked\n"
            "  placeholders and customize the metadata fields; carry the\n"
            "  rest of the template through verbatim (notably __init__,\n"
            "  imports, and the inherited-method comments). Do NOT redefine\n"
            "  load_data or override reset/step/evaluate — they are\n"
            "  inherited by design.\n"
            "  Two sub-shapes:\n"
            "    * Conditional generative — self._data is the B-shape bundle\n"
            "      {'train': {features, ground_truth}, 'test': {features,\n"
            "      ground_truth=None}, ...}. The held-out test reference is\n"
            "      verifier-only; the inherited get_reference_data() raises\n"
            "      PermissionError. Use get_train_features() and\n"
            "      get_train_reference_data() to fit a generator.\n"
            "    * Unconditional generative (split_policy='no_split') —\n"
            "      self._data has only a 'reference' key. Override\n"
            "      get_reference_data() to return self._data['reference'].\n"
            "  Set metadata.interaction_model='generative'.\n"
        )

    return (
        "You are implementing a financial time series task package in Python.\n"
        "Generate `task.py` plus three short prose blocks that the task pipeline\n"
        "will splice into a deterministically rendered `instruction.md`. The pipeline\n"
        "renders the file paths, agent-task interface, evaluator contract, resolved-\n"
        "metric list, and output filenames itself — you must NOT rewrite or restate\n"
        "those. Only the three named prose blocks below are yours to author.\n"
        "Return a JSON object with keys 'task_py', 'context_prose', 'objective_prose',\n"
        "and 'schema_description'. The `task_py` key must contain the full source of\n"
        "`task.py` as a string. The three prose keys must be plain markdown strings\n"
        "(no fences, no headings — the template wraps them under section headings).\n\n"
        f"INTERACTION MODEL: {interaction_model}\n"
        "  * 'forecasting': batch prediction task. Inherit ForecastingTask;\n"
        "    return self._data['test']['features'] from get_features();\n"
        "    train (features + labels) is reachable via the inherited\n"
        "    get_train_features() / get_train_ground_truth() accessors.\n"
        "    Test ground_truth is held out (verifier-only); do NOT expose it.\n"
        "  * 'generative': batch generation task. Inherit GenerativeTask.\n"
        "    Conditional case (B-shape): same accessor pattern as forecasting.\n"
        "    Unconditional case ({'reference': ...}): override\n"
        "    get_reference_data() to return self._data['reference'].\n\n"
        "Requirements:\n"
        f"{base_class_guidance}"
        "  CLASS NAME: the deterministic Phase 4 test harness imports the task class\n"
        "  by a canonical name derived from the task_id (PascalCase of `task_id`,\n"
        "  e.g. `task_1` -> `Task1`). Use exactly the class name shown in the\n"
        "  template reference -- do NOT rename it.\n"
        "  DATA LOADING (HARD RULE — strict static check enforces this):\n"
        "    * `load_data()` is provided by `BaseTask` (framework code). It imports\n"
        "      the bundled per-task `load.py` (installed at `/data/load.py`) via\n"
        "      `importlib.util` and caches the result on `self._data`. The\n"
        "      install-time slicer materializes `dataset.h5` and replaces\n"
        "      `load.py` with a trivial reader; the cached dict is the\n"
        "      B-shape bundle ({'train': {...}, 'test': {features only,\n"
        "      ground_truth=None}, 'split_policy': ..., 'split_metadata': ...}\n"
        "      OR the reference-shape {'reference': ndarray, 'split_policy':\n"
        "      'no_split', 'split_metadata': None}).\n"
        "    * DO NOT define `load_data` in your `task.py`. DO NOT import\n"
        "      `importlib`, do NOT use `compile`/`exec`, do NOT parse files inline.\n"
        "      The static checker blocks `importlib`, `eval`, and `exec` in user\n"
        "      code; any reimplementation here will be rejected.\n"
        "    * Test ground_truth is intentionally None inside `self._data` —\n"
        "      it is held out from agent code. The verifier reads it from\n"
        "      `/eval-data/test_ground_truth.h5` via the assembled evaluator.\n"
        "    * Use the downloaded dataset description, payload preview, and labeler\n"
        "      notes as context for the SHAPE/SEMANTICS of `self._data['train']`,\n"
        "      `self._data['test']['features']`, etc.\n"
        "      — but do NOT reimplement the file parsing.\n"
        "- context_prose: 2-4 sentences naming the paper task and the dataset(s) it\n"
        "  uses. State the prediction/generation objective in one sentence at most;\n"
        "  the next block (`objective_prose`) gives the formal one-line target.\n"
        "  Plain markdown — no headings, no list formatting.\n"
        "- objective_prose: 1-2 sentences naming exactly what each entry of the\n"
        "  agent's output array represents (e.g. \"per-minute log return at the\n"
        "  corresponding test timestamp\", \"next-day close price for each test row\",\n"
        "  \"a generated path matched to the test conditioning input at index i\").\n"
        "  This is the single most important sentence in the document — be specific\n"
        "  about units, horizon, and indexing. Do NOT mention output filenames or\n"
        "  metrics; the template emits those.\n"
        "- schema_description: a markdown bullet list describing the columns /\n"
        "  features the agent will see in `train.features` and `test.features`.\n"
        "  Derive this from the dataset preview + downloaded description above.\n"
        "  One bullet per feature group is fine; do NOT enumerate every column if\n"
        "  the data is wide. Stay factual — no claims about what the model should\n"
        "  do with each feature.\n\n"
        "Write production-quality code: no placeholders, no TODO comments.\n\n"
        "What the pipeline renders for you (so you can avoid duplicating it in\n"
        "prose): the data layout table for `/data/`, the `Agent task interface`\n"
        "section, the `Evaluation` contract, the per-metric `Metrics` block (with\n"
        "argument signatures and prediction-array semantics for every metric the\n"
        "evaluator actually scores), the canonical output filenames, and the\n"
        "Submission helper. If you mention any of those in the prose blocks, you\n"
        "will only create drift between the LLM-authored prose and the verifier's\n"
        "ground truth. Stick to paper-and-data prose.\n\n"
        f"{template_section}"
        "--- BaseTask interface ---\n"
        f"{base_task_source}\n\n"
        "--- Example task (for reference) ---\n"
        f"{example_block}\n\n"
        "--- Task specification ---\n"
        f"{task_info}\n\n"
        "--- Paper context ---\n"
        f"{paper_block}"
    )


def task_generation_schema() -> dict:
    return {
        "title": "TaskImplementation",
        "description": (
            "Task implementation package: task.py source plus three "
            "instruction.md prose slots (context_prose, objective_prose, "
            "schema_description). The deterministic instruction.md sections "
            "(file paths, agent task interface, evaluator contract, metric "
            "list, output filenames) are rendered server-side from the "
            "manifest and are NOT part of the LLM output."
        ),
        "type": "object",
        "properties": {
            "task_py": {"type": "string"},
            "context_prose": {"type": "string"},
            "objective_prose": {"type": "string"},
            "schema_description": {"type": "string"},
        },
        "required": [
            "task_py",
            "context_prose",
            "objective_prose",
            "schema_description",
        ],
        "additionalProperties": False,
    }


def build_fix_errors_prompt(
    task_info: str,
    current_code_summary: str,
    error_output: str,
    base_task_source: str = "",
    paper_context: str = "",
    *,
    template_reference: str = "",
) -> str:
    error_block = error_output.strip()[:32000] or "(no error output)"
    base_block = base_task_source.strip() if base_task_source else ""
    paper_block = paper_context.strip()[:60000] if paper_context else ""

    parts = [
        "The previously generated task code failed tests.\n"
        "Fix ALL errors shown in the pytest output below.\n"
        'Return JSON with one or more of: "task_py", "context_prose",\n'
        '"objective_prose", "schema_description".\n'
        "ONLY include fields you actually changed — omit unchanged ones to save tokens.\n"
        "If your fix to one field changes its content in a way the others depend on,\n"
        "return all affected fields.\n"
        "Do not truncate any field you return. Write the complete corrected content.\n\n"
        "The failing tests are framework checks of the task contract — they assert\n"
        "the public surface of your task.py. The deterministic sections of\n"
        "instruction.md (file paths, agent interface, evaluator contract, metrics,\n"
        "output filenames) are rendered server-side from the manifest and are NOT\n"
        "in the LLM output, so they cannot be the source of test failures —\n"
        "the fix lives in task.py or in one of the three prose slots.\n\n"
        "Do NOT modify `load_data()` — it delegates to the bundled `load.py` and the\n"
        "framework owns its body. Test failures that mention `load_data` are about\n"
        "the contract identity check (your subclass must inherit it unchanged); the\n"
        "fix is to remove your override.\n"
        "The framework checks exercise the public surface of your task class\n"
        "(metadata, the get_* accessors, and predict_and_evaluate /\n"
        "generate_and_evaluate). The original implementation prompt and template\n"
        "reference define the exact contract — re-read them before editing.\n"
        "When fixing failures, change only what the error points at; do not\n"
        "delete or rewrite unrelated parts of the file (imports, __init__,\n"
        "metadata, methods documented as inherited in the template). Removing\n"
        "such code to silence one error often re-breaks something else on the\n"
        "next round.\n\n"
    ]

    if template_reference:
        parts.append(
            "--- Template reference (expected structure) ---\n"
            "Your fixed code MUST keep the same class structure and method signatures.\n"
            "In particular: do NOT switch the base class between BaseTask,\n"
            "ForecastingTask, GenerativeTask, or TradingTask -- keep whichever the template uses.\n\n"
            f"{template_reference}\n\n"
        )

    if base_block:
        parts.append(
            "--- BaseTask interface (your code MUST conform to this) ---\n"
            f"{base_block}\n\n"
        )

    parts.append(
        "--- Task specification ---\n"
        f"{task_info}\n\n"
        "--- Errors / test output ---\n"
        f"{error_block}\n\n"
        "--- Current code ---\n"
        f"{current_code_summary}"
    )

    if paper_block:
        parts.append(
            "\n\n--- Paper context (for domain details) ---\n" f"{paper_block}"
        )

    return "".join(parts)


def task_fix_schema() -> dict:
    """Schema for fix responses — all keys optional (only changed slots needed)."""
    return {
        "title": "TaskFixPatch",
        "description": (
            "Partial fix output for correcting failing task code. The "
            "deterministic instruction.md sections are server-rendered "
            "from the manifest; the LLM only authors task.py + three "
            "instruction.md prose slots, and only the changed slots "
            "need to be returned here."
        ),
        "type": "object",
        "properties": {
            "task_py": {"type": "string"},
            "context_prose": {"type": "string"},
            "objective_prose": {"type": "string"},
            "schema_description": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }
